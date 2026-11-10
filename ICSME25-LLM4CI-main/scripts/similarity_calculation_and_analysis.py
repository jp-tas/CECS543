import pandas as pd
import yaml
import re
import os
from sentence_transformers import SentenceTransformer, util
from joblib import Parallel, delayed
from tqdm import tqdm
from sacrebleu.metrics import CHRF
from ignite.metrics import RougeL
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from zss import simple_distance, Node
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

SentenceTransformerModelPath = 'all-MiniLM-L6-v2'
                    
def remove_comments(text):
    if not isinstance(text, str):
        return ''
    
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        line_no_inline = re.sub(r'(?<!["\'])#.*', '', line)
        if line_no_inline.strip() != '':
            lines.append(line_no_inline.rstrip())
    return '\n'.join(lines)

def yaml_line_tokenizer(yaml_text):
    return [line.strip() for line in yaml_text.splitlines() if line.strip()]

# Load model on CPU only
model = SentenceTransformer(SentenceTransformerModelPath, device='cpu')
def compute_consie_similarity(code, llm_output):
    code_embedding = model.encode(str(code), convert_to_tensor=True, device='cpu')
    llm_output_embedding = model.encode(str(llm_output), convert_to_tensor=True, device='cpu')
    similarity = util.pytorch_cos_sim(code_embedding, llm_output_embedding)
    return similarity.item()

def compute_euclidean_distance(code, llm_output):
    code_embedding = model.encode(str(code), convert_to_tensor=True, device='cpu')
    llm_output_embedding = model.encode(str(llm_output), convert_to_tensor=True, device='cpu')
    distance = torch.norm(code_embedding - llm_output_embedding, p=2)
    return distance.item()

chrf_metric_sacrebleu = CHRF(lowercase=True, whitespace=True, eps_smoothing=True)

rouge_metric_ignite = RougeL(multiref="best")
def compute_rouge_ignite(code, llm_output):
    candidate = llm_output.split()
    references = [code.split()]
    rouge_metric_ignite.reset()
    rouge_metric_ignite.update(([candidate], [references]))
    scores = rouge_metric_ignite.compute()
    return max(scores['Rouge-L-P'], scores['Rouge-L-R'], scores['Rouge-L-F'])

# ROUGE-L and chrF
def compute_metrics_yaml_lines(code, llm_output):
    code_tokens = yaml_line_tokenizer(code)
    llm_output_tokens = yaml_line_tokenizer(llm_output)
    code_str = ' '.join(code_tokens)
    llm_output_str = ' '.join(llm_output_tokens)

    if not code_str.strip() or not llm_output_str.strip():
        return 0.0, 0.0, 0.0, 0.0
    rouge_l = compute_rouge_ignite(code_str, llm_output_str)
    chrf = chrf_metric_sacrebleu.sentence_score(llm_output_str, [code_str]).score
    return rouge_l, chrf

def yaml_to_tree(yaml_text):
    def build_tree(obj, name="root"):
        node = Node(str(name))
        if isinstance(obj, dict):
            for k, v in obj.items():
                node.addkid(build_tree(v, k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                node.addkid(build_tree(v, f'item_{i}'))
        elif obj is not None:
            node.addkid(Node(str(obj)))
        return node

    try:
        data = yaml.safe_load(yaml_text)
        if data is None:
            return Node("root")
        return build_tree(data)
    except Exception:
        return Node("root")
    
def normalized_tree_edit_distance(yaml1, yaml2):
    tree1 = yaml_to_tree(yaml1)
    tree2 = yaml_to_tree(yaml2)
    ted = simple_distance(tree1, tree2)
    max_size = max(len(list(Node.get_children(tree1))) + 1, len(list(Node.get_children(tree2))) + 1)
    return 1.0 if max_size == 0 else 1.0 - (ted / max_size)

def normalized_tree_edit_distance(yaml1, yaml2):
    tree1 = yaml_to_tree(yaml1)
    tree2 = yaml_to_tree(yaml2)
    
    def count_nodes(node):
        stack = [node]
        count = 0
        while stack:
            current = stack.pop()
            count += 1
            stack.extend(Node.get_children(current))
        return count
    
    # Get sizes including all nodes
    size1 = count_nodes(tree1)
    size2 = count_nodes(tree2)
    
    # Compute edit distance
    ted = simple_distance(tree1, tree2)
    
    # Handle empty trees
    if size1 == 0 and size2 == 0:
        return 1.0
    elif size1 == 0 or size2 == 0:
        return 0.0
        
    # Normalize by maximum possible edit distance
    max_ops = max(size1, size2)
    similarity = 1.0 - (ted / (2 * max_ops))
    
    return max(0.0, min(1.0, similarity))  # Make result is between 0 and 1

def process_in_batches(df, compute_func, desc, batch_size=40):
    results = []
    total_rows = len(df)
    
    # Process in batches
    for start_idx in tqdm(range(0, total_rows, batch_size), desc=f"{desc} batches"):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]
        
        try:
            # Process single batch without parallelization
            batch_results = [
                compute_func(row['Code'], row['LLM_YML_Output'])
                for _, row in batch_df.iterrows()
            ]
            results.extend(batch_results)
            
        except Exception as e:
            print(f"Error in batch {start_idx//batch_size}: {str(e)}")
            # Fill failed batch with appropriate zeros
            if desc == "Metrics":
                results.extend([(0.0, 0.0, 0.0)] * len(batch_df))
            else:
                results.extend([0.0] * len(batch_df))
            
        # Force garbage collection after each batch
        import gc
        gc.collect()
        
    return results

def calculate_all_similarity_metrics(input_folder, output_folder, ci_service_name):
    LLM_Output_Files = [
        f'{ci_service_name}_gpt-4o_output.csv',
        f'{ci_service_name}_gpt-4.1_output.csv',
        f'{ci_service_name}_gemma3-12b_output.csv',
        f'{ci_service_name}_llama3.1-8b_output.csv',
        f'{ci_service_name}_codegemma-7b_output.csv',
        f'{ci_service_name}_codellama-7b_output.csv',
    ]

    base_columns = ["ID", "URL", "Title", "Description", "Code"]
    all_dfs = []

    for llm_output in LLM_Output_Files:
        print(f"##### Processing {llm_output} #####")
        df = pd.read_csv(os.path.join(input_folder, llm_output), encoding='utf-8')
        df = df.drop_duplicates(subset=base_columns)

        num_jobs = 1

        if 'Code' not in df.columns or 'LLM_YML_Output' not in df.columns:
            raise ValueError("CSV must contain 'Code' and 'LLM_YML_Output' columns.")

        # Handle NaN values before processing
        df['Code'] = df['Code'].fillna('')
        df['LLM_YML_Output'] = df['LLM_YML_Output'].fillna('')

        df['Code'] = df['Code'].apply(remove_comments)
        df['LLM_YML_Output'] = df['LLM_YML_Output'].apply(remove_comments)

        print(f"##### Computing similarity and metrics using {num_jobs} threads on CPU for {llm_output}.")

        # Tokenize YAML lines for both reference and hypothesis
        df['Code_tokens'] = df['Code'].apply(yaml_line_tokenizer)
        df['LLM_YML_Output_tokens'] = df['LLM_YML_Output'].apply(yaml_line_tokenizer)

        # Cosine similarity
        try:
            cosine_results = process_in_batches(
                df, 
                compute_consie_similarity, 
                "Cosine",
                batch_size=30
            )
        except Exception as e:
            print(f"Error computing cosine similarity: {str(e)}")
            cosine_results = [''] * len(df)

        # Euclidean distance
        euclidean_results = process_in_batches(
            df, 
            compute_euclidean_distance, 
            "Euclidean",
            batch_size=40
        )

        # ROUGE-L and chrF
        metrics_results = process_in_batches(
            df, 
            compute_metrics_yaml_lines, 
            "Metrics",
            batch_size=40
        )
        rouge_scores, chrf_scores = zip(*metrics_results)

        # Tree Edit Distance
        tree_edit_distances = Parallel(n_jobs=num_jobs, backend='loky')(
            delayed(normalized_tree_edit_distance)(row['Code'], row['LLM_YML_Output'])
            for _, row in tqdm(df.iterrows(), total=len(df), desc="TreeEditDist")
        )

        model_name = llm_output.replace(f'{ci_service_name}_', '').replace('_output.csv', '')

        df_model = df[base_columns].copy()
        df_model[f'LLM_YML_Output ({model_name})'] = df['LLM_YML_Output']
        if 'Time' in df.columns:
            df_model[f'Time ({model_name})'] = df['Time']
        df_model[f'CosineSimilarity ({model_name})'] = cosine_results
        df_model[f'EuclideanDistance ({model_name})'] = [1 / (1 + val) for val in euclidean_results] # Normalize to similarity [0, 1]
        df_model[f'ROUGE-L ({model_name})'] = rouge_scores
        df_model[f'chrF ({model_name})'] = chrf_scores
        df_model[f'chrF ({model_name})'] = [val / 100.0 for val in chrf_scores] # Normalize to similarity [0, 1]
        df_model[f'TreeEditDistance ({model_name})'] = tree_edit_distances

        for col in base_columns:
            df_model[col] = df_model[col].astype(str).str.strip()
        df_model = df_model.drop_duplicates(subset=base_columns)
        
        all_dfs.append(df_model)

    final_df = all_dfs[0]
    for df_next in all_dfs[1:]:
        df_next_reduced = df_next.drop(columns=["URL", "Title", "Description", "Code"])
        final_df = final_df.merge(df_next_reduced, on='ID', how='inner')
    
    for col in ["URL", "Title", "Description", "Code"]:
        final_df[col] = all_dfs[0][col]

    final_df.to_csv(f"{output_folder}/{ci_service_name}_Similarity_Scores_Six_LLMs.csv", index=False)

def plot_all_similarity_metrics(input_folder, output_folder, ci_service_name):
    # Read the results
    df = pd.read_csv(f"{input_folder}/{ci_service_name}_Similarity_Scores_Six_LLMs.csv")

    # Get model names and metric names
    models = [
                'gpt-4o',
                'gpt-4.1',
                'llama3.1-8b',
                'gemma3-12b',
                'codellama-7b',
                'codegemma-7b',
             ]
    
    metrics = [
               'CosineSimilarity',
               'EuclideanDistance',
               'TreeEditDistance',
               'ROUGE-L',
               'chrF',
              ]

    # Prepare data for plotting
    plot_data = []
    for metric in metrics:
        for model in models:
            col_name = f"{metric} ({model})"
            values = df[col_name]
            plot_data.extend([(metric, model, v) for v in values])

    plot_df = pd.DataFrame(plot_data, columns=['Metric', 'Model', 'Score'])

    # Create the plot with a custom palette
    plt.figure(figsize=(14, 4))

    palette = sns.color_palette("muted")

    # Create boxplot with custom palette
    ax = sns.boxplot(data=plot_df, x='Metric', y='Score', hue='Model', palette=palette)

    # Add boxes around each metric group
    for i in range(len(metrics)):
        x_start = i - 0.4  # Adjust these values based on your plot
        x_end = i + 0.4
        y_min = ax.get_ylim()[0]
        y_max = ax.get_ylim()[1]
        
        # Draw rectangle around metric group
        rect = plt.Rectangle((x_start, y_min), 
                            x_end - x_start, 
                            y_max - y_min,
                            fill=False,
                            color='gray',
                            linestyle='--',
                            alpha=0.3)
        ax.add_patch(rect)

    # Add mean markers and text for each model within each metric
    mean_color = "#0D0101"  # bright red for visibility
    for metric in metrics:
        for i, model in enumerate(models):
            # Get data for specific metric and model
            metric_model_data = plot_df[(plot_df['Metric'] == metric) & 
                                    (plot_df['Model'] == model)]['Score']
            mean_val = metric_model_data.mean()
            
            # Calculate x position for the current box (offset for each model)
            x_pos = metrics.index(metric) + (i - 1) * 0.27
            
    # Customize the plot
    #plt.title('Similarity Metrics Comparison Across Models', pad=20, fontsize=15)
    plt.xticks(ha='center', fontsize=13)  # Increased tick label size
    plt.yticks(fontsize=13)  # Added y-axis tick label size
    plt.xlabel('Similarity Metric', labelpad=10, fontsize=15)
    plt.ylabel('Similarity Score', labelpad=10, fontsize=15)

    # Style the grid
    plt.grid(True, alpha=0.2, color='gray', linestyle='--')

    # Update legend with mean marker and larger font
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles,
            labels, title='Model', 
            title_fontsize=12,
            fontsize=11,        
            bbox_to_anchor=(1.0, 1), 
            loc='upper left')

    ax.set_axisbelow(True)

    plt.savefig(f"{output_folder}/{ci_service_name}_Similarity_All_LLMs_boxplot.pdf", dpi=300, bbox_inches='tight')

def correlations_analysis(input_folder, ci_service_name):
    file_path = f"{input_folder}/{ci_service_name}_Similarity_Scores_Six_LLMs.csv"

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Compute input length (word count of the 'Description' field)
    df['InputLength'] = df['Description'].astype(str).apply(lambda x: len(x.split()))

    # Define similarity metrics to test against input length
    metrics = {
        'CosineSimilarity (gpt-4o)': 'Cosine Similarity',
        'ROUGE-L (gpt-4o)': 'ROUGE-L',
        'TreeEditDistance (gpt-4o)': 'Tree Edit Distance',
        'EuclideanDistance (gpt-4o)': 'Euclidean Distance'
    }

    # Compute Spearman correlations
    correlation_results = []
    raw_p_values = []

    for metric_col, metric_name in metrics.items():
        df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
        valid_data = df[['InputLength', metric_col]].dropna()
        rho, p = spearmanr(valid_data['InputLength'], valid_data[metric_col])
        # Effect size: absolute value of rho and coefficient of determination (rho^2)
        abs_rho = abs(rho)
        rho_squared = rho ** 2
        correlation_results.append({
            'Metric': metric_name,
            'Spearman ρ': rho,
            'p-value': p,
            'Effect Size |ρ|': abs_rho,
            'Effect Size ρ²': rho_squared,
            'Rows Used': len(valid_data)
        })
        raw_p_values.append(p)

    # Apply Holm–Bonferroni correction
    rejected, corrected_pvals, _, _ = multipletests(raw_p_values, alpha=0.05, method='holm')

    # Attach corrected p-values to results
    for i in range(len(correlation_results)):
        correlation_results[i]['Holm–Bonferroni p-value'] = corrected_pvals[i]
        correlation_results[i]['Significant (α=0.05)'] = rejected[i]

    # Create and return results dataframe
    results_df = pd.DataFrame(correlation_results)

    # Display results
    print("\nSpearman Correlation: Input Length vs Similarity Metrics (GPT-4o) with Holm–Bonferroni Correction")
    print(results_df)
    return results_df

def wilcoxon_cosine_similarity_comparison(input_folder, ci_service_name):
    file_path = f"{input_folder}/{ci_service_name}_Similarity_Scores_Six_LLMs.csv"
    df = pd.read_csv(file_path)
    
    models = ['gpt-4o', 'gpt-4.1', 'gemma3-12b', 'llama3.1-8b', 'codegemma-7b', 'codellama-7b']
    base_model = 'gpt-4o'

    # Ensure cosine similarity columns are numeric
    for model in models:
        col = f"CosineSimilarity ({model})"
        df[col] = pd.to_numeric(df[col], errors='coerce')
    results = []
    raw_p_values = []
    for model in models:
        if model == base_model:
            continue
        col_base = f"CosineSimilarity ({base_model})"
        col_other = f"CosineSimilarity ({model})"
        paired_data = df[[col_base, col_other]].dropna()
        if len(paired_data) > 10:
            stat, p = wilcoxon(paired_data[col_base], paired_data[col_other])
            results.append({
                "Model A": base_model,
                "Model B": model,
                "Wilcoxon Statistic": stat,
                "Raw p-value": p,
                "Rows Used": len(paired_data)
            })
            raw_p_values.append(p)
    
    # Apply Holm–Bonferroni correction
    if raw_p_values:
        rejected, corrected_pvals, _, _ = multipletests(raw_p_values, alpha=0.05, method='holm')
        for i in range(len(results)):
            results[i]["Holm–Bonferroni p-value"] = corrected_pvals[i]
            results[i]["Significant (α=0.05)"] = rejected[i]
    
    results_df = pd.DataFrame(results)
    print("\nWilcoxon Test Results with Holm–Bonferroni Correction (Cosine Similarity)")
    print(results_df)
    return results_df

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    input_folder = os.path.join(script_dir, '../data')
    output_folder = os.path.join(script_dir, '../results')
    ci_service_name = "GitHubActions"

    # Set pandas display options
    pd.set_option('display.max_columns', None)

    # Calculate similarity metrics for six LLMs
    calculate_all_similarity_metrics(input_folder, output_folder, ci_service_name)
    
    # Plot all similarity metrics using boxplots
    plot_all_similarity_metrics(output_folder, output_folder, ci_service_name)

    # Run the correlations analysis
    correlations_analysis(output_folder, ci_service_name)
    
    # Run the Wilcoxon test analysis across cosine similarity of all LLMs
    wilcoxon_cosine_similarity_comparison(output_folder, ci_service_name)
