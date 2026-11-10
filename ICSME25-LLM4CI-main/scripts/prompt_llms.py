import pandas as pd
import re
import os
import time
import time
from ollama import chat, ChatResponse
from openai import OpenAI
import warnings
warnings.filterwarnings("ignore")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def Ask_LLM(service_name, model_name, context_len, input_text):
    predict_len = 4096

    messages = [
        {'role': 'system', 'content': "You are a highly skilled software developer and DevOps engineer"},
        {'role': 'user', 'content': f"Given the description below, generate a minimal and single valid {service_name} YAML configuration. Do not provide any reasoning. Only output the single YAML configuration, with no comments, markdown, or extra formatting. The YAML must match the described task exactly. Do not add any unneeded directives unless told.\n\n{input_text}"},
    ]

    start = time.time()

    if model_name.lower().startswith("gpt"): # OpenAI models
        response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
            model=model_name,
            temperature=0,
            messages=messages
        )
        output = response.choices[0].message.content or ""
    
    else: # Ollama models
        response: ChatResponse = chat(
            model=model_name,
            messages=messages,
            options={"temperature": 0, 'num_ctx': int(context_len), 'num_predict': predict_len}#, 'num_thread': 2}
        )
        output = response.message.content or ""
    
    elapsed = round(time.time() - start, 2)
    return output, elapsed

def extract_yaml_only(text):
    text = re.sub(r"^```(yml|yaml)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```.*", "", text.strip(), flags=re.DOTALL)
    return text

if __name__ == "__main__":
    # Set model and paths
    ci_service_docs = "GitHubActions_Docs"
    input_output_folder = os.path.join(os.path.dirname(__file__), '../data')

    model_contexts_df = pd.read_csv(f'{input_output_folder}/models_contexts.csv')
    models_contexts = model_contexts_df.set_index('model_name')['context_length'].to_dict()
    
    for model_name, context_len in models_contexts.items():
        input_csv_path = f'{input_output_folder}/{ci_service_docs}.csv'
        output_csv_path = f'{input_output_folder}/{ci_service_docs}_{model_name.replace(":", "-")}_output.csv'

        df = pd.read_csv(input_csv_path)
        if 'Title' not in df.columns or 'Description' not in df.columns:
            raise ValueError("Input CSV must contain 'Title' and 'Description' columns.")

        inputs = (df['Title'] + ': ' + df['Description']).tolist()

        file_exists = os.path.isfile(output_csv_path)
        existing_ids = set()
        if file_exists:
            existing_df = pd.read_csv(output_csv_path, usecols=["ID"])
            existing_ids = set(existing_df["ID"].astype(str))

        for i, input_text in enumerate(inputs):
            id = str(df.iloc[i]['ID'])

            if id in existing_ids:
                print(f"  Skipping ID {id}")
                continue

            print(f"  Processing ID {id}: ", end='')

            raw_output, elapsed_time = Ask_LLM(ci_service_docs, model_name, context_len, input_text)
            yaml_output = extract_yaml_only(raw_output) or ""

            if yaml_output:
                print("✅")
            else:
                print("⚠️")

            row = {
                "ID": id,
                "URL": df.iloc[i]['URL'],
                "Title": df.iloc[i]['Title'],
                "Description": df.iloc[i]['Description'],
                "Code": df.iloc[i]['Code'].strip(),
                "LLM_YML_Output": yaml_output,
                "Time": elapsed_time
            }

            pd.DataFrame([row]).to_csv(
                output_csv_path,
                mode='a',
                index=False,
                header=not file_exists,
                quoting=1,
                encoding='utf-8',
                lineterminator='\n'
            )
            file_exists = True  # Write header only once
            existing_ids.add(id)

            time.sleep(1)

        print("✅ All rows processed successfully.")
