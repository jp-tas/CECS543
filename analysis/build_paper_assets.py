#!/usr/bin/env python3
"""Build manuscript tables directly from the exported analysis results."""
from pathlib import Path
import json
import shutil

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['pdf.fonttype'] = 42

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'analysis/results'
TABLES = ROOT / 'paper/tables'
TABLES.mkdir(exist_ok=True)
SUMMARY = json.loads((RESULTS / 'summary.json').read_text())
MODELS = SUMMARY['models']
LABELS = dict(zip(MODELS, ['GPT-4o', 'GPT-4.1', 'Gemma3--12B', 'Llama3.1--8B', 'CodeGemma--7B', 'Code Llama--7B']))


def table(name, caption, label, columns, headers, rows, note=''):
    text = [r'\begin{table}[H]', r'\centering\small',
            r'\caption{' + caption + '}', r'\label{tab:' + label + '}',
            r'\begin{tabular}{@{}' + columns + '@{}}', r'\toprule',
            ' & '.join(headers) + r'\\', r'\midrule']
    text += [' & '.join(str(v) for v in row) + r'\\' for row in rows]
    text += [r'\bottomrule', r'\end{tabular}']
    if note:
        text += [r'\par\vspace{0.3em}{\footnotesize ' + note + '}']
    text += [r'\end{table}', '']
    (TABLES / (name + '.tex')).write_text('\n'.join(text))


def pformat(p):
    if p >= .001:
        return f'${p:.3f}$'
    power = int(f'{p:.2e}'.split('e')[1])
    mantissa = p / 10 ** power
    return rf'${mantissa:.2f}\times 10^{{{power}}}$'


means = pd.read_csv(RESULTS / 'similarity_summary.csv')
table('data_profile', 'Profile of the supplied documentation benchmark and its archived scores.', 'data', 'lr',
      ['Property', 'Count'], [
          ['Example identifiers', '1,318'], ['Archived model outputs', '7,908'],
          ['Distinct raw description--reference pairs', '690'],
          ['Repeated pair rows beyond one per group', '628'],
          ['Distinct raw reference strings', '603'], ['Distinct source pages (URLs)', '186'],
          ['Distinct pairs in transformed score table', '691']],
      'Raw pair grouping uses exact Description and Code fields in the documentation table. The separately transformed score table has a different pair count.')
rows = []
for model in MODELS:
    frame = means[means.model == model].set_index('metric')
    rows.append([LABELS[model]] + [f'{frame.loc[m,"row_mean"]:.3f}' for m in
                                 ['CosineSimilarity', 'EuclideanDistance', 'TreeEditDistance', 'ROUGE-L', 'chrF']])
table('model_scores', 'Recomputed means of the released score columns; 1,318 observations per model.', 'scores', 'lrrrrr',
      ['Model', 'Cosine', 'Euclidean sim.', 'Tree sim.', 'ROUGE-L', 'chrF'], rows,
      'Similarity scores are reused from the release; summaries are newly computed. Tree scores retain the original parser failure behavior.')
wil = pd.read_csv(RESULTS / 'wilcoxon_cosine.csv')
rows = []
for model in MODELS[1:]:
    row = wil[(wil.unit == 'row') & (wil.comparison == f'gpt-4o vs {model}')].iloc[0]
    rows.append([LABELS[model], f'{row.mean_paired_difference:+.4f}', pformat(row.p_raw), pformat(row.p_holm)])
table('wilcoxon', 'Two-sided paired Wilcoxon tests: GPT-4o cosine scores versus each alternative; Holm family of five comparisons.',
      'wilcoxon', 'lrrr', ['Alternative', 'Mean difference', 'Raw $p$', 'Adjusted $p$'], rows,
      'All comparisons contain 1,318 pairs; zero differences are excluded from signed-rank computation. Positive differences favor GPT-4o.')
corr = pd.read_csv(RESULTS / 'spearman_description_length.csv')
rows = []
for metric, label in [('CosineSimilarity', 'Cosine'), ('EuclideanDistance', 'Euclidean similarity'),
                      ('TreeEditDistance', 'Tree similarity'), ('ROUGE-L', 'ROUGE-L')]:
    row = corr[corr.metric == metric].iloc[0]
    rows.append([label, f'{row.rho:.3f}', pformat(row.p_holm_original_four)])
table('correlations', 'Spearman correlation of description word count with GPT-4o scores; 1,318 rows and the original four-test Holm family.',
      'correlation', 'lrr', ['Metric', '$\\rho$', 'Adjusted $p$'], rows)
rows = []
for model in MODELS:
    row = means[(means.model == model) & (means.metric == 'CosineSimilarity')].iloc[0]
    rows.append([LABELS[model], f'{row.row_mean:.3f}', f'{row.equal_pair_mean:.3f}', f'{row.equal_url_mean:.3f}'])
table('aggregation', 'Cosine means under alternative aggregation units.', 'aggregation', 'lrrr',
      ['Model', '1,318 rows', '690 raw task pairs', '186 source pages'], rows,
      'Pair means average all archived responses within each exact raw documentation pair before giving each pair equal weight. Page means give each URL equal weight.')
parses = pd.read_csv(RESULTS / 'yaml_parse_summary.csv')
base = pd.read_csv(RESULTS / 'yaml_baseloader_summary.csv').set_index('model')
rows = []
for model in ['reference'] + MODELS:
    row = parses[(parses.model == model) & (parses.stage == 'raw')].iloc[0]
    rows.append(['Raw documentation reference' if model == 'reference' else LABELS[model], int(row.parseable),
                 int(row.parse_errors), int(base.loc[model, 'gate_accepted'])])
table('parser_audit', 'Raw-text loading results out of 1,318 examples; loaders and nonempty acceptance are kept distinct.',
      'parser', 'lrrr', ['Text source', 'SafeLoader loads', 'SafeLoader errors', 'BaseLoader accepts'], rows,
      'BaseLoader acceptance additionally excludes empty documents. These counts are parser outcomes, not GitHub Actions validity rates.')
fallback = pd.read_csv(RESULTS / 'fallback_summary.csv')
rows = []
for _, row in fallback.iterrows():
    scope = 'All examples' if row['subset'] == 'all_rows' else 'Loadable documentation references'
    rows.append([scope, int(row.n), int(row.baseline_accepted), int(row.fallback_accepted), int(row.rescued)])
table('fallback', 'Nonempty BaseLoader acceptance for GPT-4o alone and fixed fallback over archived candidates.',
      'fallback', 'lrrrr', ['Scope', '$n$', 'GPT-4o', 'Fallback', 'Recovered'], rows,
      'The second row uses SafeLoader-loadable raw documentation references. Both policies retain 16 exact matches after outer-whitespace stripping; functional success is unmeasured.')
boot = SUMMARY['url_cluster_bootstrap_gpt4o_minus_gpt41']
allrows = SUMMARY['fallback_summary'][0]
low, high = boot['row_weighted_percentile_95_ci']
macros = {
    'CosineDifference': f'${boot["row_weighted_point_difference"]:+.4f}$',
    'ClusterInterval': f'$[{low:+.4f}, {high:+.4f}]$',
    'FalsePerfectCount': str(len(SUMMARY['false_perfect_tree_rows'])),
    'BaselineAcceptance': f'{allrows["baseline_accepted"]:,}/1,318 ({100*allrows["baseline_acceptance_rate"]:.2f}\\%)',
    'FallbackAcceptance': f'{allrows["fallback_accepted"]:,}/1,318 (100\\%)',
    'RescuedCandidates': str(allrows['rescued']),
    'AcceptanceGain': f'{100*(allrows["fallback_acceptance_rate"]-allrows["baseline_acceptance_rate"]):.2f}',
}
(TABLES / 'numbers.tex').write_text('% Generated by analysis/build_paper_assets.py\n' +
                                  '\n'.join('\\newcommand{\\' + k + '}{' + v + '}' for k,v in macros.items()) + '\n')
fig, ax = plt.subplots(figsize=(7.4, 3.4))
positions = np.arange(len(MODELS))
for offset, column, label, color, marker in [
    (-.18, 'row_mean', '1,318 rows', '#22577a', 'o'),
    (0, 'equal_pair_mean', '690 raw task pairs', '#258f80', 's'),
    (.18, 'equal_url_mean', '186 source pages', '#bb790d', '^')]:
    values = [float(means[(means.model == m) & (means.metric == 'CosineSimilarity')][column].iloc[0]) for m in MODELS]
    ax.scatter(values, positions + offset, label=label, color=color, marker=marker, s=40, zorder=3)
ax.set(yticks=positions, yticklabels=[LABELS[m].replace('--', '-') for m in MODELS],
       xlabel='Mean archived cosine similarity', xlim=(.59, .725), ylim=(5.6, -.6))
ax.grid(axis='x', alpha=.2)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, loc='lower right')
fig.tight_layout()
fig.savefig(ROOT / 'paper/figures/aggregation.pdf')
plt.close(fig)
print('Built manuscript tables, numerical macros, and the aggregation figure.')

# New empirical extension: these tables are computed from raw-input experiments.
new = pd.read_csv(RESULTS / 'recomputed_summary.csv')
new = new[new.subset == 'all_references'].set_index('method')
policy_labels = {'syntax_fallback': 'Syntax fallback', 'strict_fallback': 'Strict fallback',
                 'consensus_medoid': 'Consensus', 'minimal_structure': 'Minimal structure'}
rows = []
for method in MODELS + list(policy_labels):
    row = new.loc[method]
    rows.append([LABELS.get(method, policy_labels.get(method)), f'{row.rouge_l_f1_mean:.3f}',
                 f'{row.chrf_mean:.3f}', f'{row.path_value_f1_mean:.3f}',
                 f'{row.path_value_leaf_f1_mean:.3f}', int(row.structural_exact)])
table('raw_metrics', 'Independently recomputed agreement and reference-free selection results.', 'rawmetrics', 'lrrrrr',
      ['Model or selector', 'ROUGE F1', 'chrF', '$F_{\\mathrm{pv}}$', 'Leaf F1', 'Exact structure'], rows,
      'Lexical means use 1,318 examples. Structural means and exact counts use the same 1,293 valid references, assigning zero to invalid generated candidates. Leaf F1 removes the shared root-kind feature. Selectors use only candidate text.')

work = pd.read_csv(RESULTS / 'workflow_summary.csv')
rows = []
for method in MODELS + ['syntax_fallback','actionlint_fallback']:
    a = work[(work.scope == 'full_workflow_shape') & (work.policy_or_model == method)].iloc[0]
    b = work[(work.scope == 'reference_actionlint_accepted') & (work.policy_or_model == method)].iloc[0]
    label = LABELS.get(method, 'Syntax fallback' if method == 'syntax_fallback' else 'Checker fallback')
    rows.append([label, f'{int(a.actionlint_accepted)}/335', f'{100*a.actionlint_accepted/335:.2f}\\%',
                 int(a.abstained), f'{int(b.actionlint_accepted)}/292'])
table('workflow_validation', 'Static workflow acceptance with actionlint 1.7.7; complete-workflow-shaped task scope.',
      'workflow', 'lrrrr', ['Model or selector', 'Accepted', 'Rate', 'Abstained', 'Reference-pass stratum'], rows,
      'All 335 tasks remain in the main denominator. References themselves pass in 292 cases. The last column uses only that separately defined 292-task stratum. All generated exact-text matches are zero in both strata; acceptance is a static check.')

fig, ax = plt.subplots(figsize=(7.4,3.2))
frame = work[(work.scope == 'full_workflow_shape') & work.policy_or_model.isin(MODELS)].set_index('policy_or_model')
for index, method in enumerate(MODELS):
    parse = 100 * frame.loc[method, 'syntax_accepted'] / 335
    lint = 100 * frame.loc[method, 'actionlint_accepted'] / 335
    ax.plot([lint,parse], [index,index], color='#c8d2d8', linewidth=2, zorder=1)
ax.scatter([100*frame.loc[m,'syntax_accepted']/335 for m in MODELS], positions, marker='o', color='#258f80', label='YAML parser acceptance', zorder=3)
ax.scatter([100*frame.loc[m,'actionlint_accepted']/335 for m in MODELS], positions, marker='s', color='#22577a', label='actionlint acceptance', zorder=3)
ax.set(yticks=positions, yticklabels=[LABELS[m].replace('--','-') for m in MODELS], xlim=(0,105), ylim=(5.6,-.6), xlabel='Accepted candidates on 335 workflow-shaped tasks (%)')
ax.grid(axis='x', alpha=.2);ax.set_axisbelow(True)
ax.legend(frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,1.18),ncol=2)
fig.tight_layout();fig.savefig(ROOT/'paper/figures/workflow_validation.pdf');plt.close(fig)
print('Built raw-metric and workflow-validation tables and the static-acceptance figure.')
