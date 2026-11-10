#!/usr/bin/env python3
"""Audit weighting and candidate-count sensitivity of archived workflow checks.

Run after validate_workflows.py. This script makes no model or workflow calls:
it checks the saved per-example decisions against the unchanged input texts,
then evaluates fixed prefixes of the declared candidate order. Prefix length
is a count of distinct archived responses, not an inference-cost estimate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'ICSME25-LLM4CI-main' / 'data'
OUT = ROOT / 'analysis' / 'results'
TABLE = ROOT / 'paper' / 'tables' / 'workflow_robustness.tex'
MODELS = ['gpt-4o', 'gpt-4.1', 'gemma3-12b', 'llama3.1-8b',
          'codegemma-7b', 'codellama-7b']
SEED = 2026
BOOTSTRAPS = 10000


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode('utf-8') if isinstance(value, str) else value).hexdigest()


def point_estimates(frame: pd.DataFrame, accepted: pd.Series) -> dict:
    values = accepted.astype(float)
    return {'n_rows': len(frame), 'n_pairs': int(frame.pair_group.nunique()),
            'n_urls': int(frame.URL.nunique()), 'accepted': int(values.sum()),
            'row_rate': float(values.mean()),
            'equal_pair_rate': float(values.groupby(frame.pair_group).mean().mean()),
            'equal_url_rate': float(values.groupby(frame.URL).mean().mean())}


def cluster_comparison(frame: pd.DataFrame, difference: pd.Series) -> dict:
    """Resample paired URL blocks, preserving row counts and multiplicities."""
    blocks = difference.groupby(frame.URL, sort=True).agg(['sum', 'count', 'mean'])
    draws = np.random.default_rng(SEED).integers(0, len(blocks),
                                               size=(BOOTSTRAPS, len(blocks)))
    row_samples = (blocks['sum'].to_numpy()[draws].sum(axis=1)
                   / blocks['count'].to_numpy()[draws].sum(axis=1))
    url_samples = blocks['mean'].to_numpy()[draws].mean(axis=1)
    row_low, row_high = np.quantile(row_samples, [.025, .975])
    url_low, url_high = np.quantile(url_samples, [.025, .975])
    return {'n_rows': len(frame), 'n_urls': len(blocks),
            'row_difference': float(difference.mean()),
            'row_cluster_95_low': float(row_low), 'row_cluster_95_high': float(row_high),
            'equal_pair_difference': float(difference.groupby(frame.pair_group).mean().mean()),
            'equal_url_difference': float(blocks['mean'].mean()),
            'url_cluster_95_low': float(url_low), 'url_cluster_95_high': float(url_high),
            'rescued': int((difference > 0).sum()), 'lost': int((difference < 0).sum()),
            'bootstrap_iterations': BOOTSTRAPS, 'seed': SEED}


def main() -> None:
    docs_path = SOURCE / 'GitHubActions_Docs.csv'
    details_path = OUT / 'workflow_lint_details.csv'
    metadata_path = OUT / 'workflow_summary.json'
    docs = pd.read_csv(docs_path, keep_default_na=False)
    assert docs.ID.is_unique
    docs['pair_group'] = docs.groupby(['Description', 'Code'], sort=False,
                                      dropna=False).ngroup()
    docs = docs.set_index('ID')
    details = pd.read_csv(details_path, keep_default_na=False)
    metadata = json.loads(metadata_path.read_text())
    assert metadata['actionlint']['version'] == '1.7.7'
    assert metadata['selection_order'] == MODELS
    assert not details.duplicated(['ID', 'model']).any()
    assert set(details.model) == {'reference', *MODELS}
    input_paths = [docs_path]
    for model in ['reference', *MODELS]:
        subset = details[details.model == model].set_index('ID')
        if model == 'reference':
            expected = docs.Code
        else:
            path = SOURCE / f'GitHubActions_{model}_output.csv'
            input_paths.append(path)
            raw = pd.read_csv(path, keep_default_na=False).set_index('ID')
            assert raw.index.is_unique and set(raw.index) == set(docs.index)
            expected = raw.LLM_YML_Output
        assert subset.text_sha256.equals(expected.loc[subset.index].map(digest).rename('text_sha256'))
        assert subset.actionlint_accepted.equals(subset.returncode.eq(0).rename('actionlint_accepted'))
        assert not subset.actionlint_status.eq('tool_error').any()
    accepted = details.pivot(index='ID', columns='model', values='actionlint_accepted')
    assert not accepted.isna().any().any()
    # Verify the shape restriction independently from raw documentation text.
    expected_scope = []
    for identifier, row in docs.iterrows():
        try:
            value = yaml.load(row.Code, Loader=yaml.BaseLoader)
        except yaml.YAMLError:
            continue
        if isinstance(value, dict) and 'on' in value and isinstance(value.get('jobs'), dict):
            expected_scope.append(identifier)
    assert set(accepted.index) == set(expected_scope)
    frame = accepted.join(docs[['URL', 'pair_group']])
    scopes = {'full_workflow_shape': frame,
              'reference_actionlint_accepted': frame[frame.reference]}
    weighted, budgets, comparisons = [], [], []
    for scope, group in scopes.items():
        fallback = group[MODELS].any(axis=1)
        for model in [*MODELS, 'actionlint_fallback']:
            values = fallback if model == 'actionlint_fallback' else group[model]
            weighted.append({'scope': scope, 'method': model,
                             **point_estimates(group, values)})
        for k in range(1, len(MODELS) + 1):
            prefix = group[MODELS[:k]].to_numpy(dtype=bool)
            covered = prefix.any(axis=1)
            # Sequential static checks stop at the first passing candidate.
            attempts = np.where(covered, prefix.argmax(axis=1) + 1, k)
            budgets.append({'scope': scope, 'prefix_length': k,
                            'last_model': MODELS[k-1],
                            **point_estimates(group, pd.Series(covered, index=group.index)),
                            'mean_static_checks_until_stop': float(attempts.mean())})
        for comparator in ['gpt-4.1', 'gpt-4o']:
            difference = fallback.astype(float) - group[comparator].astype(float)
            comparisons.append({'scope': scope, 'policy': 'actionlint_fallback',
                                'comparator': comparator,
                                **cluster_comparison(group, difference)})
    for name, rows in [('weighting', weighted), ('budget', budgets),
                       ('comparisons', comparisons)]:
        pd.DataFrame(rows).to_csv(OUT / f'workflow_robustness_{name}.csv',
                                  index=False, float_format='%.12g')
    rows = [r for r in budgets if r['scope'] == 'full_workflow_shape']
    baseline = next(r for r in weighted if r['scope'] == 'full_workflow_shape'
                    and r['method'] == 'gpt-4.1')
    latex = [r'\begin{table}[H]', r'\centering\small',
             r'\caption{Static acceptance under fixed candidate prefixes and alternative weighting.}',
             r'\label{tab:workflowrobustness}',
             r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
             r'Candidate pool & Accepted & Row (\%) & Pair (\%) & Page (\%) & Checks\\',
             r'\midrule']
    latex.append(f"GPT-4.1 alone & {baseline['accepted']} & {100*baseline['row_rate']:.2f} & "
                 f"{100*baseline['equal_pair_rate']:.2f} & {100*baseline['equal_url_rate']:.2f} & 1.00"+r'\\')
    for row in rows:
        label = 'First archived candidate' if row['prefix_length'] == 1 else f"First {row['prefix_length']} archived candidates"
        latex.append(f"{label} & {row['accepted']} & "
                     f"{100*row['row_rate']:.2f} & {100*row['equal_pair_rate']:.2f} & "
                     f"{100*row['equal_url_rate']:.2f} & {row['mean_static_checks_until_stop']:.2f}"+r'\\')
    latex.extend([r'\bottomrule', r'\end{tabular}',
                  r'\par\vspace{0.3em}{\footnotesize The same 335 rows contain 175 exact description--reference pairs and 127 source pages. Pair and page columns average within each unit before averaging units. Prefixes follow the fixed model priority; checks count sequential static evaluations until acceptance or exhaustion. They do not estimate model calls, latency, or cost.}',
                  r'\end{table}'])
    TABLE.write_text('\n'.join(latex)+'\n')
    provenance = {'scope': 'Post-hoc robustness analysis of archived static-check decisions; no generation or workflow execution.',
                  'weighting': 'Row mean; equal mean of exact raw Docs Description+Code pair means; equal mean of source-URL means. All tasks, including abstentions, remain in denominators.',
                  'budget': 'Fixed ordered prefixes of distinct archived outputs; not an equal-inference-budget comparison against repeated sampling from one model.',
                  'uncertainty': 'Paired percentile bootstrap resampling URL blocks, retaining within-block rows and multiplicities. Intervals are exploratory, unadjusted, and do not remove cross-page dependence of duplicated tasks.',
                  'seed': SEED, 'bootstrap_iterations': BOOTSTRAPS, 'order': MODELS,
                  'source_sha256': {str(path.relative_to(ROOT)): digest(path.read_bytes())
                                    for path in [*input_paths, details_path, metadata_path]},
                  'weighting_results': weighted, 'budget_results': budgets,
                  'paired_comparisons': comparisons}
    (OUT / 'workflow_robustness_metadata.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == '__main__':
    main()
