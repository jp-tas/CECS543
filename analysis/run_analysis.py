#!/usr/bin/env python3
"""Offline reanalysis of the archived ICSME25-LLM4CI release.

No model requests are made and stored similarity scores are not recomputed.
Run: python3 analysis/run_analysis.py (from any working directory).

The remove_comments helper is retained from Ghaleb and Rathnayake's
ICSME25-LLM4CI package under its MIT license; see the unchanged LICENSE
and README in ../ICSME25-LLM4CI-main. All original data are attributed
in the manuscript and retained with the supplied package.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
import re
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'ICSME25-LLM4CI-main'
OUT = Path(__file__).resolve().parent / 'results'
MODELS = ['gpt-4o', 'gpt-4.1', 'gemma3-12b', 'llama3.1-8b', 'codegemma-7b', 'codellama-7b']
METRICS = ['CosineSimilarity', 'EuclideanDistance', 'TreeEditDistance', 'ROUGE-L', 'chrF']
SEED = 2026
BOOTSTRAPS = 10000


def remove_comments(text: str) -> str:
    """Exact project helper: preserve its behavior rather than silently fix it."""
    if not isinstance(text, str):
        return ''
    lines = []
    for line in text.splitlines():
        if line.strip().startswith('#'):
            continue
        line_no_inline = re.sub(r'(?<!["\'])#.*', '', line)
        if line_no_inline.strip() != '':
            lines.append(line_no_inline.rstrip())
    return '\n'.join(lines)


@lru_cache(maxsize=None)
def parse_yaml(text: str, loader=yaml.SafeLoader) -> dict:
    """Fail explicitly. Never replace a parse failure by a valid empty tree."""
    try:
        value = yaml.load(text, Loader=loader)
    except yaml.YAMLError as error:
        return {'parseable': False, 'nonempty': False, 'root_type': 'parse_error', 'error': str(error).splitlines()[0]}
    # safe_load follows YAML 1.1 scalar resolution; mapping keys are not inspected.
    root_type = ('mapping' if isinstance(value, dict) else 'sequence' if isinstance(value, list)
                 else 'null' if value is None else 'scalar')
    nonempty = value is not None and (not isinstance(value, (dict, list, str)) or len(value) > 0)
    return {'parseable': True, 'nonempty': nonempty, 'root_type': root_type, 'error': ''}


def syntax_gate(text: str) -> bool:
    parsed = parse_yaml(text, yaml.BaseLoader)
    return parsed['parseable'] and parsed['nonempty']


def select_candidate(candidates: dict[str, str]) -> tuple[str | None, str | None]:
    """Fixed order, reference-free fallback over already archived candidates."""
    for name in MODELS:
        if syntax_gate(candidates[name]):
            return name, candidates[name]
    return None, None


def write_csv(records, name):
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    frame.to_csv(OUT / name, index=False, float_format='%.12g')
    return frame


def bootstrap_difference(scores: pd.DataFrame) -> dict:
    """Resample documentation URLs as blocks; preserve paired model scores."""
    frame = scores.assign(difference=scores['CosineSimilarity (gpt-4o)'] - scores['CosineSimilarity (gpt-4.1)'])
    blocks = frame.groupby('URL', sort=True).difference.agg(['sum', 'count', 'mean'])
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(blocks), size=(BOOTSTRAPS, len(blocks)))
    sums = blocks['sum'].to_numpy()[sampled].sum(axis=1)
    counts = blocks['count'].to_numpy()[sampled].sum(axis=1)
    row_estimates = sums / counts
    macro_estimates = blocks['mean'].to_numpy()[sampled].mean(axis=1)
    write_csv(pd.DataFrame({'iteration': np.arange(1, BOOTSTRAPS + 1),
                            'row_weighted_difference': row_estimates,
                            'url_macro_difference': macro_estimates}), 'url_cluster_bootstrap.csv')
    return {'unit': 'URL', 'clusters': len(blocks), 'iterations': BOOTSTRAPS, 'seed': SEED,
            'row_weighted_point_difference': float(frame.difference.mean()),
            'row_weighted_percentile_95_ci': np.quantile(row_estimates, [.025, .975]).tolist(),
            'url_macro_point_difference': float(blocks['mean'].mean()),
            'url_macro_percentile_95_ci': np.quantile(macro_estimates, [.025, .975]).tolist()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(SOURCE / 'results/GitHubActions_Similarity_Scores_Six_LLMs.csv')
    reference = pd.read_csv(SOURCE / 'data/GitHubActions_Docs.csv').set_index('ID')
    assert scores.ID.is_unique and reference.index.is_unique
    assert set(scores.ID) == set(reference.index)
    reference = reference.loc[scores.ID].reset_index()
    base_columns = ['ID', 'URL', 'Title', 'Description', 'Code']
    raw_outputs = {}
    cached_refs = {}
    integrity = []
    for name in MODELS:
        frame = pd.read_csv(SOURCE / f'data/GitHubActions_{name}_output.csv').set_index('ID')
        assert frame.index.is_unique and set(frame.index) == set(reference.ID)
        frame = frame.loc[reference.ID].reset_index()
        pd.testing.assert_frame_equal(frame[base_columns[:-1]], reference[base_columns[:-1]])
        cached_refs[name] = frame.Code.tolist()
        for idx, (doc, cached) in enumerate(zip(reference.Code, frame.Code)):
            integrity.append({'ID':int(reference.iloc[idx].ID), 'model':name,
                              'same_after_outer_strip':doc.strip()==cached.strip(),
                              'cached_is_name_error':cached=='#NAME?',
                              'docs_safe_parseable':parse_yaml(doc)['parseable'],
                              'cached_safe_parseable':parse_yaml(cached)['parseable'],
                              'comment_stripped_cached_safe_parseable':parse_yaml(remove_comments(cached))['parseable']})
        raw_outputs[name] = frame.LLM_YML_Output.fillna('').tolist()
        # Confirm archived score columns point to these same generated texts.
        assert [remove_comments(t) for t in raw_outputs[name]] == scores[f'LLM_YML_Output ({name})'].fillna('').tolist()
    # The supplied artifacts are not perfectly synchronized; preserve and report
    # reference differences instead of assuming the merged Code is authoritative.
    score_reference_mismatches=[int(reference.iloc[i].ID) for i,(a,b) in enumerate(zip(cached_refs['gpt-4o'],scores.Code.fillna('')))
                                if remove_comments(a).strip()!=b]
    integrity = write_csv(integrity, 'reference_transformation_audit.csv')
    reference_stages={
        'raw_docs':reference.Code.tolist(),
        'strip_docs':[t.strip() for t in reference.Code],
        'remove_comments_docs':[remove_comments(t) for t in reference.Code],
        'strip_then_remove_comments_docs':[remove_comments(t.strip()) for t in reference.Code],
    }
    reference_stage_rows=[]; reference_stage_summary=[]
    for stage,texts in reference_stages.items():
        stage_parses=[parse_yaml(t) for t in texts]
        ids=[int(reference.iloc[i].ID) for i,v in enumerate(stage_parses) if not v['parseable']]
        for idx,result in enumerate(stage_parses):
            reference_stage_rows.append({'ID':int(reference.iloc[idx].ID),'stage':stage,**result})
        reference_stage_summary.append({'stage':stage,'n':len(texts),'parseable':sum(p['parseable'] for p in stage_parses),
                                        'parse_errors':len(ids),'error_ids':ids})
    write_csv(reference_stage_rows,'docs_reference_transformation_details.csv')
    write_csv([{**r,'error_ids':' '.join(map(str,r['error_ids']))} for r in reference_stage_summary],'docs_reference_transformation_summary.csv')
    raw_doc_success=np.array([parse_yaml(t)['parseable'] for t in reference.Code])
    stripped_doc_success=np.array([parse_yaml(t.strip())['parseable'] for t in reference.Code])
    strip_induced_error_ids=reference.loc[raw_doc_success & ~stripped_doc_success,'ID'].astype(int).tolist()
    scores['pair_group'] = reference.groupby(['Description', 'Code'], sort=False, dropna=False).ngroup()
    numeric = [f'{metric} ({name})' for name in MODELS for metric in METRICS]
    assert scores[numeric].notna().all().all()
    group_scores = scores.groupby('pair_group')[numeric].mean()
    score_groups=scores.groupby(['Description','Code'],sort=False,dropna=False).ngroup()
    score_pair_scores=scores.assign(score_pair_group=score_groups).groupby('score_pair_group')[numeric].mean()
    url_scores = scores.groupby('URL')[numeric].mean()
    means = []
    for name in MODELS:
        for metric in METRICS:
            col = f'{metric} ({name})'
            means.append({'model': name, 'metric': metric, 'row_mean': scores[col].mean(),
                          'row_median': scores[col].median(), 'equal_pair_mean': group_scores[col].mean(),
                          'equal_stored_score_pair_mean':score_pair_scores[col].mean(),
                          'equal_url_mean': url_scores[col].mean()})
    means = write_csv(means, 'similarity_summary.csv')
    write_csv(group_scores.reset_index(), 'pair_averaged_scores.csv')
    write_csv(score_pair_scores.reset_index(), 'stored_score_pair_averaged_scores.csv')
    write_csv(url_scores.reset_index(), 'url_averaged_scores.csv')
    sizes = scores.groupby('pair_group').size()
    write_csv(sizes.value_counts().sort_index().rename_axis('group_size').reset_index(name='number_of_groups'), 'duplicate_group_sizes.csv')
    tests = []
    for name in MODELS[1:]:
        for unit, frame in [('row', scores), ('unique_pair_mean', group_scores)]:
            diff = frame['CosineSimilarity (gpt-4o)'] - frame[f'CosineSimilarity ({name})']
            stat, p = wilcoxon(diff, alternative='two-sided', zero_method='wilcox', correction=False, method='auto')
            tests.append({'unit': unit, 'comparison': f'gpt-4o vs {name}', 'n': len(diff),
                          'nonzero_n': int((diff != 0).sum()), 'statistic': stat, 'p_raw': p,
                          'mean_paired_difference': diff.mean(), 'wins': int((diff > 0).sum()),
                          'ties': int((diff == 0).sum()), 'losses': int((diff < 0).sum())})
    tests = pd.DataFrame(tests)
    for unit in tests.unit.unique():
        mask = tests.unit == unit
        reject, adjusted, _, _ = multipletests(tests.loc[mask, 'p_raw'], alpha=.05, method='holm')
        tests.loc[mask, 'p_holm'] = adjusted
        tests.loc[mask, 'reject_at_0_05'] = reject
    write_csv(tests, 'wilcoxon_cosine.csv')
    correlations = []
    for metric in METRICS:
        rho, p = spearmanr(scores.Description.str.split().str.len(), scores[f'{metric} (gpt-4o)'])
        correlations.append({'model': 'gpt-4o', 'metric': metric, 'n': len(scores), 'rho': rho, 'p_raw': p})
    correlations = pd.DataFrame(correlations)
    correlations['p_holm_all_five'] = multipletests(correlations.p_raw, method='holm')[1]
    # Published helper omits chrF, so retain the exact four-test family as well.
    original_mask = correlations.metric != 'chrF'
    correlations.loc[original_mask, 'p_holm_original_four'] = multipletests(correlations.loc[original_mask, 'p_raw'], method='holm')[1]
    write_csv(correlations, 'spearman_description_length.csv')
    bootstrap = bootstrap_difference(scores)

    parse_rows = []
    all_texts = {'reference': reference.Code.tolist(),'reference_stored_score':scores.Code.fillna('').tolist(), **{f'reference_cached_{m}':v for m,v in cached_refs.items()}, **raw_outputs}
    parsed = {}
    for name, texts in all_texts.items():
        for stage in ['raw', 'original_comment_stripped']:
            transformed = texts if stage == 'raw' else [remove_comments(t) for t in texts]
            results = [parse_yaml(t) for t in transformed]
            parsed[name, stage] = results
            for idx, result in enumerate(results):
                parse_rows.append({'ID': int(reference.iloc[idx].ID), 'model': name, 'stage': stage,
                                   'accepted_by_syntax_gate': result['parseable'] and result['nonempty'], **result})
    parses = write_csv(parse_rows, 'yaml_parse_details.csv')
    parse_summary = []
    for (name, stage), frame in parses.groupby(['model', 'stage'], sort=False):
        counts = frame.root_type.value_counts()
        parse_summary.append({'model': name, 'stage': stage, 'n': len(frame),
                              'parseable': int(frame.parseable.sum()), 'parse_errors': int((~frame.parseable).sum()),
                              'gate_accepted': int(frame.accepted_by_syntax_gate.sum()),
                              'mapping_roots': int(counts.get('mapping', 0)), 'sequence_roots': int(counts.get('sequence', 0)),
                              'scalar_roots': int(counts.get('scalar', 0)), 'null_roots': int(counts.get('null', 0))})
    write_csv(parse_summary, 'yaml_parse_summary.csv')
    agreements = []
    false_perfect = []
    parser_aware_tree = []
    for name in MODELS:
        refs = parsed[f'reference_cached_{name}', 'original_comment_stripped']; gens = parsed[name, 'original_comment_stripped']
        raw_refs = parsed['reference', 'raw']; raw_gens = parsed[name, 'raw']
        both_invalid_ids = []
        for idx, (rr, gg) in enumerate(zip(refs, gens)):
            stored = scores.iloc[idx][f'TreeEditDistance ({name})']
            both_load = rr['parseable'] and gg['parseable']
            parser_aware_tree.append({'ID': int(reference.iloc[idx].ID), 'model': name,
                                     'reference_safe_loads': rr['parseable'],
                                     'output_safe_loads': gg['parseable'],
                                     'stored_tree_similarity': stored,
                                     'parser_aware_tree_similarity': stored if both_load else np.nan,
                                     'status': 'both_load' if both_load else 'unavailable_parse_failure'})
            if stored == 1 and (not rr['parseable'] or not gg['parseable']):
                both_invalid_ids.append(int(reference.iloc[idx].ID))
                false_perfect.append({'ID': int(reference.iloc[idx].ID), 'model': name, 'stored_tree_similarity': stored,
                                      'reference_parseable': rr['parseable'], 'generated_parseable': gg['parseable'],
                                      'reference_error': rr['error'], 'generated_error': gg['error']})
        agreements.append({'model': name,
                           'raw_exact_after_outer_strip': sum(a.strip() == b.strip() for a,b in zip(reference.Code, raw_outputs[name])),
                           'comment_stripped_exact': sum(remove_comments(a) == remove_comments(b) for a,b in zip(reference.Code, raw_outputs[name])),
                           'stored_tree_similarity_one': int((scores[f'TreeEditDistance ({name})'] == 1).sum()),
                           'stored_cosine_at_least_0_999999': int((scores[f'CosineSimilarity ({name})'] >= .999999).sum()),
                           'stored_tree_one_with_parse_error': len(both_invalid_ids),
                           'both_parseable_raw': sum(a['parseable'] and b['parseable'] for a,b in zip(raw_refs, raw_gens))})
    write_csv(agreements, 'agreement_definitions.csv')
    write_csv(parser_aware_tree, 'parser_aware_tree_scores.csv')
    write_csv(pd.DataFrame(false_perfect, columns=['ID','model','stored_tree_similarity','reference_parseable','generated_parseable','reference_error','generated_error']), 'false_perfect_tree_examples.csv')

    # BaseLoader is used only by the syntax gate: no YAML 1.1 boolean coercion
    # and no constructor/type interpretation is needed for this policy.
    base_rows=[]
    for name,texts in {'reference':reference.Code.tolist(),**raw_outputs}.items():
        for idx,text in enumerate(texts):
            result=parse_yaml(text,yaml.BaseLoader)
            base_rows.append({'ID':int(reference.iloc[idx].ID),'model':name,
                              'accepted_by_syntax_gate':result['parseable'] and result['nonempty'],**result})
    base_parses=write_csv(base_rows,'yaml_baseloader_details.csv')
    write_csv(base_parses.groupby('model',sort=False).agg(n=('ID','size'),parseable=('parseable','sum'),gate_accepted=('accepted_by_syntax_gate','sum')).reset_index(),'yaml_baseloader_summary.csv')
    selection = []
    for idx, row in reference.iterrows():
        candidates = {name: raw_outputs[name][idx] for name in MODELS}
        selected, text = select_candidate(candidates)
        baseline_accept = syntax_gate(candidates['gpt-4o'])
        selection.append({'ID': int(row.ID), 'URL': row.URL, 'reference_raw_parseable': parsed['reference','raw'][idx]['parseable'],
                          'baseline_accepted': baseline_accept, 'fallback_accepted': selected is not None,
                          'selected_model': selected or 'none', 'rescued': not baseline_accept and selected is not None,
                          'baseline_exact_raw_strip': parsed['reference','raw'][idx]['parseable'] and baseline_accept and row.Code.strip() == candidates['gpt-4o'].strip(),
                          'fallback_exact_raw_strip': parsed['reference','raw'][idx]['parseable'] and selected is not None and row.Code.strip() == text.strip(),
                          'selected_stored_cosine': scores.iloc[idx][f'CosineSimilarity ({selected})'] if selected else np.nan})
    selected = write_csv(selection, 'fallback_selection.csv')
    policy_summary = []
    for subset, frame in [('all_rows', selected), ('raw_parseable_reference', selected.loc[selected.reference_raw_parseable])]:
        policy_summary.append({'subset': subset, 'n': len(frame), 'baseline_accepted': int(frame.baseline_accepted.sum()),
                               'fallback_accepted': int(frame.fallback_accepted.sum()), 'rescued': int(frame.rescued.sum()),
                               'all_candidates_rejected': int((~frame.fallback_accepted).sum()),
                               'baseline_acceptance_rate': float(frame.baseline_accepted.mean()),
                               'fallback_acceptance_rate': float(frame.fallback_accepted.mean()),
                               'baseline_exact_raw_strip_parseable': int(frame.baseline_exact_raw_strip.sum()),
                               'fallback_exact_raw_strip_parseable': int(frame.fallback_exact_raw_strip.sum())})
    write_csv(policy_summary, 'fallback_summary.csv')
    write_csv(selected.selected_model.value_counts().rename_axis('model').reset_index(name='selected_count'), 'fallback_selection_counts.csv')
    # Save selected outputs so the framework component is concrete and inspectable.
    selected_texts = []
    for idx, row in selected.iterrows():
        name = row.selected_model
        selected_texts.append({'ID': row.ID, 'selected_model': name,
                               'selected_yaml': raw_outputs[name][idx] if name != 'none' else ''})
    write_csv(selected_texts, 'fallback_selected_outputs.csv')

    # Two reproducible vector figures: unit weighting and syntactic acceptance.
    labels = ['GPT-4o', 'GPT-4.1', 'Gemma 3', 'Llama 3.1', 'CodeGemma', 'Code Llama']
    fig, ax = plt.subplots(figsize=(7.4,3.3))
    x=np.arange(len(MODELS)); width=.24
    for i,(col,label,color) in enumerate([('row_mean','1,318 rows','#22577a'),('equal_pair_mean','690 unique task pairs','#38a3a5'),('equal_url_mean','186 source URLs','#e8a838')]):
        vals=[float(means.loc[(means.model==m)&(means.metric=='CosineSimilarity'),col].iloc[0]) for m in MODELS]
        ax.bar(x+(i-1)*width,vals,width,label=label,color=color)
    ax.set(xticks=x,xticklabels=labels,ylabel='Mean stored cosine similarity',ylim=(.5,.76))
    ax.tick_params(axis='x',labelsize=9);ax.legend(frameon=False,fontsize=8,ncol=3,loc='upper left')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(OUT/'weighting_sensitivity.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,2.9))
    raw_summary=base_parses.groupby('model').accepted_by_syntax_gate.sum()
    vals=[100*raw_summary.loc[m]/len(scores) for m in MODELS]
    ax.bar(np.arange(len(MODELS)),vals,color='#22577a')
    ax.set(xticks=np.arange(len(MODELS)),xticklabels=labels,ylabel='BaseLoader gate acceptance (%)',ylim=(90,100.5))
    for i,v in enumerate(vals):ax.text(i,v+.15,f'{v:.2f}',ha='center',fontsize=9)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(OUT/'yaml_parseability.pdf');plt.close(fig)

    source_files=sorted([SOURCE/'results/GitHubActions_Similarity_Scores_Six_LLMs.csv',*SOURCE.glob('data/*.csv'),*SOURCE.glob('scripts/*.py')])
    metadata={'scope':'Offline archived-output reanalysis; stored similarity metrics are not recomputed; no model inference or workflow execution.',
              'rows':len(scores),'models':MODELS,'unique_description_code_pairs':len(group_scores),'pair_key_source':'GitHubActions_Docs.csv exact raw Description+Code; average every output within each group before averaging groups',
              'unique_stored_score_pairs':len(score_pair_scores),'stored_score_pair_key_source':'stored score CSV exact Description+Code including missing as one group',
              'duplicate_pair_rows_beyond_first':int(sizes.sum()-len(sizes)),
              'unique_code':int(reference.Code.nunique()),'unique_description':int(reference.Description.nunique()),'unique_urls':int(reference.URL.nunique()),
              'url_cluster_bootstrap_gpt4o_minus_gpt41':bootstrap,'policy_order':MODELS,
              'policy_acceptance':'yaml.BaseLoader parse succeeds and result is non-null and nonempty. No workflow schema, behavior, reference text, or similarity is used for selection.',
              'reference_transformations':{m:{'not_equal_after_outer_strip_ids':integrity.loc[(integrity.model==m)&(~integrity.same_after_outer_strip),'ID'].astype(int).tolist(),
                                              'docs_parseable_cached_error_ids':integrity.loc[(integrity.model==m)&integrity.docs_safe_parseable&(~integrity.cached_safe_parseable),'ID'].astype(int).tolist()} for m in MODELS},
              'stored_reference_mismatch_vs_cleaned_gpt4o_cached_ids':score_reference_mismatches,
              'docs_reference_transformations':reference_stage_summary,
              'strip_induced_reference_error_count':len(strip_induced_error_ids),
              'strip_induced_reference_error_ids':strip_induced_error_ids,
              'false_perfect_tree_rows':false_perfect,'fallback_summary':policy_summary,
              'rescued_ids':selected.loc[selected.rescued,'ID'].astype(int).tolist(),
              'unaccepted_ids':selected.loc[~selected.fallback_accepted,'ID'].astype(int).tolist(),
              'environment':{'python':platform.python_version(),**{pkg:importlib.metadata.version(pkg) for pkg in ['numpy','pandas','scipy','statsmodels','PyYAML','matplotlib']}},
              'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}}
    (OUT/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:metadata[k] for k in ['rows','unique_description_code_pairs','unique_urls','url_cluster_bootstrap_gpt4o_minus_gpt41','fallback_summary','false_perfect_tree_rows']},indent=2))

if __name__=='__main__':
    main()
