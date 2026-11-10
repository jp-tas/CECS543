#!/usr/bin/env python3
"""Recompute explicit raw-text and parser-aware metrics without model calls.

Run with .venv-metrics/bin/python analysis/recompute_metrics.py.
Every reference is taken from unmodified GitHubActions_Docs.csv. No comments,
indentation, or markup are removed from the lexical inputs. Structural reference
errors are excluded; generated structural errors score zero on valid references.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rapidfuzz.distance import LCSseq
from sacrebleu.metrics import CHRF
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from yaml.nodes import MappingNode, SequenceNode, ScalarNode

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'ICSME25-LLM4CI-main'
OUT=ROOT/'analysis/results'
MODELS=['gpt-4o','gpt-4.1','gemma3-12b','llama3.1-8b','codegemma-7b','codellama-7b']
POLICIES=['syntax_fallback','strict_fallback','consensus_medoid','minimal_structure']
SCORES=['rouge_l_f1','rouge_l_max','chrf','path_value_f1','path_value_leaf_f1']
SEED=2026
BOOTSTRAPS=10000
CHRF_METRIC=CHRF(char_order=6,word_order=0,beta=2,lowercase=False,whitespace=False,eps_smoothing=False)

class StructuralError(ValueError):
    pass

@lru_cache(maxsize=None)
def structure(text):
    """Return (status, immutable typed path/value features, short diagnostic).

    BaseLoader preserves scalar spelling. Mapping order is ignored; sequence
    positions are distinct path segments. Scalar tags, a root-kind token, and
    empty-container tokens prevent empty structures from collapsing together.
    No construction/merging or domain schema interpretation is performed.
    """
    try:
        root=yaml.compose(text,Loader=yaml.BaseLoader)
    except yaml.YAMLError as error:
        return 'parse_error',frozenset(),str(error).splitlines()[0]
    if root is None:
        return 'empty_document',frozenset(),'No YAML root node'
    features=set();active=set();visits=0
    def walk(node,path):
        nonlocal visits
        visits+=1
        if visits>100000:raise StructuralError('expansion_budget')
        if id(node) in active:raise StructuralError('cyclic_alias')
        active.add(id(node))
        try:
            if isinstance(node,ScalarNode):
                features.add(('scalar',path,node.tag,node.value))
            elif isinstance(node,SequenceNode):
                if not node.value:features.add(('empty_sequence',path,node.tag,''))
                for idx,child in enumerate(node.value):walk(child,path+(('index',idx),))
            elif isinstance(node,MappingNode):
                if not node.value:features.add(('empty_mapping',path,node.tag,''))
                keys=set()
                for key,value in node.value:
                    if not isinstance(key,ScalarNode):raise StructuralError('complex_mapping_key')
                    if key.value in keys:raise StructuralError('duplicate_mapping_key')
                    keys.add(key.value)
                    walk(value,path+(('key',key.value),))
            else:raise StructuralError('unsupported_node')
        finally:active.remove(id(node))
    try:
        kind='mapping' if isinstance(root,MappingNode) else 'sequence' if isinstance(root,SequenceNode) else 'scalar'
        features.add(('root_kind',(),kind,''))
        walk(root,())
    except StructuralError as error:
        return str(error),frozenset(),str(error)
    return 'ok',frozenset(features),''

@lru_cache(maxsize=None)
def syntax_accepted(text):
    try:value=yaml.load(text,Loader=yaml.BaseLoader)
    except yaml.YAMLError:return False
    return value is not None and (not isinstance(value,(dict,list,str)) or len(value)>0)

@lru_cache(maxsize=None)
def reference_form(text):
    if structure(text)[0]!='ok':return 'invalid_or_unsupported'
    value=yaml.load(text,Loader=yaml.BaseLoader)
    return ('full_workflow_shape' if isinstance(value,dict) and 'on' in value
            and isinstance(value.get('jobs'),dict) else 'other_snippet')

@lru_cache(maxsize=None)
def rouge_f1(reference,candidate):
    a=reference.split();b=candidate.split()
    if not a or not b:return 0.0
    return 2.0*LCSseq.similarity(a,b)/(len(a)+len(b))

@lru_cache(maxsize=None)
def rouge_max(reference,candidate):
    """Original max(P,R,F1) formula, now on exactly the same raw text inputs."""
    a=reference.split();b=candidate.split()
    if not a or not b:return 0.0
    return LCSseq.similarity(a,b)/min(len(a),len(b))

@lru_cache(maxsize=None)
def chrf(reference,candidate):
    if not reference or not candidate:return 0.0
    return CHRF_METRIC.sentence_score(candidate,[reference]).score/100.0


def set_f1(reference,candidate):
    if not reference or not candidate:return 0.0
    return 2.0*len(reference & candidate)/(len(reference)+len(candidate))


def evaluate(reference,candidate):
    rs,rf,_=structure(reference);gs,gf,_=structure(candidate)
    structural=set_f1(rf,gf) if rs=='ok' and gs=='ok' else 0.0 if rs=='ok' else np.nan
    leaf_structural=(set_f1(frozenset(f for f in rf if f[0]!='root_kind'),frozenset(f for f in gf if f[0]!='root_kind'))
                     if rs=='ok' and gs=='ok' else 0.0 if rs=='ok' else np.nan)
    return {'rouge_l_f1':rouge_f1(reference,candidate),'rouge_l_max':rouge_max(reference,candidate),'chrf':chrf(reference,candidate),
            'path_value_f1':structural,
            'path_value_leaf_f1':leaf_structural,
            'path_value_f1_conditional':structural if rs=='ok' and gs=='ok' else np.nan,
            'reference_status':rs,'generated_status':gs,'reference_features':len(rf),'generated_features':len(gf),
            'structural_exact':rs=='ok' and gs=='ok' and rf==gf,
            'exact_bytes':reference==candidate,'exact_outer_strip':reference.strip()==candidate.strip()}


def choose(candidates,policy):
    if policy=='syntax_fallback':
        return next((m for m in MODELS if syntax_accepted(candidates[m])),None)
    eligible=[m for m in MODELS if structure(candidates[m])[0]=='ok' and syntax_accepted(candidates[m])]
    if not eligible:return None
    if policy=='strict_fallback':return eligible[0]
    if policy=='minimal_structure':return min(eligible,key=lambda m:len(structure(candidates[m])[1]))
    if policy=='consensus_medoid':
        if len(eligible)==1:return eligible[0]
        def agreement(model):
            f=structure(candidates[model])[1]
            return sum(set_f1(f,structure(candidates[other])[1]) for other in eligible if other!=model)/(len(eligible)-1)
        return max(eligible,key=agreement)
    raise ValueError(policy)


def write(records,name):
    frame=records if isinstance(records,pd.DataFrame) else pd.DataFrame(records)
    frame.to_csv(OUT/f'recomputed_{name}.csv',index=False,float_format='%.12g')
    return frame


def selfcheck():
    assert abs(rouge_f1('a b c','b c')-.8)<1e-12
    assert rouge_max('a b c','b c')==1
    assert rouge_f1('a b','a b')==1
    assert structure('a: 1\na: 2')[0]=='duplicate_mapping_key'
    assert structure('x: &loop\n  x: *loop')[0]=='cyclic_alias'
    assert structure('')[0]=='empty_document'
    assert set_f1(structure('{}')[1],structure('[]')[1])==0
    assert structure('on: true\nx: 1')[1]==structure('x: 1\non: true')[1]
    assert structure('x: [a,b]')[1]!=structure('x: [b,a]')[1]
    assert structure('"a/b": c')[1]!=structure('a: {b: c}')[1]
    assert structure('v: "true"')[1]==structure('v: true')[1]
    assert evaluate('a: 1','bad: [')['path_value_f1']==0
    assert np.isnan(evaluate('bad: [','a: 1')['path_value_f1'])


def cluster_interval(frame,difference):
    blocks=frame.assign(difference=difference).groupby('URL',sort=True).difference.agg(['sum','count'])
    rng=np.random.default_rng(SEED)
    draws=rng.integers(0,len(blocks),size=(BOOTSTRAPS,len(blocks)))
    estimates=blocks['sum'].to_numpy()[draws].sum(axis=1)/blocks['count'].to_numpy()[draws].sum(axis=1)
    low,high=np.quantile(estimates,[.025,.975])
    return float(low),float(high),len(blocks)


def main():
    selfcheck();OUT.mkdir(parents=True,exist_ok=True)
    docs=pd.read_csv(SOURCE/'data/GitHubActions_Docs.csv').fillna('')
    assert docs.ID.is_unique
    docs['pair_group']=docs.groupby(['Description','Code'],sort=False,dropna=False).ngroup()
    outputs={}
    for m in MODELS:
        raw=pd.read_csv(SOURCE/f'data/GitHubActions_{m}_output.csv').set_index('ID')
        assert raw.index.is_unique and set(raw.index)==set(docs.ID)
        raw=raw.loc[docs.ID].reset_index()
        assert raw.Description.tolist()==docs.Description.tolist()
        outputs[m]=raw.LLM_YML_Output.fillna('').tolist()
    records=[];diagnostics=[];selected=[];selected_outputs=[]
    for idx,row in docs.iterrows():
        ref=row.Code
        rs,rf,re=structure(ref)
        diagnostics.append({'ID':int(row.ID),'source':'reference_docs','status':rs,'diagnostic':re,'features':len(rf)})
        candidates={m:outputs[m][idx] for m in MODELS}
        for m,text in candidates.items():
            gs,gf,ge=structure(text)
            diagnostics.append({'ID':int(row.ID),'source':m,'status':gs,'diagnostic':ge,'features':len(gf)})
            records.append({'ID':int(row.ID),'URL':row.URL,'pair_group':int(row.pair_group),'reference_form':reference_form(ref),'method':m,**evaluate(ref,text)})
        for policy in POLICIES:
            chosen=choose(candidates,policy);text=candidates[chosen] if chosen else ''
            chosen_metrics=evaluate(ref,text)
            records.append({'ID':int(row.ID),'URL':row.URL,'pair_group':int(row.pair_group),'reference_form':reference_form(ref),'method':policy,**chosen_metrics})
            selected.append({'ID':int(row.ID),'policy':policy,'selected_model':chosen or 'none',
                             'syntax_accepted':bool(chosen and syntax_accepted(text)),
                             'strict_accepted':bool(chosen and structure(text)[0]=='ok'),
                             'changed_from_gpt4o':chosen!='gpt-4o'})
            selected_outputs.append({'ID':int(row.ID),'policy':policy,'selected_model':chosen or 'none','selected_yaml':text})
    pairs=write(records,'pair_metrics')
    diag=write(diagnostics,'parse_diagnostics')
    write(diag.groupby(['source','status'],sort=False).size().reset_index(name='count'),'parse_status_summary')
    selections=write(selected,'policy_selection')
    write(selected_outputs,'selected_outputs')
    summaries=[]
    for method,group in pairs.groupby('method',sort=False):
        for subset,frame in [('all_references',group),('valid_reference',group[group.reference_status=='ok']),
                             ('conditional_both_valid',group[(group.reference_status=='ok')&(group.generated_status=='ok')])]:
            summaries.append({'method':method,'subset':subset,'n_rows':len(frame),
                              'n_path_value_scored':int(frame.path_value_f1.notna().sum()),
                              'generated_strict_ok':int(frame.generated_status.eq('ok').sum()),
                              **{f'{metric}_mean':float(frame[metric].mean()) for metric in SCORES},
                              **{f'{metric}_equal_pair_mean':float(frame.groupby('pair_group')[metric].mean().mean()) for metric in SCORES},
                              **{f'{metric}_equal_url_mean':float(frame.groupby('URL')[metric].mean().mean()) for metric in SCORES},
                              'exact_bytes':int(frame.exact_bytes.sum()),'exact_outer_strip':int(frame.exact_outer_strip.sum()),
                              'structural_exact':int(frame.structural_exact.sum())})
    summary=write(summaries,'summary')
    strata=[]
    for (method,form),frame in pairs[pairs.reference_status=='ok'].groupby(['method','reference_form'],sort=False):
        strata.append({'method':method,'reference_form':form,'n':len(frame),
                       **{f'{metric}_mean':float(frame[metric].mean()) for metric in SCORES},
                       'generated_strict_ok':int(frame.generated_status.eq('ok').sum()),
                       'structural_exact':int(frame.structural_exact.sum())})
    write(strata,'reference_form_summary')
    write(selections.groupby(['policy','selected_model'],sort=False).size().reset_index(name='count'),'policy_selection_counts')
    comparisons=[]
    for comparator in ['gpt-4o','syntax_fallback','strict_fallback']:
        baseline=pairs[pairs.method==comparator].set_index('ID')
        for policy in POLICIES:
            if policy==comparator:continue
            other=pairs[pairs.method==policy].set_index('ID')
            for metric in SCORES:
                valid=baseline[metric].notna() & other[metric].notna()
                base=baseline.loc[valid];diff=other.loc[valid,metric]-base[metric]
                if np.all(diff==0):stat,p=0.0,1.0
                else:stat,p=wilcoxon(diff,alternative='two-sided',zero_method='wilcox',method='auto')
                low,high,clusters=cluster_interval(base,diff)
                comparisons.append({'policy':policy,'comparator':comparator,'metric':metric,'n':len(diff),
                                    'reference_scope':'valid_reference' if metric.startswith('path_value') else 'all_references',
                                    'mean_difference':float(diff.mean()),'median_difference':float(diff.median()),
                                    'wins':int((diff>0).sum()),'ties':int((diff==0).sum()),'losses':int((diff<0).sum()),
                                    'wilcoxon_statistic':stat,'p_raw':p,'cluster_bootstrap_95_low':low,
                                    'cluster_bootstrap_95_high':high,'clusters':clusters,'seed':SEED,'bootstraps':BOOTSTRAPS})
    comparisons=pd.DataFrame(comparisons)
    for (comparator,metric),frame in comparisons.groupby(['comparator','metric'],sort=False):
        comparisons.loc[frame.index,'p_holm_within_metric_comparator']=multipletests(frame.p_raw,method='holm')[1]
    write(comparisons,'policy_comparisons')
    old=pd.read_csv(SOURCE/'results/GitHubActions_Similarity_Scores_Six_LLMs.csv')
    ranking=[]
    for m in MODELS:
        s=summary[(summary.method==m)&(summary.subset=='all_references')].iloc[0]
        ranking.append({'model':m,'stored_cosine_mean':old[f'CosineSimilarity ({m})'].mean(),
                        'stored_rouge_variant_mean':old[f'ROUGE-L ({m})'].mean(),'stored_chrf_variant_mean':old[f'chrF ({m})'].mean(),
                        'stored_tree_variant_mean':old[f'TreeEditDistance ({m})'].mean(),
                        **{f'recomputed_{metric}_mean':s[f'{metric}_mean'] for metric in SCORES}})
    ranking=pd.DataFrame(ranking)
    for col in list(ranking.columns)[1:]:ranking[col+'_rank']=ranking[col].rank(ascending=False,method='min').astype(int)
    write(ranking,'rankings')
    files=[SOURCE/'data/GitHubActions_Docs.csv',*[SOURCE/f'data/GitHubActions_{m}_output.csv' for m in MODELS],SOURCE/'results/GitHubActions_Similarity_Scores_Six_LLMs.csv']
    metadata={
        'scope':'Recomputed raw-text metrics and offline candidate selection; no new generation or execution.',
        'rows':len(docs),'model_output_pairs':len(docs)*len(MODELS),'valid_structural_reference_count':int(diag[(diag.source=='reference_docs')&(diag.status=='ok')].shape[0]),
        'definitions':{
            'rouge_l_f1':'Whitespace tokens, case-sensitive, 2*LCS length/(reference token count+candidate token count); 0 if either token sequence empty. Raw inputs including comments.',
            'rouge_l_max':'Same RAW references, candidates, tokenization and LCS as rouge_l_f1, but LCS/min(reference token count,candidate token count), equivalent to max(P,R,F1). This isolates the scoring-formula change on fixed raw inputs.',
            'chrf':'SacreBLEU 2.5.1 CHRF(char_order=6,word_order=0,beta=2,lowercase=False,whitespace=False,eps_smoothing=False), sentence_score/100; raw inputs including comments.',
            'path_value_f1':'BaseLoader compose; set overlap F1 of scalar (typed tuple path, node tag, exact decoded scalar string), root-kind token, and empty-container tokens. Mapping order ignored; sequence indices matter. No schema/domain execution or YAML scalar typing. Duplicate scalar mapping keys, complex mapping keys, recursive aliases, expansion overflow, parse errors and empty documents are explicit invalid/unsupported statuses.',
            'path_value_leaf_f1':'Robustness ablation of path_value_f1 excluding the root-kind token. Scalar leaf path/value and empty-container tokens are retained. Policies remain unchanged; this tests whether gains depend on shared root-kind credit.',
            'structural_denominator':'Invalid/unsupported Docs references excluded (NA). On all structurally valid references, invalid/unsupported generated outputs score 0. Conditional both-valid means also exported and use smaller explicitly counted denominators.',
            'exact_bytes':'Exact raw string equality. exact_outer_strip additionally removes only surrounding whitespace; neither is functional equivalence.',
            'pair_balancing':'Exact raw Docs Description+Code groups; mean each group before averaging groups.',
            'url_balancing':'Mean within source URL before averaging URLs.',
            'reference_form':'Evaluation-only stratum: full_workflow_shape means a strictly parseable Docs root mapping contains a literal on key and a jobs value that is a mapping (same criterion as validate_workflows.classify_reference); other_snippet means all other strictly parseable nonempty references. This is not a schema or executable-workflow validity claim and is never used for candidate selection.',
            'syntax_fallback':'Existing fixed-order BaseLoader non-null/nonempty acceptance, without duplicate key rejection.',
            'strict_fallback':'First strict-parseable nonempty candidate in fixed model order.',
            'consensus_medoid':'Strict-parseable candidate maximizing mean pairwise path/value F1 with other strict-parseable candidates. Fixed-order tie breaking; no reference used.',
            'minimal_structure':'Strict-parseable candidate with fewest structural features; fixed-order tie breaking. No reference used.',
            'inference':'Two-sided Wilcoxon signed-rank; Holm correction within metric/comparator family. Paired URL-cluster bootstrap resamples URLs retaining all their rows; 10000 draws, seed2026, percentile95CI. Retrospective exploratory analysis; no functional-success claim.'},
        'fixed_order':MODELS,'policies_predefined_before_results':POLICIES,
        'reference_error_ids':{status:frame.ID.astype(int).tolist() for status,frame in diag[diag.source=='reference_docs'].groupby('status') if status!='ok'},
        'environment':{'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','statsmodels','PyYAML','sacrebleu','rapidfuzz']}},
        'chrf_signature':str(CHRF_METRIC.get_signature()),
        'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        'selfchecks':'Passed LCS formula, duplicate keys, recursive aliases, empty doc, container identity, mapping-order invariance, sequence sensitivity, path-key collision avoidance, string-preserving scalar treatment, invalid candidate zero, invalid reference NA.'}
    (OUT/'recomputed_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(summary[summary.subset=='all_references'][['method','n_path_value_scored','rouge_l_f1_mean','chrf_mean','path_value_f1_mean','structural_exact']].to_string(index=False))
    print('REFERENCE STATUSES',diag[diag.source=='reference_docs'].status.value_counts().to_dict())

if __name__=='__main__':main()
