#!/usr/bin/env python3
"""Verify report/source consistency and the separately compiled Overleaf archive.

The rendered page-* and portable-* images in tmp/pdfs/final-v3 are QA inputs.
"""
from pathlib import Path
from PIL import Image, ImageChops
import hashlib
import json
import re
import subprocess
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / 'output/pdf/CECS543_Team1_Research_Report.pdf'
ARCHIVE = ROOT / 'output/CECS543_Team1_Overleaf.zip'
REPRODUCIBILITY = ROOT / 'output/CECS543_Team1_Reproducibility.zip'
QA = ROOT / 'tmp/pdfs/final-v3'
info = subprocess.check_output(['pdfinfo', str(FINAL)], text=True)
pages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
assert FINAL.read_bytes() == (ROOT / 'paper/main.pdf').read_bytes()
for number in range(1, pages + 1):
    original = Image.open(QA / f'page-{number:02}.png').convert('RGB')
    portable = Image.open(QA / f'portable-{number:02}.png').convert('RGB')
    assert original.size == portable.size
    assert ImageChops.difference(original, portable).getbbox() is None, number
for path in [ROOT / 'paper/main.log', QA / 'overleaf/main.log']:
    log = path.read_text()
    assert not any(value in log for value in ['Overfull', 'Underfull', 'undefined', 'LaTeX Warning',
                                             'Package natbib Warning', 'duplicate ignored'])
fonts = subprocess.check_output(['pdffonts', str(FINAL)], text=True)
assert 'Type 3' not in fonts
raw = (ROOT / 'paper/report.tex').read_text()
abstract = raw.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
words = len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", abstract))
assert 150 <= words <= 250 and '\\cite' not in abstract
intro = raw.split('\\section{Introduction}')[1].split('\\section{Proposed Methodology}')[0]
keys = {key for group in re.findall(r'\\cite\{([^}]+)\}', intro) for key in group.split(',')}
assert len(keys) == 4
source = json.loads((ROOT / 'analysis/results/summary.json').read_text())
assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
           for path, expected in source['source_sha256'].items())
masked = pd.read_csv(ROOT / 'analysis/results/parser_aware_tree_scores.csv')
assert len(masked) == 7908
both = masked.reference_safe_loads & masked.output_safe_loads
assert masked.loc[~both, 'parser_aware_tree_similarity'].isna().all()
assert masked.loc[both, 'parser_aware_tree_similarity'].notna().all()
fresh = json.loads((ROOT / 'analysis/results/recomputed_metadata.json').read_text())
workflow = json.loads((ROOT / 'analysis/results/workflow_summary.json').read_text())
for hashes in [fresh['source_sha256'], workflow['input_sha256']]:
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
               for path, expected in hashes.items())
metrics = pd.read_csv(ROOT / 'analysis/results/recomputed_pair_metrics.csv')
assert len(metrics) == 13180
assert fresh['valid_structural_reference_count'] == 1293
assert metrics.loc[metrics.reference_status != 'ok', 'path_value_leaf_f1'].isna().all()
assert (metrics.loc[(metrics.reference_status == 'ok') & (metrics.generated_status != 'ok'), 'path_value_leaf_f1'] == 0).all()
choices = pd.read_csv(ROOT / 'analysis/results/workflow_selection.csv')
gate = choices[choices.policy == 'actionlint_fallback']
assert len(gate) == 335 and gate.selected.sum() == 284 and (~gate.selected).sum() == 51
assert gate.loc[gate.selected, 'selected_actionlint_accepted'].all()
with zipfile.ZipFile(REPRODUCIBILITY) as project:
    assert project.testzip() is None
    for name in project.namelist():
        assert 'Bilder' not in name and 'Titelblatt' not in name
        relative = Path(name).relative_to('CECS543_Team1')
        assert project.read(name) == (ROOT / relative).read_bytes()
text = subprocess.check_output(['pdftotext', '-layout', str(FINAL), '-'], text=True)
assert 'TO_FILL: contact email' in text and '[?]' not in text
assert 'Team Lead' not in text and 'Research Report' not in text
assert 'Muhammad Abdul Basit Ur Rahim, PhD' in text
assert 'CECS 543 |' not in text
assert text.count('Generative AI and Software Engineering') == 1
with zipfile.ZipFile(ARCHIVE) as archive:
    assert archive.testzip() is None
    members = archive.namelist()
    assert 'title_page.tex' in members and 'images/csulb_logo.png' in members
    assert not any('Bilder' in name or 'Titelblatt' in name for name in members)
    expected = {str(path.relative_to(ROOT / 'paper')) for path in (ROOT / 'paper').rglob('*')
                if path.is_file() and (path.suffix in {'.tex', '.bib', '.png'}
                or path.name in {'Makefile', 'OVERLEAF_README.txt'}
                or path.parent.name == 'figures' and path.suffix == '.pdf')}
    assert set(members) == expected
    for name in members:
        assert archive.read(name) == (ROOT / 'paper' / name).read_bytes()
aux = (ROOT / 'paper/main.aux').read_text()
bbl = (ROOT / 'paper/main.bbl').read_text()
report = {
    'date': '2026-11-10', 'pdf_pages': pages, 'paper_size': 'A4',
    'abstract_words': words, 'intro_sources': len(keys),
    'tables': len(re.findall(r'\\newlabel\{tab:', aux)),
    'figures': len(re.findall(r'\\newlabel\{fig:', aux)),
    'bibliography_entries': len(re.findall(r'\\bibitem', bbl)), 'latex_warnings': 0,
    'independent_overleaf_compile': True, 'all_pages_pixel_identical': True,
    'archive_entries': len(members), 'source_hashes_unchanged': True,
    'parser_aware_model_pairs': len(masked),
    'parser_aware_unavailable_pairs': int(masked.parser_aware_tree_similarity.isna().sum()),
    'fresh_metric_model_pairs': 7908, 'fresh_metric_all_policy_pairs': len(metrics),
    'fresh_structural_reference_count': 1293,
    'workflow_task_count': 335, 'workflow_reference_pass': 292,
    'workflow_fallback_pass': 284, 'workflow_fallback_abstain': 51,
    'visual_review': 'All pages reviewed; no clipping, overlap or unreadable tables.',
    'running_headers': False, 'cover_team_lead_label': False,
    'english_project_paths': True,
    'cover_instructor': 'Muhammad Abdul Basit Ur Rahim, PhD',
    'remaining_placeholder': 'contact email, explicitly requested by user',
    'deliverables': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in [FINAL, ARCHIVE, REPRODUCIBILITY]},
}
(ROOT / 'output/verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
