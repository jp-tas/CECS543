#!/usr/bin/env python3
"""Verify the current release with a fresh independent build and fresh renders.

No pre-existing QA images are used. Inspect tmp/pdfs/review-final/page-*.png,
then repeat with --visual-review-confirmed after reviewing this final PDF.
"""
from pathlib import Path, PurePosixPath
from PIL import Image, ImageChops
import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / 'output/pdf/CECS543_Team1_Research_Report.pdf'
ARCHIVE = ROOT / 'output/CECS543_Team1_Overleaf.zip'
REPRODUCIBILITY = ROOT / 'output/CECS543_Team1_Reproducibility.zip'
QA = ROOT / 'tmp/pdfs/review-final'
MODELS = ['gpt-4o', 'gpt-4.1', 'gemma3-12b', 'llama3.1-8b',
          'codegemma-7b', 'codellama-7b']


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(hashes: dict) -> None:
    for relative, expected in hashes.items():
        assert sha256(ROOT / relative) == expected, f'Input changed: {relative}'


def output(command: list[str], **kwargs) -> str:
    return subprocess.check_output(command, text=True, **kwargs)


def manuscript_files() -> set[str]:
    return {str(path.relative_to(ROOT / 'paper'))
            for path in (ROOT / 'paper').rglob('*') if path.is_file() and (
                path.suffix in {'.tex', '.bib', '.png'}
                or path.name in {'Makefile', 'OVERLEAF_README.txt'}
                or path.parent.name == 'figures' and path.suffix == '.pdf')}


def check_manuscript_references() -> dict:
    tex = '\n'.join(path.read_text() for path in (ROOT / 'paper').rglob('*.tex'))
    labels = re.findall(r'\\label\{([^}]+)\}', tex)
    labels += re.findall(r'\blabel\s*=\s*\{?([A-Za-z][A-Za-z0-9:_.-]*)', tex)
    assert len(labels) == len(set(labels)), 'Duplicate LaTeX labels'
    references = set(re.findall(r'\\(?:ref|autoref|pageref)\{([^}]+)\}', tex))
    assert references <= set(labels), f'Undefined labels: {references - set(labels)}'
    bibliography = (ROOT / 'paper/references.bib').read_text()
    bib_keys = set(re.findall(r'@\w+\{([^,]+),', bibliography))
    citations = {key for group in re.findall(r'\\cite\{([^}]+)\}', tex)
                 for key in group.split(',')}
    assert citations <= bib_keys, f'Undefined citations: {citations - bib_keys}'
    for name in re.findall(r'\\input\{([^}]+)\}', tex):
        path = ROOT / 'paper' / name
        assert path.is_file() or path.with_suffix('.tex').is_file(), name
    for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', tex):
        path = ROOT / 'paper' / name
        assert path.is_file() or any(path.with_suffix(ext).is_file()
                                     for ext in ['.pdf', '.png', '.jpg']), name
    checked_paths = []
    appendix = ROOT / 'paper/appendix.tex'
    if appendix.exists():
        for value in re.findall(r'\\path\{([^}]+)\}', appendix.read_text()):
            relative = value.replace(r'\_', '_')
            if relative.startswith(('analysis/', 'paper/', 'output/', 'ICSME25-LLM4CI-main/')):
                if any(char in relative for char in '*?['):
                    assert list(ROOT.glob(relative)), f'Missing appendix path: {relative}'
                else:
                    assert (ROOT / relative).exists(), f'Missing appendix path: {relative}'
                checked_paths.append(relative)
    return {'internal_labels_checked': len(references),
            'citation_keys_checked': len(citations),
            'appendix_file_references_checked': sorted(set(checked_paths))}


def check_workflow_robustness(details: pd.DataFrame) -> dict:
    metadata_path = ROOT / 'analysis/results/workflow_robustness_metadata.json'
    if not metadata_path.exists():
        return {'workflow_robustness_checked': False}
    metadata = json.loads(metadata_path.read_text())
    check_hashes(metadata['source_sha256'])
    assert metadata['order'] == MODELS
    docs = pd.read_csv(ROOT / 'ICSME25-LLM4CI-main/data/GitHubActions_Docs.csv',
                       keep_default_na=False)
    docs['pair_group'] = docs.groupby(['Description', 'Code'], sort=False,
                                      dropna=False).ngroup()
    frame = details.pivot(index='ID', columns='model', values='actionlint_accepted')
    assert not frame.isna().any().any()
    frame = frame.join(docs.set_index('ID')[['URL', 'pair_group']])
    scopes = {'full_workflow_shape': frame,
              'reference_actionlint_accepted': frame[frame.reference]}
    for filename, records in [('weighting', metadata['weighting_results']),
                              ('budget', metadata['budget_results']),
                              ('comparisons', metadata['paired_comparisons'])]:
        table = pd.read_csv(ROOT / f'analysis/results/workflow_robustness_{filename}.csv')
        assert list(table.columns) == list(pd.DataFrame(records).columns)
        assert len(table) == len(records)
        for actual, expected in zip(table.to_dict('records'), records):
            for key, value in expected.items():
                if isinstance(value, (int, float)):
                    assert math.isclose(actual[key], value, rel_tol=1e-10, abs_tol=1e-11), key
                else:
                    assert actual[key] == value, key
    for row in metadata['weighting_results']:
        group = scopes[row['scope']]
        values = (group[MODELS].any(axis=1) if row['method'] == 'actionlint_fallback'
                  else group[row['method']])
        assert row['n_rows'] == len(group) and row['accepted'] == int(values.sum())
        assert row['n_pairs'] == group.pair_group.nunique()
        assert row['n_urls'] == group.URL.nunique()
        for key, expected in [('row_rate', values.mean()),
                              ('equal_pair_rate', values.groupby(group.pair_group).mean().mean()),
                              ('equal_url_rate', values.groupby(group.URL).mean().mean())]:
            assert math.isclose(row[key], expected, abs_tol=1e-12), key
    for row in metadata['budget_results']:
        group = scopes[row['scope']]
        prefix = group[MODELS[:row['prefix_length']]]
        assert row['accepted'] == int(prefix.any(axis=1).sum())
        assert row['n_rows'] == len(group)
        attempts = prefix.apply(lambda values: next((i + 1 for i, accepted in enumerate(values)
                                                      if accepted), len(values)), axis=1)
        assert math.isclose(row['mean_static_checks_until_stop'], attempts.mean(), abs_tol=1e-12)
    for row in metadata['paired_comparisons']:
        group = scopes[row['scope']]
        difference = group[MODELS].any(axis=1).astype(float) - group[row['comparator']].astype(float)
        assert math.isclose(row['row_difference'], difference.mean(), abs_tol=1e-12)
        assert row['rescued'] == int((difference > 0).sum())
        assert row['lost'] == int((difference < 0).sum())
        for weighting in ['row', 'url']:
            low, high = row[f'{weighting}_cluster_95_low'], row[f'{weighting}_cluster_95_high']
            assert math.isfinite(low) and math.isfinite(high) and -1 <= low <= high <= 1
    return {'workflow_robustness_checked': True,
            'workflow_robustness_weighting_rows': len(metadata['weighting_results']),
            'workflow_robustness_budget_rows': len(metadata['budget_results']),
            'workflow_robustness_comparison_rows': len(metadata['paired_comparisons'])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual-review-confirmed', action='store_true',
                        help='Record that all pages of the current final PDF have been manually reviewed.')
    args = parser.parse_args()
    assert FINAL.read_bytes() == (ROOT / 'paper/main.pdf').read_bytes()
    reference_checks = check_manuscript_references()
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        members = archive.namelist()
        assert len(members) == len(set(members))
        assert set(members) == manuscript_files()
        assert 'title_page.tex' in members and 'images/csulb_logo.png' in members
        for name in members:
            member = PurePosixPath(name)
            assert not member.is_absolute() and '..' not in member.parts
            assert 'Bilder' not in name and 'Titelblatt' not in name
            assert archive.read(name) == (ROOT / 'paper' / name).read_bytes()
        if QA.exists():
            shutil.rmtree(QA)
        portable = QA / 'overleaf'
        portable.mkdir(parents=True)
        archive.extractall(portable)
    build = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode',
                            '-halt-on-error', 'main.tex'], cwd=portable,
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (QA / 'overleaf-build.txt').write_text(build.stdout)
    assert build.returncode == 0, f'Independent build failed; see {QA / "overleaf-build.txt"}'
    info = output(['pdfinfo', str(FINAL)])
    pages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
    assert '(A4)' in info
    portable_pages = int(re.search(r'Pages:\s+(\d+)',
                                   output(['pdfinfo', str(portable / 'main.pdf')])).group(1))
    assert portable_pages == pages
    for pdf, prefix in [(FINAL, 'page'), (portable / 'main.pdf', 'portable')]:
        subprocess.run(['pdftoppm', '-r', '120', '-png', str(pdf), str(QA / prefix)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    original_images = sorted(QA.glob('page-*.png'))
    portable_images = sorted(QA.glob('portable-*.png'))
    assert len(original_images) == len(portable_images) == pages
    for number, (original_path, portable_path) in enumerate(zip(original_images, portable_images), 1):
        with Image.open(original_path) as original_image, Image.open(portable_path) as portable_image:
            original, counterpart = original_image.convert('RGB'), portable_image.convert('RGB')
            assert original.size == counterpart.size
            assert ImageChops.difference(original, counterpart).getbbox() is None, number
    for path in [ROOT / 'paper/main.log', portable / 'main.log']:
        log = path.read_text()
        assert not any(value in log for value in ['Overfull', 'Underfull', 'undefined', 'LaTeX Warning',
                                                 'Package natbib Warning', 'duplicate ignored']), path
    fonts = output(['pdffonts', str(FINAL)])
    assert 'Type 3' not in fonts
    assert all(line.split()[-5] == 'yes' for line in fonts.splitlines()[2:] if line.strip())
    pdf_urls = set(re.findall(r'https?://\S+', output(['pdfinfo', '-url', str(FINAL)])))
    bib_urls = set(re.findall(r'url=\{([^}]+)\}', (ROOT / 'paper/references.bib').read_text()))
    assert bib_urls <= pdf_urls, f'Missing PDF URL annotations: {bib_urls - pdf_urls}'
    raw = (ROOT / 'paper/report.tex').read_text()
    abstract = raw.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
    words = len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", abstract))
    assert 150 <= words <= 250 and '\\cite' not in abstract
    intro = raw.split('\\section{Introduction}')[1].split('\\section{Proposed Methodology}')[0]
    keys = {key for group in re.findall(r'\\cite\{([^}]+)\}', intro) for key in group.split(',')}
    assert len(keys) == 4
    source = json.loads((ROOT / 'analysis/results/summary.json').read_text())
    check_hashes(source['source_sha256'])
    audit = json.loads((ROOT / 'analysis/results/reproduction_audit.json').read_text())
    assert audit['csv_files_compared'] == len(audit['csv_files']) == 39
    assert audit['all_csvs_byte_identical'] and audit['fresh_actionlint_cache_logically_identical']
    for row in audit['csv_files']:
        assert row['byte_identical'] and sha256(ROOT / row['file']) == row['sha256']
    assert sha256(ROOT / 'analysis/results/workflow_lint_cache.json') == audit['archived_cache_sha256']
    masked = pd.read_csv(ROOT / 'analysis/results/parser_aware_tree_scores.csv')
    assert len(masked) == 7908
    both = masked.reference_safe_loads & masked.output_safe_loads
    assert masked.loc[~both, 'parser_aware_tree_similarity'].isna().all()
    assert masked.loc[both, 'parser_aware_tree_similarity'].notna().all()
    fresh = json.loads((ROOT / 'analysis/results/recomputed_metadata.json').read_text())
    workflow = json.loads((ROOT / 'analysis/results/workflow_summary.json').read_text())
    for hashes in [fresh['source_sha256'], workflow['input_sha256']]:
        check_hashes(hashes)
    metrics = pd.read_csv(ROOT / 'analysis/results/recomputed_pair_metrics.csv')
    assert len(metrics) == 13180
    assert fresh['valid_structural_reference_count'] == 1293
    assert metrics.loc[metrics.reference_status != 'ok', 'path_value_leaf_f1'].isna().all()
    assert (metrics.loc[(metrics.reference_status == 'ok') & (metrics.generated_status != 'ok'),
                        'path_value_leaf_f1'] == 0).all()
    choices = pd.read_csv(ROOT / 'analysis/results/workflow_selection.csv')
    gate = choices[choices.policy == 'actionlint_fallback']
    assert len(gate) == 335 and gate.selected.sum() == 284 and (~gate.selected).sum() == 51
    assert gate.loc[gate.selected, 'selected_actionlint_accepted'].all()
    details = pd.read_csv(ROOT / 'analysis/results/workflow_lint_details.csv')
    assert not details.duplicated(['ID', 'model']).any()
    assert len(details) == 335 * 7
    assert details.actionlint_accepted.equals(details.returncode.eq(0).rename('actionlint_accepted'))
    robustness_checks = check_workflow_robustness(details)
    with zipfile.ZipFile(REPRODUCIBILITY) as project:
        assert project.testzip() is None
        assert len(project.namelist()) == len(set(project.namelist()))
        for name in project.namelist():
            assert 'Bilder' not in name and 'Titelblatt' not in name
            relative = Path(name).relative_to('CECS543_Team1')
            assert project.read(name) == (ROOT / relative).read_bytes(), name
        assert 'CECS543_Team1/output/verification.json' not in project.namelist()
    text = output(['pdftotext', '-layout', str(FINAL), '-'])
    assert '[?]' not in text
    cover = text.split('\f', 1)[0]
    assert 'Team Lead' not in cover and 'Research Report' not in cover
    assert 'Muhammad Abdul Basit Ur Rahim, PhD' in cover
    assert 'CECS 543 |' not in text
    assert text.count('Generative AI and Software Engineering') == 1
    placeholders = sorted(set(re.findall(r'\[TO_FILL:[^\]]+\]', text)))
    aux = (ROOT / 'paper/main.aux').read_text()
    bbl = (ROOT / 'paper/main.bbl').read_text()
    report = {
        'date': '2026-11-10', 'pdf_pages': pages, 'paper_size': 'A4',
        'abstract_words': words, 'intro_sources': len(keys),
        'tables': len(re.findall(r'\\newlabel\{tab:', aux)),
        'figures': len(re.findall(r'\\newlabel\{fig:', aux)),
        'bibliography_entries': len(re.findall(r'\\bibitem', bbl)), 'latex_warnings': 0,
        'independent_overleaf_compile': True, 'all_pages_pixel_identical': True,
        'fresh_render_directory': str(QA.relative_to(ROOT)), 'render_dpi': 120,
        'rendered_pages_per_pdf': len(original_images),
        'archive_entries': len(members), 'source_hashes_unchanged': True,
        'independently_reproduced_csv_files': audit['csv_files_compared'],
        'fresh_actionlint_cache_reproduced': audit['fresh_actionlint_cache_logically_identical'],
        'pdf_bibliography_urls_checked': len(bib_urls),
        **reference_checks, **robustness_checks,
        'parser_aware_model_pairs': len(masked),
        'parser_aware_unavailable_pairs': int(masked.parser_aware_tree_similarity.isna().sum()),
        'fresh_metric_model_pairs': 7908, 'fresh_metric_all_policy_pairs': len(metrics),
        'fresh_structural_reference_count': 1293,
        'workflow_task_count': len(gate),
        'workflow_reference_pass': int(details[details.model == 'reference'].actionlint_accepted.sum()),
        'workflow_fallback_pass': int(gate.selected.sum()),
        'workflow_fallback_abstain': int((~gate.selected).sum()),
        'visual_review': ('Manually reviewed current final PDF; no clipping, overlap or unreadable tables.'
                          if args.visual_review_confirmed else 'Rendered; manual review required.'),
        'visual_review_confirmed': args.visual_review_confirmed,
        'running_headers': False, 'cover_team_lead_label': False,
        'english_project_paths': True,
        'cover_instructor': 'Muhammad Abdul Basit Ur Rahim, PhD',
        'contact_line': bool(re.search(r'\bContact\s*:', cover)),
        'remaining_placeholders': placeholders,
        'submission_metadata_complete': not placeholders,
        'verification_in_reproducibility_archive': False,
        'deliverables': {str(path.relative_to(ROOT)): sha256(path)
                         for path in [FINAL, ARCHIVE, REPRODUCIBILITY]},
    }
    (ROOT / 'output/verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
