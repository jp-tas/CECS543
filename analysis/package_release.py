#!/usr/bin/env python3
"""Package the final PDF, standalone Overleaf sources, and reproducible project."""
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / 'paper'
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
(OUT / 'pdf').mkdir(exist_ok=True)
shutil.copyfile(PAPER / 'main.pdf', OUT / 'pdf/CECS543_Team1_Research_Report.pdf')


def manuscript_files():
    return sorted(path for path in PAPER.rglob('*') if path.is_file() and (
        path.suffix in {'.tex', '.bib', '.png'}
        or path.name in {'Makefile', 'OVERLEAF_README.txt'}
        or path.parent.name == 'figures' and path.suffix == '.pdf'))


def archive(path, entries):
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as target:
        for source, member in sorted(entries, key=lambda entry: str(entry[1])):
            target.write(source, str(member))
    with zipfile.ZipFile(path) as target:
        assert target.testzip() is None
        assert len(target.namelist()) == len(set(target.namelist()))
        for source, member in entries:
            assert target.read(str(member)) == source.read_bytes()
    print(f'{path.name}: {len(entries)} files, {path.stat().st_size:,} bytes')


manuscript = manuscript_files()
archive(OUT / 'CECS543_Team1_Overleaf.zip', [(path, path.relative_to(PAPER)) for path in manuscript])
project = manuscript + [ROOT/'README.md', ROOT/'.gitignore',
                        ROOT/'2507.17165v1.pdf',
                        ROOT/'Topic, Guidelines and Template for Research Report - Section 1.pdf',
                        OUT/'pdf/CECS543_Team1_Research_Report.pdf']
project += [path for path in (ROOT/'analysis').rglob('*') if path.is_file()
            and '__pycache__' not in path.parts and path.suffix in {'.py','.txt','.csv','.json','.pdf'}]
project += [path for path in (ROOT/'ICSME25-LLM4CI-main').rglob('*')
            if path.is_file() and not any(part.startswith('.') for part in path.relative_to(ROOT/'ICSME25-LLM4CI-main').parts)]
project = sorted(set(project))
archive(OUT / 'CECS543_Team1_Reproducibility.zip',
        [(path, Path('CECS543_Team1')/path.relative_to(ROOT)) for path in project])
