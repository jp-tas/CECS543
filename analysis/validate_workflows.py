#!/usr/bin/env python3
"""Static GitHub Actions validation of the archived full-workflow task subset.

No workflow, action, shell command, or model is executed. Only the pinned
actionlint executable is invoked, with its external shellcheck/pyflakes
integrations disabled. Run from any directory with Python and PyYAML 6.0.2.
The public check_workflow(text, binary) API is also used by the selector CLI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import subprocess
import tarfile
import tempfile
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'ICSME25-LLM4CI-main' / 'data'
OUT = ROOT / 'analysis' / 'results'
VERSION = '1.7.7'
RELEASE_DATE = '2025-01-19T12:01:22Z'
TOOL_DIR = ROOT / '.tools' / 'actionlint' / VERSION
DEFAULT_BINARY = TOOL_DIR / ('actionlint.exe' if platform.system() == 'Windows' else 'actionlint')
MODELS = ['gpt-4o', 'gpt-4.1', 'gemma3-12b', 'llama3.1-8b', 'codegemma-7b', 'codellama-7b']
FLAGS = ['-no-color', '-shellcheck=', '-pyflakes=', '-format', '{{json .}}',
         '-stdin-filename', 'workflow.yml', '-']
CACHE_FORMAT = 1


def sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode('utf-8') if isinstance(data, str) else data).hexdigest()


def ensure_binary() -> Path:
    """Fetch only the pinned official release; verify its official checksum."""
    if DEFAULT_BINARY.is_file():
        provenance_file = TOOL_DIR / 'provenance.json'
        if not provenance_file.is_file():
            raise RuntimeError('Existing local binary has no recorded official-release provenance.')
        provenance = json.loads(provenance_file.read_text())
        if provenance.get('binary_sha256') != sha256(DEFAULT_BINARY.read_bytes()):
            raise RuntimeError('Existing local binary does not match its recorded official-release hash.')
        return DEFAULT_BINARY
    operating_system = {'Darwin': 'darwin', 'Linux': 'linux'}.get(platform.system())
    architecture = {'arm64': 'arm64', 'aarch64': 'arm64', 'x86_64': 'amd64',
                    'AMD64': 'amd64'}.get(platform.machine())
    if not operating_system or not architecture:
        raise RuntimeError('Supply --binary for this platform; automatic download supports macOS/Linux arm64/amd64.')
    name = f'actionlint_{VERSION}_{operating_system}_{architecture}.tar.gz'
    base = f'https://github.com/rhysd/actionlint/releases/download/v{VERSION}/'
    checksum_url = base + f'actionlint_{VERSION}_checksums.txt'
    checksums = urllib.request.urlopen(checksum_url, timeout=60).read()
    expected = {line.split()[1]: line.split()[0] for line in checksums.decode().splitlines()}[name]
    archive = urllib.request.urlopen(base + name, timeout=60).read()
    if sha256(archive) != expected:
        raise RuntimeError('Official release archive checksum mismatch.')
    TOOL_DIR.mkdir(parents=True, exist_ok=True)
    (TOOL_DIR / name).write_bytes(archive)
    (TOOL_DIR / 'checksums.txt').write_bytes(checksums)
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as bundle:
        # Extract named regular files only, never archive paths or symlinks.
        for member_name in ['actionlint', 'LICENSE.txt']:
            try:
                member = bundle.getmember(member_name)
            except KeyError:
                continue
            if not member.isfile():
                raise RuntimeError(f'Unexpected archive member: {member_name}')
            (TOOL_DIR / member_name).write_bytes(bundle.extractfile(member).read())
    DEFAULT_BINARY.chmod(0o755)
    version_output = subprocess.check_output([str(DEFAULT_BINARY), '-version'], text=True)
    provenance = {'version': VERSION, 'release_published_at': RELEASE_DATE,
                  'source': base + name, 'checksums_source': checksum_url,
                  'archive_sha256': sha256(archive), 'binary_sha256': sha256(DEFAULT_BINARY.read_bytes()),
                  'platform': f'{operating_system}_{architecture}', 'version_output': version_output}
    (TOOL_DIR / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return DEFAULT_BINARY


def check_workflow(text: str, binary: Path | str = DEFAULT_BINARY) -> dict:
    """Check raw text in an isolated cwd, retaining complete JSON diagnostics.

    accepted is strictly actionlint exit 0, not functional correctness. A
    tool_error status is distinct from an ordinary static rejection.
    No repository configuration or local/reusable workflow files are supplied.
    """
    executable = str(Path(binary).expanduser().resolve())
    with tempfile.TemporaryDirectory(prefix='llm4ci-static-') as neutral_directory:
        process = subprocess.run([executable, *FLAGS], input=text, text=True,
                                 capture_output=True, cwd=neutral_directory, timeout=30,
                                 check=False)
    diagnostics = []
    protocol_error = ''
    if process.stdout.strip():
        try:
            diagnostics = json.loads(process.stdout)
            if diagnostics is None:
                diagnostics = []
            if not isinstance(diagnostics, list):
                protocol_error = 'actionlint JSON output is not an array'
        except json.JSONDecodeError as error:
            protocol_error = str(error)
    tool_error = process.returncode not in (0, 1) or bool(protocol_error)
    return {'accepted': process.returncode == 0 and not tool_error,
            'status': 'tool_error' if tool_error else 'accepted' if process.returncode == 0 else 'rejected',
            'returncode': process.returncode, 'diagnostics': diagnostics,
            'stdout': process.stdout, 'stderr': process.stderr,
            'protocol_error': protocol_error, 'text_sha256': sha256(text)}


def parse_text(text: str) -> tuple[object, str]:
    try:
        return yaml.load(text, Loader=yaml.BaseLoader), ''
    except yaml.YAMLError as error:
        return None, str(error)


def syntax_gate(text: str) -> bool:
    value, error = parse_text(text)
    return not error and value is not None and (not isinstance(value, (dict, list, str)) or bool(value))


def classify_reference(text: str) -> tuple[str, str]:
    value, error = parse_text(text)
    if error:
        return 'parse_error', error
    if isinstance(value, dict) and 'on' in value and isinstance(value.get('jobs'), dict):
        return 'full_workflow_shape', ''
    if isinstance(value, dict) and 'runs' in value:
        return 'action_definition_shape', ''
    if isinstance(value, dict):
        return 'mapping_fragment', ''
    if isinstance(value, list):
        return 'sequence_fragment', ''
    return 'scalar_or_empty', ''


def dependency_flags(text: str) -> dict:
    """Flag unavailable repository context, without assuming dependencies exist."""
    uses = re.findall(r'^\s*(?:-\s*)?uses:\s*[\'\"]?([^\s\'\"#]+)', text, flags=re.MULTILINE)
    local = [value for value in uses if value.startswith('./')]
    reusable = [value for value in uses if '/.github/workflows/' in value]
    return {'uses_local_dependency': bool(local), 'uses_reusable_workflow': bool(reusable),
            'local_dependencies': local, 'reusable_workflows': reusable}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        path.write_text('')
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, help='Pinned actionlint 1.7.7 binary; otherwise use/download the local binary.')
    parser.add_argument('--refresh', action='store_true', help='Re-run all unique texts instead of using the verified content cache.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--install-only', action='store_true', help='Install or verify the pinned local binary, then exit.')
    args = parser.parse_args()
    binary = (args.binary or ensure_binary()).resolve()
    version_output = subprocess.check_output([str(binary), '-version'], text=True)
    if version_output.splitlines()[0] != VERSION:
        raise RuntimeError(f'Expected actionlint {VERSION}, received {version_output!r}')
    binary_hash = sha256(binary.read_bytes())
    provenance_path = binary.parent / 'provenance.json'
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    if provenance.get('binary_sha256') and provenance['binary_sha256'] != binary_hash:
        raise RuntimeError('Binary no longer matches its recorded provenance.')
    if args.install_only:
        print(json.dumps({'binary': str(binary), 'version_output': version_output,
                          'binary_sha256': binary_hash, 'provenance': provenance}, indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    docs = read_rows(SOURCE / 'GitHubActions_Docs.csv')
    outputs = {model: {row['ID']: row for row in read_rows(SOURCE / f'GitHubActions_{model}_output.csv')}
               for model in MODELS}
    assert len({row['ID'] for row in docs}) == len(docs)
    assert all(set(rows) == {row['ID'] for row in docs} for rows in outputs.values())
    scope_rows, selected_docs = [], []
    for doc in docs:
        category, error = classify_reference(doc['Code'])
        scope_rows.append({'ID': doc['ID'], 'category': category, 'in_full_workflow_scope': category == 'full_workflow_shape',
                           'URL': doc['URL'], 'Title': doc['Title'], 'reference_sha256': sha256(doc['Code']),
                           'classification_error': error})
        if category == 'full_workflow_shape':
            selected_docs.append(doc)
    write_csv(OUT / 'workflow_scope.csv', scope_rows)
    texts = {}
    for doc in selected_docs:
        texts[sha256(doc['Code'])] = doc['Code']
        for model in MODELS:
            text = outputs[model][doc['ID']]['LLM_YML_Output']
            texts[sha256(text)] = text
    cache_path = OUT / 'workflow_lint_cache.json'
    cache_metadata = {'format': CACHE_FORMAT, 'actionlint_version': VERSION, 'flags': FLAGS}
    cache = {'metadata': cache_metadata, 'results': {}}
    if cache_path.exists() and not args.refresh:
        previous = json.loads(cache_path.read_text())
        if previous.get('metadata') == cache_metadata:
            cache = previous
    pending = [(digest, text) for digest, text in texts.items() if digest not in cache['results']]
    def lint_item(item):
        digest, text = item
        return digest, check_workflow(text, binary)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for digest, result in executor.map(lint_item, pending):
            cache['results'][digest] = result
    errors = [digest for digest in texts if cache['results'][digest]['status'] == 'tool_error']
    if errors:
        raise RuntimeError(f'Linter tool failures for {len(errors)} unique texts; cannot compute acceptance statistics.')
    cache['results'] = {digest: cache['results'][digest] for digest in sorted(texts)}
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True) + '\n')
    detail_rows, diagnostic_rows, selection_rows = [], [], []
    per_row = {}
    for doc in selected_docs:
        identifier = doc['ID']
        candidate_results = {}
        for model in ['reference', *MODELS]:
            text = doc['Code'] if model == 'reference' else outputs[model][identifier]['LLM_YML_Output']
            digest = sha256(text)
            result = cache['results'][digest]
            dependencies = dependency_flags(text)
            record = {'ID': identifier, 'model': model, 'text_sha256': digest,
                      'syntax_accepted': syntax_gate(text), 'actionlint_accepted': result['accepted'],
                      'actionlint_status': result['status'], 'returncode': result['returncode'],
                      'diagnostic_count': len(result['diagnostics']),
                      'diagnostic_kinds': '|'.join(sorted({error['kind'] for error in result['diagnostics']})),
                      **{key: json.dumps(value) if isinstance(value, list) else value for key, value in dependencies.items()},
                      'exact_match_outer_strip': text.strip() == doc['Code'].strip(),
                      'diagnostics_json': json.dumps(result['diagnostics'], ensure_ascii=False, sort_keys=True)}
            detail_rows.append(record)
            for diagnostic in result['diagnostics']:
                diagnostic_rows.append({'ID': identifier, 'model': model, 'kind': diagnostic.get('kind', ''),
                                        'line': diagnostic.get('line', ''), 'column': diagnostic.get('column', ''),
                                        'message': diagnostic.get('message', ''), 'snippet': diagnostic.get('snippet', '')})
            candidate_results[model] = record
        per_row[identifier] = candidate_results
        for policy, accepted_key in [('syntax_fallback', 'syntax_accepted'), ('actionlint_fallback', 'actionlint_accepted')]:
            chosen = next((model for model in MODELS if candidate_results[model][accepted_key]), None)
            chosen_record = candidate_results[chosen] if chosen else None
            selection_rows.append({'ID': identifier, 'policy': policy, 'selected_model': chosen or '',
                                   'selected': chosen is not None, 'reference_actionlint_accepted': candidate_results['reference']['actionlint_accepted'],
                                   'selected_syntax_accepted': chosen_record['syntax_accepted'] if chosen else False,
                                   'selected_actionlint_accepted': chosen_record['actionlint_accepted'] if chosen else False,
                                   'exact_match_outer_strip': chosen_record['exact_match_outer_strip'] if chosen else False,
                                   'uses_local_dependency': chosen_record['uses_local_dependency'] if chosen else False,
                                   'uses_reusable_workflow': chosen_record['uses_reusable_workflow'] if chosen else False,
                                   'selected_text_sha256': chosen_record['text_sha256'] if chosen else '',
                                   'gpt4o_actionlint_accepted': candidate_results['gpt-4o']['actionlint_accepted']})
    write_csv(OUT / 'workflow_lint_details.csv', detail_rows)
    write_csv(OUT / 'workflow_lint_diagnostics.csv', diagnostic_rows)
    write_csv(OUT / 'workflow_selection.csv', selection_rows)
    diagnostic_summary = []
    for model in ['reference', *MODELS]:
        records = [row for row in diagnostic_rows if row['model'] == model]
        for kind in sorted({row['kind'] for row in records}):
            matching = [row for row in records if row['kind'] == kind]
            diagnostic_summary.append({'model': model, 'kind': kind, 'diagnostic_count': len(matching),
                                       'affected_rows': len({row['ID'] for row in matching})})
    write_csv(OUT / 'workflow_diagnostic_summary.csv', diagnostic_summary)
    summary_rows, model_counts = [], []
    scopes = {'full_workflow_shape': [doc['ID'] for doc in selected_docs],
              'reference_actionlint_accepted': [doc['ID'] for doc in selected_docs if per_row[doc['ID']]['reference']['actionlint_accepted']]}
    for scope, identifiers in scopes.items():
        id_set = set(identifiers)
        for model in ['reference', *MODELS]:
            records = [per_row[identifier][model] for identifier in identifiers]
            summary_rows.append({'scope': scope, 'policy_or_model': model, 'n': len(records), 'selected': len(records), 'abstained': 0,
                                 'syntax_accepted': sum(row['syntax_accepted'] for row in records),
                                 'actionlint_accepted': sum(row['actionlint_accepted'] for row in records),
                                 'exact_matches_outer_strip': sum(row['exact_match_outer_strip'] for row in records),
                                 'selected_local_dependency': sum(row['uses_local_dependency'] for row in records),
                                 'selected_reusable_workflow': sum(row['uses_reusable_workflow'] for row in records)})
        for policy in ['syntax_fallback', 'actionlint_fallback']:
            records = [row for row in selection_rows if row['ID'] in id_set and row['policy'] == policy]
            summary_rows.append({'scope': scope, 'policy_or_model': policy, 'n': len(records),
                                 'selected': sum(row['selected'] for row in records), 'abstained': sum(not row['selected'] for row in records),
                                 'syntax_accepted': sum(row['selected_syntax_accepted'] for row in records),
                                 'actionlint_accepted': sum(row['selected_actionlint_accepted'] for row in records),
                                 'exact_matches_outer_strip': sum(row['exact_match_outer_strip'] for row in records),
                                 'selected_local_dependency': sum(row['uses_local_dependency'] for row in records),
                                 'selected_reusable_workflow': sum(row['uses_reusable_workflow'] for row in records)})
            for model, count in Counter(row['selected_model'] or 'abstain' for row in records).items():
                model_counts.append({'scope': scope, 'policy': policy, 'selected_model': model, 'count': count})
    write_csv(OUT / 'workflow_summary.csv', summary_rows)
    write_csv(OUT / 'workflow_chosen_models.csv', model_counts)
    input_files = [SOURCE / 'GitHubActions_Docs.csv', *[SOURCE / f'GitHubActions_{model}_output.csv' for model in MODELS]]
    category_counts = dict(Counter(row['category'] for row in scope_rows))
    metadata = {'experiment': 'Static domain validation of archived complete-workflow-shaped tasks',
                'scope_definition': 'Original Docs Code loads with PyYAML BaseLoader as a mapping containing a jobs mapping and the string key on.',
                'category_definitions': {
                    'full_workflow_shape': 'BaseLoader mapping containing string key on and a mapping-valued jobs key; tested first.',
                    'action_definition_shape': 'Other BaseLoader mapping containing a runs key; a shape indicator, not proof of a complete or valid action metadata file.',
                    'mapping_fragment': 'Other BaseLoader mapping.',
                    'sequence_fragment': 'BaseLoader sequence.',
                    'parse_error': 'PyYAML BaseLoader raises YAMLError.',
                    'scalar_or_empty': 'Remaining scalar or null BaseLoader result.'},
                'classification_counts': category_counts, 'selection_order': MODELS,
                'selection_uses_reference': False,
                'reference_use': 'Reference shape declares task scope; reference actionlint status defines a separately reported stratum; reference text is used only for retrospective exact matching.',
                'actionlint': {'version': VERSION, 'release_published_at': RELEASE_DATE,
                               'flags': FLAGS, 'binary_sha256': binary_hash, 'version_output': version_output,
                               'official_provenance': provenance},
                'pyyaml_version': yaml.__version__, 'unique_linted_texts': len(texts),
                'row_text_checks': len(detail_rows), 'summary': summary_rows,
                'scope_distinct_reference_texts': len({doc['Code'] for doc in selected_docs}),
                'scope_distinct_source_urls': len({doc['URL'] for doc in selected_docs}),
                'diagnostic_kinds': dict(Counter(row['kind'] for row in diagnostic_rows)),
                'input_sha256': {str(path.relative_to(ROOT)): sha256(path.read_bytes()) for path in input_files},
                'limitations': [
                    'Passing actionlint is a static intermediate gate, not evidence of functional correctness or task-intent satisfaction.',
                    'No workflows, actions, generated shell code, Python code, model requests, or external linter integrations are executed.',
                    'Inputs are raw archived texts. No comments, indentation, placeholders, action versions, or generated configurations are repaired.',
                    'Each text is checked through stdin in a fresh neutral temporary directory, with no repository configuration or companion files.',
                    'Local actions and reusable workflows may require missing repository context; local dependency and reusable-workflow indicators are separately exported.',
                    'Remote action/reusable-workflow existence, secrets, permissions at runtime, external services, and behavioral outcomes are not validated.',
                    'The complete-workflow-shaped reference subset includes duplicated documentation tasks and is not an independent test set.',
                    'The fixed order was declared for the offline experiment on this archived candidate pool; gains do not establish generalization or latency/cost benefits.',
                    'The 2025-01-19 linter version reduces temporal drift but cannot exactly reconstruct the original authors environment.'
                ]}
    (OUT / 'workflow_summary.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'scope_counts': category_counts, 'unique_linted_texts': len(texts),
                      'summary': summary_rows}, indent=2))


if __name__ == '__main__':
    main()
