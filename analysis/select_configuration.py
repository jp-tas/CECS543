#!/usr/bin/env python3
"""Select the first statically acceptable archived YAML candidate.

Candidate paths define the fixed priority order. The tool never executes
workflow commands and never consults a reference answer.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml
from validate_workflows import DEFAULT_BINARY, VERSION, check_workflow, ensure_binary


def select(paths: list[Path], scope: str, binary: Path) -> tuple[str | None, dict]:
    decisions = []
    for path in paths:
        candidate = path.read_text(encoding='utf-8')
        try:
            value = yaml.load(candidate, Loader=yaml.BaseLoader)
            accepted = value is not None and (not hasattr(value, '__len__') or len(value) > 0)
            result = {'accepted': accepted, 'status': 'parser_accepted' if accepted else 'empty_document'}
        except yaml.YAMLError as error:
            result = {'accepted': False, 'status': 'parse_failure', 'diagnostics': [str(error)]}
        if scope == 'workflow' and result['accepted']:
            result = check_workflow(candidate, binary)
        decisions.append({'candidate': str(path), **result})
        if result['accepted']:
            return candidate, {'scope': scope, 'selected': str(path), 'attempted': len(decisions),
                               'status': 'accepted', 'decisions': decisions}
    return None, {'scope': scope, 'selected': None, 'attempted': len(decisions),
                  'status': 'abstained', 'decisions': decisions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidates', type=Path, nargs='+', help='Candidate files in fixed priority order.')
    parser.add_argument('--scope', choices=['fragment', 'workflow'], required=True,
                        help='Declared task scope; complete workflows receive actionlint checks.')
    parser.add_argument('--actionlint', type=Path, default=DEFAULT_BINARY)
    parser.add_argument('--output', type=Path, help='Write selected text here instead of stdout.')
    parser.add_argument('--decision-json', type=Path, help='Write selection diagnostics here.')
    args = parser.parse_args()
    if args.scope == 'workflow':
        if args.actionlint == DEFAULT_BINARY and not args.actionlint.is_file():
            args.actionlint = ensure_binary()
        version = subprocess.check_output([str(args.actionlint), '-version'], text=True).splitlines()[0]
        if version != VERSION:
            parser.error(f'Expected actionlint {VERSION}, received {version}.')
    text, result = select(args.candidates, args.scope, args.actionlint)
    if args.decision_json:
        args.decision_json.write_text(json.dumps(result, indent=2) + '\n')
    if text is None:
        print('No candidate passed the declared static checks.', file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    else:
        sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
