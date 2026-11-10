# CECS 543 Research Report - Team 1

**Research topic:** Generative AI and Software Engineering.

**Selected article:** Taher A. Ghaleb and Dulina Rathnayake (2025),
*Can LLMs Write CI? A Study on Automatic Generation of GitHub Actions Configurations*.

**Report title:** *Replication and Evaluation of LLM-Generated GitHub Actions Workflows*.

The completed English report uses the established IS545/HW10 cover layout,
adapted to CECS 543 and the four authors: Revathi Eedala Venkatesulu,
Jean-Pierre Tas, Chi Kien Ngo, and Sahiti Vuyyuru. The cover includes the
research topic and instructor. Body pages use centered page numbers without
running headers.

## Project structure

```text
paper/
  main.tex
  title_page.tex
  preamble.tex
  report.tex
  appendix.tex
  references.bib
  images/
  figures/
  tables/
analysis/
  results/
output/
  pdf/
ICSME25-LLM4CI-main/
  data/
  scripts/
  results/
```

## Deliverables

- `output/pdf/CECS543_Team1_Research_Report.pdf`: final report.
- `output/CECS543_Team1_Overleaf.zip`: standalone LaTeX project; upload to Overleaf and compile `main.tex` with pdfLaTeX.
- `output/CECS543_Team1_Reproducibility.zip`: report sources, analysis code, numerical results, original data, source article, assignment guidelines, and original license.
- `output/verification.json`: build, archive, input-integrity, and numerical checks.

## Completed experiments

1. Reanalysis of all archived similarity columns, paired model tests, description-length correlations, duplicate-aware aggregation, and URL-cluster bootstrap intervals.
2. Independent raw-text ROUGE-L F1, raw maximum-style ROUGE ablation, chrF, structural path/value overlap, and a root-feature sensitivity analysis for all 7,908 model/reference pairs.
3. Four reference-free selection policies, evaluated on the same tasks and accompanied by lexical/structural tradeoffs and explicit failure states.
4. Static workflow validation with official, checksum-verified actionlint 1.7.7 on a predefined 335-example complete-workflow-shaped subset, including all six models and references.
5. Fixed checker-guided fallback, explicit abstention, and a reusable selector CLI.
6. Workflow acceptance under equal task-pair and source-page weighting, paired URL-cluster intervals, and fixed candidate prefixes from one to six archived outputs.

The main structural result is a consensus-selector gain that survives removal
of root-kind credit. Its lexical agreement decreases. Checker fallback accepts
284/335 archived workflow candidates, compared with 264/335 for the strongest
individual model; it abstains on 51 examples. These are retrospective structural
and static-check outcomes. No new model requests or workflow executions are
claimed, and functional correctness is not inferred from checker acceptance.
The 335 workflow rows contain 175 exact task pairs and 127 source pages.
Fallback improves coverage over GPT-4.1 under all three weighting schemes;
five candidates already reach the full six-candidate coverage. This uses a
larger candidate pool than the individual-model baseline and does not establish
a benefit under equal inference budgets.

The appendix maps reported findings to evidence files and includes raw,
automatically extracted reference/candidate examples for IDs 82, 28, and 42.

## Reproduce

Use Python 3.12 and a TeX Live installation with pdfLaTeX, BibTeX, `latexmk`,
Latin Modern, TikZ, natbib, and the IEEEtran bibliography style.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r analysis/requirements.txt -r analysis/requirements-metrics.txt
.venv/bin/python analysis/run_analysis.py
.venv/bin/python analysis/recompute_metrics.py
.venv/bin/python analysis/validate_workflows.py
.venv/bin/python analysis/audit_workflow_robustness.py
.venv/bin/python analysis/build_paper_assets.py
make -C paper
.venv/bin/python analysis/package_release.py
.venv/bin/python analysis/verify_release.py
```

The workflow-validation script automatically obtains actionlint **1.7.7** from
its official GitHub release on macOS/Linux arm64/amd64, verifies the release
archive checksum, and stores provenance in `.tools/actionlint/1.7.7/`.
Alternatively use `--binary /path/to/actionlint`; the version must match.
`--install-only` prepares the checker; `--refresh` repeats every check without
the content cache. A complete uncached repetition reproduced the workflow
result files byte for byte. ShellCheck and Pyflakes integrations are disabled.

To compile only the manuscript from its included tables and figures, run
`make -C paper`. The Overleaf ZIP requires neither Python nor dataset access.

The verifier requires Poppler (`pdfinfo`, `pdffonts`, `pdftoppm`, `pdftotext`)
and freshly extracts, compiles, renders, and compares the Overleaf archive.
After visually inspecting the current rendered pages, repeat verification with
`--visual-review-confirmed` to record that manual check. The separate
`output/verification.json` is not included inside the reproducibility ZIP.

## Use the selection framework

Supply candidates in the fixed priority order appropriate to the application:

```sh
.venv/bin/python analysis/select_configuration.py --scope workflow --output selected.yml --decision-json decision.json candidate1.yml candidate2.yml candidate3.yml
```

`--scope workflow` applies parsing and the pinned checker; `--scope fragment`
uses the nonempty parser gate. The CLI selects the first candidate accepted
under the declared scope. It exits with status 2 and emits a diagnostic record
if every candidate is rejected. It never executes generated commands or
consults a reference answer. Structural consensus/minimality experiments are
implemented separately in `recompute_metrics.py`.

## Evidence

The analysis scripts export per-example CSVs, settings, dependency
versions, and SHA-256 input hashes in `analysis/results/`. The original
`ICSME25-LLM4CI-main/` data and code remain unchanged. Its README states MIT for
code and CC BY 4.0 for data. The report credits the original authors and
distinguishes their experiment from the new analyses.

`analysis/results/reproduction_audit.json` records an independent isolated
repetition: all 39 prior CSV result files were byte-identical, and a fresh
uncached actionlint run reproduced the saved checker decisions.

The contact line was removed at the author's request. No contact placeholder
or Overleaf invitation information appears in the manuscript.
