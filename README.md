# Forge

Minimal end-to-end LLM distillation pipeline: question generation → teacher supervision collection →
verification → rejection-sampling selection → SFT compilation → isolated evaluation → report.

## Quick start (smoke test)

```bash
brew install python@3.11          # system python3 is too old (needs >=3.10 for the type hints used here)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env              # fill in TEACHER_API_KEY and TEACHER_MODEL
python run.py smoke               # one command: gen -> collect -> verify -> select -> compile -> eval -> report
cat reports/smoke_report.md
```

Runs in minutes for pennies of API cost — teacher answers come from a real LLM API (via LiteLLM),
the student is mocked (`MockStudent`, mode=weak by default) so no GPU is required. Ctrl-C mid-run
and re-running `python run.py smoke` resumes from whatever's already on disk (all writes are
content-addressed and idempotent).

`pip install -e .` alone installs only what the smoke path actually imports. `bespokelabs-curator`/
`duckdb`/`datasketch` are heavier, larger-scale tooling unused by the smoke path — `bespokelabs-curator`
in particular pulls a very heavy transitive tree (full Google Cloud AI Platform SDK, matplotlib, pandas,
datasets/pyarrow, mistralai, anthropic, instructor). Install them only once you need them:
`pip install -e '.[p1]'`.

Individual stages can also be run on their own: `python run.py gen|collect|verify|select|compile|eval|report`,
plus `python run.py trace <sample_id>` to walk a sample's full ledger chain back to its source question.

## Quick start (teacher-generated dataset + value selection vs. random baseline)

```bash
python run.py p1
cat reports/p1_report.md
```

Generates a larger question set directly from the teacher LLM across a coverage matrix (topic ×
difficulty), dedups it, collects teacher/student attempts, selects a training set two ways (teacher-
student-gap value selection vs. a stratified-random baseline matched cell-by-cell), and reports
per-cell success rates with confidence intervals across base/value/baseline/teacher arms. Training
itself is currently mocked (validates the data → checkpoint → eval plumbing without needing a GPU);
swapping in real training doesn't require touching anything upstream of it.

## Status legend

| Symbol | Meaning |
|---|---|
| ✅ | real, working implementation |
| 🔶 | file exists with real signatures, body raises `NotImplementedError` (ships later) |
| ⬜ | pure placeholder — interface not yet finalized, raises `NotImplementedError` |

(One deliberate exception: `datagen/dedup.py`'s `exact_dedup()` is real despite the file's overall 🔶
tag, because the smoke flow must demonstrably catch a duplicated seed question. Only
`minhash_semantic_dedup()` in that file was a stub before the value-selection work landed.)

## Module map

- `core/` — shared schemas (pydantic), the unified `ModelClient` interface (teacher via LiteLLM, mock
  student), and the content-hash-based ledger/ID system that makes every stage resumable and traceable.
- `taxonomy/` — coverage-matrix definition and quota allocation.
- `datagen/` — question generation (seed jsonl and teacher-driven generation) and dedup.
- `collect/` — per-question k-sample orchestration against teacher/student clients.
- `envs/` — Agent environment abstraction (interface only for now).
- `verify/` — pluggable verifiers; `math_answer.py` is the sympy-based checker.
- `select_/` — rejection sampling, teacher-student-gap value selection, stratified-random baseline.
- `compile_/` — renders selected samples into TRL-ready SFT jsonl.
- `train/` — training scripts; SFT is currently a mock that validates the pipeline without a GPU.
- `evals/` — isolated holdout eval + report generation.
