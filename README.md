# distill-mvp

Minimal end-to-end LLM distillation pipeline: question generation → teacher supervision collection →
verification → rejection-sampling selection → SFT compilation → isolated evaluation → report. Built
per `docs/90-MVP搭建指南.md`.

## Quick start (P0 smoke test)

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

`pip install -e .` alone installs only what P0 actually imports. `bespokelabs-curator`/`duckdb`/`datasketch`
are the P1+ tech-stack choices named in the build spec but unused by any P0 code path — `bespokelabs-curator`
in particular pulls a very heavy transitive tree (full Google Cloud AI Platform SDK, matplotlib, pandas,
datasets/pyarrow, mistralai, anthropic, instructor). Install them only once P1 work needs them:
`pip install -e '.[p1]'`.

Individual stages can also be run on their own: `python run.py gen|collect|verify|select|compile|eval|report`,
plus `python run.py trace <sample_id>` to walk a sample's full ledger chain back to its source question.

## Status legend

| Symbol | Meaning |
|---|---|
| ✅ | P0 — real, working implementation |
| 🔶 | P1/P2/P3 — file exists with real signatures, body raises `NotImplementedError` |
| ⬜ | Pure placeholder — interface not yet finalized, raises `NotImplementedError` |

(One deliberate exception: `datagen/dedup.py`'s `exact_dedup()` is real despite the file's overall 🔶
tag, because the P0 smoke flow must demonstrably catch a duplicated seed question. Only
`minhash_semantic_dedup()` in that file is a stub.)

## Module map

- `core/` — shared schemas (pydantic), the unified `ModelClient` interface (teacher via LiteLLM, mock
  student), and the content-hash-based ledger/ID system that makes every stage resumable and traceable.
- `taxonomy/` — coverage-matrix definition and quota allocation.
- `datagen/` — question generation (P0: reads seed jsonl) and dedup.
- `collect/` — per-question k-sample orchestration against teacher/student clients.
- `envs/` — Agent environment abstraction (interface only in P0; `toy_calc` etc. ship in P2).
- `verify/` — pluggable verifiers; `math_answer.py` is the P0 sympy-based checker.
- `select_/` — rejection sampling (P0) and value-based selection (P1).
- `compile_/` — renders selected samples into TRL-ready SFT jsonl.
- `train/` — TRL training scripts (P1+, needs GPU).
- `evals/` — isolated holdout eval + report generation.
- `docs/` — this project's build spec (`90-MVP搭建指南.md`) plus general distillation theory in
  `docs/reference/` (background reading, not specific to this codebase).
