"""Build reports: P0 smoke report + P1 per-cell A/B comparison report."""
from __future__ import annotations

import math
from pathlib import Path

from collect.sample import CollectStats
from core.ledger import Ledger
from core.schemas import EvalRecord


def build_smoke_report(
    *,
    seed_count: int,
    dedup_dropped: int,
    train_count: int,
    holdout_count: int,
    collect_stats: CollectStats,
    total_teacher_cost_usd: float,
    pass_rates: dict,
    sft_count: int,
    eval_records: list[EvalRecord],
    ledger: Ledger,
    example_sample_id: str | None,
) -> str:
    cost_per_question = total_teacher_cost_usd / train_count if train_count else 0.0

    cell_rows = []
    for rec in eval_records:
        cell_rows.append(f"| {rec.cell} | {rec.question_id} | {'PASS' if rec.passed else 'FAIL'} |")
    cell_table = "\n".join(cell_rows) if cell_rows else "| (no holdout eval records) | | |"

    trace_block = "(no sample available to trace)"
    if example_sample_id:
        chain = ledger.trace(example_sample_id)
        trace_block = "\n".join(f"- `{e.type}` **{e.id}** (upstream: {e.upstream or 'none'})" for e in chain)

    return f"""# Smoke Report

## Question generation
- Seed lines read: {seed_count}
- Exact-dedup dropped: {dedup_dropped}
- Train questions: {train_count}
- Holdout questions: {holdout_count}

## Teacher collection
- Teacher attempts total (this run's train pool): {collect_stats.teacher_total} (newly collected: {collect_stats.teacher_new})
- Student(weak) attempts total: {collect_stats.student_total} (newly collected: {collect_stats.student_new})
- Teacher cost total: ${total_teacher_cost_usd:.4f}
- Cost per question: ${cost_per_question:.4f}

## Verification / selection
- Teacher pass rate (p_T, high-temp attempts): {pass_rates.get('p_T', 0):.2%}
- Student(weak) pass rate (p_S): {pass_rates.get('p_S', 0):.2%}
- SFT samples selected (rejection sampling, 1 per solved question): {sft_count}

## Holdout eval (MockStudent)
| cell | question_id | result |
|---|---|---|
{cell_table}

## Ledger traceability example
Sample `{example_sample_id or 'N/A'}` traced root-first:

{trace_block}
"""


def write_smoke_report(report_text: str, path: Path = Path("reports/smoke_report.md")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion -- more honest than a normal approximation
    at the small per-cell n docs/reference/06 warns about ("<50题时置信区间宽得惊人")."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def _cell_success(records: list[EvalRecord], cell: str) -> tuple[int, int]:
    cell_records = [r for r in records if r.cell == cell]
    return sum(1 for r in cell_records if r.passed), len(cell_records)


def build_p1_report(
    *,
    matrix: dict,
    arm_eval_records: dict[str, list[EvalRecord]],
    ledger: Ledger,
    example_sample_ids: dict[str, str | None],
) -> str:
    """Per-cell success rate + 95% Wilson CI for each arm, macro average, worst-3 cells (per
    docs/reference/06 §二: "报告单位是能力格子"). arm_eval_records keys are typically
    base/baseline_random/value_selected/teacher (docs/reference/06 §3.2's standard control groups,
    minus "上轮学生" which only applies from round 2 onward).
    """
    arms = list(arm_eval_records.keys())
    cells = [c["id"] for c in matrix["cells"]]

    header = "| cell | " + " | ".join(arms) + " |"
    sep = "|---|" + "|".join(["---"] * len(arms)) + "|"
    rows = []
    macro: dict[str, list[float]] = {arm: [] for arm in arms}
    for cell in cells:
        row = [cell]
        for arm in arms:
            successes, n = _cell_success(arm_eval_records[arm], cell)
            if n == 0:
                row.append("n/a")
                continue
            rate = successes / n
            lo, hi = wilson_ci(successes, n)
            row.append(f"{rate:.0%} (n={n}, CI [{lo:.0%},{hi:.0%}])")
            macro[arm].append(rate)
        rows.append("| " + " | ".join(row) + " |")

    macro_row = ["**macro avg**"] + [
        f"{(sum(macro[arm]) / len(macro[arm])):.0%}" if macro[arm] else "n/a" for arm in arms
    ]
    rows.append("| " + " | ".join(macro_row) + " |")

    ranking_arm = "value_selected" if "value_selected" in arm_eval_records else arms[0]
    cell_rates = []
    for cell in cells:
        successes, n = _cell_success(arm_eval_records[ranking_arm], cell)
        if n > 0:
            cell_rates.append((cell, successes / n))
    worst = sorted(cell_rates, key=lambda x: x[1])[:3]
    worst_block = "\n".join(f"- {cell}: {rate:.0%}" for cell, rate in worst) if worst else "(no data)"

    trace_lines = []
    for arm, sid in example_sample_ids.items():
        if not sid:
            continue
        chain = ledger.trace(sid)
        trace_lines.append(f"- **{arm}**: " + " -> ".join(f"{e.type}:{e.id}" for e in chain))
    trace_block = "\n".join(trace_lines) if trace_lines else "(no samples to trace)"

    return f"""# P1 Report

> **MOCK TRAINING NOTICE**: the `base`/`baseline_random`/`value_selected` arms below share the exact
> same (untrained) MockStudent weights -- `train/sft.py` is in mock mode this round, so their eval
> numbers are expected to come out numerically identical. This validates that the full pipeline
> (generation -> dedup -> collection -> selection -> compilation -> "training" -> evaluation ->
> reporting) works end to end. It does NOT show whether value-selected data beats random data --
> that question can only be answered once `mode="real"` training runs on actual GPU compute.
> The **teacher** arm is a real number (the teacher LLM itself, evaluated on the same holdout set).

## Per-cell success rate ({len(cells)} cells)
{header}
{sep}
{chr(10).join(rows)}

## Worst 3 cells (ranked by `{ranking_arm}` arm)
{worst_block}

## Ledger traceability (one sample per arm)
{trace_block}
"""
