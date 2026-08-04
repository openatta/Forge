from datetime import datetime, timezone

from core.ledger import Ledger
from core.schemas import EvalRecord
from evals.report import build_p1_report, wilson_ci


def _eval(cell: str, passed: bool, question_id: str, run_id: str = "run-1") -> EvalRecord:
    return EvalRecord(
        id=f"eval-{run_id}-{question_id}", model_id="m", question_id=question_id,
        passed=passed, cell=cell, run_id=run_id, ts=datetime.now(timezone.utc),
    )


def test_wilson_ci_zero_n_returns_zero_interval():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_bounds_stay_within_unit_interval():
    lo, hi = wilson_ci(3, 5)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_ci_is_conservative_at_small_n():
    lo, hi = wilson_ci(1, 1)  # 100% observed on a single trial
    assert hi <= 1.0
    assert lo < 0.5  # still a wide interval despite a 100% observed rate at n=1


def test_build_p1_report_computes_macro_average_and_worst_cells(tmp_path):
    matrix = {"cells": [{"id": "a"}, {"id": "b"}]}
    arm_eval_records = {
        "value_selected": [_eval("a", True, "qa"), _eval("b", False, "qb")],
        "baseline_random": [_eval("a", True, "qa2"), _eval("b", True, "qb2")],
    }
    ledger = Ledger(tmp_path / "ledger.jsonl")
    report = build_p1_report(
        matrix=matrix, arm_eval_records=arm_eval_records, ledger=ledger, example_sample_ids={}
    )
    assert "| a | 100%" in report
    assert "Worst 3 cells" in report
    assert "b: 0%" in report
