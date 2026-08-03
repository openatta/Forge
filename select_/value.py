"""Teacher-student gap + coverage quota combined selection — P1.

Per docs/reference/04 §四, value signals ranked by empirical validity: teacher-student gap is
strongest, coverage marginal contribution (empty > weak > saturated cells) second, student-NLL
learnability is explicitly skipped here (the doc calls it noisy in practice).
"""
from __future__ import annotations

from core.ledger import sample_id
from core.schemas import Attempt, Question, Sample, VerifyResult
from select_.rejection import pick_best_passing_attempt
from taxonomy.quota import allocate_quota


def compute_question_stats(
    train_questions: list[Question],
    attempts: list[Attempt],
    verify_by_attempt: dict[str, VerifyResult],
) -> dict[str, dict]:
    """Per-question p_T (teacher high-temp pass rate), p_S (student pass fraction), gap=max(0,p_T-p_S)."""
    stats: dict[str, dict] = {}
    for q in train_questions:
        q_attempts = [a for a in attempts if a.question_id == q.id]
        teacher_hi = [a for a in q_attempts if a.actor_role == "teacher" and a.sampling.strategy == "high_temp"]
        student = [a for a in q_attempts if a.actor_role == "student"]

        def pass_rate(subset: list[Attempt]) -> float:
            if not subset:
                return 0.0
            passed = sum(1 for a in subset if verify_by_attempt.get(a.id) and verify_by_attempt[a.id].passed)
            return passed / len(subset)

        p_t = pass_rate(teacher_hi)
        p_s = pass_rate(student)
        stats[q.id] = {"p_T": p_t, "p_S": p_s, "gap": max(0.0, p_t - p_s)}
    return stats


def select_by_value(
    train_questions: list[Question],
    attempts: list[Attempt],
    verify_by_attempt: dict[str, VerifyResult],
    matrix: dict,
    target_size: int,
    batch_id: str,
    config_hash: str,
) -> list[Sample]:
    """Two-pass selection up to target_size:
    Pass 1 guarantees each cell up to its quota share (allocate_quota), sorted by gap within the
    cell -- this is what protects empty/weak cells from being crowded out by a few high-gap cells.
    Pass 2 fills any remaining budget from the best leftover candidates globally by gap, regardless
    of cell -- saturated cells only get extras once every cell's fair share is met.
    """
    stats = compute_question_stats(train_questions, attempts, verify_by_attempt)
    quota = allocate_quota(matrix, total=target_size)

    eligible_by_cell: dict[str, list[Question]] = {}
    for q in train_questions:
        if pick_best_passing_attempt(q, attempts, verify_by_attempt) is None:
            continue  # no passing teacher attempt to train on -- not eligible regardless of value score
        eligible_by_cell.setdefault(q.cell, []).append(q)
    for questions in eligible_by_cell.values():
        questions.sort(key=lambda q: stats[q.id]["gap"], reverse=True)

    selected: list[Question] = []
    selected_ids: set[str] = set()

    for cell, questions in eligible_by_cell.items():
        take = quota.get(cell, 0)
        for q in questions[:take]:
            selected.append(q)
            selected_ids.add(q.id)

    if len(selected) < target_size:
        leftover = [q for questions in eligible_by_cell.values() for q in questions if q.id not in selected_ids]
        leftover.sort(key=lambda q: stats[q.id]["gap"], reverse=True)
        selected.extend(leftover[: target_size - len(selected)])

    samples: list[Sample] = []
    for q in selected[:target_size]:
        best = pick_best_passing_attempt(q, attempts, verify_by_attempt)
        samples.append(
            Sample(
                id=sample_id(q.id, "sft", batch_id, best.id),
                question_id=q.id,
                attempt_id=best.id,
                kind="sft",
                payload={"content": best.content},
                batch_id=batch_id,
                config_hash=config_hash,
            )
        )
    return samples
