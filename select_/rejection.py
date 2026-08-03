"""Rejection sampling: k candidates -> verify filter -> pick shortest correct solution."""
from __future__ import annotations

from core.ledger import sample_id
from core.schemas import Attempt, Question, Sample, VerifyResult


def pick_best_passing_attempt(
    question: Question,
    attempts: list[Attempt],
    verify_by_attempt: dict[str, VerifyResult],
) -> Attempt | None:
    """Among this question's teacher attempts, keep only passing ones and pick the shortest by
    len(content); ties broken by lowest sampling.index for determinism. None if none passed.

    Shared by every selection strategy (rejection sampling, value-based, stratified-random baseline)
    so they only ever differ in *which questions* get included, never in *how the training text for
    a chosen question is rendered* -- isolating the selection variable, per docs/reference/04's
    confounding-variable discipline for preference pairs, generalized here to arm comparisons.
    """
    teacher_attempts = [a for a in attempts if a.question_id == question.id and a.actor_role == "teacher"]
    passing = [a for a in teacher_attempts if verify_by_attempt.get(a.id) is not None and verify_by_attempt[a.id].passed]
    if not passing:
        return None
    return min(passing, key=lambda a: (len(a.content), a.sampling.index))


def select_rejection(
    question: Question,
    attempts: list[Attempt],
    verify_by_attempt: dict[str, VerifyResult],
    batch_id: str,
    config_hash: str,
) -> Sample | None:
    best = pick_best_passing_attempt(question, attempts, verify_by_attempt)
    if best is None:
        return None
    return Sample(
        id=sample_id(question.id, "sft", batch_id, best.id),
        question_id=question.id,
        attempt_id=best.id,
        kind="sft",
        payload={"content": best.content},
        batch_id=batch_id,
        config_hash=config_hash,
    )


def latest_samples(samples: list[Sample]) -> list[Sample]:
    """samples.jsonl is an append-only history (sample_id includes attempt_id, so a re-selected
    attempt for the same question produces a new row rather than overwriting the old one). This
    resolves "what should actually be compiled right now": last-write-wins per question_id, relying
    on JsonlStore.all() returning rows in append (i.e. chronological) order."""
    by_question: dict[str, Sample] = {}
    for s in samples:
        by_question[s.question_id] = s
    return list(by_question.values())


def compute_pass_rates(
    train_questions: list[Question],
    attempts: list[Attempt],
    verify_by_attempt: dict[str, VerifyResult],
) -> dict:
    """Returns {p_T: teacher pass rate (high-temp attempts only), p_S: student(weak) pass rate}."""
    train_ids = {q.id for q in train_questions}

    def pass_rate(role: str, strategy: str | None) -> float:
        relevant = [
            a for a in attempts
            if a.question_id in train_ids and a.actor_role == role and (strategy is None or a.sampling.strategy == strategy)
        ]
        if not relevant:
            return 0.0
        passed = sum(1 for a in relevant if verify_by_attempt.get(a.id) and verify_by_attempt[a.id].passed)
        return passed / len(relevant)

    return {
        "p_T": pass_rate("teacher", "high_temp"),
        "p_S": pass_rate("student", "weak"),
    }
