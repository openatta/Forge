"""Stratified-random control-group export, matched cell-by-cell to the value-selected set — P1."""
from __future__ import annotations

import random

from core.ledger import sample_id
from core.schemas import Attempt, Question, Sample, VerifyResult
from select_.rejection import pick_best_passing_attempt


def stratified_random_baseline(
    train_questions: list[Question],
    attempts: list[Attempt],
    verify_by_attempt: dict[str, VerifyResult],
    target_cell_counts: dict[str, int],
    batch_id: str,
    config_hash: str,
    seed: int,
) -> list[Sample]:
    """For each cell, randomly draw target_cell_counts[cell] questions (seeded RNG, reproducible)
    from those with >=1 passing teacher attempt -- the same eligibility bar as every other selection
    strategy. Rendered via pick_best_passing_attempt, same as select_by_value(), so the ONLY
    difference between this baseline and the value-selected set is *which questions* were chosen,
    not answer quality (docs/reference/04's confounding-variable discipline) -- and target_cell_counts
    should come from the value-selected set's actual per-cell distribution so the two are the same
    size cell-by-cell (docs/reference/06 §3.1 fixed-budget principle).
    """
    rng = random.Random(seed)
    eligible_by_cell: dict[str, list[Question]] = {}
    for q in train_questions:
        if pick_best_passing_attempt(q, attempts, verify_by_attempt) is None:
            continue
        eligible_by_cell.setdefault(q.cell, []).append(q)

    samples: list[Sample] = []
    for cell, want in target_cell_counts.items():
        pool = sorted(eligible_by_cell.get(cell, []), key=lambda q: q.id)  # deterministic pre-sample order
        chosen = rng.sample(pool, min(want, len(pool)))
        for q in chosen:
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
