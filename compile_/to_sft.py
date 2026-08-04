"""Render selected Samples into TRL-ready SFT jsonl (student chat template)."""
from __future__ import annotations

from core.ledger import JsonlStore
from core.schemas import Attempt, Question, Sample, SftRecord
from collect.sample import SYSTEM_PROMPT


def render_sft_example(question: Question, attempt: Attempt) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.text},
        {"role": "assistant", "content": attempt.content},
    ]


def compile_sft(
    samples: list[Sample],
    attempts_by_id: dict[str, Attempt],
    questions_by_id: dict[str, Question],
    sft_store: JsonlStore,
) -> int:
    """Renders and persists each sample; returns the count of newly written sft records."""
    written = 0
    for sample in samples:
        question = questions_by_id[sample.question_id]
        attempt = attempts_by_id[sample.attempt_id]
        messages = render_sft_example(question, attempt)
        record = SftRecord(
            id=sample.id,
            messages=messages,
            meta={
                "question_id": question.id,
                "attempt_id": attempt.id,
                "sample_id": sample.id,
                "cell": question.cell,
            },
        )
        if sft_store.append(record):
            written += 1
    return written


def latest_sft_records(records: list[SftRecord]) -> list[SftRecord]:
    """sft.jsonl is an append-only history keyed by sample.id (which itself embeds attempt_id, see
    core.ledger.sample_id), so a re-selected attempt for the same question produces a new row instead
    of overwriting the old one. This resolves "what should actually be trained on right now":
    last-write-wins per question_id, relying on JsonlStore.all() returning rows in append order."""
    by_question: dict[str, SftRecord] = {}
    for r in records:
        by_question[r.meta["question_id"]] = r
    return list(by_question.values())
