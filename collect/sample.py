"""Per-question k-sample orchestration: asyncio + resume + full persist.

Each attempt's id is a deterministic function of (question_id, role, sampling_key, sampling
fingerprint), so this module skips any attempt already present in attempts_store before making an
API call, and persists each successful call immediately (under a lock) rather than batching at the
end -- a Ctrl-C mid-run leaves all completed attempts durably on disk for the next invocation to
resume from. The fingerprint covers model_id/temperature/max_tokens so editing a config value (or
switching the underlying model) for an already-collected slot produces a new id and gets recollected,
rather than silently resuming as if nothing changed.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from core.hashing import short_hash
from core.ledger import JsonlStore, Ledger, attempt_id, now
from core.model_client import ModelClient
from core.schemas import Attempt, Question, Sampling

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a careful math problem solver. Work through the problem, then end your response with "
    "a line of the form 'Final Answer: <answer>'. For multiple solutions, separate them with commas."
)


@dataclass
class CollectStats:
    teacher_new: int
    teacher_total: int
    student_new: int
    student_total: int


def _build_messages(question_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]


async def _collect_one(
    question: Question,
    role: str,
    sampling_key: str,
    sampling: Sampling,
    client: ModelClient,
    attempts_store: JsonlStore,
    ledger: Ledger,
    config_hash: str,
    write_lock: asyncio.Lock,
) -> bool:
    """Returns True if a new attempt was actually written (False if already present or the call failed)."""
    fingerprint = short_hash(
        {"model_id": client.model_id, "temperature": sampling.temperature, "max_tokens": sampling.max_tokens}, 6
    )
    aid = attempt_id(question.id, role, sampling_key, fingerprint)
    if attempts_store.has(aid):
        return False
    messages = _build_messages(question.text)
    try:
        result = await client.chat(messages, temperature=sampling.temperature, max_tokens=sampling.max_tokens)
    except Exception as exc:  # noqa: BLE001
        logger.error("collect failed for %s (%s): %s -- will retry on next run", aid, role, exc)
        return False
    attempt = Attempt(
        id=aid,
        question_id=question.id,
        actor_model=client.model_id,
        actor_role=role,
        sampling=sampling,
        messages=messages,
        content=result.content,
        usage=result.usage,
        ts=now(),
        config_hash=config_hash,
    )
    async with write_lock:
        written = attempts_store.append(attempt)
        if written:
            ledger.record(attempt.id, "attempt", upstream=[question.id], config_hash=config_hash)
    return written


async def run_collect(
    config: dict,
    train_questions: list[Question],
    teacher: ModelClient,
    student: ModelClient,
    attempts_store: JsonlStore,
    ledger: Ledger,
    config_hash: str,
) -> CollectStats:
    teacher_cfg = config["teacher"]
    student_cfg = config["student"]["weak"]
    write_lock = asyncio.Lock()

    teacher_coros = []
    student_coros = []
    for q in train_questions:
        for i in range(teacher_cfg["k_high_temp"]):
            sampling = Sampling(
                strategy="high_temp", index=i,
                temperature=teacher_cfg["temperature_high"], max_tokens=teacher_cfg["max_tokens"],
            )
            teacher_coros.append(
                _collect_one(q, "teacher", f"hi-{i}", sampling, teacher, attempts_store, ledger, config_hash, write_lock)
            )
        low_sampling = Sampling(
            strategy="low_temp", index=0,
            temperature=teacher_cfg["temperature_low"], max_tokens=teacher_cfg["max_tokens"],
        )
        teacher_coros.append(
            _collect_one(q, "teacher", "lo-0", low_sampling, teacher, attempts_store, ledger, config_hash, write_lock)
        )

        weak_sampling = Sampling(
            strategy="weak", index=0,
            temperature=student_cfg["temperature"], max_tokens=student_cfg["max_tokens"],
        )
        student_coros.append(
            _collect_one(q, "student", "weak-0", weak_sampling, student, attempts_store, ledger, config_hash, write_lock)
        )

    teacher_results, student_results = await asyncio.gather(
        asyncio.gather(*teacher_coros), asyncio.gather(*student_coros)
    )

    all_attempts = attempts_store.all()
    train_ids = {q.id for q in train_questions}
    teacher_total = sum(1 for a in all_attempts if a.question_id in train_ids and a.actor_role == "teacher")
    student_total = sum(1 for a in all_attempts if a.question_id in train_ids and a.actor_role == "student")

    return CollectStats(
        teacher_new=sum(teacher_results),
        teacher_total=teacher_total,
        student_new=sum(student_results),
        student_total=student_total,
    )
