"""Run eval: model client x question set x verifier -> per-cell results."""
from __future__ import annotations

from collect.sample import SYSTEM_PROMPT
from core.ledger import JsonlStore, Ledger, eval_id, now
from core.model_client import ModelClient
from core.schemas import Attempt, EvalRecord, Question, Sampling
from verify.base import Verifier


async def run_holdout_eval(
    student: ModelClient,
    holdout_questions: list[Question],
    verifier: Verifier,
    run_id: str,
    eval_store: JsonlStore,
    ledger: Ledger,
    config_hash: str,
    temperature: float = 0.0,
) -> list[EvalRecord]:
    """temperature defaults to 0 (greedy decoding): eval should be reproducible, and P1's A/B report
    relies on it -- the base/value_selected/baseline_random arms share identical (mocked, untrained)
    weights, so their eval numbers are only meaningfully "expected to match" if decoding is
    deterministic. At temperature>0, MockStudent(mode=weak)'s real stochastic LLM call would make
    every arm an independent sample, and any observed difference would be noise, not signal.
    """
    existing = {r.id: r for r in eval_store.all()}
    records: list[EvalRecord] = []
    for question in holdout_questions:
        eid = eval_id(run_id, question.id)
        if eid in existing:
            records.append(existing[eid])
            continue

        # Must match collect/sample.py's message shape -- the verifier's answer extraction depends on
        # the model being told to end with "Final Answer: ...", same as during collection.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.text},
        ]
        result = await student.chat(messages, temperature=temperature, max_tokens=512)

        probe_attempt = Attempt(
            id=f"probe-{eid}",
            question_id=question.id,
            actor_model=student.model_id,
            actor_role="student",
            sampling=Sampling(strategy="weak", index=0, temperature=temperature, max_tokens=512),
            messages=messages,
            content=result.content,
            usage=result.usage,
            ts=now(),
            config_hash=config_hash,
        )
        verify_result = verifier.verify(probe_attempt, question)

        record = EvalRecord(
            id=eid,
            model_id=student.model_id,
            question_id=question.id,
            passed=verify_result.passed,
            cell=question.cell,
            run_id=run_id,
            detail=verify_result.detail,
            ts=now(),
        )
        eval_store.append(record)
        ledger.record(record.id, "eval", upstream=[question.id], config_hash=config_hash)
        records.append(record)
    return records
