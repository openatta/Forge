"""Small builders for the pydantic models shared across P0/P1 tests, so each test file doesn't
have to restate every required field just to get a valid Question/Attempt/VerifyResult.
"""
from __future__ import annotations

from datetime import datetime, timezone

from core.schemas import Attempt, Question, Sampling, Usage, VerifyResult


def make_question(
    id: str = "q-1",
    text: str = "What is 2+2?",
    cell: str = "arithmetic/easy",
    family_id: str = "fam-1",
    verify_method: str = "numeric",
    gold: str = "4",
    split: str = "train",
    config_hash: str = "cfg",
    source: str = "seed",
) -> Question:
    return Question(
        id=id, text=text, cell=cell, family_id=family_id, source=source,
        verify_method=verify_method, gold=gold, split=split, config_hash=config_hash,
    )


def make_attempt(
    id: str,
    question_id: str,
    content: str = "ok",
    role: str = "teacher",
    strategy: str = "high_temp",
    index: int = 0,
    model: str = "test-model",
    config_hash: str = "cfg",
) -> Attempt:
    return Attempt(
        id=id, question_id=question_id, actor_model=model, actor_role=role,
        sampling=Sampling(strategy=strategy, index=index, temperature=0.8, max_tokens=512),
        messages=[{"role": "user", "content": "hi"}], content=content,
        usage=Usage(), ts=datetime.now(timezone.utc), config_hash=config_hash,
    )


def make_verify(
    attempt_id: str, passed: bool, verifier_id: str = "math_answer_v1", detail: str = ""
) -> VerifyResult:
    return VerifyResult(
        id=f"ver-{attempt_id}-{verifier_id}-v1", attempt_id=attempt_id, passed=passed,
        detail=detail, verifier_id=verifier_id, ts=datetime.now(timezone.utc),
    )
