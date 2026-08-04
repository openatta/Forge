"""Regression tests for generate_questions_via_teacher()'s resume behavior: it must top a cell up
to its full quota across multiple rounds instead of treating "cell has >=1 question" as "done",
must give up after a round produces nothing new rather than looping forever, and a fully-satisfied
cell must not call the teacher again on a resumed run.
"""
from __future__ import annotations

import json

from core.ledger import Ledger, JsonlStore
from core.schemas import ChatResult, Question, Usage
from datagen.generate import generate_questions_via_teacher


class ScriptedTeacher:
    """Fake ModelClient returning one scripted JSON-array response per call, so a test can control
    exactly what the "teacher" produces on each top-up round."""

    def __init__(self, responses: list[list[dict]]):
        self.model_id = "scripted-teacher"
        self._responses = responses
        self.call_count = 0

    async def chat(self, messages, *, tools=None, temperature=0.7, max_tokens=4096) -> ChatResult:
        assert self.call_count < len(self._responses), "teacher called more times than scripted"
        items = self._responses[self.call_count]
        self.call_count += 1
        return ChatResult(content=json.dumps(items), finish_reason="stop", usage=Usage())


class ExplodingTeacher:
    """Fails the test if called at all -- used to prove a fully-satisfied cell never re-queries."""

    model_id = "should-not-be-called"

    async def chat(self, *args, **kwargs):
        raise AssertionError("teacher should not be called once the cell's quota is already met")


def _matrix() -> dict:
    return {"cells": [{"id": "cell-a", "topic": "arithmetic", "difficulty": "easy"}]}


def _item(text: str, gold: str = "1") -> dict:
    return {"text": text, "gold": gold, "verify_method": "numeric"}


async def test_generation_tops_up_a_cell_that_fell_short_of_quota(tmp_path):
    # quota_per_cell=2, holdout_per_cell=1 -> round 1 asks for 3, but 2 of the 3 items collapse to
    # the same exact-dedup text, leaving only 2 unique -> cell is short by 1 after round 1. Round 2
    # should ask for just the remaining shortfall and top the cell up to full quota.
    round1 = [_item("What is 1+1?"), _item("what is 1+1?"), _item("What is 2+2?")]
    round2 = [_item("What is 3+3?")]
    teacher = ScriptedTeacher([round1, round2])

    ledger = Ledger(tmp_path / "ledger.jsonl")
    questions_store = JsonlStore(tmp_path / "questions.jsonl", Question)
    holdout_store = JsonlStore(tmp_path / "holdout.jsonl", Question)

    result = await generate_questions_via_teacher(
        matrix=_matrix(), teacher=teacher, quota_per_cell=2, holdout_per_cell=1,
        questions_store=questions_store, holdout_store=holdout_store, ledger=ledger, config_hash="cfg",
    )

    assert len(result.train) == 2
    assert len(result.holdout) == 1
    assert teacher.call_count == 2  # topped up in round 2, didn't need a 3rd round


async def test_generation_stops_after_a_round_produces_nothing_new(tmp_path):
    # Every round returns an item that fails validation (no gold answer) -> zero usable questions
    # ever, every round -- must stop after the first empty round instead of burning all 3 rounds.
    invalid_item = [{"text": "What is 1+1?", "gold": None, "verify_method": "numeric"}]
    teacher = ScriptedTeacher([invalid_item, invalid_item, invalid_item])

    ledger = Ledger(tmp_path / "ledger.jsonl")
    questions_store = JsonlStore(tmp_path / "questions.jsonl", Question)
    holdout_store = JsonlStore(tmp_path / "holdout.jsonl", Question)

    result = await generate_questions_via_teacher(
        matrix=_matrix(), teacher=teacher, quota_per_cell=2, holdout_per_cell=1,
        questions_store=questions_store, holdout_store=holdout_store, ledger=ledger, config_hash="cfg",
    )

    assert len(result.train) == 0
    assert len(result.holdout) == 0
    assert teacher.call_count == 1  # round 1 produced nothing usable -> stopped immediately


async def test_resumed_run_skips_a_cell_that_already_met_quota(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    questions_path = tmp_path / "questions.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"

    first_teacher = ScriptedTeacher([[_item("What is 1+1?"), _item("What is 2+2?"), _item("What is 3+3?")]])
    await generate_questions_via_teacher(
        matrix=_matrix(), teacher=first_teacher, quota_per_cell=2, holdout_per_cell=1,
        questions_store=JsonlStore(questions_path, Question), holdout_store=JsonlStore(holdout_path, Question),
        ledger=ledger, config_hash="cfg",
    )

    result = await generate_questions_via_teacher(
        matrix=_matrix(), teacher=ExplodingTeacher(), quota_per_cell=2, holdout_per_cell=1,
        questions_store=JsonlStore(questions_path, Question), holdout_store=JsonlStore(holdout_path, Question),
        ledger=ledger, config_hash="cfg",
    )

    assert len(result.train) == 2
    assert len(result.holdout) == 1
