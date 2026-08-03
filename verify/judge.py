"""Calibrated LLM judge — P3."""
from __future__ import annotations

from core.schemas import Attempt, Question, VerifyResult
from verify.base import Verifier


class JudgeVerifier(Verifier):
    id = "judge_v1"
    version = "unimplemented"

    def verify(self, attempt: Attempt, question: Question) -> VerifyResult:
        raise NotImplementedError("calibrated LLM judge ships in P3.")
