"""Unit-test execution verifier (subprocess to start, sandboxed in P2) — P1."""
from __future__ import annotations

from core.schemas import Attempt, Question, VerifyResult
from verify.base import Verifier


class CodeExecVerifier(Verifier):
    id = "code_exec_v1"
    version = "unimplemented"

    def verify(self, attempt: Attempt, question: Question) -> VerifyResult:
        raise NotImplementedError("unit-test execution verifier ships in P1.")
