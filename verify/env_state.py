"""Environment terminal-state verifier, delegates to env.verify() — P2."""
from __future__ import annotations

from core.schemas import Attempt, Question, VerifyResult
from verify.base import Verifier


class EnvStateVerifier(Verifier):
    id = "env_state_v1"
    version = "unimplemented"

    def verify(self, attempt: Attempt, question: Question) -> VerifyResult:
        raise NotImplementedError("environment terminal-state verifier ships in P2.")
