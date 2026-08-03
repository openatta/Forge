"""Evidence localization (answer must be traceable to a supporting passage in the source doc) — P3."""
from __future__ import annotations

from core.schemas import Attempt, Question, VerifyResult
from verify.base import Verifier


class EvidenceVerifier(Verifier):
    id = "evidence_v1"
    version = "unimplemented"

    def verify(self, attempt: Attempt, question: Question) -> VerifyResult:
        raise NotImplementedError("evidence localization verifier ships in P3.")
