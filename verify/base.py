"""Verifier abstraction. Every verifier maps (attempt, question) -> VerifyResult."""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.schemas import Attempt, Question, VerifyResult


class Verifier(ABC):
    id: str
    version: str  # must change whenever verify()'s logic changes; invalidates cached VerifyResults

    @abstractmethod
    def verify(self, attempt: Attempt, question: Question) -> VerifyResult: ...
