"""Simulated multi-turn user for tau-bench-style environments — P2+."""
from __future__ import annotations


class SimUser:
    def respond(self, transcript: list[dict]) -> str:
        raise NotImplementedError("simulated user ships in P2+.")
