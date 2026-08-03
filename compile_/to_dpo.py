"""Compile chosen/rejected attempt pairs into DPO-ready jsonl — not yet scoped to a phase."""
from __future__ import annotations

from core.schemas import Sample


def compile_dpo(samples: list[Sample]) -> None:
    raise NotImplementedError("DPO compilation ships alongside train/dpo.py.")
