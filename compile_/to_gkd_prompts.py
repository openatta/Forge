"""Compile failure-mode prompts for on-policy GKD training — P3."""
from __future__ import annotations

from core.schemas import Question


def compile_gkd_prompts(questions: list[Question]) -> None:
    raise NotImplementedError("GKD prompt compilation ships in P3.")
