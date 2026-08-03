"""All pydantic models shared across the pipeline. See docs/90-MVP搭建指南.md §4.3 for the mandated core set."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Sampling(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["high_temp", "low_temp", "weak"]
    index: int
    temperature: float
    max_tokens: int


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class ChatResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    tool_calls: list[dict] | None = None
    finish_reason: str
    usage: Usage
    logprobs: Any | None = None


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    cell: str
    family_id: str
    source: Literal["seed", "curator", "evol", "teacher_gen"] = "seed"
    verify_method: Literal["numeric", "numeric_set"]
    gold: Any | None = None
    split: Literal["train", "holdout"] = "train"
    config_hash: str


class Attempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question_id: str
    actor_model: str
    actor_role: Literal["teacher", "student"]
    sampling: Sampling
    messages: list[dict]
    content: str
    usage: Usage
    ts: datetime
    config_hash: str


class VerifyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str  # deterministic: f"ver-{attempt_id}-{verifier_id}"
    attempt_id: str
    passed: bool
    detail: str
    verifier_id: str
    ts: datetime


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question_id: str
    attempt_id: str
    kind: Literal["sft", "dpo", "gkd_prompt"] = "sft"
    payload: dict
    batch_id: str
    config_hash: str


class EvalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str  # deterministic: f"eval-{run_id}-{question_id}"
    model_id: str
    question_id: str
    passed: bool
    cell: str
    run_id: str
    detail: str | None = None
    ts: datetime


class SftRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str  # == sample.id
    messages: list[dict]
    meta: dict


class LedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["question", "attempt", "verify", "sample", "eval", "checkpoint"]
    upstream: list[str] = Field(default_factory=list)
    config_hash: str
    ts: datetime
