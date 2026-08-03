"""Environment abstraction, aligned with the verifiers project's reset/step/verify semantics so P2+
environments can migrate there with minimal friction. P0 only defines the interface; P2 implements it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

Observation = Any
ToolCall = Any


class RewardInfo:
    def __init__(self, success: bool, detail: str = ""):
        self.success = success
        self.detail = detail


class TaskSpec:
    def __init__(self, task_id: str, payload: dict):
        self.task_id = task_id
        self.payload = payload


class Environment(ABC):
    version: str

    @abstractmethod
    def reset(self, task_spec: TaskSpec) -> Observation: ...

    @abstractmethod
    def step(self, action: ToolCall) -> tuple[Observation, bool]: ...

    @abstractmethod
    def verify(self) -> RewardInfo: ...

    @abstractmethod
    def snapshot(self) -> bytes: ...
