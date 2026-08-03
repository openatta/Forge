"""P2 smoke environment: in-process calculator + file tools, no Docker required. Not implemented yet."""
from __future__ import annotations

from envs.base import Environment, Observation, RewardInfo, TaskSpec, ToolCall


class ToyCalcEnv(Environment):
    version = "toy_calc-unimplemented"

    def reset(self, task_spec: TaskSpec) -> Observation:
        raise NotImplementedError("toy_calc in-process env ships in P2.")

    def step(self, action: ToolCall) -> tuple[Observation, bool]:
        raise NotImplementedError("toy_calc in-process env ships in P2.")

    def verify(self) -> RewardInfo:
        raise NotImplementedError("toy_calc in-process env ships in P2.")

    def snapshot(self) -> bytes:
        raise NotImplementedError("toy_calc in-process env ships in P2.")
