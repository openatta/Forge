"""Step/token/timeout budget enforcement for agent rollouts — P2."""
from __future__ import annotations


class Budget:
    def __init__(self, max_steps: int, max_tokens: int, tool_timeout_s: float):
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.tool_timeout_s = tool_timeout_s

    def check(self, steps_used: int, tokens_used: int) -> bool:
        raise NotImplementedError("step/token/timeout budget enforcement ships in P2.")
