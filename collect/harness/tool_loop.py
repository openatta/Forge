"""OpenAI-tools-protocol tool loop, shared by teacher and student rollouts — P2."""
from __future__ import annotations

from core.model_client import ModelClient


class ToolLoop:
    def __init__(self, client: ModelClient, tools: list[dict], env):
        self.client = client
        self.tools = tools
        self.env = env

    async def run(self, task_spec):
        raise NotImplementedError("tool_loop (OpenAI tools protocol, teacher+student shared) ships in P2.")
