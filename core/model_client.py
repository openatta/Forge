"""Unified model client — the mock boundary. Every pipeline stage only ever talks to a ModelClient;
STUDENT_MODE is read exactly once, in build_student_client(), so switching mock<->vllm needs zero
changes anywhere else. See docs/90-MVP搭建指南.md §4.1.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from core.ledger import normalize
from core.schemas import ChatResult, Usage

logger = logging.getLogger(__name__)


@runtime_checkable
class ModelClient(Protocol):
    model_id: str

    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ChatResult: ...


class TeacherClient:
    """litellm.acompletion wrapper: bounded concurrency, retry with exponential backoff, cost tracking."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        concurrency: int = 4,
        max_retries: int = 3,
    ):
        self.model_id = model
        self._api_key = api_key
        self._api_base = api_base
        self._sem = asyncio.Semaphore(concurrency)
        self._max_retries = max_retries
        self._debug_dir = Path("data/raw/teacher")

    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ChatResult:
        import litellm

        kwargs: dict = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if tools:
            kwargs["tools"] = tools

        delay = 1.0
        last_exc: Exception | None = None
        async with self._sem:
            for attempt in range(self._max_retries):
                try:
                    response = await litellm.acompletion(**kwargs)
                    break
                except Exception as exc:  # noqa: BLE001 - litellm raises many provider-specific types
                    last_exc = exc
                    logger.warning("teacher call failed (attempt %d/%d): %s", attempt + 1, self._max_retries, exc)
                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(delay)
                        delay *= 2
            else:
                raise RuntimeError(f"teacher call failed after {self._max_retries} attempts") from last_exc

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = None
        raw_tool_calls = getattr(choice.message, "tool_calls", None)
        if raw_tool_calls:
            tool_calls = [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in raw_tool_calls]

        usage_obj = getattr(response, "usage", None)
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:  # noqa: BLE001 - cost lookup is best-effort
            cost = 0.0

        usage = Usage(
            prompt_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage_obj, "total_tokens", 0) or 0,
            cost_usd=cost or 0.0,
        )

        await asyncio.to_thread(self._persist_debug, response)

        return ChatResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )

    def _persist_debug(self, response) -> None:
        try:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            path = self._debug_dir / f"{response.id}.json"
            path.write_text(json.dumps(response.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8")
        except Exception:  # noqa: BLE001 - debug persistence must never break the pipeline
            pass


class VLLMClient:
    """P1+: OpenAI-compatible client against a vLLM server. Stub for P0."""

    def __init__(self, base_url: str, model: str = "student"):
        self.model_id = model
        self._base_url = base_url

    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ChatResult:
        raise NotImplementedError("vLLM client ships in P1 once a real student is served.")


class MockStudent:
    """P0 student stand-in. mode=echo (fixed template, tests data flow), mode=weak (delegates to a
    cheap API model, tests eval/gap logic), mode=replay (deterministic jsonl playback, CI-usable).
    """

    def __init__(
        self,
        mode: str = "weak",
        weak_model: str | None = None,
        weak_api_key: str | None = None,
        weak_api_base: str | None = None,
        replay_path: str | None = None,
    ):
        self.model_id = f"mock-student:{mode}"
        self._mode = mode
        if mode == "weak":
            if not weak_model:
                raise ValueError("MOCK_STUDENT_MODE=weak requires STUDENT_WEAK_MODEL to be set")
            self._delegate = TeacherClient(
                model=weak_model, api_key=weak_api_key, api_base=weak_api_base, concurrency=4
            )
        elif mode == "replay":
            if not replay_path:
                raise ValueError("MOCK_STUDENT_MODE=replay requires STUDENT_REPLAY_PATH to be set")
            self._replay: dict[str, str] = {}
            path = Path(replay_path)
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    self._replay[normalize(record["question"])] = record["content"]
        elif mode != "echo":
            raise ValueError(f"unknown MockStudent mode: {mode!r}")

    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ChatResult:
        if self._mode == "echo":
            return ChatResult(
                content="Final Answer: 0",
                finish_reason="stop",
                usage=Usage(),
            )
        if self._mode == "weak":
            return await self._delegate.chat(messages, tools=tools, temperature=temperature, max_tokens=max_tokens)
        if self._mode == "replay":
            key = normalize(_last_user_content(messages))
            if key not in self._replay:
                raise RuntimeError(f"replay mode: no stored answer for question matching {key!r}")
            return ChatResult(content=self._replay[key], finish_reason="stop", usage=Usage())
        raise RuntimeError(f"unreachable mode: {self._mode!r}")


def _last_user_content(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    raise RuntimeError("replay mode requires at least one user message")


def build_teacher_client(
    model: str | None = None, api_key: str | None = None, api_base: str | None = None, concurrency: int = 4
) -> TeacherClient:
    model = model or os.environ.get("TEACHER_MODEL")
    api_key = api_key if api_key is not None else os.environ.get("TEACHER_API_KEY")
    api_base = api_base if api_base is not None else os.environ.get("TEACHER_API_BASE")
    if not model:
        raise ValueError("TEACHER_MODEL is not set (env var or explicit argument required)")
    return TeacherClient(model=model, api_key=api_key, api_base=api_base, concurrency=concurrency)


def build_student_client() -> "ModelClient":
    student_mode = os.environ.get("STUDENT_MODE", "mock")
    if student_mode == "mock":
        mock_mode = os.environ.get("MOCK_STUDENT_MODE", "weak")
        return MockStudent(
            mode=mock_mode,
            weak_model=os.environ.get("STUDENT_WEAK_MODEL"),
            # Weak student reuses the teacher's credentials/endpoint by default -- a smoke setup
            # normally only has one provider account configured. Override with STUDENT_WEAK_API_KEY/
            # STUDENT_WEAK_API_BASE if the weak student should hit a different provider.
            weak_api_key=os.environ.get("STUDENT_WEAK_API_KEY") or os.environ.get("TEACHER_API_KEY"),
            weak_api_base=os.environ.get("STUDENT_WEAK_API_BASE") or os.environ.get("TEACHER_API_BASE"),
            replay_path=os.environ.get("STUDENT_REPLAY_PATH"),
        )
    if student_mode == "vllm":
        base_url = os.environ.get("VLLM_BASE_URL")
        if not base_url:
            raise ValueError("STUDENT_MODE=vllm requires VLLM_BASE_URL to be set")
        return VLLMClient(base_url=base_url)
    raise ValueError(f"unknown STUDENT_MODE: {student_mode!r}")
