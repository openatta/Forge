"""Content hashing for configs/prompts/tool schemas — the basis of deterministic IDs across the ledger."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def content_hash(obj: Any) -> str:
    payload = json.dumps(_to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def short_hash(obj: Any, n: int = 8) -> str:
    return content_hash(obj)[:n]
