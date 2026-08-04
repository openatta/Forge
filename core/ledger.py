"""ID system, idempotent jsonl storage, and traceability. Every ID is a pure function of upstream
content, so re-running the same seeds/config against an existing data/ dir reproduces identical IDs —
that's what makes JsonlStore.append() a safe no-op on resume. See docs/90-MVP搭建指南.md §4.3.
"""
from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

from core.hashing import short_hash
from core.schemas import LedgerEntry

try:
    import fcntl
except ImportError:  # pragma: no cover - POSIX-only; this project targets macOS/Linux (see README)
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def now() -> datetime:
    return datetime.now(timezone.utc)


class JsonlStore(Generic[T]):
    """Append-only jsonl store keyed by record.id. append() is idempotent: a record whose id is
    already present is skipped (logged at debug level), never rewritten or duplicated.

    append() also holds an OS-level advisory lock (fcntl.flock) for its check-and-write and
    re-reads the file fresh under that lock rather than trusting the in-memory `_ids` set built
    at construction time -- otherwise two run.py processes pointed at the same data_dir could
    both pass a stale membership check and duplicate-append the same id. Re-reading the whole
    file on every append is O(n) rather than incremental, which is fine at the row counts P0/P1
    actually produce (tens to low thousands); revisit if a stage starts writing at real scale.
    has()/all() stay lock-free reads of whatever's on disk right now -- append() is the only
    writer, so it's the only place actual corruption could happen.
    """

    def __init__(self, path: Path, model: type[T]):
        self.path = Path(path)
        self.model = model
        self._lock = threading.Lock()
        self._ids: set[str] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                self._ids.add(self.model.model_validate_json(line).id)

    def has(self, record_id: str) -> bool:
        return record_id in self._ids

    def append(self, record: T) -> bool:
        """Returns True if the record was newly written, False if it was already present."""
        with self._lock:
            with self.path.open("a+", encoding="utf-8") as f:
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    for line in f:
                        if line.strip():
                            self._ids.add(self.model.model_validate_json(line).id)
                    if record.id in self._ids:
                        logger.debug("skip (already present): %s", record.id)
                        return False
                    f.write(record.model_dump_json() + "\n")
                    f.flush()
                finally:
                    if fcntl is not None:
                        fcntl.flock(f, fcntl.LOCK_UN)
            self._ids.add(record.id)
            return True

    def all(self) -> list[T]:
        if not self.path.exists():
            return []
        return [
            self.model.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def question_id(text: str) -> str:
    return f"q-{short_hash(normalize(text), 12)}"


def attempt_id(question_id_: str, role: str, sampling_key: str, sampling_fingerprint: str) -> str:
    """sampling_fingerprint must cover whatever actually changes the call (temperature, max_tokens,
    ...) so that editing a config value for an already-collected slot produces a new id instead of
    silently resuming as if nothing changed."""
    return f"att-{question_id_}-{role}-{sampling_key}-{sampling_fingerprint}"


def verify_id(attempt_id_: str, verifier_id: str, verifier_version: str) -> str:
    """verifier_version should change whenever the verifier's actual logic changes (e.g. a source
    hash), so fixing a verifier bug invalidates old cached results instead of reusing stale verdicts."""
    return f"ver-{attempt_id_}-{verifier_id}-{verifier_version}"


def sample_id(question_id_: str, kind: str, batch_id_: str, attempt_id_: str) -> str:
    """Includes attempt_id so a re-selected (different) best attempt for the same question/batch
    produces a distinct id — see select_.rejection.latest_samples for how the current selection is
    resolved from the resulting append-only history."""
    return f"smp-{question_id_}-{kind}-{batch_id_}-{short_hash(attempt_id_, 6)}"


def batch_id(config_hash: str) -> str:
    return f"batch-{config_hash[:10]}"


def baseline_batch_id(config_hash: str) -> str:
    """Separate ID namespace from batch_id() so the stratified-random baseline arm's samples/sft
    records never collide with the value-selected arm's, even though both derive from the same
    config_hash."""
    return f"baseline-{config_hash[:10]}"


def checkpoint_id(dataset_record_ids: list[str], config_hash: str, base_model: str, arm: str) -> str:
    return f"ckpt-{arm}-{short_hash({'ids': sorted(dataset_record_ids), 'config_hash': config_hash, 'model': base_model}, 10)}"


def run_id(model_id: str, config_hash: str) -> str:
    return f"run-{short_hash(model_id + config_hash, 8)}"


def eval_id(run_id_: str, question_id_: str) -> str:
    return f"eval-{run_id_}-{question_id_}"


class Ledger:
    def __init__(self, path: Path = Path("data/ledger.jsonl")):
        self.store = JsonlStore(path, LedgerEntry)

    def record(self, id: str, type_: str, upstream: list[str], config_hash: str) -> None:
        entry = LedgerEntry(id=id, type=type_, upstream=upstream, config_hash=config_hash, ts=now())
        self.store.append(entry)

    def trace(self, id: str) -> list[LedgerEntry]:
        """Recursive root-first walk of upstream[] for a given id — used by the smoke report and
        `run.py trace` to demonstrate full-chain traceability."""
        by_id = {e.id: e for e in self.store.all()}
        chain: list[LedgerEntry] = []
        seen: set[str] = set()

        def visit(entry_id: str) -> None:
            if entry_id in seen or entry_id not in by_id:
                return
            seen.add(entry_id)
            entry = by_id[entry_id]
            for up in entry.upstream:
                visit(up)
            chain.append(entry)

        visit(id)
        return chain
