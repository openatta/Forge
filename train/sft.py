"""TRL SFTTrainer script (multi-GPU accelerate config in configs/) — P1, needs GPU for mode="real".

mode="mock" performs no training at all. It validates the dataset and writes a checkpoint manifest
so the rest of the P1 pipeline (eval, A/B report) can be exercised end-to-end before real compute is
available -- see project plan: this round keeps training mocked and builds out everything around it
for real, so swapping in mode="real" later needs no changes anywhere downstream.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.ledger import Ledger, checkpoint_id, now
from core.schemas import SftRecord

logger = logging.getLogger(__name__)


def train_sft(
    sft_records: list[SftRecord],
    base_model: str,
    arm: str,
    config_hash: str,
    ledger: Ledger,
    checkpoints_dir: Path,
    mode: str = "mock",
) -> str:
    """Returns a checkpoint_id. mode="mock": validates the dataset, writes a manifest, records the
    checkpoint in the ledger (upstream = the sft record ids it was "trained" on), does NOT change any
    model weights. mode="real": not implemented; ships once GPU + accelerate compute is available.
    """
    if mode == "real":
        raise NotImplementedError("real TRL SFT training ships once GPU/accelerate compute is available.")
    if mode != "mock":
        raise ValueError(f"unknown train mode: {mode!r}")

    if not sft_records:
        raise ValueError(f"train_sft(mode=mock, arm={arm!r}): no sft records to train on")
    for r in sft_records:
        if not r.messages:
            raise ValueError(f"train_sft(mode=mock): record {r.id} has no messages")

    record_ids = [r.id for r in sft_records]
    ckpt_id = checkpoint_id(record_ids, config_hash, base_model, arm)
    ckpt_dir = checkpoints_dir / ckpt_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "checkpoint_id": ckpt_id,
        "arm": arm,
        "base_model": base_model,
        "sample_count": len(sft_records),
        "config_hash": config_hash,
        "mode": "mock",
        "ts": now().isoformat(),
        "note": (
            "MOCK TRAINING: no gradient update occurred. This checkpoint is just the untouched base "
            "model plus this manifest; it exists only to validate pipeline plumbing (dataset -> "
            "checkpoint id -> eval). Do not treat downstream eval numbers as real training results."
        ),
    }
    manifest_path = ckpt_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    ledger.record(ckpt_id, "checkpoint", upstream=record_ids, config_hash=config_hash)
    logger.warning(
        "MOCK TRAINING (arm=%s): checkpoint %s from %d samples -- no gradient update occurred",
        arm, ckpt_id, len(sft_records),
    )
    return ckpt_id
