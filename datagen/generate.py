"""Question generation. P0: reads seed jsonl directly. P1 generates via the teacher LLM directly
(see generate_questions_via_teacher below) -- not Bespoke Curator, see project plan for why: the
caching/retry/batch-orchestration value Curator would add is already covered by our own
TeacherClient + ledger idempotency, and its exact API surface couldn't be verified in this
environment without pulling its heavy transitive dependency tree.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from core import ledger as ledger_mod
from core.ledger import JsonlStore, Ledger
from core.model_client import ModelClient
from core.schemas import Question
from datagen.dedup import exact_dedup, minhash_semantic_dedup, template_family_cap
from taxonomy.quota import allocate_quota

logger = logging.getLogger(__name__)


@dataclass
class GenResult:
    train: list[Question]
    holdout: list[Question]
    dedup_dropped_count: int


def load_seeds(path: Path) -> list[dict]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def generate_questions(
    seeds_path: Path,
    data_dir: Path,
    holdout_dir: Path,
    config_hash: str,
    ledger: Ledger,
) -> GenResult:
    raw = load_seeds(seeds_path)

    all_questions = [
        Question(
            id=ledger_mod.question_id(r["text"]),
            text=r["text"],
            cell=r["cell"],
            family_id=r["family_id"],
            source="seed",
            verify_method=r["verify_method"],
            gold=r["gold"],
            split=r["split"],
            config_hash=config_hash,
        )
        for r in raw
    ]

    train_candidates = [q for q in all_questions if q.split == "train"]
    holdout_questions = [q for q in all_questions if q.split == "holdout"]

    train_questions, dropped = exact_dedup(train_candidates)
    if dropped:
        logger.info("exact_dedup dropped %d duplicate question(s): %s", len(dropped), [q.id for q in dropped])

    questions_store = JsonlStore(data_dir / "questions.jsonl", Question)
    for q in train_questions:
        questions_store.append(q)
        ledger.record(q.id, "question", upstream=[], config_hash=config_hash)

    holdout_store = JsonlStore(holdout_dir / "questions.jsonl", Question)
    for q in holdout_questions:
        holdout_store.append(q)
        ledger.record(q.id, "question", upstream=[], config_hash=config_hash)

    matrix = yaml.safe_load(Path("taxonomy/matrix.yaml").read_text(encoding="utf-8"))
    quota = allocate_quota(matrix, total=len(train_questions))
    logger.info("target coverage quota (informational, P0 doesn't generate to it yet): %s", quota)

    return GenResult(train=train_questions, holdout=holdout_questions, dedup_dropped_count=len(dropped))


GEN_SYSTEM_PROMPT = (
    "You are a math item-writer for an LLM distillation pipeline. Given a topic and difficulty level, "
    "produce distinct, self-contained math questions, each with a single unambiguous machine-checkable "
    "answer. Output ONLY a JSON array (no markdown fences, no commentary) of objects with exactly these "
    "fields:\n"
    '  "text": the question, fully self-contained (no external context needed)\n'
    '  "gold": the correct answer as a string; if verify_method is "numeric_set", comma-separate ALL '
    "solutions\n"
    '  "verify_method": either "numeric" (single-value answer) or "numeric_set" (multiple values, e.g. '
    "all roots of an equation)\n"
    "If you cannot produce a question with an unambiguous checkable answer for a slot, omit it rather "
    "than force one -- an item with no clear verification method is worse than one fewer item."
)


def _build_gen_messages(topic: str, difficulty: str, n: int) -> list[dict]:
    return [
        {"role": "system", "content": GEN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Topic: {topic}\nDifficulty: {difficulty}\nGenerate exactly {n} distinct questions as a JSON array.",
        },
    ]


def _extract_json_array(text: str) -> list:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def _validate_gen_item(item) -> bool:
    return (
        isinstance(item, dict)
        and isinstance(item.get("text"), str) and item["text"].strip()
        and item.get("verify_method") in ("numeric", "numeric_set")
        and item.get("gold") is not None and str(item["gold"]).strip()
    )


# A cell that keeps coming back all-duplicates/all-invalid shouldn't retry forever within one
# run -- cap top-up attempts per cell and let the *next* `run.py p1` invocation pick up where
# this one left off (same resume story collect/sample.py already relies on for failed calls).
_MAX_TOPUP_ROUNDS_PER_CELL = 3


async def generate_questions_via_teacher(
    matrix: dict,
    teacher: ModelClient,
    quota_per_cell: int,
    holdout_per_cell: int,
    questions_store: JsonlStore,
    holdout_store: JsonlStore,
    ledger: Ledger,
    config_hash: str,
    max_tokens: int = 8192,
) -> GenResult:
    """P1: generates quota_per_cell (+holdout_per_cell) questions per coverage-matrix cell by asking
    the teacher LLM for "question + gold answer + verify_method" triples (docs/reference/02 §3.2) --
    any item with an unclear/missing verify_method is discarded, never guessed.

    Dedup order per cell (docs/reference/02 §五, layers 1+3; layer 2/semantic is out of scope --
    see project plan): exact -> MinHash near-dup -> template-family cap, all applied to the round's
    candidate pool BEFORE splitting into train/holdout. This is what keeps a train question and a
    holdout question from the same cell from being near-duplicates of each other -- template-family
    isolation between train and eval is a hard requirement (docs/reference/06), and family_id here
    is just the cell id (a coarse stand-in for real template-family clustering, which free-form
    LLM-generated text doesn't give us for free).

    Resume: generation is sampled at temperature=0.9 for diversity, so unlike collect/sample.py's
    attempt_id there's no way to know a question's id before calling the teacher -- there's nothing
    to check existence of ahead of time. This resumes at cell *quota* granularity rather than cell
    *presence*: a cell only gets skipped once it actually has >=quota_per_cell train questions and
    >=holdout_per_cell holdout questions on disk. A cell sitting below quota (e.g. because a prior
    round's candidates mostly collided with MinHash near-dupes) re-requests just the remaining
    shortfall, for up to _MAX_TOPUP_ROUNDS_PER_CELL rounds per invocation -- a round that lands zero
    usable new questions stops early rather than burning the rest of its rounds on a cell that's
    clearly stuck (e.g. an exhausted or repetitive topic), leaving it for the next `run.py p1` run.

    max_tokens defaults to 8192, not the smaller budget collection/eval calls use elsewhere: this
    teacher model spends hidden reasoning tokens before emitting visible output, and empirically
    "hard"-difficulty cells alone can consume 4-5k completion tokens on reasoning before any JSON
    appears -- at 2048 or 4096 every "hard" cell came back with finish_reason="length" and zero
    visible content (confirmed via direct probe), silently losing a third of the coverage matrix.
    """
    all_train: list[Question] = list(questions_store.all())
    all_holdout: list[Question] = list(holdout_store.all())
    total_dropped = 0

    def _cell_texts(cell_id: str) -> set[str]:
        return {ledger_mod.normalize(q.text) for q in all_train + all_holdout if q.cell == cell_id}

    for cell in matrix["cells"]:
        cell_id = cell["id"]
        train_have = sum(1 for q in all_train if q.cell == cell_id)
        holdout_have = sum(1 for q in all_holdout if q.cell == cell_id)
        if train_have >= quota_per_cell and holdout_have >= holdout_per_cell:
            logger.info(
                "gen(p1) cell=%s: quota already met (train=%d/%d holdout=%d/%d), skipping (resume)",
                cell_id, train_have, quota_per_cell, holdout_have, holdout_per_cell,
            )
            continue

        for round_num in range(1, _MAX_TOPUP_ROUNDS_PER_CELL + 1):
            train_need = max(0, quota_per_cell - train_have)
            holdout_need = max(0, holdout_per_cell - holdout_have)
            if train_need == 0 and holdout_need == 0:
                break
            want = train_need + holdout_need
            messages = _build_gen_messages(cell["topic"], cell["difficulty"], want)
            try:
                result = await teacher.chat(messages, temperature=0.9, max_tokens=max_tokens)
                items = _extract_json_array(result.content)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "generation failed for cell %s (round %d/%d): %s -- stopping this cell for this run",
                    cell_id, round_num, _MAX_TOPUP_ROUNDS_PER_CELL, exc,
                )
                break

            candidates: list[Question] = []
            round_dropped = 0
            for item in items:
                if not _validate_gen_item(item):
                    round_dropped += 1
                    continue
                candidates.append(
                    Question(
                        id=ledger_mod.question_id(item["text"]),
                        text=item["text"],
                        cell=cell_id,
                        family_id=cell_id,
                        source="teacher_gen",
                        verify_method=item["verify_method"],
                        gold=str(item["gold"]),
                        split="train",  # placeholder; real split decided below after dedup
                        config_hash=config_hash,
                    )
                )

            valid_count = len(candidates)
            candidates, dropped_exact = exact_dedup(candidates)
            # Also drop anything that duplicates a question this cell already accepted in an
            # earlier round (this run) or an earlier run -- exact_dedup() above only catches
            # duplicates *within* this round's candidates.
            existing_texts = _cell_texts(cell_id)
            candidates = [q for q in candidates if ledger_mod.normalize(q.text) not in existing_texts]
            dropped_prior_round = valid_count - len(dropped_exact) - len(candidates)
            candidates, dropped_minhash = minhash_semantic_dedup(candidates)
            candidates, dropped_family = template_family_cap(candidates, max_per_family=want)
            round_dropped += len(dropped_exact) + dropped_prior_round + len(dropped_minhash) + len(dropped_family)
            total_dropped += round_dropped

            cell_holdout = candidates[:holdout_need]
            cell_train = candidates[holdout_need : holdout_need + train_need]

            new_holdout, new_train = 0, 0
            for q in cell_holdout:
                q = q.model_copy(update={"split": "holdout"})
                if holdout_store.append(q):
                    ledger.record(q.id, "question", upstream=[], config_hash=config_hash)
                    all_holdout.append(q)
                    new_holdout += 1
            for q in cell_train:
                if questions_store.append(q):
                    ledger.record(q.id, "question", upstream=[], config_hash=config_hash)
                    all_train.append(q)
                    new_train += 1

            train_have = sum(1 for q in all_train if q.cell == cell_id)
            holdout_have = sum(1 for q in all_holdout if q.cell == cell_id)
            logger.info(
                "gen(p1) cell=%s round=%d/%d: raw=%d valid=%d dropped=%d -> +train=%d +holdout=%d "
                "(now train=%d/%d holdout=%d/%d)",
                cell_id, round_num, _MAX_TOPUP_ROUNDS_PER_CELL,
                len(items) if isinstance(items, list) else 0, valid_count, round_dropped,
                new_train, new_holdout, train_have, quota_per_cell, holdout_have, holdout_per_cell,
            )

            if new_train == 0 and new_holdout == 0:
                logger.warning(
                    "gen(p1) cell=%s: round %d produced no usable new questions -- stopping this cell "
                    "for this run (still train=%d/%d holdout=%d/%d)",
                    cell_id, round_num, train_have, quota_per_cell, holdout_have, holdout_per_cell,
                )
                break

    return GenResult(train=all_train, holdout=all_holdout, dedup_dropped_count=total_dropped)
