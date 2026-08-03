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
    see project plan): exact -> MinHash near-dup -> template-family cap, all applied to the WHOLE
    per-cell candidate pool (quota_per_cell + holdout_per_cell items) BEFORE splitting into
    train/holdout. This is what keeps a train question and a holdout question from the same cell
    from being near-duplicates of each other -- template-family isolation between train and eval is
    a hard requirement (docs/reference/06), and family_id here is just the cell id (a coarse stand-in
    for real template-family clustering, which free-form LLM-generated text doesn't give us for free).

    Resume: generation is sampled at temperature=0.9 for diversity, so unlike collect/sample.py's
    attempt_id there's no way to know a question's id before calling the teacher -- there's nothing
    to check existence of ahead of time. Instead this resumes at cell granularity: a cell already
    represented in questions_store/holdout_store is skipped outright (no new API call, no re-roll of
    already-accepted questions); a cell that failed last run (e.g. unparseable JSON) wrote nothing and
    is retried automatically, mirroring collect/sample.py's "failed calls retry on next run".

    max_tokens defaults to 8192, not the smaller budget collection/eval calls use elsewhere: this
    teacher model spends hidden reasoning tokens before emitting visible output, and empirically
    "hard"-difficulty cells alone can consume 4-5k completion tokens on reasoning before any JSON
    appears -- at 2048 or 4096 every "hard" cell came back with finish_reason="length" and zero
    visible content (confirmed via direct probe), silently losing a third of the coverage matrix.
    """
    all_train: list[Question] = list(questions_store.all())
    all_holdout: list[Question] = list(holdout_store.all())
    existing_cells = {q.cell for q in all_train} | {q.cell for q in all_holdout}
    total_dropped = 0

    for cell in matrix["cells"]:
        if cell["id"] in existing_cells:
            logger.info("gen(p1) cell=%s: already generated, skipping (resume)", cell["id"])
            continue
        want = quota_per_cell + holdout_per_cell
        messages = _build_gen_messages(cell["topic"], cell["difficulty"], want)
        try:
            result = await teacher.chat(messages, temperature=0.9, max_tokens=max_tokens)
            items = _extract_json_array(result.content)
        except Exception as exc:  # noqa: BLE001
            logger.error("generation failed for cell %s: %s -- skipping cell", cell["id"], exc)
            continue

        candidates: list[Question] = []
        cell_dropped = 0
        for item in items:
            if not _validate_gen_item(item):
                cell_dropped += 1
                continue
            candidates.append(
                Question(
                    id=ledger_mod.question_id(item["text"]),
                    text=item["text"],
                    cell=cell["id"],
                    family_id=cell["id"],
                    source="teacher_gen",
                    verify_method=item["verify_method"],
                    gold=str(item["gold"]),
                    split="train",  # placeholder; real split decided below after dedup
                    config_hash=config_hash,
                )
            )

        valid_count = len(candidates)
        candidates, dropped_exact = exact_dedup(candidates)
        candidates, dropped_minhash = minhash_semantic_dedup(candidates)
        candidates, dropped_family = template_family_cap(candidates, max_per_family=want)
        cell_dropped += len(dropped_exact) + len(dropped_minhash) + len(dropped_family)
        total_dropped += cell_dropped

        cell_holdout = candidates[:holdout_per_cell]
        cell_train = candidates[holdout_per_cell : holdout_per_cell + quota_per_cell]

        for q in cell_holdout:
            q = q.model_copy(update={"split": "holdout"})
            holdout_store.append(q)
            ledger.record(q.id, "question", upstream=[], config_hash=config_hash)
            all_holdout.append(q)
        for q in cell_train:
            questions_store.append(q)
            ledger.record(q.id, "question", upstream=[], config_hash=config_hash)
            all_train.append(q)

        logger.info(
            "gen(p1) cell=%s: raw=%d valid=%d dedup_dropped=%d -> train=%d holdout=%d",
            cell["id"], len(items) if isinstance(items, list) else 0, valid_count, cell_dropped, len(cell_train), len(cell_holdout),
        )

    return GenResult(train=all_train, holdout=all_holdout, dedup_dropped_count=total_dropped)
