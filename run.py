"""CLI entry point: smoke / gen / collect / verify / select / compile / eval / report / trace.

See docs/90-MVP搭建指南.md §5 for the full P0 smoke-flow spec this file implements.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from collect.sample import run_collect
from compile_.to_sft import compile_sft, latest_sft_records
from core.hashing import content_hash
from core.ledger import (
    JsonlStore,
    Ledger,
    batch_id as make_batch_id,
    baseline_batch_id as make_baseline_batch_id,
    run_id as make_run_id,
    verify_id,
)
from core.model_client import build_student_client, build_teacher_client
from core.schemas import Attempt, EvalRecord, Question, Sample, SftRecord, VerifyResult
from datagen.generate import generate_questions, generate_questions_via_teacher
from envs.registry import get as get_env
from envs.base import TaskSpec
from evals.report import build_p1_report, build_smoke_report, write_smoke_report
from evals.run_eval import run_holdout_eval
from select_.baselines import stratified_random_baseline
from select_.rejection import compute_pass_rates, latest_samples, select_rejection
from select_.value import select_by_value
from train.sft import train_sft
from verify.math_answer import MathAnswerVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run")

app = typer.Typer(add_completion=False)


def load_config(config_path: str = "configs/smoke.yaml") -> tuple[dict, str]:
    load_dotenv()
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return config, content_hash(config)


def _questions_by_id(config: dict) -> dict[str, Question]:
    train = JsonlStore(Path(config["data_dir"]) / "questions.jsonl", Question).all()
    holdout = JsonlStore(Path(config["holdout_dir"]) / "questions.jsonl", Question).all()
    return {q.id: q for q in [*train, *holdout]}


def stage_gen(config: dict, config_hash: str, ledger: Ledger):
    result = generate_questions(
        seeds_path=Path(config["seeds_path"]),
        data_dir=Path(config["data_dir"]),
        holdout_dir=Path(config["holdout_dir"]),
        config_hash=config_hash,
        ledger=ledger,
    )
    logger.info(
        "gen: train=%d holdout=%d dedup_dropped=%d",
        len(result.train), len(result.holdout), result.dedup_dropped_count,
    )
    return result


async def stage_collect(config: dict, config_hash: str, ledger: Ledger, train_questions: list[Question]):
    teacher = build_teacher_client(concurrency=config["teacher"]["concurrency"])
    student = build_student_client()
    attempts_store = JsonlStore(Path(config["data_dir"]) / "attempts.jsonl", Attempt)
    stats = await run_collect(config, train_questions, teacher, student, attempts_store, ledger, config_hash)
    logger.info(
        "collect: teacher_total=%d (new %d) student_total=%d (new %d) teacher_cost=$%.4f",
        stats.teacher_total, stats.teacher_new, stats.student_total, stats.student_new,
        sum(a.usage.cost_usd for a in attempts_store.all() if a.actor_role == "teacher"),
    )
    return attempts_store, stats


def stage_verify(config: dict, config_hash: str, ledger: Ledger, attempts_store: JsonlStore):
    verifier = MathAnswerVerifier()
    verify_store = JsonlStore(Path(config["data_dir"]) / "verify.jsonl", VerifyResult)
    existing = {v.id: v for v in verify_store.all()}
    all_questions = _questions_by_id(config)

    verify_by_attempt: dict[str, VerifyResult] = {}
    for attempt in attempts_store.all():
        question = all_questions.get(attempt.question_id)
        if question is None:
            continue
        vid = verify_id(attempt.id, verifier.id, verifier.version)
        if vid in existing:
            vr = existing[vid]
        else:
            vr = verifier.verify(attempt, question)
            verify_store.append(vr)
            ledger.record(vr.id, "verify", upstream=[attempt.id], config_hash=config_hash)
            existing[vid] = vr
        verify_by_attempt[attempt.id] = vr

    passed = sum(1 for v in verify_by_attempt.values() if v.passed)
    logger.info("verify: %d/%d attempts passed", passed, len(verify_by_attempt))
    return verify_store, verify_by_attempt


def stage_select(
    config: dict, config_hash: str, ledger: Ledger,
    train_questions: list[Question], attempts_store: JsonlStore, verify_by_attempt: dict,
):
    bid = make_batch_id(config_hash)
    samples_store = JsonlStore(Path(config["data_dir"]) / "samples.jsonl", Sample)
    attempts = attempts_store.all()
    for q in train_questions:
        sample = select_rejection(q, attempts, verify_by_attempt, bid, config_hash)
        if sample is not None and samples_store.append(sample):
            ledger.record(sample.id, "sample", upstream=[sample.attempt_id], config_hash=config_hash)

    pass_rates = compute_pass_rates(train_questions, attempts, verify_by_attempt)
    current = latest_samples(samples_store.all())
    logger.info("select: %d samples selected, p_T=%.2f p_S=%.2f", len(current), pass_rates["p_T"], pass_rates["p_S"])
    return samples_store, pass_rates


def stage_compile(config: dict, config_hash: str, ledger: Ledger, samples_store: JsonlStore, attempts_store: JsonlStore):
    sft_store = JsonlStore(Path(config["data_dir"]) / "sft.jsonl", SftRecord)
    attempts_by_id = {a.id: a for a in attempts_store.all()}
    questions_by_id = _questions_by_id(config)
    current_samples = latest_samples(samples_store.all())
    written = compile_sft(current_samples, attempts_by_id, questions_by_id, sft_store, ledger)
    sft_records = latest_sft_records(sft_store.all())
    logger.info("compile: %d new sft records (current total %d)", written, len(sft_records))
    return sft_store, sft_records


async def stage_eval(config: dict, config_hash: str, ledger: Ledger, holdout_questions: list[Question]):
    student = build_student_client()
    verifier = MathAnswerVerifier()
    eval_store = JsonlStore(Path(config["data_dir"]) / "eval.jsonl", EvalRecord)
    rid = make_run_id(student.model_id, config_hash)
    records = await run_holdout_eval(student, holdout_questions, verifier, rid, eval_store, ledger, config_hash)
    logger.info("eval: %d holdout records, %d passed", len(records), sum(1 for r in records if r.passed))
    return records


def stage_report(
    config: dict, ledger: Ledger, gen_result, collect_stats, pass_rates: dict,
    sft_records: list[SftRecord], eval_records: list[EvalRecord], attempts_store: JsonlStore,
):
    example_sample_id = sft_records[0].id if sft_records else None
    teacher_cost = sum(a.usage.cost_usd for a in attempts_store.all() if a.actor_role == "teacher")
    report_text = build_smoke_report(
        seed_count=len(gen_result.train) + len(gen_result.holdout) + gen_result.dedup_dropped_count,
        dedup_dropped=gen_result.dedup_dropped_count,
        train_count=len(gen_result.train),
        holdout_count=len(gen_result.holdout),
        collect_stats=collect_stats,
        total_teacher_cost_usd=teacher_cost,
        pass_rates=pass_rates,
        sft_count=len(sft_records),
        eval_records=eval_records,
        ledger=ledger,
        example_sample_id=example_sample_id,
    )
    write_smoke_report(report_text, Path(config["reports_dir"]) / "smoke_report.md")
    logger.info("report: written to %s/smoke_report.md", config["reports_dir"])
    return report_text


# ---------------------------------------------------------------------------
# P1: teacher-generated questions -> dedup -> collect/verify (shared with P0) -> value-based
# selection + stratified-random baseline -> compile both arms -> mock "train" both -> eval
# base/baseline/value/teacher arms -> A/B report. See docs/90-MVP搭建指南.md §六 and the project
# plan for scope (math only, training mocked, no Curator).
# ---------------------------------------------------------------------------

def _load_matrix(config: dict) -> dict:
    return yaml.safe_load(Path(config["matrix_path"]).read_text(encoding="utf-8"))


async def stage_gen_p1(config: dict, config_hash: str, ledger: Ledger, teacher):
    matrix = _load_matrix(config)
    questions_store = JsonlStore(Path(config["data_dir"]) / "questions.jsonl", Question)
    holdout_store = JsonlStore(Path(config["holdout_dir"]) / "questions.jsonl", Question)
    result = await generate_questions_via_teacher(
        matrix=matrix,
        teacher=teacher,
        quota_per_cell=config["gen"]["quota_per_cell"],
        holdout_per_cell=config["gen"]["holdout_per_cell"],
        questions_store=questions_store,
        holdout_store=holdout_store,
        ledger=ledger,
        config_hash=config_hash,
        max_tokens=config["gen"].get("max_tokens", 8192),
    )
    logger.info(
        "gen(p1): train=%d holdout=%d dedup_dropped=%d",
        len(result.train), len(result.holdout), result.dedup_dropped_count,
    )
    return result


def stage_select_p1(
    config: dict, config_hash: str, ledger: Ledger,
    train_questions: list[Question], attempts_store: JsonlStore, verify_by_attempt: dict,
):
    matrix = _load_matrix(config)
    attempts = attempts_store.all()
    bid = make_batch_id(config_hash)
    baseline_bid = make_baseline_batch_id(config_hash)

    value_store = JsonlStore(Path(config["data_dir"]) / "samples_value.jsonl", Sample)
    baseline_store = JsonlStore(Path(config["data_dir"]) / "samples_baseline.jsonl", Sample)

    value_samples = select_by_value(
        train_questions, attempts, verify_by_attempt, matrix, config["select"]["value_target_size"], bid, config_hash
    )
    for s in value_samples:
        if value_store.append(s):
            ledger.record(s.id, "sample", upstream=[s.attempt_id], config_hash=config_hash)

    questions_by_id = {q.id: q for q in train_questions}
    target_cell_counts: dict[str, int] = {}
    for s in value_samples:
        cell = questions_by_id[s.question_id].cell
        target_cell_counts[cell] = target_cell_counts.get(cell, 0) + 1

    baseline_samples = stratified_random_baseline(
        train_questions, attempts, verify_by_attempt, target_cell_counts,
        baseline_bid, config_hash, seed=config["select"]["baseline_seed"],
    )
    for s in baseline_samples:
        if baseline_store.append(s):
            ledger.record(s.id, "sample", upstream=[s.attempt_id], config_hash=config_hash)

    logger.info(
        "select(p1): value=%d baseline=%d target_cell_counts=%s",
        len(value_samples), len(baseline_samples), target_cell_counts,
    )
    return value_store, baseline_store


def stage_compile_p1(
    config: dict, ledger: Ledger, value_store: JsonlStore, baseline_store: JsonlStore,
    attempts_store: JsonlStore, train_questions: list[Question], holdout_questions: list[Question],
):
    attempts_by_id = {a.id: a for a in attempts_store.all()}
    questions_by_id = {q.id: q for q in [*train_questions, *holdout_questions]}

    value_sft_store = JsonlStore(Path(config["data_dir"]) / "sft_value.jsonl", SftRecord)
    baseline_sft_store = JsonlStore(Path(config["data_dir"]) / "sft_baseline.jsonl", SftRecord)

    compile_sft(latest_samples(value_store.all()), attempts_by_id, questions_by_id, value_sft_store, ledger)
    compile_sft(latest_samples(baseline_store.all()), attempts_by_id, questions_by_id, baseline_sft_store, ledger)

    value_sft = latest_sft_records(value_sft_store.all())
    baseline_sft = latest_sft_records(baseline_sft_store.all())
    logger.info("compile(p1): value_sft=%d baseline_sft=%d", len(value_sft), len(baseline_sft))
    return value_sft, baseline_sft


def stage_train_p1(config: dict, config_hash: str, ledger: Ledger, value_sft, baseline_sft):
    checkpoints_dir = Path(config["data_dir"]) / "checkpoints"
    mode = config["train"]["mode"]
    base_model = config["train"]["base_model"]
    ckpt_value = train_sft(value_sft, base_model, "value_selected", config_hash, ledger, checkpoints_dir, mode=mode)
    ckpt_baseline = train_sft(baseline_sft, base_model, "baseline_random", config_hash, ledger, checkpoints_dir, mode=mode)
    logger.info("train(p1) [%s]: value_checkpoint=%s baseline_checkpoint=%s", mode, ckpt_value, ckpt_baseline)
    return ckpt_value, ckpt_baseline


async def stage_eval_p1(
    config: dict, config_hash: str, ledger: Ledger, holdout_questions: list[Question],
    teacher, ckpt_value: str, ckpt_baseline: str,
):
    student = build_student_client()
    verifier = MathAnswerVerifier()
    eval_store = JsonlStore(Path(config["data_dir"]) / "eval.jsonl", EvalRecord)

    arm_run_ids = {
        "base": make_run_id(f"base:{student.model_id}", config_hash),
        "value_selected": make_run_id(f"value_selected:{ckpt_value}", config_hash),
        "baseline_random": make_run_id(f"baseline_random:{ckpt_baseline}", config_hash),
    }
    arm_eval_records: dict[str, list[EvalRecord]] = {}
    for arm, rid in arm_run_ids.items():
        records = await run_holdout_eval(student, holdout_questions, verifier, rid, eval_store, ledger, config_hash)
        arm_eval_records[arm] = records
        logger.info("eval(p1) arm=%s: %d records, %d passed", arm, len(records), sum(1 for r in records if r.passed))

    teacher_rid = make_run_id(f"teacher:{teacher.model_id}", config_hash)
    teacher_records = await run_holdout_eval(teacher, holdout_questions, verifier, teacher_rid, eval_store, ledger, config_hash)
    arm_eval_records["teacher"] = teacher_records
    logger.info(
        "eval(p1) arm=teacher: %d records, %d passed (real, non-mocked)",
        len(teacher_records), sum(1 for r in teacher_records if r.passed),
    )
    return arm_eval_records


def stage_report_p1(config: dict, ledger: Ledger, arm_eval_records: dict, example_sample_ids: dict):
    matrix = _load_matrix(config)
    report_text = build_p1_report(
        matrix=matrix, arm_eval_records=arm_eval_records, ledger=ledger, example_sample_ids=example_sample_ids
    )
    write_smoke_report(report_text, Path(config["reports_dir"]) / "p1_report.md")
    logger.info("report(p1): written to %s/p1_report.md", config["reports_dir"])
    return report_text


@app.command()
def smoke(env: str = typer.Option(None, help="If set, resolve+probe an envs/ implementation instead of running the smoke pipeline (P2 interface check).")):
    if env is not None:
        env_cls = get_env(env)
        env_cls().reset(TaskSpec(task_id="probe", payload={}))
        return

    config, config_hash = load_config()
    ledger = Ledger(Path(config["data_dir"]) / "ledger.jsonl")

    gen_result = stage_gen(config, config_hash, ledger)
    attempts_store, collect_stats = asyncio.run(stage_collect(config, config_hash, ledger, gen_result.train))
    _, verify_by_attempt = stage_verify(config, config_hash, ledger, attempts_store)
    samples_store, pass_rates = stage_select(config, config_hash, ledger, gen_result.train, attempts_store, verify_by_attempt)
    _, sft_records = stage_compile(config, config_hash, ledger, samples_store, attempts_store)
    eval_records = asyncio.run(stage_eval(config, config_hash, ledger, gen_result.holdout))
    stage_report(config, ledger, gen_result, collect_stats, pass_rates, sft_records, eval_records, attempts_store)

    typer.echo(f"smoke complete. train={len(gen_result.train)} sft_samples={len(sft_records)} report=reports/smoke_report.md")


@app.command()
def gen():
    config, config_hash = load_config()
    ledger = Ledger(Path(config["data_dir"]) / "ledger.jsonl")
    stage_gen(config, config_hash, ledger)


@app.command()
def collect():
    config, config_hash = load_config()
    ledger = Ledger(Path(config["data_dir"]) / "ledger.jsonl")
    train_questions = JsonlStore(Path(config["data_dir"]) / "questions.jsonl", Question).all()
    asyncio.run(stage_collect(config, config_hash, ledger, train_questions))


@app.command()
def verify():
    config, config_hash = load_config()
    ledger = Ledger(Path(config["data_dir"]) / "ledger.jsonl")
    attempts_store = JsonlStore(Path(config["data_dir"]) / "attempts.jsonl", Attempt)
    stage_verify(config, config_hash, ledger, attempts_store)


@app.command(name="select")
def select_cmd():
    config, config_hash = load_config()
    ledger = Ledger(Path(config["data_dir"]) / "ledger.jsonl")
    train_questions = JsonlStore(Path(config["data_dir"]) / "questions.jsonl", Question).all()
    attempts_store = JsonlStore(Path(config["data_dir"]) / "attempts.jsonl", Attempt)
    verify_store = JsonlStore(Path(config["data_dir"]) / "verify.jsonl", VerifyResult)
    verify_by_attempt = {v.attempt_id: v for v in verify_store.all()}
    stage_select(config, config_hash, ledger, train_questions, attempts_store, verify_by_attempt)


@app.command(name="compile")
def compile_cmd():
    config, config_hash = load_config()
    ledger = Ledger(Path(config["data_dir"]) / "ledger.jsonl")
    samples_store = JsonlStore(Path(config["data_dir"]) / "samples.jsonl", Sample)
    attempts_store = JsonlStore(Path(config["data_dir"]) / "attempts.jsonl", Attempt)
    stage_compile(config, config_hash, ledger, samples_store, attempts_store)


@app.command(name="eval")
def eval_cmd():
    config, config_hash = load_config()
    ledger = Ledger(Path(config["data_dir"]) / "ledger.jsonl")
    holdout_questions = JsonlStore(Path(config["holdout_dir"]) / "questions.jsonl", Question).all()
    asyncio.run(stage_eval(config, config_hash, ledger, holdout_questions))


@app.command()
def report():
    from collect.sample import CollectStats

    config, config_hash = load_config()
    ledger = Ledger(Path(config["data_dir"]) / "ledger.jsonl")
    # Re-running gen is idempotent (JsonlStore.append no-ops on ids already on disk) and is the only
    # way to get an accurate dedup_dropped_count when `report` is run standalone rather than via `smoke`.
    gen_result = stage_gen(config, config_hash, ledger)
    attempts_store = JsonlStore(Path(config["data_dir"]) / "attempts.jsonl", Attempt)
    verify_store = JsonlStore(Path(config["data_dir"]) / "verify.jsonl", VerifyResult)
    verify_by_attempt = {v.attempt_id: v for v in verify_store.all()}
    sft_records = latest_sft_records(JsonlStore(Path(config["data_dir"]) / "sft.jsonl", SftRecord).all())
    eval_store = JsonlStore(Path(config["data_dir"]) / "eval.jsonl", EvalRecord)

    train_ids = {q.id for q in gen_result.train}
    teacher_total = sum(1 for a in attempts_store.all() if a.question_id in train_ids and a.actor_role == "teacher")
    student_total = sum(1 for a in attempts_store.all() if a.question_id in train_ids and a.actor_role == "student")
    collect_stats = CollectStats(teacher_new=0, teacher_total=teacher_total, student_new=0, student_total=student_total)
    pass_rates = compute_pass_rates(gen_result.train, attempts_store.all(), verify_by_attempt)

    stage_report(config, ledger, gen_result, collect_stats, pass_rates, sft_records, eval_store.all(), attempts_store)


@app.command(name="p1")
def p1_cmd():
    config, config_hash = load_config("configs/p1_math_code.yaml")
    ledger = Ledger(Path(config["data_dir"]) / "ledger.jsonl")
    teacher = build_teacher_client(concurrency=config["teacher"]["concurrency"])

    gen_result = asyncio.run(stage_gen_p1(config, config_hash, ledger, teacher))
    attempts_store, collect_stats = asyncio.run(stage_collect(config, config_hash, ledger, gen_result.train))
    _, verify_by_attempt = stage_verify(config, config_hash, ledger, attempts_store)

    value_store, baseline_store = stage_select_p1(
        config, config_hash, ledger, gen_result.train, attempts_store, verify_by_attempt
    )
    value_sft, baseline_sft = stage_compile_p1(
        config, ledger, value_store, baseline_store, attempts_store, gen_result.train, gen_result.holdout
    )
    ckpt_value, ckpt_baseline = stage_train_p1(config, config_hash, ledger, value_sft, baseline_sft)
    arm_eval_records = asyncio.run(
        stage_eval_p1(config, config_hash, ledger, gen_result.holdout, teacher, ckpt_value, ckpt_baseline)
    )

    example_sample_ids = {
        "value_selected": value_sft[0].id if value_sft else None,
        "baseline_random": baseline_sft[0].id if baseline_sft else None,
    }
    stage_report_p1(config, ledger, arm_eval_records, example_sample_ids)

    typer.echo(
        f"p1 complete. train={len(gen_result.train)} value_samples={len(value_sft)} "
        f"baseline_samples={len(baseline_sft)} report=reports/p1_report.md"
    )


@app.command()
def trace(sample_id: str, config_path: str = typer.Option("configs/smoke.yaml", "--config")):
    config, _ = load_config(config_path)
    ledger = Ledger(Path(config["data_dir"]) / "ledger.jsonl")
    chain = ledger.trace(sample_id)
    if not chain:
        typer.echo(f"no ledger entries found for id={sample_id!r}")
        raise typer.Exit(code=1)
    for entry in chain:
        typer.echo(f"[{entry.type}] {entry.id}  (upstream: {entry.upstream or 'none'})")


if __name__ == "__main__":
    app()
