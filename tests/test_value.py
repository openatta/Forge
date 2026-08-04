from select_.value import compute_question_stats, select_by_value
from tests.factories import make_attempt, make_question, make_verify


def _matrix(cell_ids):
    return {"cells": [{"id": cid, "topic": "t", "difficulty": "easy"} for cid in cell_ids]}


def test_compute_question_stats_basic():
    q = make_question(id="q-1", cell="a")
    teacher_ok = make_attempt(id="att-t", question_id="q-1", role="teacher", strategy="high_temp")
    student_ok = make_attempt(id="att-s", question_id="q-1", role="student", strategy="weak")
    verify_by_attempt = {"att-t": make_verify("att-t", True), "att-s": make_verify("att-s", True)}
    stats = compute_question_stats([q], [teacher_ok, student_ok], verify_by_attempt)
    assert stats["q-1"] == {"p_T": 1.0, "p_S": 1.0, "gap": 0.0}


def test_compute_question_stats_gap_clamped_to_zero():
    # p_T=0 (teacher fails), p_S=1 (student passes) -- gap must not go negative.
    q = make_question(id="q-1", cell="a")
    teacher_fail = make_attempt(id="att-t", question_id="q-1", role="teacher", strategy="high_temp")
    student_ok = make_attempt(id="att-s", question_id="q-1", role="student", strategy="weak")
    verify_by_attempt = {"att-t": make_verify("att-t", False), "att-s": make_verify("att-s", True)}
    stats = compute_question_stats([q], [teacher_fail, student_ok], verify_by_attempt)
    assert stats["q-1"]["gap"] == 0.0


def test_select_by_value_guarantees_per_cell_quota_before_leftover_fill():
    # 2 cells, target_size=2 -> quota 1 each. Cell "a" has one eligible question, cell "b" has two.
    # Pass 1 must still take 1 from "b" (its fair share) rather than letting "a"'s absence starve it.
    matrix = _matrix(["a", "b"])
    qa1 = make_question(id="qa-1", cell="a")
    qb1 = make_question(id="qb-1", cell="b")
    qb2 = make_question(id="qb-2", cell="b")

    def teacher_ok(qid, aid):
        return make_attempt(id=aid, question_id=qid, role="teacher", strategy="high_temp")

    attempts = [teacher_ok("qa-1", "att-a1"), teacher_ok("qb-1", "att-b1"), teacher_ok("qb-2", "att-b2")]
    verify_by_attempt = {a.id: make_verify(a.id, True) for a in attempts}

    samples = select_by_value(
        [qa1, qb1, qb2], attempts, verify_by_attempt, matrix, target_size=2, batch_id="b1", config_hash="cfg"
    )
    assert {s.question_id for s in samples} == {"qa-1", "qb-1"}


def test_select_by_value_respects_target_size_via_leftover_fill():
    matrix = _matrix(["a"])
    qs = [make_question(id=f"q-{i}", cell="a") for i in range(4)]
    attempts = [make_attempt(id=f"att-{i}", question_id=f"q-{i}", role="teacher", strategy="high_temp") for i in range(4)]
    verify_by_attempt = {a.id: make_verify(a.id, True) for a in attempts}

    samples = select_by_value(qs, attempts, verify_by_attempt, matrix, target_size=3, batch_id="b1", config_hash="cfg")
    assert len(samples) == 3


def test_select_by_value_excludes_questions_without_passing_teacher_attempt():
    matrix = _matrix(["a"])
    q_ok = make_question(id="q-ok", cell="a")
    q_bad = make_question(id="q-bad", cell="a")
    ok_attempt = make_attempt(id="att-ok", question_id="q-ok", role="teacher", strategy="high_temp")
    bad_attempt = make_attempt(id="att-bad", question_id="q-bad", role="teacher", strategy="high_temp")
    verify_by_attempt = {"att-ok": make_verify("att-ok", True), "att-bad": make_verify("att-bad", False)}

    samples = select_by_value(
        [q_ok, q_bad], [ok_attempt, bad_attempt], verify_by_attempt, matrix,
        target_size=5, batch_id="b1", config_hash="cfg",
    )
    assert [s.question_id for s in samples] == ["q-ok"]
