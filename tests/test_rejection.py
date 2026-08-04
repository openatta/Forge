from core.ledger import sample_id
from select_.rejection import compute_pass_rates, pick_best_passing_attempt, select_rejection
from tests.factories import make_attempt, make_question, make_verify


def test_pick_best_passing_attempt_prefers_shortest_passing():
    q = make_question(id="q-1")
    long_ok = make_attempt(id="att-long", question_id="q-1", content="a long correct solution", index=0)
    short_ok = make_attempt(id="att-short", question_id="q-1", content="ok", index=1)
    failing = make_attempt(id="att-fail", question_id="q-1", content="x", index=2)
    attempts = [long_ok, short_ok, failing]
    verify_by_attempt = {
        "att-long": make_verify("att-long", True),
        "att-short": make_verify("att-short", True),
        "att-fail": make_verify("att-fail", False),
    }
    best = pick_best_passing_attempt(q, attempts, verify_by_attempt)
    assert best.id == "att-short"


def test_pick_best_passing_attempt_ties_broken_by_lowest_index():
    q = make_question(id="q-1")
    a0 = make_attempt(id="att-0", question_id="q-1", content="same", index=0)
    a1 = make_attempt(id="att-1", question_id="q-1", content="same", index=1)
    verify_by_attempt = {"att-0": make_verify("att-0", True), "att-1": make_verify("att-1", True)}
    best = pick_best_passing_attempt(q, [a1, a0], verify_by_attempt)
    assert best.id == "att-0"


def test_pick_best_passing_attempt_returns_none_when_all_fail():
    q = make_question(id="q-1")
    a0 = make_attempt(id="att-0", question_id="q-1", content="wrong")
    verify_by_attempt = {"att-0": make_verify("att-0", False)}
    assert pick_best_passing_attempt(q, [a0], verify_by_attempt) is None


def test_pick_best_passing_attempt_ignores_student_attempts():
    q = make_question(id="q-1")
    student_ok = make_attempt(id="att-s", question_id="q-1", content="ok", role="student")
    verify_by_attempt = {"att-s": make_verify("att-s", True)}
    assert pick_best_passing_attempt(q, [student_ok], verify_by_attempt) is None


def test_select_rejection_builds_expected_sample():
    q = make_question(id="q-1")
    a0 = make_attempt(id="att-0", question_id="q-1", content="ok")
    verify_by_attempt = {"att-0": make_verify("att-0", True)}
    sample = select_rejection(q, [a0], verify_by_attempt, batch_id="batch-x", config_hash="cfg")
    assert sample.id == sample_id("q-1", "sft", "batch-x", "att-0")
    assert sample.question_id == "q-1"
    assert sample.attempt_id == "att-0"
    assert sample.payload == {"content": "ok"}


def test_select_rejection_none_when_nothing_passes():
    q = make_question(id="q-1")
    a0 = make_attempt(id="att-0", question_id="q-1", content="bad")
    verify_by_attempt = {"att-0": make_verify("att-0", False)}
    assert select_rejection(q, [a0], verify_by_attempt, "batch-x", "cfg") is None


def test_compute_pass_rates():
    q = make_question(id="q-1")
    teacher_hi_pass = make_attempt(id="att-t1", question_id="q-1", role="teacher", strategy="high_temp", index=0)
    teacher_hi_fail = make_attempt(id="att-t2", question_id="q-1", role="teacher", strategy="high_temp", index=1)
    student_pass = make_attempt(id="att-s1", question_id="q-1", role="student", strategy="weak")
    attempts = [teacher_hi_pass, teacher_hi_fail, student_pass]
    verify_by_attempt = {
        "att-t1": make_verify("att-t1", True),
        "att-t2": make_verify("att-t2", False),
        "att-s1": make_verify("att-s1", True),
    }
    rates = compute_pass_rates([q], attempts, verify_by_attempt)
    assert rates == {"p_T": 0.5, "p_S": 1.0}
