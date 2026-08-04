from select_.baselines import stratified_random_baseline
from tests.factories import make_attempt, make_question, make_verify


def _eligible_pool(cell: str, n: int):
    questions, attempts, verify_by_attempt = [], [], {}
    for i in range(n):
        q = make_question(id=f"{cell}-q{i}", cell=cell)
        a = make_attempt(id=f"{cell}-att{i}", question_id=q.id, role="teacher", strategy="high_temp")
        questions.append(q)
        attempts.append(a)
        verify_by_attempt[a.id] = make_verify(a.id, True)
    return questions, attempts, verify_by_attempt


def test_stratified_random_baseline_respects_target_counts():
    questions, attempts, verify_by_attempt = _eligible_pool("a", 5)
    samples = stratified_random_baseline(questions, attempts, verify_by_attempt, {"a": 3}, "batch-x", "cfg", seed=1)
    assert len(samples) == 3
    assert all(s.question_id.startswith("a-q") for s in samples)


def test_stratified_random_baseline_is_reproducible_for_same_seed():
    questions, attempts, verify_by_attempt = _eligible_pool("a", 5)
    s1 = stratified_random_baseline(questions, attempts, verify_by_attempt, {"a": 3}, "batch-x", "cfg", seed=7)
    s2 = stratified_random_baseline(questions, attempts, verify_by_attempt, {"a": 3}, "batch-x", "cfg", seed=7)
    assert [s.question_id for s in s1] == [s.question_id for s in s2]


def test_stratified_random_baseline_caps_at_pool_size():
    questions, attempts, verify_by_attempt = _eligible_pool("a", 2)
    samples = stratified_random_baseline(questions, attempts, verify_by_attempt, {"a": 5}, "batch-x", "cfg", seed=1)
    assert len(samples) == 2


def test_stratified_random_baseline_only_draws_from_eligible():
    q_ok = make_question(id="q-ok", cell="a")
    q_bad = make_question(id="q-bad", cell="a")
    ok_attempt = make_attempt(id="att-ok", question_id="q-ok", role="teacher", strategy="high_temp")
    bad_attempt = make_attempt(id="att-bad", question_id="q-bad", role="teacher", strategy="high_temp")
    verify_by_attempt = {"att-ok": make_verify("att-ok", True), "att-bad": make_verify("att-bad", False)}
    samples = stratified_random_baseline(
        [q_ok, q_bad], [ok_attempt, bad_attempt], verify_by_attempt, {"a": 5}, "batch-x", "cfg", seed=1
    )
    assert [s.question_id for s in samples] == ["q-ok"]
