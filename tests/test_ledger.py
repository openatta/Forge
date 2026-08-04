from core.ledger import Ledger, JsonlStore, attempt_id, question_id
from core.schemas import Question
from tests.factories import make_question


def test_append_is_idempotent(tmp_path):
    store = JsonlStore(tmp_path / "q.jsonl", Question)
    q = make_question(id="q-1")
    assert store.append(q) is True
    assert store.append(q) is False
    assert len(store.all()) == 1


def test_has_reflects_appended_ids(tmp_path):
    store = JsonlStore(tmp_path / "q.jsonl", Question)
    assert store.has("q-1") is False
    store.append(make_question(id="q-1"))
    assert store.has("q-1") is True


def test_all_preserves_append_order(tmp_path):
    store = JsonlStore(tmp_path / "q.jsonl", Question)
    for i in range(3):
        store.append(make_question(id=f"q-{i}", text=f"question {i}"))
    assert [q.id for q in store.all()] == ["q-0", "q-1", "q-2"]


def test_concurrent_processes_do_not_duplicate_same_id(tmp_path):
    """Simulates two run.py processes pointed at the same data_dir: both construct a JsonlStore
    before either writes, so their in-memory `_ids` sets start out equally stale. Regression test
    for the cross-process append() fix -- previously each store only ever trusted the id set it
    loaded at construction time, so both processes could pass the check and duplicate the line.
    """
    path = tmp_path / "q.jsonl"
    store_a = JsonlStore(path, Question)
    store_b = JsonlStore(path, Question)
    q = make_question(id="q-shared")

    assert store_a.append(q) is True
    assert store_b.append(q) is False  # must see store_a's write despite store_b's stale _ids

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_concurrent_processes_can_append_different_ids(tmp_path):
    path = tmp_path / "q.jsonl"
    store_a = JsonlStore(path, Question)
    store_b = JsonlStore(path, Question)

    assert store_a.append(make_question(id="q-a")) is True
    assert store_b.append(make_question(id="q-b")) is True

    ids = {q.id for q in JsonlStore(path, Question).all()}
    assert ids == {"q-a", "q-b"}


def test_ledger_trace_walks_upstream_root_first(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record("q-1", "question", upstream=[], config_hash="cfg")
    ledger.record("att-1", "attempt", upstream=["q-1"], config_hash="cfg")
    ledger.record("smp-1", "sample", upstream=["att-1"], config_hash="cfg")

    chain = ledger.trace("smp-1")
    assert [e.id for e in chain] == ["q-1", "att-1", "smp-1"]


def test_id_helpers_are_deterministic_and_normalize_text():
    qid = question_id("What is 2+2?")
    assert qid == question_id("  what IS 2+2?  ")  # normalize() folds case/whitespace

    aid = attempt_id(qid, "teacher", "hi-0", "fp123456")
    assert aid == attempt_id(qid, "teacher", "hi-0", "fp123456")
    assert aid != attempt_id(qid, "teacher", "hi-1", "fp123456")
