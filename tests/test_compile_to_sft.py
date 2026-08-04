from collect.sample import SYSTEM_PROMPT
from compile_.to_sft import compile_sft, latest_sft_records, render_sft_example
from core.ledger import JsonlStore, sample_id
from core.schemas import Sample, SftRecord
from tests.factories import make_attempt, make_question


def test_render_sft_example_shape():
    q = make_question(id="q-1", text="2+2?")
    a = make_attempt(id="att-1", question_id="q-1", content="4")
    messages = render_sft_example(q, a)
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
    ]


def test_compile_sft_writes_and_is_idempotent(tmp_path):
    q = make_question(id="q-1", text="2+2?")
    a = make_attempt(id="att-1", question_id="q-1", content="4")
    sample = Sample(
        id=sample_id("q-1", "sft", "batch-x", "att-1"), question_id="q-1", attempt_id="att-1",
        kind="sft", payload={"content": "4"}, batch_id="batch-x", config_hash="cfg",
    )
    sft_store = JsonlStore(tmp_path / "sft.jsonl", SftRecord)

    written_first = compile_sft([sample], {"att-1": a}, {"q-1": q}, sft_store)
    written_second = compile_sft([sample], {"att-1": a}, {"q-1": q}, sft_store)

    assert written_first == 1
    assert written_second == 0
    assert len(sft_store.all()) == 1


def test_latest_sft_records_last_write_wins():
    r1 = SftRecord(id="smp-1", messages=[], meta={"question_id": "q-1"})
    r2 = SftRecord(id="smp-2", messages=[], meta={"question_id": "q-1"})
    assert [r.id for r in latest_sft_records([r1, r2])] == ["smp-2"]
