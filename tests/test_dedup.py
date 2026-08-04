import pytest

from datagen.dedup import exact_dedup, template_family_cap
from tests.factories import make_question


def test_exact_dedup_drops_case_and_whitespace_duplicates():
    q1 = make_question(id="q-1", text="What is 2+2?")
    q2 = make_question(id="q-2", text="  what IS 2+2?  ")
    kept, dropped = exact_dedup([q1, q2])
    assert [q.id for q in kept] == ["q-1"]
    assert [q.id for q in dropped] == ["q-2"]


def test_exact_dedup_keeps_distinct_questions():
    q1 = make_question(id="q-1", text="What is 2+2?")
    q2 = make_question(id="q-2", text="What is 3+3?")
    kept, dropped = exact_dedup([q1, q2])
    assert len(kept) == 2
    assert dropped == []


def test_template_family_cap_keeps_first_n_per_family():
    qs = [make_question(id=f"q-{i}", text=f"q{i}", family_id="fam-a") for i in range(5)]
    kept, dropped = template_family_cap(qs, max_per_family=2)
    assert [q.id for q in kept] == ["q-0", "q-1"]
    assert [q.id for q in dropped] == ["q-2", "q-3", "q-4"]


def test_template_family_cap_independent_across_families():
    qs = [make_question(id="a-1", family_id="fam-a"), make_question(id="b-1", family_id="fam-b")]
    kept, dropped = template_family_cap(qs, max_per_family=1)
    assert [q.id for q in kept] == ["a-1", "b-1"]
    assert dropped == []


def test_minhash_semantic_dedup_drops_near_duplicates():
    pytest.importorskip("datasketch")
    from datagen.dedup import minhash_semantic_dedup

    base = "the quick brown fox jumps over the lazy dog near the river bank today"
    near_dup = base + " indeed"
    distinct = "completely unrelated sentence about something else entirely different"

    qs = [
        make_question(id="q-1", text=base),
        make_question(id="q-2", text=near_dup),
        make_question(id="q-3", text=distinct),
    ]
    kept, dropped = minhash_semantic_dedup(qs, threshold=0.5)
    assert [q.id for q in kept] == ["q-1", "q-3"]
    assert [q.id for q in dropped] == ["q-2"]
