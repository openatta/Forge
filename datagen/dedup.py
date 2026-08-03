"""Dedup. NOTE the one deliberate exception to the P1/P2-stub tag on this file: exact_dedup() is a
real P0 implementation because the smoke flow must demonstrably catch a duplicated seed question
(see docs/90-MVP搭建指南.md §5.2). minhash_semantic_dedup()/template_family_cap() are the real P1
implementations (see docs/reference/02 §五 for the four-layer dedup this covers layers 1 and 3 of;
layer 2, embedding-based semantic dedup, is deliberately deferred -- see project plan).

datasketch is imported lazily inside minhash_semantic_dedup(), not at module top level: it lives in
the optional `p1` extra (`pip install -e '.[p1]'`), and this module must stay importable for the P0
smoke path even when that extra isn't installed.
"""
from __future__ import annotations

from core.ledger import normalize
from core.schemas import Question


def exact_dedup(questions: list[Question]) -> tuple[list[Question], list[Question]]:
    """Drop exact-duplicate (same normalized text) questions. Returns (kept, dropped)."""
    seen: set[str] = set()
    kept: list[Question] = []
    dropped: list[Question] = []
    for q in questions:
        key = normalize(q.text)
        if key in seen:
            dropped.append(q)
        else:
            seen.add(key)
            kept.append(q)
    return kept, dropped


def _word_shingles(text: str, k: int = 3) -> set[str]:
    words = text.split()
    if len(words) < k:
        return {text} if text else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def minhash_semantic_dedup(questions: list[Question], threshold: float = 0.85) -> tuple[list[Question], list[Question]]:
    """MinHash near-duplicate detection over word 3-shingles of normalized text. Returns (kept, dropped).

    Only the "MinHash" half of the file's original name is implemented -- true semantic (embedding-
    based) dedup would need an extra embedding-model call and is deliberately out of scope for this
    pass; template-family dedup lives in template_family_cap() below instead.
    """
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept: list[Question] = []
    dropped: list[Question] = []

    for q in questions:
        mh = MinHash(num_perm=128)
        for shingle in _word_shingles(normalize(q.text)):
            mh.update(shingle.encode("utf-8"))
        if lsh.query(mh):
            dropped.append(q)
        else:
            lsh.insert(q.id, mh)
            kept.append(q)

    return kept, dropped


def template_family_cap(questions: list[Question], max_per_family: int) -> tuple[list[Question], list[Question]]:
    """Cap how many questions sharing the same family_id survive, keeping the first max_per_family
    seen. Per docs/reference/02 §五 this is "最容易漏的一层" -- easy to miss because exact/near-dup
    checks alone don't catch "same solving pattern, different surface numbers"."""
    counts: dict[str, int] = {}
    kept: list[Question] = []
    dropped: list[Question] = []
    for q in questions:
        counts[q.family_id] = counts.get(q.family_id, 0) + 1
        if counts[q.family_id] <= max_per_family:
            kept.append(q)
        else:
            dropped.append(q)
    return kept, dropped
