from core.hashing import content_hash, short_hash


def test_content_hash_is_order_independent():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_content_hash_changes_with_content():
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_short_hash_is_a_prefix_of_content_hash():
    h = short_hash({"x": 1}, 10)
    assert len(h) == 10
    assert h == content_hash({"x": 1})[:10]
