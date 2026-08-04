from taxonomy.quota import allocate_quota


def test_allocate_quota_splits_evenly():
    matrix = {"cells": [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]}
    assert allocate_quota(matrix, total=8) == {"a": 2, "b": 2, "c": 2, "d": 2}


def test_allocate_quota_distributes_remainder_to_first_cells():
    matrix = {"cells": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}
    quota = allocate_quota(matrix, total=7)
    assert quota == {"a": 3, "b": 2, "c": 2}
    assert sum(quota.values()) == 7


def test_allocate_quota_empty_matrix():
    assert allocate_quota({"cells": []}, total=10) == {}
