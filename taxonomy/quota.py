"""Coverage-matrix quota allocation. In P0 this only logs the target distribution (smoke reads seeds
directly rather than generating to quota); P1's generate.py consumes it to drive actual generation.
"""
from __future__ import annotations


def allocate_quota(matrix: dict, total: int) -> dict[str, int]:
    cells = matrix["cells"]
    n = len(cells)
    if n == 0:
        return {}
    base, remainder = divmod(total, n)
    quota: dict[str, int] = {}
    for i, cell in enumerate(cells):
        quota[cell["id"]] = base + (1 if i < remainder else 0)
    return quota
