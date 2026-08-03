"""Docker container pool management (pre-warmed pool, network isolation, resource limits) — P2."""
from __future__ import annotations


class ContainerPool:
    def acquire(self):
        raise NotImplementedError("Docker sandbox pool ships in P2.")

    def release(self, container) -> None:
        raise NotImplementedError("Docker sandbox pool ships in P2.")
