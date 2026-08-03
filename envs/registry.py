"""Environment registry. P0 registers "toy_calc" pointing at its (stubbed) P2 implementation, so
`run.py smoke --env toy_calc` resolves through a real lookup before hitting the stub's NotImplementedError.

Registration is a manual dict assignment below rather than a decorator on each Environment class:
a decorator defined here (`register(name)`) and applied inside envs/toy_calc/__init__.py would need
that module to import back from this one, which is circular since this module already imports
envs.toy_calc for its side effect below. Revisit if/when more than a couple of environments exist.
"""
from __future__ import annotations

from envs.base import Environment

ENV_REGISTRY: dict[str, type[Environment]] = {}


def get(name: str) -> type[Environment]:
    if name not in ENV_REGISTRY:
        raise KeyError(f"unknown environment: {name!r} (registered: {sorted(ENV_REGISTRY)})")
    return ENV_REGISTRY[name]


from envs.toy_calc import ToyCalcEnv  # noqa: E402

ENV_REGISTRY["toy_calc"] = ToyCalcEnv
