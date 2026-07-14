"""System adapters. SB0 ships the no-deps oracles; SB3/SB4 add JuneApiSystem / CogneeSystem
(behind extras) — each a one-file adapter registered here."""
from __future__ import annotations

from june_bench.systems.base import EchoSystem, NullSystem


def _june_api_factory():
    from june_bench.systems.june_api import from_env   # lazy: only [june-api] needs httpx
    return from_env()


def _june_local_factory():
    from june_bench.systems.june_local import JuneLocalSystem
    return JuneLocalSystem()


def _cognee_factory():
    from june_bench.systems.cognee import from_cognee   # lazy: only [cognee] needs cognee
    return from_cognee()


# name → zero-arg factory (heavy adapters import lazily inside their factory so the base install
# needs no optional deps).
_SYSTEMS: dict = {
    "null": NullSystem,
    "echo": EchoSystem,
    "june-api": _june_api_factory,
    "june": _june_api_factory,          # alias for the default June path
    "june-local": _june_local_factory,
    "cognee": _cognee_factory,          # first pluggable competitor ([cognee] extra)
}


def get(name: str):
    if name not in _SYSTEMS:
        raise KeyError(f"unknown system {name!r}; available: {names()}")
    return _SYSTEMS[name]()


def names() -> list[str]:
    return sorted(_SYSTEMS)


def register(name: str, factory) -> None:  # noqa: ANN001
    _SYSTEMS[name] = factory


__all__ = ["NullSystem", "EchoSystem", "get", "names", "register"]
