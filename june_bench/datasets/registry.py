"""Dataset registry (SB0) — name → Dataset. Adding a benchmark is one entry, never a runner edit."""
from __future__ import annotations

from june_bench.ports import Dataset
from june_bench.datasets.loaders import (
    HotpotFormatDataset,
    JuneFormatDataset,
    SmokeDataset,
)

_DATASETS: dict[str, Dataset] = {}


def register(dataset: Dataset) -> None:
    if not getattr(dataset, "name", ""):
        raise ValueError("dataset must have a non-empty name")
    _DATASETS[dataset.name] = dataset


def get(name: str) -> Dataset:
    if name not in _DATASETS:
        raise KeyError(f"unknown dataset {name!r}; registered: {names()}")
    return _DATASETS[name]


def names() -> list[str]:
    return sorted(_DATASETS)


register(SmokeDataset())
# SB2 — the four benchmarks (six datasets, two formats). Full files resolve from the data dir
# (JUNE_BENCH_DATA / repo); the bundled `smoke` split needs no data.
register(HotpotFormatDataset("hotpot", "hotpot_dev.json"))
register(HotpotFormatDataset("2wiki", "2wiki_dev.json"))
register(HotpotFormatDataset("musique", "musique_dev.json"))
register(JuneFormatDataset("locomo", "locomo.june.json"))
register(JuneFormatDataset("longmemeval", "lme.june.json"))
register(JuneFormatDataset("financebench", "financebench.june.json"))

__all__ = ["register", "get", "names"]
