"""Per-dataset scoring profiles (SB1) — which axes are the *headline* for each benchmark.

All benchmarks are scored on the same three axes (accuracy/coverage/cost, `score.py`), but they don't
all *report* the same headline: HotpotQA/2Wiki/MuSiQue/FinanceBench are EM/F1; the memory benchmarks
(LoCoMo, LongMemEval) pair accuracy with coverage / retrieval-recall. A profile is just the ordered
metric keys a reporter surfaces — adding a dataset's profile is one entry, never a scorer change."""
from __future__ import annotations

_PROFILES: dict[str, tuple[str, ...]] = {
    "default": ("em", "f1", "coverage"),
    "hotpot": ("em", "f1"),
    "2wiki": ("em", "f1"),
    "musique": ("em", "f1"),
    "locomo": ("em", "f1", "coverage"),
    "longmemeval": ("em", "f1", "context_recall"),
    "financebench": ("em", "f1"),
    "smoke": ("em", "f1"),
}


def headline_metrics(dataset: str) -> tuple[str, ...]:
    """The ordered headline metric keys for a dataset (falls back to the default profile)."""
    return _PROFILES.get(dataset, _PROFILES["default"])


def headline(summary: dict, dataset: str) -> dict:
    """Project a full `score()` summary down to a dataset's headline metrics (order preserved)."""
    return {k: summary[k] for k in headline_metrics(dataset) if k in summary}


def register_profile(dataset: str, metrics: tuple[str, ...]) -> None:
    if not metrics:
        raise ValueError("a profile needs at least one metric")
    _PROFILES[dataset] = tuple(metrics)


__all__ = ["headline_metrics", "headline", "register_profile"]
