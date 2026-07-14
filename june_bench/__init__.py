"""june_bench — a pip-installable, reproducible benchmark suite for memory/QA systems.

`run(system, dataset) → records → score`. Two typed ports (`System`, `Dataset`) are the only
extension points; June and pluggable competitors plug in behind `System`, the four benchmarks behind
`Dataset`. The harness core is pure (no model, no network) and unit-tests with the bundled `smoke`
fixtures; June is reached over its REST API by default (no June source shipped).
"""
from __future__ import annotations

from june_bench.ports import Dataset, Example, Prediction, Record, System
from june_bench.runner import run, run_async
from june_bench.score import score

__version__ = "0.0.1"

__all__ = [
    "Example", "Prediction", "Record", "System", "Dataset",
    "run", "run_async", "score", "__version__",
]
