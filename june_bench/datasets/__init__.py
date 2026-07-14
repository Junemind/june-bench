"""Datasets. SB0 ships the bundled `smoke` set; SB2 registers the four real benchmarks."""
from __future__ import annotations

from june_bench.datasets import registry
from june_bench.datasets.loaders import SmokeDataset

__all__ = ["registry", "SmokeDataset"]
