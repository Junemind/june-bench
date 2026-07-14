"""june_bench ports — the two typed contracts the whole suite extends through (SB0).

A benchmark is ``run(system, dataset) → records → score``. Only two things vary: the **System**
being benchmarked and the **Dataset** it runs on. Everything else (runner, scorer, reporter) is
fixed. So a new competitor or a new dataset is exactly one adapter behind one of these Protocols —
never an edit to the harness. Pure: stdlib only, so the core installs and unit-tests with no model,
no datasets, no network (the ``NullSystem``/``EchoSystem`` smoke).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Example:
    """One benchmark item. ``context`` is in-prompt passages (context-QA like HotpotQA);
    ``corpus`` is documents a *memory* system ingests before answering (LoCoMo/LongMemEval).
    A system uses whichever its modality needs. ``meta`` carries dataset-specific fields
    (category, session id, …) so per-category scoring needs no schema change."""
    qid: str
    question: str
    golds: tuple[str, ...]
    context: tuple[str, ...] = ()
    corpus: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Prediction:
    """A system's answer + optional provenance (``calls``/``cost``/``abstained`` in ``meta``)
    so the cost axis is measured, not guessed."""
    text: str
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Record:
    """One scored row: the prediction joined to its gold + provenance. The scorer reads only
    these, so scoring is decoupled from how the answer was produced (any System, any Dataset)."""
    qid: str
    question: str
    golds: tuple[str, ...]
    prediction: str
    context: tuple[str, ...] = ()
    calls: int = 0
    cost: float = 0.0
    abstained: bool = False
    meta: dict = field(default_factory=dict)


@runtime_checkable
class System(Protocol):
    """A thing being benchmarked. ``answer`` is the one required method; a memory system may also
    expose ``ingest(corpus)`` (the runner calls it when present). Async so real adapters (June over
    HTTP, an LLM) fit without blocking; sync logic just returns. The model/HTTP/heavy deps live in
    the adapter, never in this contract."""
    name: str

    async def answer(self, example: Example) -> Prediction: ...


@runtime_checkable
class Dataset(Protocol):
    """A benchmark's data. ``load(split)`` returns the examples for a named split (e.g. ``smoke``,
    ``dev100``, ``full``). Loaders are pure readers over the bundled fixtures or the fetched dumps."""
    name: str

    def load(self, split: str, limit: int | None = None) -> Sequence[Example]: ...


__all__ = ["Example", "Prediction", "Record", "System", "Dataset"]
