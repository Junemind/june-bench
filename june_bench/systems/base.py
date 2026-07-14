"""No-deps reference systems (SB0) — the wiring oracles.

These let the *entire* pipeline (load → run → score → report) be exercised with no model, no
network, no datasets — the in-sandbox smoke that proves the harness before any real System lands
(mirrors the answer-kernel's no-LLM smoke). ``NullSystem`` is the floor (always abstains → EM 0);
``EchoSystem`` is the oracle (returns a gold → EM 1), so a green smoke proves the scorer reports
both ends correctly.
"""
from __future__ import annotations

from june_bench.ports import Example, Prediction


class NullSystem:
    """Answers nothing — the degenerate floor. Useful to prove the scorer reports 0 honestly."""
    name = "null"

    async def answer(self, example: Example) -> Prediction:
        return Prediction(text="", meta={"calls": 0, "abstained": True})


class EchoSystem:
    """The wiring **oracle**: returns the example's first gold (or, if none, the first context
    line). Not a real system — it exists so a no-deps end-to-end smoke yields a known score (EM 1),
    proving load→run→score→report are wired correctly."""
    name = "echo"

    async def answer(self, example: Example) -> Prediction:
        if example.golds:
            return Prediction(text=example.golds[0], meta={"calls": 1})
        first = example.context[0] if example.context else ""
        return Prediction(text=first, meta={"calls": 1})


__all__ = ["NullSystem", "EchoSystem"]
