"""JuneLocalSystem (SB3 stub) — the in-process June adapter, behind the `[june-local]` extra.

For runs that don't want an HTTP hop, June can run in-process — but **only via a source-protected
(compiled) wheel**, never `june_ai` source. That wheel is the output of the separate GitHub+pip
source-protected phase; until it's published this adapter is a clear-erroring stub, so the public
package never depends on June's source. Use `--system june-api` (the default HTTP path) meanwhile.
"""
from __future__ import annotations

_MSG = (
    "JuneLocalSystem needs the source-protected June wheel (a compiled `june` distribution, no "
    "source). Install it via `pip install june-bench[june-local]` once the GitHub+pip "
    "source-protected phase has published it. For now use `--system june-api` (HTTP to /v1/answer)."
)


class JuneLocalSystem:
    name = "june-local"

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        try:
            import june as _june_protected_wheel  # noqa: F401  (the COMPILED wheel, not june_ai source)
        except Exception as exc:  # not installed → clear, actionable error
            raise RuntimeError(_MSG) from exc
        raise NotImplementedError(
            "JuneLocalSystem wiring lands when the source-protected wheel API is finalized; "
            "use --system june-api until then.")


__all__ = ["JuneLocalSystem"]
