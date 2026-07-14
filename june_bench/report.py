"""Reporter (SB0) — a scored summary → a self-describing markdown row + JSON.

Every row records *system · dataset · split · model + the three axes*, so a published number is
reproducible and self-describing (the honesty discipline). SB5 extends this to a full multi-arm
`RESULTS.md` table; SB0 ships the single-run row.
"""
from __future__ import annotations

import json


def to_json(summary: dict, *, system: str, dataset: str, split: str, model: str = "") -> str:
    head = {"system": system, "dataset": dataset, "split": split, "model": model}
    return json.dumps({**head, **summary}, indent=2)


def _cell(v) -> str:  # noqa: ANN001
    """Escape a value for a markdown table cell — a literal `|` or newline in an error string / name
    would otherwise break the row (audit L16)."""
    return str(v).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def to_markdown(summary: dict, *, system: str, dataset: str, split: str, model: str = "") -> str:
    s = summary
    jh, jd, jv = ((" judge |", "---|", f" {s['judge']} |") if "judge" in s else ("", "", ""))
    return (
        f"### {system} · {dataset}/{split}" + (f" · {model}" if model else "") + "\n\n"
        "| n | answered | EM | F1 | coverage | ctx-recall | calls/ans | cost/ans |" + jh + "\n"
        "|---|---|---|---|---|---|---|---|" + jd + "\n"
        f"| {s['n']} | {s['answered']} | {s['em']} | {s['f1']} | {s['coverage']} | "
        f"{s['context_recall']} | {s['calls_per_answer']} | {s['cost_per_answer']} |" + jv + "\n"
    )


def suite_markdown(rows: list[dict], *, split: str = "", model: str = "") -> str:
    """A systems × datasets matrix → one `RESULTS.md`-ready table. Each row is
    ``{system, dataset, summary}`` or ``{system, dataset, error}`` (fail-soft per cell). The
    per-dataset *headline* metrics (profiles) are noted, but every axis is shown for honesty."""
    from june_bench.profiles import headline_metrics
    head = "# june-bench results"
    if split:
        head += f" · split `{split}`"
    if model:
        head += f" · model `{model}`"
    has_judge = any("summary" in r and "judge" in r["summary"] for r in rows)
    jh = " judge |" if has_judge else ""
    jd = "---|" if has_judge else ""
    lines = [head, "",
             "| system | dataset | headline | n | EM | F1 | coverage | ctx-recall | calls/ans | cost/ans |" + jh,
             "|---|---|---|---|---|---|---|---|---|---|" + jd]
    for r in rows:
        sysn, dsn = _cell(r.get("system", "?")), _cell(r.get("dataset", "?"))
        if "error" in r:
            lines.append(f"| {sysn} | {dsn} | — | — | _error_ | {_cell(str(r['error'])[:60])} | | | | |"
                         + (" |" if has_judge else ""))
            continue
        s = r["summary"]
        hl = "/".join(headline_metrics(dsn))
        jv = f" {s['judge']} |" if has_judge else ""
        lines.append(f"| {sysn} | {dsn} | {hl} | {s['n']} | {s['em']} | {s['f1']} | "
                     f"{s['coverage']} | {s['context_recall']} | {s['calls_per_answer']} | "
                     f"{s['cost_per_answer']} |" + jv)
    lines += ["", "_Same data, same scorer; every cell records the system, dataset, split and model. "
              "Headline = the dataset's reported metric(s); all axes shown for honesty._"]
    return "\n".join(lines) + "\n"


def suite_json(rows: list[dict], *, split: str = "", model: str = "") -> str:
    return json.dumps({"split": split, "model": model, "rows": rows}, indent=2)


__all__ = ["to_json", "to_markdown", "suite_markdown", "suite_json"]
