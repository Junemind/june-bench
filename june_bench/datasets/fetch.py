"""Fetch I/O for the full splits (SB-fetch) — the network/disk edge around the pure normalizers.

`fetch_one` downloads a `Source`'s mirror (or reads a local `--from` raw file), parses it (zip member
/ jsonl / json), runs the source's pure `normalize`, writes the loader's `full_file`, then **round-trips
it through the real loader** — a file that doesn't parse into ≥1 Example is deleted and reported, so a
wrong URL or format drift can never masquerade as data. Stdlib only (urllib/zipfile/json): this runs on
the user's machine after `pip install june-bench`, no extra dependency.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import tempfile
import urllib.request
import zipfile
from collections.abc import Sequence

from june_bench.datasets.fetch_sources import SOURCES, Source

_UA = {"User-Agent": "june-bench/fetch (+https://pypi.org/project/june-bench)"}


def default_data_dir() -> pathlib.Path:
    """Where fetched files land: JUNE_BENCH_DATA if set, else ~/.cache/june-bench/data."""
    env = os.environ.get("JUNE_BENCH_DATA")
    if env:
        return pathlib.Path(env)
    return pathlib.Path.home() / ".cache" / "june-bench" / "data"


def _download(url: str, dest: pathlib.Path, *, timeout: float = 120.0) -> None:
    req = urllib.request.Request(url, headers=_UA)  # noqa: S310 — pinned canonical mirrors only
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as fh:  # noqa: S310
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)


def _parse_payload(raw: bytes, src: Source):
    """Bytes → a Python payload, honouring the source's container (zip member / jsonl / json)."""
    if src.member:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = [n for n in zf.namelist() if n.endswith(src.member) or n == src.member]
            if not names:
                raise ValueError(f"zip has no member ending in {src.member!r} (has: {zf.namelist()[:5]}…)")
            raw = zf.read(names[0])
    text = raw.decode("utf-8")
    if src.jsonl:
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    return json.loads(text)


def _roundtrip_ok(name: str, data_dir: pathlib.Path) -> int:
    """Load the just-written file through the REAL loader (full split), count Examples, and **validate
    gold ∈ corpus**. Sets JUNE_BENCH_DATA so the loader resolves the file we wrote. Returns the count.

    The gold check catches the id-space class of normalizer bug: if a retrieval query's gold ids never
    appear among its ingested corpus doc-ids, recall is 0 by construction — so a normalizer that emits
    gold in a different id space than the docs must FAIL the fetch loudly, not cache silent recall-0 data.
    """
    from june_bench.datasets import registry

    prev = os.environ.get("JUNE_BENCH_DATA")
    os.environ["JUNE_BENCH_DATA"] = str(data_dir)
    try:
        exs = list(registry.get(name).load("full"))
    finally:
        if prev is None:
            os.environ.pop("JUNE_BENCH_DATA", None)
        else:
            os.environ["JUNE_BENCH_DATA"] = prev

    scored = [e for e in exs if e.meta.get("gold_ids")]
    if scored:                                   # a retrieval set (QA-only sets carry no gold_ids)
        with_gold = sum(
            1 for e in scored
            if set(e.meta.get("gold_ids", [])) & {i for i, _t in (e.meta.get("corpus_docs") or [])})
        frac = with_gold / len(scored)
        if frac < 0.5:                           # gold and docs are in different id spaces → recall ≈ 0
            raise ValueError(
                f"only {with_gold}/{len(scored)} queries have their gold in the corpus ({frac:.0%}) — "
                f"gold ids and doc ids are in different id spaces, so recall would be ~0. "
                f"The normalizer for {name!r} is emitting mismatched ids.")
    return len(exs)


def fetch_one(src: Source, data_dir: pathlib.Path, *, raw_from: str | None = None,
              force: bool = False) -> tuple[str, str]:
    """Returns (status, message). status ∈ {skipped, ok, failed}. Never raises — fail-soft per dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / src.full_file
    if out.exists() and not force:
        return "skipped", f"{src.name}: {out.name} already present (use --force to refetch)"

    # 1) obtain raw bytes — a local --from file, or the first reachable mirror.
    raw: bytes | None = None
    err = ""
    if raw_from:
        try:
            raw = pathlib.Path(raw_from).read_bytes()
        except OSError as exc:
            return "failed", f"{src.name}: cannot read --from {raw_from!r}: {exc}"
    else:
        for url in src.urls:
            tmp_path: pathlib.Path | None = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, dir=str(data_dir)) as tmp:
                    tmp_path = pathlib.Path(tmp.name)
                _download(url, tmp_path)
                raw = tmp_path.read_bytes()
                break
            except Exception as exc:  # noqa: BLE001 — try the next mirror, report at the end
                err = f"{type(exc).__name__}: {exc}"
                continue
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)   # M9: never leave a temp file behind on any path
    if raw is None:
        return "failed", (f"{src.name}: could not download ({err or 'no mirror'}).\n"
                          f"  → {src.manual}")

    # 2) parse → normalize → write ATOMICALLY → validate. A bad transform must not leave a file behind.
    tmp_out = out.with_name(out.name + ".tmp")
    try:
        payload = _parse_payload(raw, src)
        normalized = src.normalize(payload)
        tmp_out.write_text(json.dumps(normalized), encoding="utf-8")
        tmp_out.replace(out)               # M10: atomic rename — a crash mid-write never leaves a truncated file
    except Exception as exc:  # noqa: BLE001
        tmp_out.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
        return "failed", f"{src.name}: parse/normalize failed ({type(exc).__name__}: {exc}).\n  → {src.manual}"
    try:
        n = _roundtrip_ok(src.name, data_dir)
    except Exception as exc:  # noqa: BLE001
        out.unlink(missing_ok=True)
        return "failed", f"{src.name}: wrote {out.name} but it did not load ({exc}); removed.\n  → {src.manual}"
    if n < 1:
        out.unlink(missing_ok=True)
        return "failed", f"{src.name}: produced 0 examples; removed {out.name}.\n  → {src.manual}"
    return "ok", f"{src.name}: {out.name} ✓ ({n} examples)"


def fetch(names: Sequence[str], data_dir: pathlib.Path, *, raw_from: str | None = None,
          force: bool = False) -> list[tuple[str, str, str]]:
    """Fetch each named source. Returns [(name, status, message)]. Fail-soft per dataset."""
    results = []
    for name in names:
        src = SOURCES.get(name)
        if src is None:
            results.append((name, "failed", f"{name}: no fetch source (known: {sorted(SOURCES)})"))
            continue
        status, msg = fetch_one(src, data_dir, raw_from=raw_from, force=force)
        results.append((name, status, msg))
    return results


__all__ = ["fetch", "fetch_one", "default_data_dir"]
