#!/usr/bin/env python3
"""
Performance Diagnostic V2.3.2
Minimal single-request cache behavior probe.

Research-only. Does NOT modify production source, does NOT place orders,
and does NOT execute the production scanner.

Purpose:
    Make two identical SPY requests through production Market Data V2 and
    determine, with minimal instrumentation, whether request #2 reuses the
    persistent cache.

Default request:
    SPY / 5y / 1d / auto_adjust=True / actions=False / group_by=column

The probe uses a temporary cache directory by default, so it is isolated from
the production cache. Pass --cache-dir explicitly only when you intentionally
want to inspect the real cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from app.data import market_data as md


VERSION = "2.3.2"
PRODUCTION_SOURCE = Path("/app/app/data/market_data.py")
DEFAULT_TICKER = "SPY"
DEFAULT_PERIOD = "5y"
DEFAULT_INTERVAL = "1d"
DEFAULT_AUTO_ADJUST = True
DEFAULT_ACTIONS = False
DEFAULT_GROUP_BY = "column"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _request_key(
    ticker: str,
    period: str,
    interval: str,
    auto_adjust: bool,
    actions: bool,
    group_by: str,
) -> str:
    payload = {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "start": None,
        "end": None,
        "auto_adjust": bool(auto_adjust),
        "actions": bool(actions),
        "group_by": group_by,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_probe(path: Path, ttl: int) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "age_seconds": None, "size_bytes": None}
    age = max(0.0, time.time() - path.stat().st_mtime)
    return {
        "exists": True,
        "age_seconds": age,
        "size_bytes": path.stat().st_size,
        "fresh_for_ttl": age <= ttl,
    }


def _run_one(
    ticker: str,
    period: str,
    interval: str,
    auto_adjust: bool,
    actions: bool,
    group_by: str,
    cache_dir: Path,
    ttl_seconds: int,
) -> dict[str, Any]:
    before_files = len(list(cache_dir.glob("market_*.pkl")))
    key = _request_key(
        ticker, period, interval, auto_adjust, actions, group_by
    )
    cache_path = cache_dir / f"market_{key}.pkl"
    before = _cache_probe(cache_path, ttl_seconds)

    started = time.perf_counter()
    error = None
    try:
        df = md.get_history(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            actions=actions,
            group_by=group_by,
            threads=False,
            ttl_seconds=ttl_seconds,
            cache_dir=cache_dir,
        )
        elapsed = time.perf_counter() - started
        rows = int(len(df)) if isinstance(df, pd.DataFrame) else 0
    except Exception as exc:
        elapsed = time.perf_counter() - started
        rows = 0
        error = f"{type(exc).__name__}: {exc}"

    after = _cache_probe(cache_path, ttl_seconds)
    after_files = len(list(cache_dir.glob("market_*.pkl")))

    return {
        "elapsed_seconds": elapsed,
        "rows": rows,
        "error": error,
        "cache_path": str(cache_path),
        "request_key": key,
        "cache_before": before,
        "cache_after": after,
        "cache_file_count_before": before_files,
        "cache_file_count_after": after_files,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not PRODUCTION_SOURCE.exists():
        raise RuntimeError(f"Production source not found: {PRODUCTION_SOURCE}")

    # Research safety: force execution mode off in this process.
    os.environ["AI_TRADER_AUTONOMOUS_EXECUTION"] = "0"
    os.environ["AI_TRADER_LIVE_TRADING"] = "0"

    if args.cache_dir:
        cache_dir = Path(args.cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_isolated = False
    else:
        # Isolated cache makes the experiment deterministic and prevents any
        # mutation of production cache files.
        import tempfile
        cache_dir = Path(tempfile.mkdtemp(prefix="ai_trader_v232_"))
        cache_isolated = True

    ttl = int(args.ttl_seconds)

    r1 = _run_one(
        args.ticker, args.period, args.interval, args.auto_adjust,
        args.actions, args.group_by, cache_dir, ttl
    )
    r2 = _run_one(
        args.ticker, args.period, args.interval, args.auto_adjust,
        args.actions, args.group_by, cache_dir, ttl
    )

    expected_key = _request_key(
        args.ticker, args.period, args.interval,
        args.auto_adjust, args.actions, args.group_by
    )

    result = {
        "version": VERSION,
        "method": "Minimal Two-Request Cache Probe",
        "research_only": True,
        "production_changes": False,
        "orders_allowed": False,
        "production_sha256": _sha256(PRODUCTION_SOURCE),
        "safety": {
            "autonomous_execution": os.environ["AI_TRADER_AUTONOMOUS_EXECUTION"],
            "live_trading": os.environ["AI_TRADER_LIVE_TRADING"],
            "orders_allowed": False,
        },
        "request": {
            "ticker": args.ticker,
            "period": args.period,
            "interval": args.interval,
            "auto_adjust": args.auto_adjust,
            "actions": args.actions,
            "group_by": args.group_by,
            "ttl_seconds": ttl,
        },
        "cache": {
            "directory": str(cache_dir),
            "isolated": cache_isolated,
        },
        "expected_request_key": expected_key,
        "request_1": r1,
        "request_2": r2,
        "verdict": {
            "same_request_key": r1["request_key"] == r2["request_key"] == expected_key,
            "request_1_created_cache": (
                not r1["cache_before"]["exists"]
                and r1["cache_after"]["exists"]
            ),
            "request_2_cache_was_present": r2["cache_before"]["exists"],
            "request_2_cache_was_fresh": r2["cache_before"].get("fresh_for_ttl"),
            "request_2_fast_relative_to_request_1": (
                r2["elapsed_seconds"] < r1["elapsed_seconds"]
            ),
            "request_2_error": r2["error"],
        },
    }

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )

    print("=== PERFORMANCE DIAGNOSTIC V2.3.2 ===")
    print("Minimal two-request cache probe | research-only | no orders")
    print(f"Production SHA256: {result['production_sha256']}")
    print(f"Cache directory: {cache_dir}")
    print(f"Request key: {expected_key}")
    print()
    print(
        f"REQUEST 1: {r1['elapsed_seconds']:.3f}s | "
        f"rows={r1['rows']} | error={r1['error']}"
    )
    print(
        f"REQUEST 2: {r2['elapsed_seconds']:.3f}s | "
        f"rows={r2['rows']} | error={r2['error']}"
    )
    print()
    print("CACHE AFTER REQUEST 1:", r1["cache_after"])
    print("CACHE BEFORE REQUEST 2:", r2["cache_before"])
    print("CACHE AFTER REQUEST 2:", r2["cache_after"])
    print()
    if r2["cache_before"].get("fresh_for_ttl") and not r2["error"]:
        print("VERDICT: REQUEST 2 ENTERED WITH A FRESH CACHE ENTRY.")
        print("The production get_history cache path is reusable for identical requests.")
    elif r2["error"]:
        print("VERDICT: REQUEST 2 FAILED; cache reuse could not be evaluated.")
    else:
        print("VERDICT: REQUEST 2 DID NOT ENTER WITH A FRESH CACHE ENTRY.")

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default=DEFAULT_TICKER)
    p.add_argument("--period", default=DEFAULT_PERIOD)
    p.add_argument("--interval", default=DEFAULT_INTERVAL)
    p.add_argument("--auto-adjust", action=argparse.BooleanOptionalAction,
                   default=DEFAULT_AUTO_ADJUST)
    p.add_argument("--actions", action=argparse.BooleanOptionalAction,
                   default=DEFAULT_ACTIONS)
    p.add_argument("--group-by", default=DEFAULT_GROUP_BY)
    p.add_argument("--ttl-seconds", type=int, default=86400)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output", default="/app/performance_diagnostic_v2_3_2_results.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
