#!/usr/bin/env python3
"""
Performance Diagnostic V2.3.3
Real-flow SPY cache lineage diagnostic.

Research-only:
- Does NOT modify production source files.
- Does NOT submit orders.
- Does NOT call Alpaca trading/order endpoints.
- Runs the production Watchlist Scanner flow, but limits the diagnostic
  instrumentation to Market Data get_history/get_daily_history calls.
- Captures cache lineage for SPY requests:
    request_key, cache_dir, cache_path, TTL, cache existence/age/freshness,
    caller, elapsed time, and whether production cache returned data.
- Does not replace _request_key/_cache_path/_read_cache/_write_cache.
- Does not replace yfinance.download.
- Uses the production implementations as-is.

Output:
    performance_diagnostic_v2_3_3_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT = PROJECT_ROOT / "performance_diagnostic_v2_3_3_results.json"

VERSION = "2.3.3"
PRODUCTION_SHA256 = "24d2ccc22624d831f217ec9cfecaed4fb86a7245ca335d0e917f654a7f522eae"

# Force safety regardless of the caller's shell environment.
os.environ["AI_TRADER_AUTONOMOUS_EXECUTION"] = "0"
os.environ["AI_TRADER_LIVE_TRADING"] = "0"

TARGET_TICKER = "SPY"


def _source_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return "<unreprable>"


def _caller_name() -> str:
    """
    Identify the first caller outside this diagnostic and app.data.market_data.
    """
    try:
        stack = inspect.stack()
        for frame_info in stack[2:]:
            filename = str(frame_info.filename)
            module = frame_info.frame.f_globals.get("__name__", "")
            if filename == __file__:
                continue
            if module == "app.data.market_data":
                continue
            return f"{module}:{frame_info.function}"
    except Exception:
        pass
    return "unknown"


def _normalize_for_key(value: Any) -> Any:
    return value


def _exact_production_request_key(
    ticker: str,
    period: str | None,
    interval: str | None,
    start: Any,
    end: Any,
    auto_adjust: bool,
    actions: bool,
    group_by: str,
) -> str:
    """
    Exact shadow implementation of production market_data._request_key().
    This function does NOT replace or monkey-patch production code.
    """
    payload = {
        "ticker": str(ticker).strip().upper(),
        "period": period,
        "interval": interval,
        "start": start,
        "end": end,
        "auto_adjust": bool(auto_adjust),
        "actions": bool(actions),
        "group_by": group_by,
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cache_state(
    md,
    *,
    ticker: str,
    period: str | None,
    interval: str | None,
    start: Any,
    end: Any,
    auto_adjust: bool,
    actions: bool,
    group_by: str,
    ttl_seconds: int | None,
) -> Dict[str, Any]:
    """
    Inspect production cache path without replacing production helpers.
    """
    result: Dict[str, Any] = {}

    try:
        request_key = _exact_production_request_key(
            ticker, period, interval, start, end,
            auto_adjust, actions, group_by,
        )
        result["shadow_request_key"] = request_key
    except Exception as exc:
        result["shadow_request_key_error"] = str(exc)
        request_key = None

    try:
        # Production cache directory resolution.
        cache_dir = md.cache_directory(None)
        result["cache_dir"] = str(cache_dir)
    except Exception as exc:
        result["cache_dir_error"] = str(exc)
        return result

    try:
        if request_key is not None:
            cache_path = md._cache_path(cache_dir, request_key)
            result["cache_path"] = str(cache_path)
            result["cache_exists"] = bool(cache_path.exists())
            if cache_path.exists():
                stat = cache_path.stat()
                age = max(0.0, time.time() - stat.st_mtime)
                result["cache_age_seconds"] = age
                result["cache_size_bytes"] = stat.st_size

                # Use the same TTL resolver as production, but do not replace it.
                try:
                    ttl = md._ttl_seconds(interval, ttl_seconds)
                    result["ttl_seconds"] = ttl
                    result["cache_fresh"] = age <= ttl
                except Exception as exc:
                    result["ttl_error"] = str(exc)
    except Exception as exc:
        result["cache_inspection_error"] = str(exc)

    return result


def _extract_get_history_call(args, kwargs) -> Dict[str, Any]:
    """
    Capture semantic arguments using the public get_history signature.
    """
    fields = (
        "ticker", "period", "interval", "start", "end",
        "auto_adjust", "actions", "group_by", "ttl_seconds",
        "force_refresh", "cache_dir",
    )
    data: Dict[str, Any] = {}

    try:
        sig = inspect.signature(_ORIGINAL_GET_HISTORY)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        for field in fields:
            if field in bound.arguments:
                data[field] = bound.arguments[field]
    except Exception:
        # Fallback for unusual wrappers; do not fail the production call.
        names = list(fields)
        for i, value in enumerate(args[: len(names)]):
            data[names[i]] = value
        data.update(kwargs)

    return data


def _extract_get_daily_history_call(args, kwargs) -> Dict[str, Any]:
    fields = (
        "ticker", "period", "interval", "start", "end",
        "auto_adjust", "actions", "group_by", "ttl_seconds",
        "force_refresh", "cache_dir",
    )
    data: Dict[str, Any] = {}

    try:
        sig = inspect.signature(_ORIGINAL_GET_DAILY_HISTORY)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        for field in fields:
            if field in bound.arguments:
                data[field] = bound.arguments[field]
    except Exception:
        names = list(fields)
        for i, value in enumerate(args[: len(names)]):
            data[names[i]] = value
        data.update(kwargs)

    return data


def _probe_call(
    kind: str,
    original,
    extractor,
    args,
    kwargs,
    events: List[Dict[str, Any]],
):
    params = extractor(args, kwargs)
    ticker = str(params.get("ticker", "")).upper()

    if ticker != TARGET_TICKER:
        return original(*args, **kwargs)

    period = params.get("period")
    interval = params.get("interval")
    start = params.get("start")
    end = params.get("end")
    auto_adjust = bool(params.get("auto_adjust", True))
    actions = bool(params.get("actions", False))
    group_by = params.get("group_by", "column")
    ttl_seconds = params.get("ttl_seconds")
    force_refresh = bool(params.get("force_refresh", False))

    before = _cache_state(
        MD,
        ticker=ticker,
        period=period,
        interval=interval,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        actions=actions,
        group_by=group_by,
        ttl_seconds=ttl_seconds,
    )

    t0 = time.perf_counter()
    error = None
    result = None
    try:
        result = original(*args, **kwargs)
        return_value = result
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
        return_value = None
    finally:
        elapsed = time.perf_counter() - t0

        after = _cache_state(
            MD,
            ticker=ticker,
            period=period,
            interval=interval,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            actions=actions,
            group_by=group_by,
            ttl_seconds=ttl_seconds,
        )

        event: Dict[str, Any] = {
            "sequence": len(events) + 1,
            "kind": kind,
            "caller": _caller_name(),
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "start": _safe_repr(start),
            "end": _safe_repr(end),
            "auto_adjust": auto_adjust,
            "actions": actions,
            "group_by": group_by,
            "ttl_seconds_argument": ttl_seconds,
            "force_refresh": force_refresh,
            "elapsed_seconds": elapsed,
            "error": error,
            "rows": int(len(return_value)) if hasattr(return_value, "__len__") else None,
            "before": before,
            "after": after,
        }

        if before.get("shadow_request_key"):
            event["request_key"] = before["shadow_request_key"]

        # Helpful classification without altering production behavior.
        event["diagnostic_cache_classification"] = (
            "FRESH_BEFORE_CALL"
            if before.get("cache_exists") and before.get("cache_fresh")
            else "MISS_OR_STALE_BEFORE_CALL"
        )

        events.append(event)

    if error:
        # Preserve production behavior: re-raise the original exception.
        raise RuntimeError(
            f"Observed production {kind} failure for {ticker}: "
            f"{error['type']}: {error['message']}"
        )
    return return_value


def _install_safe_wrappers(events: List[Dict[str, Any]]):
    """
    Wrap only public market-data functions. We retain originals and call them
    directly; no internal cache helper or Yahoo function is replaced.
    """
    global _ORIGINAL_GET_HISTORY, _ORIGINAL_GET_DAILY_HISTORY

    _ORIGINAL_GET_HISTORY = MD.get_history
    _ORIGINAL_GET_DAILY_HISTORY = MD.get_daily_history

    def traced_get_history(*args, **kwargs):
        return _probe_call(
            "get_history",
            _ORIGINAL_GET_HISTORY,
            _extract_get_history_call,
            args,
            kwargs,
            events,
        )

    def traced_get_daily_history(*args, **kwargs):
        return _probe_call(
            "get_daily_history",
            _ORIGINAL_GET_DAILY_HISTORY,
            _extract_get_daily_history_call,
            args,
            kwargs,
            events,
        )

    MD.get_history = traced_get_history
    MD.get_daily_history = traced_get_daily_history


def _restore_safe_wrappers():
    MD.get_history = _ORIGINAL_GET_HISTORY
    MD.get_daily_history = _ORIGINAL_GET_DAILY_HISTORY


def _run_production_flow() -> Dict[str, Any]:
    """
    Import and run the production main flow.
    The main flow itself remains unmodified; only public market-data calls
    are observed.
    """
    import app.main as production_main

    # The production main is expected to own the normal market-hours guard.
    # No artificial market-data request is injected.
    return_value = production_main.main()
    return {
        "main_return_type": type(return_value).__name__,
        "main_return": _safe_repr(return_value),
    }


def _summarize(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    spy = events

    by_key: Dict[str, Dict[str, Any]] = {}
    for e in spy:
        key = e.get("request_key", "<missing>")
        bucket = by_key.setdefault(
            key,
            {
                "count": 0,
                "calls": [],
                "fresh_before_count": 0,
                "stale_or_missing_before_count": 0,
                "total_elapsed_seconds": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["calls"].append(e["sequence"])
        bucket["total_elapsed_seconds"] += float(e["elapsed_seconds"])
        if e["diagnostic_cache_classification"] == "FRESH_BEFORE_CALL":
            bucket["fresh_before_count"] += 1
        else:
            bucket["stale_or_missing_before_count"] += 1

    return {
        "spy_event_count": len(spy),
        "unique_request_keys": len(by_key),
        "request_key_groups": by_key,
        "total_spy_elapsed_seconds": sum(
            float(e["elapsed_seconds"]) for e in spy
        ),
        "fresh_before_call_count": sum(
            e["diagnostic_cache_classification"] == "FRESH_BEFORE_CALL"
            for e in spy
        ),
        "miss_or_stale_before_call_count": sum(
            e["diagnostic_cache_classification"] == "MISS_OR_STALE_BEFORE_CALL"
            for e in spy
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(OUTPUT),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    global MD
    import app.data.market_data as MD

    events: List[Dict[str, Any]] = []

    metadata = {
        "version": VERSION,
        "research_only": True,
        "production_changes": False,
        "orders_allowed": False,
        "target_ticker": TARGET_TICKER,
        "production_sha256_expected": PRODUCTION_SHA256,
        "production_sha256_actual": _source_sha256(
            PROJECT_ROOT / "app" / "data" / "market_data.py"
        ),
        "safety_env": {
            "AI_TRADER_AUTONOMOUS_EXECUTION":
                os.getenv("AI_TRADER_AUTONOMOUS_EXECUTION"),
            "AI_TRADER_LIVE_TRADING":
                os.getenv("AI_TRADER_LIVE_TRADING"),
        },
        "scope": (
            "Real production main flow; observe only SPY public "
            "market-data calls. No replacement of internal cache helpers "
            "or yfinance.download."
        ),
    }

    t0 = time.perf_counter()
    flow = None
    status = "OK"
    error = None

    _install_safe_wrappers(events)
    try:
        flow = _run_production_flow()
    except Exception as exc:
        status = "ERROR"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }
    finally:
        _restore_safe_wrappers()

    elapsed = time.perf_counter() - t0

    result = {
        "metadata": metadata,
        "status": status,
        "elapsed_seconds": elapsed,
        "error": error,
        "flow": flow,
        "summary": _summarize(events),
        "spy_events": events,
    }

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8",
    )

    print("=== PERFORMANCE DIAGNOSTIC V2.3.3 ===")
    print("Real-flow SPY cache lineage | research-only | no orders")
    print(f"Status: {status}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"SPY observed calls: {len(events)}")
    print(f"Unique request keys: {result['summary']['unique_request_keys']}")
    print(
        "Fresh-cache-before-call: "
        f"{result['summary']['fresh_before_call_count']}"
    )
    print(
        "Miss/stale-before-call: "
        f"{result['summary']['miss_or_stale_before_call_count']}"
    )
    print(f"Output: {output_path}")

    if error:
        print(
            f"ERROR: {error['type']}: {error['message']}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
