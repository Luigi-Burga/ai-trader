"""AI Trader Performance Diagnostic V2.3.1 — Safe Cache Lineage Diagnostic.

Research-only. Production source is never edited. Trading/order endpoints are not used.
V2.3.1 fixes V2.3 instrumentation interference by:
1) instrumenting only the real Market Data public boundary first;
2) recording exceptions and tracebacks instead of allowing diagnostic wrappers to
   obscure the original failure;
3) avoiding nested replacement of _request_key/_cache_path/_read_cache/_write_cache;
4) optionally tracing the internal cache helpers only after the public call has
   returned, using a shadow calculation that never replaces production helpers;
5) preserving exact production request semantics.

Primary objective: prove request -> request_key -> cache_path -> cache read/write ->
Yahoo lineage for benchmark requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VERSION = "2.3.1"
RESEARCH_ONLY = True
PRODUCTION_CHANGES = False
ORDERS_ALLOWED = False

PROJECT_ROOT = Path("/app")
PRODUCTION_MAIN = PROJECT_ROOT / "app" / "main.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "performance_diagnostic_v2_3_1_results.json"

BENCHMARKS = {
    "SPY","QQQ","GDX","GDXJ","IGV","SMH","XLK","IWM","DIA",
    "XLE","XLF","XLI","XLP","XLV",
}


class State:
    def __init__(self):
        self.calls = Counter()
        self.seconds = Counter()
        self.errors = Counter()
        self.requests = []
        self.cache_events = []
        self.yahoo = []
        self.public_failures = []
        self.asset = None
        self.stage = None

    @staticmethod
    def is_benchmark(ticker):
        return str(ticker or "").strip().upper() in BENCHMARKS


S = State()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def caller():
    return {"module": None, "function": None, "line": None}


def request_payload(args, kwargs):
    def pick(name, pos, default=None):
        if name in kwargs:
            return kwargs[name]
        return args[pos] if len(args) > pos else default

    return {
        "ticker": str(pick("ticker", 0, "") or "").strip().upper(),
        "period": pick("period", 1, None),
        "interval": pick("interval", 2, "1d"),
        "start": pick("start", 3, None),
        "end": pick("end", 4, None),
        "auto_adjust": bool(pick("auto_adjust", 5, True)),
        "actions": bool(pick("actions", 6, False)),
        "group_by": pick("group_by", 7, "column"),
    }


def request_key(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def record_request(source, payload, key):
    S.requests.append({
        "source": source,
        "request_key": key,
        "request": payload,
        "is_benchmark": S.is_benchmark(payload["ticker"]),
        "asset_context": S.asset,
        "stage": S.stage,
        "caller": caller(),
        "timestamp": time.time(),
    })


def cache_snapshot():
    """Inspect cache metadata without replacing any production cache function."""
    try:
        import app.data.market_data as md
        directory = md.cache_directory()
        key_rows = []
        for p in directory.glob("market_*.pkl"):
            key = p.stem.removeprefix("market_")
            try:
                age = max(0.0, time.time() - p.stat().st_mtime)
                size = p.stat().st_size
            except OSError:
                age, size = None, None
            key_rows.append({
                "request_key": key,
                "cache_path": str(p),
                "exists": True,
                "age_seconds": age,
                "size_bytes": size,
            })
        return {"files": len(key_rows), "entries": key_rows}
    except Exception as exc:
        return {"files": 0, "entries": [], "error": repr(exc)}


def instrument_public_market_data():
    """Instrument only get_history/get_daily_history/yfinance.

    Deliberately do NOT replace _request_key/_cache_path/_read_cache/_write_cache.
    V2.3 showed that replacing those helpers interfered with production imports/
    internal references. We compute the same key formula in shadow mode instead.
    """
    import app.data.market_data as md

    for name in ("get_history", "get_daily_history"):
        fn = getattr(md, name, None)
        if fn is None or getattr(fn, "_v231", False):
            continue

        def make_public(original, function_name):
            def wrapper(*args, **kwargs):
                payload = request_payload(args, kwargs)
                key = request_key(payload)
                record_request(function_name, payload, key)

                old_asset, old_stage = S.asset, S.stage
                if payload["ticker"]:
                    S.asset = payload["ticker"]
                S.stage = "market_data." + function_name
                started = time.perf_counter()

                try:
                    result = original(*args, **kwargs)
                    elapsed = time.perf_counter() - started

                    # Shadow cache metadata: no production helper is replaced.
                    cache_path = None
                    cache_exists = None
                    cache_age = None
                    try:
                        cache_dir = md.cache_directory()
                        cache_path = str(md._cache_path(cache_dir, key))
                        p = Path(cache_path)
                        cache_exists = p.exists()
                        if cache_exists:
                            cache_age = max(0.0, time.time() - p.stat().st_mtime)
                    except Exception as shadow_exc:
                        S.cache_events.append({
                            "event": "shadow_cache_inspection_error",
                            "request_key": key,
                            "error": repr(shadow_exc),
                        })

                    S.cache_events.append({
                        "event": "public_return",
                        "source": function_name,
                        "request_key": key,
                        "request": payload,
                        "cache_path": cache_path,
                        "cache_exists": cache_exists,
                        "cache_age_seconds": cache_age,
                        "result_empty": bool(getattr(result, "empty", False)),
                        "result_rows": int(len(result)) if hasattr(result, "__len__") else None,
                        "elapsed_seconds": elapsed,
                        "asset_context": S.asset,
                        "stage": S.stage,
                    })
                    return result

                except Exception as exc:
                    elapsed = time.perf_counter() - started
                    entry = {
                        "event": "public_failure",
                        "source": function_name,
                        "request_key": key,
                        "request": payload,
                        "elapsed_seconds": elapsed,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "traceback": traceback.format_exc(),
                        "asset_context": S.asset,
                        "stage": S.stage,
                    }
                    S.public_failures.append(entry)
                    S.errors["market_data." + function_name] += 1
                    S.cache_events.append(entry)
                    raise

                finally:
                    S.calls["market_data." + function_name] += 1
                    S.seconds["market_data." + function_name] += time.perf_counter() - started
                    S.asset, S.stage = old_asset, old_stage

            wrapper._v231 = True
            return wrapper

        setattr(md, name, make_public(fn, name))

    original_yahoo = md.yf.download
    if not getattr(original_yahoo, "_v231", False):

        def yahoo_wrapper(*args, **kwargs):
            ticker = str(
                kwargs.get("tickers",
                kwargs.get("ticker", args[0] if args else "")) or ""
            ).strip().upper()

            request = {
                "ticker": ticker,
                "period": kwargs.get("period"),
                "interval": kwargs.get("interval"),
                "start": kwargs.get("start"),
                "end": kwargs.get("end"),
                "auto_adjust": kwargs.get("auto_adjust"),
                "actions": kwargs.get("actions"),
                "group_by": kwargs.get("group_by"),
                "threads": kwargs.get("threads"),
            }

            started = time.perf_counter()
            error = None
            try:
                return original_yahoo(*args, **kwargs)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                S.yahoo.append({
                    **request,
                    "is_benchmark": S.is_benchmark(ticker),
                    "asset_context": S.asset,
                    "stage": S.stage,
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": bool(error),
                    "error_message": error or "",
                })

        yahoo_wrapper._v231 = True
        md.yf.download = yahoo_wrapper


def instrument_orchestrator():
    import app.ai.asset_analysis_orchestrator_v1_5_1 as m
    fn = m.analyze_ticker
    if getattr(fn, "_v231", False):
        return

    def wrapper(*args, **kwargs):
        old = S.asset, S.stage
        S.asset = str(kwargs.get("ticker", args[0] if args else "")).upper()
        S.stage = "asset_analysis"
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        except Exception:
            S.errors["orchestrator.analyze_ticker"] += 1
            raise
        finally:
            S.calls["orchestrator.analyze_ticker"] += 1
            S.seconds["orchestrator.analyze_ticker"] += time.perf_counter() - started
            S.asset, S.stage = old

    wrapper._v231 = True
    m.analyze_ticker = wrapper


def instrument_scanner():
    import app.scanners.watchlist_scanner_v2_2_2 as m
    fn = m.scan_buy_opportunity
    if getattr(fn, "_v231", False):
        return

    def wrapper(*args, **kwargs):
        old = S.asset, S.stage
        S.asset = str(kwargs.get("ticker", args[0] if args else "")).upper()
        S.stage = "watchlist_scanner"
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        except Exception:
            S.errors["scanner.scan_buy_opportunity"] += 1
            raise
        finally:
            S.calls["scanner.scan_buy_opportunity"] += 1
            S.seconds["scanner.scan_buy_opportunity"] += time.perf_counter() - started
            S.asset, S.stage = old

    wrapper._v231 = True
    m.scan_buy_opportunity = wrapper


def instrument_main_and_tracker():
    import app.main as main_module
    import app.prediction_tracker as tracker

    for module, names, prefix in (
        (main_module, (
            "_run_prediction_tracker", "monitor_position", "scan_buy_opportunity",
            "_log_prediction_snapshot", "_execution_sizing",
            "calculate_fundamental_score", "process_fundamental_alert"
        ), "main."),
        (tracker, (
            "update_all_predictions", "_build_market_data_cache",
            "update_prediction_file", "evaluate_snapshot"
        ), "tracker."),
    ):
        for name in names:
            fn = getattr(module, name, None)
            if fn is None or getattr(fn, "_v231", False):
                continue

            def make(original, function_name, label):
                def wrapper(*args, **kwargs):
                    started = time.perf_counter()
                    try:
                        return original(*args, **kwargs)
                    except Exception:
                        S.errors[label + function_name] += 1
                        raise
                    finally:
                        S.calls[label + function_name] += 1
                        S.seconds[label + function_name] += time.perf_counter() - started
                wrapper._v231 = True
                return wrapper

            setattr(module, name, make(fn, name, prefix))


def build_lineage():
    groups = {}

    for r in S.requests:
        key = r["request_key"]
        g = groups.setdefault(key, {
            "request_key": key,
            "request": r["request"],
            "assets": Counter(),
            "sources": Counter(),
            "public_returns": [],
            "public_failures": [],
            "yahoo_calls": [],
            "cache_paths": [],
        })
        if r.get("asset_context"):
            g["assets"][r["asset_context"]] += 1
        g["sources"][r["source"]] += 1

    for e in S.cache_events:
        key = e.get("request_key")
        if key in groups:
            if e.get("event") == "public_return":
                groups[key]["public_returns"].append(e)
                if e.get("cache_path"):
                    groups[key]["cache_paths"].append(e["cache_path"])
            elif e.get("event") == "public_failure":
                groups[key]["public_failures"].append(e)

    for y in S.yahoo:
        if not y["is_benchmark"]:
            continue

        matches = []
        for key, g in groups.items():
            q = g["request"]
            if all(q.get(n) == y.get(n) for n in (
                "ticker", "period", "interval", "start", "end",
                "auto_adjust", "actions", "group_by"
            )):
                matches.append(key)

        if len(matches) == 1:
            groups[matches[0]]["yahoo_calls"].append(y)
        elif matches:
            y["lineage_match"] = "AMBIGUOUS"
        else:
            y["lineage_match"] = "NONE"

    result = []
    for key, g in groups.items():
        result.append({
            "request_key": key,
            "request": g["request"],
            "assets": dict(g["assets"]),
            "sources": dict(g["sources"]),
            "cache_paths": sorted(set(g["cache_paths"])),
            "public_returns": g["public_returns"],
            "public_failures": g["public_failures"],
            "yahoo_calls": g["yahoo_calls"],
            "counts": {
                "request_events": sum(g["sources"].values()),
                "public_returns": len(g["public_returns"]),
                "public_failures": len(g["public_failures"]),
                "yahoo_calls": len(g["yahoo_calls"]),
            },
        })

    return sorted(
        result,
        key=lambda x: (
            -x["counts"]["yahoo_calls"],
            x["request"].get("ticker", ""),
            x["request_key"],
        ),
    )


def run(output, force_scan=False):
    os.environ["AI_TRADER_AUTONOMOUS_EXECUTION"] = "0"
    os.environ["AI_TRADER_LIVE_TRADING"] = "0"
    if force_scan:
        os.environ["AI_TRADER_FORCE_SCAN"] = "1"

    before = cache_snapshot()

    import app.main  # noqa: F401

    instrument_public_market_data()
    instrument_orchestrator()
    instrument_scanner()
    instrument_main_and_tracker()

    started = time.perf_counter()
    status = "OK"
    code = 0
    error = None

    try:
        runpy.run_path(str(PRODUCTION_MAIN), run_name="__main__")
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
        if code:
            status = "ERROR"
            error = repr(exc)
    except Exception as exc:
        status = "ERROR"
        code = 1
        error = repr(exc)

    elapsed = time.perf_counter() - started
    after = cache_snapshot()

    lineage = build_lineage()
    benchmark_yahoo = [x for x in S.yahoo if x["is_benchmark"]]

    benchmark_summary = defaultdict(
        lambda: {
            "calls": 0, "seconds": 0.0, "errors": 0,
            "assets": Counter(), "signatures": Counter()
        }
    )

    for x in benchmark_yahoo:
        z = benchmark_summary[x["ticker"]]
        z["calls"] += 1
        z["seconds"] += x["elapsed_seconds"]
        z["errors"] += int(x["error"])
        if x.get("asset_context"):
            z["assets"][x["asset_context"]] += 1
        signature = json.dumps(
            {k: x.get(k) for k in (
                "ticker", "period", "interval", "start", "end",
                "auto_adjust", "actions", "group_by", "threads"
            )},
            sort_keys=True,
        )
        z["signatures"][signature] += 1

    exact_keys = Counter(
        x["request_key"] for x in S.requests if x["is_benchmark"]
    )

    report = {
        "version": VERSION,
        "method": "Safe Cache Key / Request Lineage Diagnostic",
        "research_only": True,
        "production_changes": False,
        "orders_allowed": False,
        "production_sha256": sha256(PRODUCTION_MAIN),
        "run": {
            "status": status,
            "return_code": code,
            "elapsed_seconds": elapsed,
            "safety": {
                "autonomous_execution": "0",
                "live_trading": "0",
                "orders_allowed": False,
            },
            "calls": dict(S.calls),
            "seconds": dict(S.seconds),
            "errors": dict(S.errors),
        },
        "market_data": {
            "cache_files_before": before["files"],
            "cache_files_after": after["files"],
            "cache_reads_observed": len(S.cache_events),
            "public_failures": len(S.public_failures),
            "yfinance_calls": len(S.yahoo),
            "yfinance_seconds": sum(x["elapsed_seconds"] for x in S.yahoo),
            "benchmark_yfinance_calls": len(benchmark_yahoo),
            "benchmark_yfinance_seconds": sum(
                x["elapsed_seconds"] for x in benchmark_yahoo
            ),
        },
        "duplicate_exact_request_keys": {
            k: v for k, v in exact_keys.items() if v > 1
        },
        "benchmark_summary": {
            k: {
                **v,
                "assets": dict(v["assets"]),
                "signatures": dict(v["signatures"]),
            }
            for k, v in benchmark_summary.items()
        },
        "benchmark_request_lineage": [
            x for x in lineage if x["request"]["ticker"] in BENCHMARKS
        ],
        "all_request_lineage": lineage,
        "all_requests": S.requests,
        "all_cache_events": S.cache_events,
        "all_public_failures": S.public_failures,
        "all_yahoo_calls": S.yahoo,
        "cache_snapshot_before": before,
        "cache_snapshot_after": after,
        "error": error,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("=" * 78)
    print("AI TRADER - PERFORMANCE DIAGNOSTIC V2.3.1")
    print("Safe Cache Key / Request Lineage Diagnostic")
    print("Research-only | production source unchanged | orders blocked")
    print("=" * 78)
    print(f"Status: {status} | Elapsed: {elapsed:.2f}s")
    print(f"Production SHA256: {report['production_sha256']}")
    print(f"Public failures: {len(S.public_failures)}")
    print(f"Yahoo calls: {len(S.yahoo)} / {sum(x['elapsed_seconds'] for x in S.yahoo):.2f}s")
    print("BENCHMARKS:")
    for ticker, item in sorted(
        benchmark_summary.items(),
        key=lambda kv: kv[1]["seconds"],
        reverse=True,
    ):
        print(
            f"  {ticker}: {item['calls']} calls / "
            f"{item['seconds']:.2f}s / assets={len(item['assets'])}"
        )
    print("DUPLICATE EXACT REQUEST KEYS:",
          sum(v > 1 for v in exact_keys.values()))
    print(f"JSON: {output}")
    print("=" * 78)

    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--force-scan", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    if not args.run:
        parser.error("Use --run")
    return run(Path(args.output), args.force_scan)


if __name__ == "__main__":
    raise SystemExit(main())
