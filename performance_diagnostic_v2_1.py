"""
AI Trader - Performance Diagnostic V2.1

Research-only diagnostic focused on the Watchlist Scanner bottleneck.

V2.1 answers:
1. Which tickers are slowest?
2. How much time is spent in raw vs adjusted 5y history?
3. How many Market Data cache reads/hits/misses occur per ticker?
4. How many actual yfinance.download calls occur per ticker?
5. How much time is spent in CCE/orchestrator and scanner?
6. Are duplicate semantic requests occurring for the same ticker?
7. What is the end-to-end cost per watchlist ticker?

No production source is edited.
No Alpaca order endpoint is imported or called.
Autonomous execution is forcibly disabled.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import json
import os
import re
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

VERSION = "2.1"
PROJECT_ROOT = Path(__file__).resolve().parent
PRODUCTION_MAIN = PROJECT_ROOT / "app" / "main.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "performance_diagnostic_v2_1_results.json"


class DiagnosticSafetyError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_safety() -> dict[str, Any]:
    value = os.getenv("AI_TRADER_AUTONOMOUS_EXECUTION", "")
    if value.strip().lower() in {"1", "true", "yes", "on"}:
        raise DiagnosticSafetyError(
            "AI_TRADER_AUTONOMOUS_EXECUTION is enabled. Disable it before running V2.1."
        )
    os.environ["AI_TRADER_AUTONOMOUS_EXECUTION"] = "0"
    return {"autonomous_execution": "0", "orders_allowed": False, "live_trading": False}


def _ticker_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    value = kwargs.get("ticker")
    if value is None and args and isinstance(args[0], str):
        value = args[0]
    return str(value).upper() if value else None


def _request_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Capture semantic Market Data request dimensions without secrets."""
    ticker = _ticker_from_args(args, kwargs)
    period = kwargs.get("period")
    interval = kwargs.get("interval")
    auto_adjust = kwargs.get("auto_adjust")
    return {
        "ticker": ticker,
        "period": str(period) if period is not None else None,
        "interval": str(interval) if interval is not None else None,
        "auto_adjust": auto_adjust,
    }


class Metrics:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.calls = defaultdict(int)
        self.seconds = defaultdict(float)
        self.errors = defaultdict(int)
        self.tickers = defaultdict(lambda: {
            "calls": 0, "seconds": 0.0, "errors": 0,
            "stages": defaultdict(lambda: {"calls": 0, "seconds": 0.0, "errors": 0}),
            "market_requests": [],
            "yfinance_requests": [],
        })
        self.md = {
            "get_history_calls": 0, "get_history_seconds": 0.0,
            "get_daily_history_calls": 0, "get_daily_history_seconds": 0.0,
            "cache_read_calls": 0, "cache_hits": 0, "cache_misses": 0,
            "yfinance_download_calls": 0, "yfinance_download_seconds": 0.0,
            "yfinance_download_errors": 0,
        }
        self.request_counts = defaultdict(int)

    def add(self, label: str, elapsed: float, ticker: str | None = None, error: bool = False) -> None:
        self.calls[label] += 1
        self.seconds[label] += elapsed
        if error:
            self.errors[label] += 1
        if ticker:
            row = self.tickers[ticker]
            row["calls"] += 1
            row["seconds"] += elapsed
            if error:
                row["errors"] += 1
            stage = row["stages"][label]
            stage["calls"] += 1
            stage["seconds"] += elapsed
            if error:
                stage["errors"] += 1

    def add_market_request(self, ticker: str | None, signature: dict[str, Any], elapsed: float, error: bool) -> None:
        if ticker:
            self.tickers[ticker]["market_requests"].append({
                **signature, "elapsed_seconds": round(elapsed, 6), "error": error
            })

    def add_yfinance_request(self, ticker: str | None, signature: dict[str, Any], elapsed: float, error: bool) -> None:
        self.request_counts[json.dumps(signature, sort_keys=True, default=str)] += 1
        if ticker:
            self.tickers[ticker]["yfinance_requests"].append({
                **signature, "elapsed_seconds": round(elapsed, 6), "error": error
            })

    def report(self) -> dict[str, Any]:
        return {
            "calls": dict(self.calls),
            "seconds": {k: round(v, 6) for k, v in self.seconds.items()},
            "errors": dict(self.errors),
            "market_data": dict(self.md),
            "by_ticker": {
                t: {
                    "calls": v["calls"],
                    "seconds": round(v["seconds"], 6),
                    "errors": v["errors"],
                    "stages": {
                        s: {
                            "calls": x["calls"],
                            "seconds": round(x["seconds"], 6),
                            "errors": x["errors"],
                        } for s, x in v["stages"].items()
                    },
                    "market_requests": v["market_requests"],
                    "yfinance_requests": v["yfinance_requests"],
                } for t, v in self.tickers.items()
            },
            "duplicate_yfinance_signatures": {
                k: n for k, n in self.request_counts.items() if n > 1
            },
        }


def wrap_callable(owner: Any, attr: str, label: str, metrics: Metrics, ticker_aware: bool = False) -> bool:
    if owner is None or not hasattr(owner, attr):
        return False
    original = getattr(owner, attr)
    if not callable(original) or getattr(original, "__perf_diag_v21_wrapped__", False):
        return False

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        ticker = _ticker_from_args(args, kwargs) if ticker_aware else None
        started = time.perf_counter()
        failed = False
        try:
            return original(*args, **kwargs)
        except Exception:
            failed = True
            raise
        finally:
            elapsed = time.perf_counter() - started
            metrics.add(label, elapsed, ticker, failed)
            if label in {"market_data.get_history", "market_data.get_daily_history"}:
                if label.endswith("get_history"):
                    metrics.md["get_history_calls"] += 1
                    metrics.md["get_history_seconds"] += elapsed
                else:
                    metrics.md["get_daily_history_calls"] += 1
                    metrics.md["get_daily_history_seconds"] += elapsed
                metrics.add_market_request(ticker, _request_signature(args, kwargs), elapsed, failed)

    wrapped.__perf_diag_v21_wrapped__ = True
    setattr(owner, attr, wrapped)
    return True


def instrument(metrics: Metrics) -> dict[str, Any]:
    import app.main as prod_main
    installed = []

    for attr, label in [
        ("_run_prediction_tracker", "prediction_tracker"),
        ("monitor_position", "portfolio"),
        ("scan_buy_opportunity", "watchlist_scanner"),
        ("_log_prediction_snapshot", "prediction_logger"),
        ("_execution_sizing", "sizing"),
        ("calculate_fundamental_score", "fundamentals"),
        ("process_fundamental_alert", "fundamental_alert"),
    ]:
        if wrap_callable(prod_main, attr, label, metrics, True):
            installed.append(f"app.main.{attr}")

    md = importlib.import_module("app.data.market_data")
    for attr in ("get_history", "get_daily_history"):
        if wrap_callable(md, attr, f"market_data.{attr}", metrics, True):
            installed.append(f"app.data.market_data.{attr}")

    original_read = getattr(md, "_read_cache", None)
    if callable(original_read) and not getattr(original_read, "__perf_diag_v21_wrapped__", False):
        @functools.wraps(original_read)
        def read_cache(*args: Any, **kwargs: Any) -> Any:
            metrics.md["cache_read_calls"] += 1
            result = original_read(*args, **kwargs)
            if result is None:
                metrics.md["cache_misses"] += 1
            else:
                metrics.md["cache_hits"] += 1
            return result
        read_cache.__perf_diag_v21_wrapped__ = True
        md._read_cache = read_cache
        installed.append("app.data.market_data._read_cache")

    yf = getattr(md, "yf", None)
    if yf is not None and callable(getattr(yf, "download", None)):
        original_download = yf.download
        if not getattr(original_download, "__perf_diag_v21_wrapped__", False):
            @functools.wraps(original_download)
            def download(*args: Any, **kwargs: Any) -> Any:
                sig = _request_signature(args, kwargs)
                ticker = sig.get("ticker")
                started = time.perf_counter()
                failed = False
                try:
                    return original_download(*args, **kwargs)
                except Exception:
                    failed = True
                    raise
                finally:
                    elapsed = time.perf_counter() - started
                    metrics.md["yfinance_download_calls"] += 1
                    metrics.md["yfinance_download_seconds"] += elapsed
                    if failed:
                        metrics.md["yfinance_download_errors"] += 1
                    metrics.add_yfinance_request(ticker, sig, elapsed, failed)
            download.__perf_diag_v21_wrapped__ = True
            yf.download = download
            installed.append("app.data.market_data.yf.download")

    for module_name, attrs in [
        ("app.prediction_tracker", ["update_all_predictions", "_build_market_data_cache", "update_prediction_file", "evaluate_snapshot"]),
        ("app.scanners.watchlist_scanner_v2_2_2", ["scan_buy_opportunity"]),
        ("app.ai.asset_analysis_orchestrator_v1_5_1", ["analyze_ticker"]),
    ]:
        try:
            module = importlib.import_module(module_name)
            for attr in attrs:
                if wrap_callable(module, attr, f"{module_name}.{attr}", metrics, True):
                    installed.append(f"{module_name}.{attr}")
        except Exception as exc:
            installed.append(f"{module_name} instrumentation error: {type(exc).__name__}: {exc}")

    return {"count": len(installed), "instrumented": installed}


def cache_snapshot() -> dict[str, Any]:
    try:
        md = importlib.import_module("app.data.market_data")
        stats = md.cache_stats()
        return {"cache_dir": str(stats.get("cache_dir")), "files": int(stats.get("files", 0)), "size_bytes": int(stats.get("size_bytes", 0))}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def run(force_scan: bool) -> dict[str, Any]:
    safety = check_safety()
    old_force = os.environ.get("AI_TRADER_FORCE_SCAN")
    if force_scan:
        os.environ["AI_TRADER_FORCE_SCAN"] = "1"
    else:
        os.environ.pop("AI_TRADER_FORCE_SCAN", None)

    metrics = Metrics()
    started = time.perf_counter()
    before = cache_snapshot()
    status, return_code, error = "OK", 0, None

    try:
        instrumentation = instrument(metrics)
        import app.main as prod_main
        prod_main.main()
    except SystemExit as exc:
        return_code = int(exc.code or 0)
        if return_code:
            status = "ERROR"
    except Exception as exc:
        status, return_code = "ERROR", 1
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        instrumentation = locals().get("instrumentation", {})
    finally:
        if old_force is None:
            os.environ.pop("AI_TRADER_FORCE_SCAN", None)
        else:
            os.environ["AI_TRADER_FORCE_SCAN"] = old_force

    elapsed = time.perf_counter() - started
    return {
        "status": status,
        "return_code": return_code,
        "elapsed_seconds": round(elapsed, 6),
        "safety": safety,
        "force_scan": force_scan,
        "instrumentation": instrumentation,
        "cache_before": before,
        "cache_after": cache_snapshot(),
        "metrics": metrics.report(),
        "error": error,
    }


def static_map() -> dict[str, Any]:
    source = PRODUCTION_MAIN.read_text(encoding="utf-8")
    return {
        "production_main": str(PRODUCTION_MAIN),
        "sha256": sha256_file(PRODUCTION_MAIN),
        "watchlist_targets": {
            "main_stage": "app.main.scan_buy_opportunity",
            "scanner": "app.scanners.watchlist_scanner_v2_2_2.scan_buy_opportunity",
            "orchestrator": "app.ai.asset_analysis_orchestrator_v1_5_1.analyze_ticker",
            "market_data": "app.data.market_data.get_history",
            "yahoo": "app.data.market_data.yf.download",
            "cache": "app.data.market_data._read_cache",
        },
        "function_definitions": [
            {"line": n, "definition": line.strip()}
            for n, line in enumerate(source.splitlines(), 1)
            if line.strip().startswith(("def ", "async def "))
        ],
    }


def print_report(result: dict[str, Any]) -> None:
    print("=" * 82)
    print(f"AI TRADER - PERFORMANCE DIAGNOSTIC V{VERSION}")
    print("=" * 82)
    print("Research-only | production source unchanged | orders blocked")
    print("Autonomous execution: DISABLED")
    print()
    print("Production main:")
    print(f"  {result['static_production_map']['production_main']}")
    print(f"  SHA256: {result['static_production_map']['sha256']}")

    run_result = result.get("run")
    if not run_result:
        print("\nSTATIC MODE: no production execution was run. Use --run.")
        return

    print("\nREAL PRODUCTION EXECUTION PROFILE")
    print("-" * 82)
    print(f"Status:        {run_result['status']}")
    print(f"Elapsed:       {run_result['elapsed_seconds']:.2f} s")
    print(f"Return code:   {run_result['return_code']}")

    m = run_result["metrics"]
    print("\nWATCHLIST / MARKET DATA")
    print("-" * 82)
    print(f"watchlist scanner: {m['seconds'].get('watchlist_scanner', 0):.2f}s | {m['calls'].get('watchlist_scanner', 0)} calls")
    print(f"orchestrator:      {m['seconds'].get('app.ai.asset_analysis_orchestrator_v1_5_1.analyze_ticker', 0):.2f}s | {m['calls'].get('app.ai.asset_analysis_orchestrator_v1_5_1.analyze_ticker', 0)} calls")
    print(f"get_history:       {m['market_data']['get_history_calls']} calls | {m['market_data']['get_history_seconds']:.2f}s")
    print(f"cache reads:       {m['market_data']['cache_read_calls']} | hits={m['market_data']['cache_hits']} | misses={m['market_data']['cache_misses']}")
    print(f"yfinance.download: {m['market_data']['yfinance_download_calls']} calls | {m['market_data']['yfinance_download_seconds']:.2f}s")

    print("\nPER-TICKER WATCHLIST BREAKDOWN")
    print("-" * 82)
    rows = sorted(m["by_ticker"].items(), key=lambda kv: kv[1]["seconds"], reverse=True)
    for ticker, row in rows[:30]:
        stages = row["stages"]
        scanner = stages.get("watchlist_scanner", {}).get("seconds", 0.0)
        orch = stages.get("app.ai.asset_analysis_orchestrator_v1_5_1.analyze_ticker", {}).get("seconds", 0.0)
        yf_time = sum(x.get("elapsed_seconds", 0) for x in row["yfinance_requests"])
        print(f"{ticker:>6} | total={row['seconds']:7.2f}s | scanner={scanner:7.2f}s | CCE/orch={orch:7.2f}s | Yahoo={yf_time:7.2f}s | calls={row['calls']:3d}")

    print("\nDUPLICATE YFINANCE REQUEST SIGNATURES")
    print("-" * 82)
    dup = m["duplicate_yfinance_signatures"]
    if not dup:
        print("None")
    else:
        for signature, count in sorted(dup.items(), key=lambda kv: kv[1], reverse=True):
            print(f"{count:3d}x | {signature}")

    print("\nCACHE STATE")
    print("-" * 82)
    print(f"Before: {run_result['cache_before']}")
    print(f"After:  {run_result['cache_after']}")
    if run_result.get("error"):
        print("\nERROR:")
        print(run_result["error"]["message"])


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Trader Performance Diagnostic V2.1")
    parser.add_argument("--run", action="store_true", help="Run one real production scan")
    parser.add_argument("--force-scan", action="store_true", help="Set AI_TRADER_FORCE_SCAN=1")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    result = {
        "version": VERSION,
        "research_only": True,
        "production_changes": False,
        "orders_allowed": False,
        "static_production_map": static_map(),
    }
    if args.run:
        result["run"] = run(args.force_scan)

    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    result["result_file"] = str(output)
    print_report(result)


if __name__ == "__main__":
    main()
