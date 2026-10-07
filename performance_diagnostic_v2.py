"""
AI Trader - Performance Diagnostic V2.0

Research-only runtime diagnostic for the current production app/main.py.

Design goals:
- Measure the real production cycle without modifying production source.
- Avoid cProfile import-startup noise by importing app.main first and then
  calling main.main() directly.
- Time production stages and selected hot functions.
- Measure Market Data V2 cache hits/misses and actual yfinance downloads.
- Measure Prediction Tracker, Watchlist Scanner, CCE/asset analysis,
  fundamentals, and per-ticker costs when those functions are present.
- Capture cache state before/after the run.
- Never enable autonomous execution and never submit orders.

No production source is edited by this script.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import inspect
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


VERSION = "2.0"
PROJECT_ROOT = Path(__file__).resolve().parent
PRODUCTION_MAIN = PROJECT_ROOT / "app" / "main.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "performance_diagnostic_v2_results.json"


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
            "AI_TRADER_AUTONOMOUS_EXECUTION is enabled. "
            "Disable it before running the diagnostic."
        )

    os.environ["AI_TRADER_AUTONOMOUS_EXECUTION"] = "0"

    return {
        "autonomous_execution": "0",
        "orders_allowed": False,
        "live_trading": False,
    }


def cache_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "cache_dir": None,
        "files": 0,
        "size_bytes": 0,
    }

    try:
        md = importlib.import_module("app.data.market_data")
        stats = md.cache_stats()
        result.update(
            {
                "cache_dir": str(stats.get("cache_dir")),
                "files": int(stats.get("files", 0)),
                "size_bytes": int(stats.get("size_bytes", 0)),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


class RuntimeMetrics:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.calls: dict[str, int] = defaultdict(int)
        self.seconds: dict[str, float] = defaultdict(float)
        self.errors: dict[str, int] = defaultdict(int)
        self.by_ticker: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "calls": 0,
                "seconds": 0.0,
                "errors": 0,
            }
        )
        self.market_data = {
            "get_history_calls": 0,
            "get_history_seconds": 0.0,
            "get_history_errors": 0,
            "cache_read_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "yfinance_download_calls": 0,
            "yfinance_download_seconds": 0.0,
            "yfinance_download_errors": 0,
        }
        self.stage_order: list[str] = []

    def add(
        self,
        name: str,
        elapsed: float,
        *,
        ticker: str | None = None,
        error: bool = False,
    ) -> None:
        self.calls[name] += 1
        self.seconds[name] += elapsed
        if error:
            self.errors[name] += 1

        if ticker:
            row = self.by_ticker[ticker]
            row["calls"] += 1
            row["seconds"] += elapsed
            if error:
                row["errors"] += 1

    def mark_stage(self, name: str) -> None:
        if name not in self.stage_order:
            self.stage_order.append(name)

    def report(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": round(time.perf_counter() - self.started, 6),
            "calls": dict(self.calls),
            "seconds": {
                k: round(v, 6) for k, v in self.seconds.items()
            },
            "errors": dict(self.errors),
            "by_ticker": {
                ticker: {
                    "calls": int(row["calls"]),
                    "seconds": round(float(row["seconds"]), 6),
                    "errors": int(row["errors"]),
                }
                for ticker, row in sorted(self.by_ticker.items())
            },
            "market_data": {
                **self.market_data,
                "get_history_seconds": round(
                    self.market_data["get_history_seconds"], 6
                ),
                "yfinance_download_seconds": round(
                    self.market_data["yfinance_download_seconds"], 6
                ),
            },
            "stage_order": list(self.stage_order),
        }


def _ticker_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    value = kwargs.get("ticker")
    if value is None and args:
        first = args[0]
        if isinstance(first, str):
            value = first
    if value is None:
        return None
    return str(value).upper().replace(".", ".")


def wrap_callable(
    owner: Any,
    attr: str,
    label: str,
    metrics: RuntimeMetrics,
    *,
    stage: bool = False,
    ticker_aware: bool = False,
) -> bool:
    if owner is None or not hasattr(owner, attr):
        return False

    original = getattr(owner, attr)
    if not callable(original):
        return False
    if getattr(original, "__perf_diag_v2_wrapped__", False):
        return False

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if stage:
            metrics.mark_stage(label)

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
            metrics.add(
                label,
                elapsed,
                ticker=ticker,
                error=failed,
            )

    wrapped.__perf_diag_v2_wrapped__ = True
    setattr(owner, attr, wrapped)
    return True


def instrument_runtime(metrics: RuntimeMetrics) -> dict[str, Any]:
    """
    Import production main without executing main(), then wrap runtime
    callables. This keeps the diagnostic out of the production source and
    avoids profiling the Python import tree as the primary result.
    """
    import app.main as prod_main

    installed: list[str] = []

    # Top-level production stages. These names are based on the current
    # production main.py contract; missing names are recorded rather than
    # treated as a failure so the diagnostic remains version-tolerant.
    main_targets = [
        ("_run_prediction_tracker", "prediction_tracker", True),
        ("monitor_position", "portfolio", True),
        ("scan_buy_opportunity", "watchlist_scanner", True),
        ("_log_prediction_snapshot", "prediction_logger", True),
        ("_execution_sizing", "sizing", True),
        ("calculate_fundamental_score", "fundamentals", True),
        ("process_fundamental_alert", "fundamental_alert", True),
    ]

    for attr, label, is_stage in main_targets:
        if wrap_callable(
            prod_main,
            attr,
            label,
            metrics,
            stage=is_stage,
            ticker_aware=True,
        ):
            installed.append(f"app.main.{attr}")

    # Market Data V2.
    try:
        md = importlib.import_module("app.data.market_data")
        for attr, label in [
            ("get_history", "market_data.get_history"),
            ("get_daily_history", "market_data.get_daily_history"),
        ]:
            if wrap_callable(
                md,
                attr,
                label,
                metrics,
                ticker_aware=True,
            ):
                installed.append(f"app.data.market_data.{attr}")

        # Cache classifier: _read_cache returning None means miss/expired;
        # non-None means a real persistent cache hit.
        original_read = getattr(md, "_read_cache", None)
        if callable(original_read) and not getattr(
            original_read, "__perf_diag_v2_wrapped__", False
        ):
            @functools.wraps(original_read)
            def read_cache_wrapped(*args: Any, **kwargs: Any) -> Any:
                metrics.market_data["cache_read_calls"] += 1
                result = original_read(*args, **kwargs)
                if result is None:
                    metrics.market_data["cache_misses"] += 1
                else:
                    metrics.market_data["cache_hits"] += 1
                return result

            read_cache_wrapped.__perf_diag_v2_wrapped__ = True
            setattr(md, "_read_cache", read_cache_wrapped)
            installed.append("app.data.market_data._read_cache")

        # Actual Yahoo download boundary.
        yf = getattr(md, "yf", None)
        if yf is not None and callable(getattr(yf, "download", None)):
            original_download = yf.download
            if not getattr(
                original_download, "__perf_diag_v2_wrapped__", False
            ):
                @functools.wraps(original_download)
                def download_wrapped(*args: Any, **kwargs: Any) -> Any:
                    started = time.perf_counter()
                    failed = False
                    try:
                        return original_download(*args, **kwargs)
                    except Exception:
                        failed = True
                        raise
                    finally:
                        elapsed = time.perf_counter() - started
                        metrics.market_data[
                            "yfinance_download_calls"
                        ] += 1
                        metrics.market_data[
                            "yfinance_download_seconds"
                        ] += elapsed
                        if failed:
                            metrics.market_data[
                                "yfinance_download_errors"
                            ] += 1

                download_wrapped.__perf_diag_v2_wrapped__ = True
                yf.download = download_wrapped
                installed.append("app.data.market_data.yf.download")
    except Exception as exc:
        installed.append(
            f"market_data instrumentation error: {type(exc).__name__}: {exc}"
        )

    # Prediction Tracker internals.
    try:
        tracker = importlib.import_module("app.prediction_tracker")
        for attr, label in [
            ("update_all_predictions", "prediction_tracker.update_all_predictions"),
            ("_build_market_data_cache", "prediction_tracker._build_market_data_cache"),
            ("update_prediction_file", "prediction_tracker.update_prediction_file"),
            ("evaluate_snapshot", "prediction_tracker.evaluate_snapshot"),
        ]:
            if wrap_callable(
                tracker,
                attr,
                label,
                metrics,
                ticker_aware=(attr != "_build_market_data_cache"),
            ):
                installed.append(f"app.prediction_tracker.{attr}")
    except Exception as exc:
        installed.append(
            f"prediction_tracker instrumentation error: "
            f"{type(exc).__name__}: {exc}"
        )

    # Watchlist scanner + orchestrator.
    for module_name, targets in [
        (
            "app.scanners.watchlist_scanner_v2_2_2",
            [
                ("scan_buy_opportunity", "watchlist.scan_buy_opportunity"),
            ],
        ),
        (
            "app.ai.asset_analysis_orchestrator_v1_5_1",
            [
                ("analyze_ticker", "asset_analysis.analyze_ticker"),
            ],
        ),
    ]:
        try:
            module = importlib.import_module(module_name)
            for attr, label in targets:
                if wrap_callable(
                    module,
                    attr,
                    label,
                    metrics,
                    ticker_aware=True,
                ):
                    installed.append(f"{module_name}.{attr}")
        except Exception as exc:
            installed.append(
                f"{module_name} instrumentation error: "
                f"{type(exc).__name__}: {exc}"
            )

    # Fundamentals module-level calls, when available.
    for module_name in [
        "app.fundamentals.score_engine",
        "app.alerts.fundamental_alert_engine_v1",
    ]:
        try:
            module = importlib.import_module(module_name)
            for attr in [
                "calculate_fundamental_score",
                "process_fundamental_alert",
            ]:
                if wrap_callable(
                    module,
                    attr,
                    f"{module_name}.{attr}",
                    metrics,
                    ticker_aware=True,
                ):
                    installed.append(f"{module_name}.{attr}")
        except Exception:
            pass

    return {
        "production_main_module": "app.main",
        "instrumented": installed,
        "count": len(installed),
    }


def run_production(metrics: RuntimeMetrics, force_scan: bool) -> dict[str, Any]:
    safety = check_safety()

    old_force = os.environ.get("AI_TRADER_FORCE_SCAN")
    if force_scan:
        os.environ["AI_TRADER_FORCE_SCAN"] = "1"
    else:
        os.environ.pop("AI_TRADER_FORCE_SCAN", None)

    started = time.perf_counter()
    status = "OK"
    return_code = 0
    error: dict[str, Any] | None = None

    try:
        instrumentation = instrument_runtime(metrics)
        import app.main as prod_main
        prod_main.main()
    except SystemExit as exc:
        return_code = int(exc.code or 0)
        if return_code != 0:
            status = "ERROR"
    except Exception as exc:
        return_code = 1
        status = "ERROR"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        instrumentation = locals().get("instrumentation", {})
    finally:
        elapsed = time.perf_counter() - started
        if old_force is None:
            os.environ.pop("AI_TRADER_FORCE_SCAN", None)
        else:
            os.environ["AI_TRADER_FORCE_SCAN"] = old_force

    report = metrics.report()
    report["elapsed_seconds"] = round(elapsed, 6)
    report["status"] = status
    report["return_code"] = return_code
    report["safety"] = safety
    report["force_scan"] = force_scan
    report["instrumentation"] = instrumentation
    report["cache_before"] = cache_snapshot()
    # cache_after is captured by the caller after the run so it also works
    # when an exception occurs.

    if error:
        report["error"] = error

    return report


def static_map() -> dict[str, Any]:
    if not PRODUCTION_MAIN.exists():
        raise FileNotFoundError(
            f"Production main.py not found: {PRODUCTION_MAIN}"
        )

    source = PRODUCTION_MAIN.read_text(encoding="utf-8")
    functions = []
    for number, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith(("def ", "async def ")):
            functions.append(
                {"line": number, "definition": stripped}
            )

    return {
        "production_main": str(PRODUCTION_MAIN),
        "sha256": sha256_file(PRODUCTION_MAIN),
        "known_stage_targets": {
            "prediction_tracker": "_run_prediction_tracker",
            "portfolio": "monitor_position",
            "watchlist_scanner": "scan_buy_opportunity",
            "prediction_logger": "_log_prediction_snapshot",
            "sizing": "_execution_sizing",
            "fundamentals": "calculate_fundamental_score",
            "fundamental_alert": "process_fundamental_alert",
            "market_data": "app.data.market_data.get_history",
            "yahoo_download": "app.data.market_data.yf.download",
            "tracker_cache_builder": (
                "app.prediction_tracker._build_market_data_cache"
            ),
            "asset_analysis": (
                "app.ai.asset_analysis_orchestrator_v1_5_1.analyze_ticker"
            ),
        },
        "functions": functions,
    }


def print_report(result: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(f"AI TRADER - PERFORMANCE DIAGNOSTIC V{VERSION}")
    print("=" * 78)
    print("Research-only | production source unchanged | orders blocked")
    print("Autonomous execution: DISABLED")
    print()

    static = result["static_production_map"]
    print("Production main:")
    print(f"  {static['production_main']}")
    print(f"  SHA256: {static['sha256']}")

    run = result.get("run")
    if not run:
        print()
        print("STATIC MODE: no production execution was run.")
        print("Use --run to profile one real production execution.")
    else:
        print()
        print("REAL PRODUCTION EXECUTION PROFILE")
        print("-" * 78)
        print(f"Status:        {run['status']}")
        print(f"Elapsed:       {run['elapsed_seconds']:.2f} s")
        print(f"Return code:   {run['return_code']}")
        print()

        print("STAGE / HOT FUNCTION TIMINGS")
        print("-" * 78)
        rows = sorted(
            [
                (name, seconds, run["calls"].get(name, 0))
                for name, seconds in run["seconds"].items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        for name, seconds, calls in rows[:30]:
            print(f"{seconds:9.2f}s | {calls:6d} calls | {name}")

        md = run["market_data"]
        print()
        print("MARKET DATA")
        print("-" * 78)
        print(
            f"get_history:       {md['get_history_calls']:6d} calls | "
            f"{md['get_history_seconds']:9.2f}s"
        )
        print(
            f"cache reads:       {md['cache_read_calls']:6d} | "
            f"hits={md['cache_hits']:6d} | "
            f"misses={md['cache_misses']:6d}"
        )
        print(
            f"yfinance.download: {md['yfinance_download_calls']:6d} calls | "
            f"{md['yfinance_download_seconds']:9.2f}s"
        )

        print()
        print("TOP TICKERS BY INSTRUMENTED TIME")
        print("-" * 78)
        ticker_rows = sorted(
            run["by_ticker"].items(),
            key=lambda kv: kv[1]["seconds"],
            reverse=True,
        )
        for ticker, row in ticker_rows[:20]:
            print(
                f"{row['seconds']:9.2f}s | "
                f"{row['calls']:6d} calls | "
                f"{ticker}"
            )

        print()
        print("CACHE STATE")
        print("-" * 78)
        print(f"Before: {run['cache_before']}")
        print(f"After:  {result['cache_after']}")

        if run.get("error"):
            print()
            print("ERROR:")
            print(run["error"]["message"])

    print()
    print(f"JSON: {result['result_file']}")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure AI Trader production runtime stages without "
            "modifying production source."
        )
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute one real production cycle under runtime instrumentation.",
    )
    parser.add_argument(
        "--force-scan",
        action="store_true",
        help=(
            "Bypass market-hours guard for this process only. "
            "Orders remain blocked."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON output file.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    result: dict[str, Any] = {
        "diagnostic": "AI Trader Performance Diagnostic",
        "version": VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changes": False,
        "orders_allowed": False,
        "live_trading": False,
        "result_file": str(output),
        "static_production_map": {},
        "run": None,
        "cache_after": None,
    }

    try:
        result["static_production_map"] = static_map()

        if args.run:
            metrics = RuntimeMetrics()
            result["run"] = run_production(
                metrics=metrics,
                force_scan=args.force_scan,
            )
            result["cache_after"] = cache_snapshot()
            result["status"] = result["run"]["status"]
        else:
            print("STATIC MODE: no production execution was run.")
            print("Use --run to profile one real production execution.")
            result["status"] = "OK"

    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print_report(result)

    if result["status"] != "OK":
        print()
        print("DIAGNOSTIC FAILED")
        print(result["error"]["message"])
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
