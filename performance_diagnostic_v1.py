"""
AI Trader - Performance Diagnostic V1.0

Research-only performance profiler for production app/main.py V2.14.
It does not modify production source and forces autonomous execution OFF.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import runpy
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parent
PRODUCTION_MAIN = PROJECT_ROOT / "app" / "main.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "performance_diagnostic_v1_results.json"
DEFAULT_PROFILE = PROJECT_ROOT / "performance_diagnostic_v1.prof"


class DiagnosticSafetyError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_safety() -> dict:
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


def static_map() -> dict:
    if not PRODUCTION_MAIN.exists():
        raise FileNotFoundError(f"Production main.py not found: {PRODUCTION_MAIN}")

    source = PRODUCTION_MAIN.read_text(encoding="utf-8")
    imports = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("from app.", "import app."))
    ]

    functions = []
    for number, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith(("def ", "async def ")):
            functions.append({"line": number, "definition": stripped})

    return {
        "production_main": str(PRODUCTION_MAIN),
        "sha256": sha256_file(PRODUCTION_MAIN),
        "imports": imports,
        "functions": functions,
        "known_stages": {
            "market_hours": "is_market_open",
            "prediction_tracker": "_run_prediction_tracker",
            "portfolio": "monitor_position",
            "watchlist_scanner": "scan_buy_opportunity",
            "prediction_logger": "_log_prediction_snapshot",
            "sizing": "_execution_sizing",
            "risk_gate": "risk_gate.evaluate",
            "executor": "executor.execute",
            "fundamentals": "calculate_fundamental_score",
            "fundamental_alert": "process_fundamental_alert",
        },
    }


def run_profile(force_scan: bool, profile_path: Path) -> dict:
    safety = check_safety()

    old_force = os.environ.get("AI_TRADER_FORCE_SCAN")
    if force_scan:
        os.environ["AI_TRADER_FORCE_SCAN"] = "1"
    else:
        os.environ.pop("AI_TRADER_FORCE_SCAN", None)

    started = time.perf_counter()
    profiler = cProfile.Profile()
    error = None
    return_code = 0

    try:
        profiler.enable()
        runpy.run_path(str(PRODUCTION_MAIN), run_name="__main__")
        profiler.disable()
    except SystemExit as exc:
        profiler.disable()
        return_code = int(exc.code or 0)
    except Exception as exc:
        profiler.disable()
        return_code = 1
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        elapsed = time.perf_counter() - started
        if old_force is None:
            os.environ.pop("AI_TRADER_FORCE_SCAN", None)
        else:
            os.environ["AI_TRADER_FORCE_SCAN"] = old_force

    profiler.dump_stats(str(profile_path))

    stats = pstats.Stats(str(profile_path))
    stats.sort_stats("cumulative")

    rows = []
    for func, values in stats.stats.items():
        primitive_calls, calls, total, cumulative, callers = values
        filename, line, function = func
        rows.append({
            "filename": str(filename),
            "line": int(line),
            "function": str(function),
            "primitive_calls": int(primitive_calls),
            "calls": int(calls),
            "tottime_seconds": round(float(total), 6),
            "cumulative_seconds": round(float(cumulative), 6),
        })

    rows.sort(key=lambda item: item["cumulative_seconds"], reverse=True)

    return {
        "status": "OK" if return_code == 0 else "ERROR",
        "return_code": return_code,
        "elapsed_seconds": round(elapsed, 6),
        "safety": safety,
        "force_scan": force_scan,
        "profile_file": str(profile_path),
        "top_functions": rows[:200],
        "error": error,
    }


def print_report(result: dict) -> None:
    print()
    print("=" * 72)
    print(f"AI TRADER - PERFORMANCE DIAGNOSTIC V{VERSION}")
    print("=" * 72)
    print("Research-only | production source unchanged | orders blocked")
    print("Autonomous execution: DISABLED")
    print()

    static = result["static_production_map"]
    print("Production main:")
    print(f"  {static['production_main']}")
    print(f"  SHA256: {static['sha256']}")

    if result.get("run"):
        run = result["run"]
        print()
        print("REAL PRODUCTION EXECUTION PROFILE")
        print("-" * 72)
        print(f"Status:       {run['status']}")
        print(f"Elapsed:      {run['elapsed_seconds']:.2f} s")
        print(f"Return code:  {run['return_code']}")
        print(f"Profile:      {run['profile_file']}")
        print()
        print("TOP FUNCTIONS BY CUMULATIVE TIME")
        print("-" * 72)

        for row in run["top_functions"][:30]:
            print(
                f"{row['cumulative_seconds']:9.2f}s | "
                f"{row['calls']:7d} calls | "
                f"{row['function']} | "
                f"{row['filename']}:{row['line']}"
            )

        if run["error"]:
            print()
            print("ERROR:")
            print(run["error"]["message"])

    print()
    print("KNOWN PRODUCTION STAGES")
    print("-" * 72)
    for name, target in static["known_stages"].items():
        print(f"{name:24s} -> {target}")

    print()
    print(f"JSON: {result['result_file']}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile AI Trader production main.py without modifying it."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the real production main.py under cProfile.",
    )
    parser.add_argument(
        "--force-scan",
        action="store_true",
        help="Bypass market-hours guard for this process only. Orders remain blocked.",
    )
    parser.add_argument(
        "--profile-file",
        default=str(DEFAULT_PROFILE),
        help="cProfile output file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON output file.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    profile_file = Path(args.profile_file)

    result = {
        "diagnostic": "AI Trader Performance Diagnostic",
        "version": VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changes": False,
        "orders_allowed": False,
        "result_file": str(output),
        "static_production_map": {},
        "run": None,
    }

    try:
        result["static_production_map"] = static_map()

        if args.run:
            result["run"] = run_profile(
                force_scan=args.force_scan,
                profile_path=profile_file,
            )
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
