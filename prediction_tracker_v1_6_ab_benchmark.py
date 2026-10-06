"""
AI Trader - Prediction Tracker V1.6 A/B Benchmark
==================================================

Research-only benchmark:
- Compares current production Prediction Tracker against V1.6.
- Does NOT modify production source.
- Does NOT modify production prediction files.
- Does NOT modify production Market Data cache.
- Uses isolated temporary copies of prediction files and market cache.
- Uses real Docker dependencies / real Market Data layer when executed there.
- No Alpaca trading endpoints, orders, or execution.

The benchmark deliberately starts both A and B from equivalent stale cache
copies (default age: 20 minutes) to reproduce the current scheduler condition:
production Market Data TTL = 15 minutes, while V1.6 tracker-specific TTL = 24h.

It also compares output semantics after removing only dynamic evaluation
timestamps.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PRODUCTION_MODULE = ROOT / "app" / "prediction_tracker.py"
EXPERIMENTAL_MODULE = ROOT / "prediction_tracker_v1_6.py"
DEFAULT_PREDICTIONS = ROOT / "data" / "predictions"
DEFAULT_CACHE = ROOT / "data" / "market_cache"


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise RuntimeError(f"Source directory not found: {src}")
    shutil.copytree(src, dst)


def age_files(root: Path, age_seconds: int) -> int:
    now = time.time()
    count = 0
    for path in root.rglob("*"):
        if path.is_file():
            ts = now - age_seconds
            os.utime(path, (ts, ts))
            count += 1
    return count


def canonical_record(record: Any) -> Any:
    if isinstance(record, dict):
        result = {}
        for key, value in record.items():
            if key in {
                "evaluated_at_utc",
            }:
                continue
            result[key] = canonical_record(value)
        return result
    if isinstance(record, list):
        return [canonical_record(value) for value in record]
    return record


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
    return records


def compare_prediction_dirs(a: Path, b: Path) -> dict[str, Any]:
    files_a = {p.name: p for p in a.glob("predictions_*.jsonl")}
    files_b = {p.name: p for p in b.glob("predictions_*.jsonl")}

    all_names = sorted(set(files_a) | set(files_b))
    mismatches = []
    total_a = 0
    total_b = 0

    for name in all_names:
        ra = load_jsonl_records(files_a[name]) if name in files_a else []
        rb = load_jsonl_records(files_b[name]) if name in files_b else []

        total_a += len(ra)
        total_b += len(rb)

        ca = [canonical_record(x) for x in ra]
        cb = [canonical_record(x) for x in rb]

        if ca != cb:
            mismatches.append(
                {
                    "file": name,
                    "records_a": len(ra),
                    "records_b": len(rb),
                }
            )

    return {
        "files_compared": len(all_names),
        "records_a": total_a,
        "records_b": total_b,
        "semantic_match": not mismatches,
        "mismatches": mismatches[:20],
    }


def run_tracker(module, prediction_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    summary = module.update_all_predictions(
        prediction_dir,
        include_today=False,
    )
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": round(elapsed, 3),
        "summary": summary,
    }


def cache_stats(root: Path) -> dict[str, Any]:
    files = list(root.glob("market_*.pkl"))
    return {
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument(
        "--stale-minutes",
        type=int,
        default=20,
        help="Age isolated cache files before each benchmark (default 20).",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep temporary benchmark directory for inspection.",
    )
    args = parser.parse_args()

    predictions = Path(args.predictions).resolve()
    cache = Path(args.cache).resolve()

    if not PRODUCTION_MODULE.exists():
        raise RuntimeError(f"Production module not found: {PRODUCTION_MODULE}")
    if not EXPERIMENTAL_MODULE.exists():
        raise RuntimeError(f"Experimental module not found: {EXPERIMENTAL_MODULE}")
    if not predictions.exists():
        raise RuntimeError(f"Prediction directory not found: {predictions}")
    if not cache.exists():
        raise RuntimeError(f"Market cache directory not found: {cache}")

    print("=" * 72)
    print("AI TRADER - PREDICTION TRACKER V1.6 A/B BENCHMARK")
    print("Research-only | production source unchanged | orders blocked")
    print("=" * 72)
    print(f"Production module : {PRODUCTION_MODULE}")
    print(f"Experimental      : {EXPERIMENTAL_MODULE}")
    print(f"Predictions       : {predictions}")
    print(f"Market cache      : {cache}")
    print(f"Stale cache age   : {args.stale_minutes} minutes")
    print()

    production = load_module(PRODUCTION_MODULE, "prediction_tracker_production")
    experimental = load_module(EXPERIMENTAL_MODULE, "prediction_tracker_v16")

    print(f"Production VERSION: {getattr(production, 'VERSION', 'UNKNOWN')}")
    print(f"V1.6 VERSION      : {getattr(experimental, 'VERSION', 'UNKNOWN')}")
    print()

    tmp_parent = Path(tempfile.mkdtemp(prefix="ai_trader_tracker_ab_"))
    pred_a = tmp_parent / "predictions_a"
    pred_b = tmp_parent / "predictions_b"
    cache_a = tmp_parent / "cache_a"
    cache_b = tmp_parent / "cache_b"

    try:
        copy_tree(predictions, pred_a)
        copy_tree(predictions, pred_b)
        copy_tree(cache, cache_a)
        copy_tree(cache, cache_b)

        aged_a = age_files(cache_a, args.stale_minutes * 60)
        aged_b = age_files(cache_b, args.stale_minutes * 60)

        print(f"Isolated cache A files aged: {aged_a}")
        print(f"Isolated cache B files aged: {aged_b}")
        print()

        # ------------------------------------------------------------------
        # A = current production
        # ------------------------------------------------------------------
        print(">>> A — CURRENT PRODUCTION")
        os.environ["AI_TRADER_MARKET_CACHE_DIR"] = str(cache_a)
        os.environ["AI_TRADER_MARKET_CACHE_TTL_SECONDS"] = "900"
        os.environ.pop("AI_TRADER_TRACKER_MARKET_CACHE_TTL_SECONDS", None)

        result_a = run_tracker(production, pred_a)
        print(f"A elapsed: {result_a['elapsed_seconds']:.3f}s")
        print(f"A summary: {result_a['summary']}")
        print(f"A cache  : {cache_stats(cache_a)}")
        print()

        # ------------------------------------------------------------------
        # B = V1.6
        # ------------------------------------------------------------------
        print(">>> B — PREDICTION TRACKER V1.6")
        os.environ["AI_TRADER_MARKET_CACHE_DIR"] = str(cache_b)
        os.environ["AI_TRADER_MARKET_CACHE_TTL_SECONDS"] = "900"
        os.environ["AI_TRADER_TRACKER_MARKET_CACHE_TTL_SECONDS"] = "86400"

        result_b = run_tracker(experimental, pred_b)
        print(f"B elapsed: {result_b['elapsed_seconds']:.3f}s")
        print(f"B summary: {result_b['summary']}")
        print(f"B cache  : {cache_stats(cache_b)}")
        print()

        a = result_a["elapsed_seconds"]
        b = result_b["elapsed_seconds"]
        improvement = ((a - b) / a * 100.0) if a > 0 else 0.0
        speedup = (a / b) if b > 0 else None

        comparison = compare_prediction_dirs(pred_a, pred_b)

        print("=" * 72)
        print("A/B RESULT")
        print("=" * 72)
        print(f"Production V1.5 time : {a:.3f}s")
        print(f"V1.6 time            : {b:.3f}s")
        print(f"Improvement          : {improvement:.2f}%")
        print(f"Speedup              : {speedup:.2f}x" if speedup else "Speedup              : N/A")
        print(f"Semantic output match: {comparison['semantic_match']}")
        print(f"Files compared       : {comparison['files_compared']}")
        print(f"Records A/B          : {comparison['records_a']} / {comparison['records_b']}")

        if comparison["mismatches"]:
            print("Mismatches (max 20):")
            for item in comparison["mismatches"]:
                print(f"  {item}")

        result = {
            "benchmark": "Prediction Tracker V1.6 A/B",
            "research_only": True,
            "production_source_modified": False,
            "orders_submitted": False,
            "stale_minutes": args.stale_minutes,
            "production": result_a,
            "v1_6": result_b,
            "improvement_percent": round(improvement, 3),
            "speedup": round(speedup, 3) if speedup else None,
            "semantic_comparison": comparison,
            "temporary_directory": str(tmp_parent),
        }

        output = ROOT / "prediction_tracker_v1_6_ab_results.json"
        output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nResults JSON: {output}")

        return 0 if comparison["semantic_match"] else 2

    finally:
        # Restore environment for the shell/process.
        os.environ.pop("AI_TRADER_MARKET_CACHE_DIR", None)
        os.environ.pop("AI_TRADER_MARKET_CACHE_TTL_SECONDS", None)
        os.environ.pop("AI_TRADER_TRACKER_MARKET_CACHE_TTL_SECONDS", None)

        if not args.keep:
            shutil.rmtree(tmp_parent, ignore_errors=True)
        else:
            print(f"Temporary benchmark directory kept: {tmp_parent}")


if __name__ == "__main__":
    raise SystemExit(main())
