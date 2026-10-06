"""
AI Trader - Prediction Tracker V1.6 Functional Equivalence Diagnostic
======================================================================

Research-only diagnostic.

Purpose
-------
Explain every semantic difference observed in the V1.6 A/B benchmark without
modifying production files.

The diagnostic:
1. Creates isolated A/B copies of prediction JSONL files and Market Data cache.
2. Ages both caches equally so V1.5 experiences the production 15-minute TTL
   behavior while V1.6 uses its tracker-specific 24h TTL.
3. Runs production Prediction Tracker and V1.6 against identical inputs.
4. Compares records field-by-field after ignoring only dynamic timestamps.
5. Classifies differences, with special attention to:
   - COMPLETE fast-path behavior
   - evaluation.status
   - horizon observations (+1d ... +60d)
   - risk_x
   - post_exit_analysis
   - evaluation metadata
6. Produces a JSON report.

Safety
------
- Does not modify app/prediction_tracker.py.
- Does not modify original data/predictions files.
- Does not modify original Market Data cache.
- No Alpaca endpoints.
- No orders.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PRODUCTION_MODULE = ROOT / "app" / "prediction_tracker.py"
EXPERIMENTAL_MODULE = ROOT / "prediction_tracker_v1_6.py"
DEFAULT_PREDICTIONS = ROOT / "data" / "predictions"
DEFAULT_CACHE = ROOT / "data" / "market_cache"

DYNAMIC_KEYS = {
    "evaluated_at_utc",
}

HORIZONS = ("+1d", "+3d", "+5d", "+10d", "+20d", "+60d")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise RuntimeError(f"Missing source directory: {src}")
    shutil.copytree(src, dst)


def age_files(root: Path, age_seconds: int) -> int:
    now = time.time()
    count = 0
    for p in root.rglob("*"):
        if p.is_file():
            ts = now - age_seconds
            os.utime(p, (ts, ts))
            count += 1
    return count


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
    return records


def strip_dynamic(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: strip_dynamic(v)
            for k, v in value.items()
            if k not in DYNAMIC_KEYS
        }
    if isinstance(value, list):
        return [strip_dynamic(v) for v in value]
    return value


def index_records(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result = {}
    for i, rec in enumerate(records):
        key = (
            str(rec.get("ticker", "")),
            str(rec.get("prediction_timestamp_utc", "")),
            str(rec.get("prediction_id", rec.get("id", ""))),
        )
        # If no explicit id exists, include ordinal to avoid accidental
        # collisions among otherwise identical records.
        if not key[2]:
            key = (key[0], key[1], f"ordinal:{i}")
        result[key] = rec
    return result


def value_at(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = record
    for part in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def compare_record(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    diffs = []

    # Top-level and nested comparison, preserving exact values.
    def walk(x: Any, y: Any, path: str) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            keys = sorted(set(x) | set(y))
            for k in keys:
                if k in DYNAMIC_KEYS:
                    continue
                if k not in x:
                    diffs.append({"path": f"{path}.{k}", "type": "missing_in_a", "a": None, "b": y[k]})
                elif k not in y:
                    diffs.append({"path": f"{path}.{k}", "type": "missing_in_b", "a": x[k], "b": None})
                else:
                    walk(x[k], y[k], f"{path}.{k}")
            return

        if isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                diffs.append({
                    "path": path,
                    "type": "list_length",
                    "a": len(x),
                    "b": len(y),
                })
                # Compare common prefix.
            for i, (vx, vy) in enumerate(zip(x, y)):
                walk(vx, vy, f"{path}[{i}]")
            return

        if x != y:
            diffs.append({
                "path": path,
                "type": "value",
                "a": x,
                "b": y,
            })

    walk(strip_dynamic(a), strip_dynamic(b), "$")
    return diffs


def classify_diff(diffs: list[dict[str, Any]], a: dict[str, Any], b: dict[str, Any]) -> str:
    if not diffs:
        return "IDENTICAL"

    paths = [d["path"] for d in diffs]

    status_a = value_at(a, ("evaluation", "status"))
    status_b = value_at(b, ("evaluation", "status"))

    obs_paths = [p for p in paths if ".evaluation.observations." in p]
    if obs_paths:
        return "HORIZON_OBSERVATION_DIFFERENCE"

    if status_a != status_b:
        return "EVALUATION_STATUS_DIFFERENCE"

    if any(".risk_x" in p or ".risk" in p for p in paths):
        return "RISK_DIFFERENCE"

    if any(".post_exit_analysis" in p for p in paths):
        return "POST_EXIT_DIFFERENCE"

    if any(p.startswith("$.evaluation.") for p in paths):
        return "EVALUATION_METADATA_DIFFERENCE"

    return "OTHER_FIELD_DIFFERENCE"


def summarize_horizon_diffs(diffs: list[dict[str, Any]]) -> Counter:
    counts = Counter()
    for d in diffs:
        path = d["path"]
        for h in HORIZONS:
            if h in path:
                counts[h] += 1
                break
    return counts


def run_tracker(module, pred_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    summary = module.update_all_predictions(pred_dir, include_today=False)
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--stale-minutes", type=int, default=20)
    parser.add_argument("--max-details", type=int, default=100)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    predictions = Path(args.predictions).resolve()
    cache = Path(args.cache).resolve()

    if not PRODUCTION_MODULE.exists():
        raise RuntimeError(f"Production module not found: {PRODUCTION_MODULE}")
    if not EXPERIMENTAL_MODULE.exists():
        raise RuntimeError(f"V1.6 module not found: {EXPERIMENTAL_MODULE}")
    if not predictions.exists():
        raise RuntimeError(f"Prediction directory not found: {predictions}")
    if not cache.exists():
        raise RuntimeError(f"Market cache directory not found: {cache}")

    print("=" * 78)
    print("AI TRADER - PREDICTION TRACKER V1.6 FUNCTIONAL EQUIVALENCE DIAGNOSTIC")
    print("Research-only | production source unchanged | orders blocked")
    print("=" * 78)

    prod = load_module(PRODUCTION_MODULE, "prediction_tracker_prod_diag")
    v16 = load_module(EXPERIMENTAL_MODULE, "prediction_tracker_v16_diag")

    tmp = Path(tempfile.mkdtemp(prefix="ai_trader_tracker_equiv_"))
    pred_a = tmp / "predictions_a"
    pred_b = tmp / "predictions_b"
    cache_a = tmp / "cache_a"
    cache_b = tmp / "cache_b"

    try:
        copy_tree(predictions, pred_a)
        copy_tree(predictions, pred_b)
        copy_tree(cache, cache_a)
        copy_tree(cache, cache_b)

        age_files(cache_a, args.stale_minutes * 60)
        age_files(cache_b, args.stale_minutes * 60)

        print(f"Prediction source : {predictions}")
        print(f"Cache source      : {cache}")
        print(f"Stale cache age   : {args.stale_minutes} min")
        print()

        os.environ["AI_TRADER_MARKET_CACHE_DIR"] = str(cache_a)
        os.environ["AI_TRADER_MARKET_CACHE_TTL_SECONDS"] = "900"
        os.environ.pop("AI_TRADER_TRACKER_MARKET_CACHE_TTL_SECONDS", None)

        print(">>> Running A — production V1.5")
        result_a = run_tracker(prod, pred_a)
        print(f"A elapsed: {result_a['elapsed_seconds']:.3f}s")
        print(f"A summary: {result_a['summary']}")
        print()

        os.environ["AI_TRADER_MARKET_CACHE_DIR"] = str(cache_b)
        os.environ["AI_TRADER_MARKET_CACHE_TTL_SECONDS"] = "900"
        os.environ["AI_TRADER_TRACKER_MARKET_CACHE_TTL_SECONDS"] = "86400"

        print(">>> Running B — V1.6")
        result_b = run_tracker(v16, pred_b)
        print(f"B elapsed: {result_b['elapsed_seconds']:.3f}s")
        print(f"B summary: {result_b['summary']}")
        print()

        files_a = {p.name: p for p in pred_a.glob("predictions_*.jsonl")}
        files_b = {p.name: p for p in pred_b.glob("predictions_*.jsonl")}

        category_counts = Counter()
        differing_records = 0
        identical_records = 0
        total_records = 0
        horizon_counts = Counter()
        details = []

        for name in sorted(set(files_a) | set(files_b)):
            ra = load_jsonl(files_a[name]) if name in files_a else []
            rb = load_jsonl(files_b[name]) if name in files_b else []

            ia = index_records(ra)
            ib = index_records(rb)

            keys = sorted(set(ia) | set(ib))
            for key in keys:
                total_records += 1
                if key not in ia or key not in ib:
                    category = "RECORD_MISSING"
                    diffs = [{
                        "path": "$",
                        "type": "record_missing",
                        "a": key in ia,
                        "b": key in ib,
                    }]
                    differing_records += 1
                else:
                    diffs = compare_record(ia[key], ib[key])
                    if not diffs:
                        identical_records += 1
                        continue
                    differing_records += 1
                    category = classify_diff(diffs, ia[key], ib[key])
                    horizon_counts.update(summarize_horizon_diffs(diffs))

                category_counts[category] += 1

                if len(details) < args.max_details:
                    detail = {
                        "file": name,
                        "key": key,
                        "category": category,
                        "diff_count": len(diffs),
                        "diffs": diffs[:30],
                    }
                    details.append(detail)

        print("=" * 78)
        print("FUNCTIONAL EQUIVALENCE RESULT")
        print("=" * 78)
        print(f"Total records compared : {total_records}")
        print(f"Identical              : {identical_records}")
        print(f"Different              : {differing_records}")
        print()
        print("Difference categories:")
        for category, count in category_counts.most_common():
            print(f"  {category}: {count}")

        print()
        print("Horizon difference counts:")
        for horizon in HORIZONS:
            if horizon_counts[horizon]:
                print(f"  {horizon}: {horizon_counts[horizon]}")

        print()
        print("Representative differences:")
        for item in details[:20]:
            print(
                f"  {item['file']} | {item['key']} | "
                f"{item['category']} | diffs={item['diff_count']}"
            )
            for diff in item["diffs"][:5]:
                print(f"      {diff['path']}: A={diff['a']!r} B={diff['b']!r}")

        report = {
            "diagnostic": "Prediction Tracker V1.6 Functional Equivalence",
            "research_only": True,
            "production_source_modified": False,
            "orders_submitted": False,
            "stale_minutes": args.stale_minutes,
            "production": result_a,
            "v1_6": result_b,
            "comparison": {
                "total_records": total_records,
                "identical_records": identical_records,
                "different_records": differing_records,
                "difference_categories": dict(category_counts),
                "horizon_difference_counts": dict(horizon_counts),
                "details": details,
            },
            "temporary_directory": str(tmp),
        }

        output = ROOT / "prediction_tracker_v1_6_functional_equivalence_results.json"
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()
        print(f"Results JSON: {output}")

        # Exit 0 means diagnostic completed, not that equivalence passed.
        return 0

    finally:
        os.environ.pop("AI_TRADER_MARKET_CACHE_DIR", None)
        os.environ.pop("AI_TRADER_MARKET_CACHE_TTL_SECONDS", None)
        os.environ.pop("AI_TRADER_TRACKER_MARKET_CACHE_TTL_SECONDS", None)

        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
