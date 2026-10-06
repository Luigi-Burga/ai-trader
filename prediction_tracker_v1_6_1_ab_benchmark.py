#!/usr/bin/env python3
"""
Prediction Tracker V1.6.1 — A/B Functional Equivalence Benchmark

Research-only benchmark.
Compares production Prediction Tracker V1.5 against V1.6.1.

Safety:
- Does NOT modify production source.
- Does NOT modify original prediction files.
- Does NOT modify original market cache.
- Does NOT call Alpaca/trading/order endpoints.
- Uses isolated temporary copies of predictions and cache.
- Ignores ONLY dynamic evaluation.evaluated_at_utc for semantic comparison.

The benchmark deliberately ages isolated cache files so V1.5's 15-minute
Market Data TTL behaves like the prior scheduler condition, while V1.6.1's
tracker-specific 24-hour TTL can reuse the same cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


VERSION = "1.0"
TRACKER_V16_1 = "1.6.1"
PRODUCTION_MODULE = "app/prediction_tracker.py"
HORIZONS = ("+1d", "+3d", "+5d", "+10d", "+20d", "+60d")
TARGET_DATE = "2026-09-26"
TARGET_DATE_EXPECTED_RECORDS = 78


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst)


def age_cache_files(cache_dir: Path, age_seconds: int) -> int:
    now = time.time()
    count = 0
    for p in cache_dir.rglob("*"):
        if p.is_file():
            ts = now - age_seconds
            os.utime(p, (ts, ts))
            count += 1
    return count


def canonicalize(value: Any) -> Any:
    """
    Canonicalize for semantic comparison.
    ONLY dynamic evaluated_at_utc is ignored.
    No business fields are ignored.
    """
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "evaluated_at_utc":
                continue
            out[k] = canonicalize(v)
        return out
    if isinstance(value, list):
        return [canonicalize(v) for v in value]
    return value


def record_key(row: Dict[str, Any], fallback_index: int) -> str:
    for key in ("prediction_id", "id", "snapshot_id"):
        if row.get(key) is not None:
            return str(row[key])
    # Prediction files are JSONL snapshots; ticker + timestamp is the natural
    # stable identity when no explicit ID exists.
    ticker = row.get("ticker", "")
    for ts_key in ("timestamp", "as_of", "created_at", "evaluated_at_utc"):
        if row.get(ts_key):
            return f"{ticker}|{row[ts_key]}"
    return f"__index__{fallback_index}"


def index_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for i, row in enumerate(rows):
        key = record_key(row, i)
        if key in out:
            raise RuntimeError(f"Duplicate logical record key: {key}")
        out[key] = row
    return out


def run_tracker(
    root: Path,
    module_path: str,
    prediction_dir: Path,
    cache_dir: Path,
    env_extra: Dict[str, str],
) -> Tuple[float, str, int]:
    env = os.environ.copy()
    env.update(env_extra)
    env["AI_TRADER_PREDICTIONS_DIR"] = str(prediction_dir)
    env["AI_TRADER_MARKET_CACHE_DIR"] = str(cache_dir)
    env["PYTHONUNBUFFERED"] = "1"

    # The production tracker is a module. V1.6.1 is supplied as a root script.
    if module_path.startswith("app/"):
        cmd = [sys.executable, "-m", module_path[:-3].replace("/", ".")]
    else:
        cmd = [sys.executable, str(root / module_path)]

    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - started
    return elapsed, proc.stdout, proc.returncode


def compare(
    prod_files: Dict[str, List[Dict[str, Any]]],
    new_files: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "semantic_output_match": True,
        "files_compared": 0,
        "records_production": 0,
        "records_v1_6_1": 0,
        "identical_records": 0,
        "different_records": 0,
        "missing_in_v1_6_1": 0,
        "missing_in_production": 0,
        "difference_categories": {},
        "horizon_difference_counts": {h: 0 for h in HORIZONS},
        "risk_x_difference_count": 0,
        "post_exit_analysis_difference_count": 0,
        "target_date": {
            "date": TARGET_DATE,
            "expected_records": TARGET_DATE_EXPECTED_RECORDS,
            "production_records": 0,
            "v1_6_1_records": 0,
            "different_records": 0,
            "horizon_difference_counts": {h: 0 for h in HORIZONS},
        },
        "examples": [],
    }

    all_names = sorted(set(prod_files) | set(new_files))
    report["files_compared"] = len(all_names)

    for name in all_names:
        a = prod_files.get(name, [])
        b = new_files.get(name, [])
        report["records_production"] += len(a)
        report["records_v1_6_1"] += len(b)

        ai = index_rows(a)
        bi = index_rows(b)

        keys = sorted(set(ai) | set(bi))
        for key in keys:
            if key not in ai:
                report["missing_in_production"] += 1
                report["semantic_output_match"] = False
                continue
            if key not in bi:
                report["missing_in_v1_6_1"] += 1
                report["semantic_output_match"] = False
                continue

            ca = canonicalize(ai[key])
            cb = canonicalize(bi[key])

            if ca == cb:
                report["identical_records"] += 1
                continue

            report["different_records"] += 1
            report["semantic_output_match"] = False

            diffs = []
            ea = ai[key].get("evaluation", {}) or {}
            eb = bi[key].get("evaluation", {}) or {}

            for h in HORIZONS:
                if ea.get("observations", {}).get(h) != eb.get("observations", {}).get(h):
                    report["horizon_difference_counts"][h] += 1
                    diffs.append(f"HORIZON_OBSERVATION_DIFFERENCE:{h}")

            if ea.get("risk_x") != eb.get("risk_x"):
                report["risk_x_difference_count"] += 1
                diffs.append("RISK_X_DIFFERENCE")

            if ea.get("post_exit_analysis") != eb.get("post_exit_analysis"):
                report["post_exit_analysis_difference_count"] += 1
                diffs.append("POST_EXIT_ANALYSIS_DIFFERENCE")

            if not diffs:
                diffs.append("OTHER_FIELD_DIFFERENCE")

            for d in diffs:
                report["difference_categories"][d] = (
                    report["difference_categories"].get(d, 0) + 1
                )

            if len(report["examples"]) < 10:
                report["examples"].append({
                    "file": name,
                    "record_key": key,
                    "differences": diffs,
                    "production_evaluation": ea,
                    "v1_6_1_evaluation": eb,
                })

            # Explicit target-date verification.
            raw = ai[key]
            date_text = str(
                raw.get("timestamp")
                or raw.get("as_of")
                or raw.get("created_at")
                or raw.get("evaluated_at_utc")
                or ""
            )
            if TARGET_DATE in date_text:
                report["target_date"]["different_records"] += 1
                for h in HORIZONS:
                    if ea.get("observations", {}).get(h) != eb.get("observations", {}).get(h):
                        report["target_date"]["horizon_difference_counts"][h] += 1

    # Count target-date records independently across both sides.
    for source_name, source_files, out_key in (
        ("production", prod_files, "production_records"),
        ("v1_6_1", new_files, "v1_6_1_records"),
    ):
        count = 0
        for rows in source_files.values():
            for row in rows:
                text_value = str(
                    row.get("timestamp")
                    or row.get("as_of")
                    or row.get("created_at")
                    or row.get("evaluated_at_utc")
                    or ""
                )
                if TARGET_DATE in text_value:
                    count += 1
        report["target_date"][out_key] = count

    return report


def collect_prediction_files(prediction_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    result = {}
    for p in sorted(prediction_dir.glob("predictions_*.jsonl")):
        result[p.name] = load_jsonl(p)
    if not result:
        raise RuntimeError(f"No predictions_*.jsonl found in {prediction_dir}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="AI Trader project root")
    parser.add_argument("--age-cache-seconds", type=int, default=1200)
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep isolated A/B work directory for forensic inspection",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    production = root / PRODUCTION_MODULE
    candidate = root / "prediction_tracker_v1_6_1.py"
    source_predictions = root / "data" / "predictions"
    source_cache = root / "data" / "market_cache"

    if not production.exists():
        raise FileNotFoundError(production)
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    if not source_predictions.exists():
        raise FileNotFoundError(source_predictions)
    if not source_cache.exists():
        raise FileNotFoundError(source_cache)

    print("=" * 78)
    print("Prediction Tracker V1.6.1 — A/B Functional Equivalence Benchmark")
    print(f"Benchmark version: {VERSION}")
    print("Production: app/prediction_tracker.py")
    print("Candidate : prediction_tracker_v1_6_1.py")
    print("Research-only | production unchanged | orders blocked")
    print("Original predictions/cache are NEVER modified")
    print("=" * 78)

    original_predictions = collect_prediction_files(source_predictions)
    total_original = sum(len(v) for v in original_predictions.values())
    print(f"Prediction files: {len(original_predictions)}")
    print(f"Original records: {total_original}")

    if TARGET_DATE in "\n".join(original_predictions.keys()):
        pass

    work_parent = Path(tempfile.mkdtemp(prefix="pt_v161_ab_"))
    print(f"Isolated workdir: {work_parent}")

    try:
        a_pred = work_parent / "A_predictions"
        b_pred = work_parent / "B_predictions"
        a_cache = work_parent / "A_cache"
        b_cache = work_parent / "B_cache"

        copy_tree(source_predictions, a_pred)
        copy_tree(source_predictions, b_pred)
        copy_tree(source_cache, a_cache)
        copy_tree(source_cache, b_cache)

        aged_a = age_cache_files(a_cache, args.age_cache_seconds)
        aged_b = age_cache_files(b_cache, args.age_cache_seconds)
        print(f"Aged isolated cache files: A={aged_a} B={aged_b}")

        common_env = {
            # Prevent accidental tracker writes to the real directories.
            "AI_TRADER_PREDICTIONS_DIR": "",
            "AI_TRADER_MARKET_CACHE_DIR": "",
            # Explicitly disable any future autonomous trading behavior.
            "AI_TRADER_AUTONOMOUS_PAPER_EXECUTION": "false",
        }

        print("\n--- A: Production V1.5 ---")
        ta, out_a, rc_a = run_tracker(
            root, PRODUCTION_MODULE, a_pred, a_cache, common_env
        )
        print(out_a)
        print(f"A exit_code={rc_a} elapsed={ta:.3f}s")
        if rc_a != 0:
            raise RuntimeError("Production V1.5 benchmark failed")

        print("\n--- B: V1.6.1 ---")
        tb, out_b, rc_b = run_tracker(
            root, "prediction_tracker_v1_6_1.py", b_pred, b_cache, common_env
        )
        print(out_b)
        print(f"B exit_code={rc_b} elapsed={tb:.3f}s")
        if rc_b != 0:
            raise RuntimeError("V1.6.1 benchmark failed")

        a_files = collect_prediction_files(a_pred)
        b_files = collect_prediction_files(b_pred)
        report = compare(a_files, b_files)

        speedup = ta / tb if tb > 0 else None
        improvement = ((ta - tb) / ta * 100.0) if ta > 0 else None

        report.update({
            "benchmark_version": VERSION,
            "candidate_version": TRACKER_V16_1,
            "production_module": PRODUCTION_MODULE,
            "candidate_module": "prediction_tracker_v1_6_1.py",
            "source_sha256": {
                "production": sha256_file(production),
                "candidate": sha256_file(candidate),
            },
            "timing_seconds": {
                "production_v1_5": ta,
                "v1_6_1": tb,
            },
            "performance": {
                "improvement_percent": improvement,
                "speedup": speedup,
            },
            "age_cache_seconds": args.age_cache_seconds,
            "dynamic_fields_ignored": ["evaluated_at_utc"],
            "orders_or_alpaca": False,
            "production_source_modified": False,
        })

        out_path = root / "prediction_tracker_v1_6_1_ab_results.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)

        print("\n" + "=" * 78)
        print("FINAL A/B RESULT")
        print("=" * 78)
        print(f"Production V1.5 time : {ta:.3f}s")
        print(f"V1.6.1 time          : {tb:.3f}s")
        print(f"Improvement           : {improvement:.2f}%")
        print(f"Speedup               : {speedup:.2f}x")
        print(f"Semantic output match : {report['semantic_output_match']}")
        print(f"Files compared       : {report['files_compared']}")
        print(f"Records A/B           : {report['records_production']} / {report['records_v1_6_1']}")
        print(f"Identical records     : {report['identical_records']}")
        print(f"Different records     : {report['different_records']}")
        print(f"risk_x differences    : {report['risk_x_difference_count']}")
        print(
            "post_exit differences: "
            f"{report['post_exit_analysis_difference_count']}"
        )
        print("\nHorizon differences:")
        for h, n in report["horizon_difference_counts"].items():
            print(f"  {h:>4}: {n}")

        td = report["target_date"]
        print(f"\nTARGET DATE {TARGET_DATE}")
        print(f"  Expected records    : {TARGET_DATE_EXPECTED_RECORDS}")
        print(f"  Production records  : {td['production_records']}")
        print(f"  V1.6.1 records      : {td['v1_6_1_records']}")
        print(f"  Different records   : {td['different_records']}")
        print("  Horizon differences:")
        for h, n in td["horizon_difference_counts"].items():
            print(f"    {h:>4}: {n}")

        print(f"\nResults JSON: {out_path}")

        if report["semantic_output_match"]:
            print("\nFUNCTIONAL EQUIVALENCE: PASS")
        else:
            print("\nFUNCTIONAL EQUIVALENCE: FAIL")
            print("Differences by category:")
            for k, v in sorted(report["difference_categories"].items()):
                print(f"  {k}: {v}")

        # The benchmark itself succeeds even when equivalence fails; the result
        # is research evidence. Exit non-zero only for execution errors.
        return 0

    finally:
        if args.keep_workdir:
            print(f"Keeping isolated workdir: {work_parent}")
        else:
            shutil.rmtree(work_parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
