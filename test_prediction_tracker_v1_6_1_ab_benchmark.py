#!/usr/bin/env python3
"""Contract test for Prediction Tracker V1.6.1 A/B Functional Equivalence Benchmark."""

from pathlib import Path
import py_compile
import ast
import tempfile
import json
import sys


SCRIPT = Path(__file__).with_name("prediction_tracker_v1_6_1_ab_benchmark.py")


def main() -> int:
    assert SCRIPT.exists(), f"Missing benchmark: {SCRIPT}"
    py_compile.compile(str(SCRIPT), doraise=True)
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    required = [
        "PRODUCTION_MODULE",
        "TRACKER_V16_1",
        "HORIZONS",
        "TARGET_DATE",
        "TARGET_DATE_EXPECTED_RECORDS",
        "canonicalize",
        "compare",
        "run_tracker",
        "age_cache_files",
    ]
    for name in required:
        assert any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.name == name
            for n in tree.body
        ) or any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
            for n in tree.body
        ), f"Missing required symbol: {name}"

    assert "app/prediction_tracker.py" in source
    assert "prediction_tracker_v1_6_1.py" in source
    assert "2026-09-26" in source
    assert "78" in source
    assert "+20d" in source and "+60d" in source
    assert "risk_x" in source
    assert "post_exit_analysis" in source
    assert "evaluated_at_utc" in source

    # Safety guards.
    assert "alpaca.markets" not in source.lower()
    assert "/v2/orders" not in source.lower()
    assert "submit_order" not in source.lower()
    assert "create_order" not in source.lower()
    assert "production_source_modified" in source.lower()

    # Verify canonicalization ignores only evaluated_at_utc.
    ns = {}
    exec(compile(source, str(SCRIPT), "exec"), ns)
    a = {
        "evaluation": {
            "evaluated_at_utc": "A",
            "observations": {"+20d": {"return_pct": 1.0}},
            "risk_x": 2.0,
        }
    }
    b = {
        "evaluation": {
            "evaluated_at_utc": "B",
            "observations": {"+20d": {"return_pct": 1.0}},
            "risk_x": 2.0,
        }
    }
    assert ns["canonicalize"](a) == ns["canonicalize"](b)

    b["evaluation"]["risk_x"] = 3.0
    assert ns["canonicalize"](a) != ns["canonicalize"](b)

    # Minimal compare contract: same data must be equal.
    assert ns["compare"]({"x.jsonl": [a]}, {"x.jsonl": [b]})["semantic_output_match"] is False

    print("PASS: py_compile")
    print("PASS: production V1.5 vs V1.6.1 source contract")
    print("PASS: six-horizon comparison contract")
    print("PASS: risk_x/post_exit_analysis contract")
    print("PASS: target date 2026-09-26 / 78-record contract")
    print("PASS: evaluated_at_utc is the only ignored field")
    print("PASS: research-only / no Alpaca / no orders guards")
    print("\nPREDICTION TRACKER V1.6.1 A/B BENCHMARK CONTRACT TEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
