import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ai.characteristic_curve_v1_7 import analyze_dataframe
from app.data import market_data


# ============================================================
# CCE V1.7 - Defensive input validation
# ============================================================

def test_cce_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="CCE_INVALID_INPUT_DATA"):
        analyze_dataframe("TEST", pd.DataFrame())


def test_cce_rejects_missing_ohlcv_columns():
    df = pd.DataFrame({
        "Close": [100.0, 101.0, 102.0]
    })

    with pytest.raises(ValueError, match="CCE_INVALID_INPUT_DATA"):
        analyze_dataframe("TEST", df)


# ============================================================
# MARKET DATA - Negative cache
# ============================================================

def test_market_data_failure_defers_repeated_download(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AI_TRADER_NEGATIVE_CACHE_TTL_SECONDS", "60")

    calls = []

    def failing_download(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("Simulated Yahoo failure")

    monkeypatch.setattr(
        market_data.yf, "download", failing_download
    )

    with pytest.raises(RuntimeError, match="Simulated Yahoo failure"):
        market_data.get_history(
            "TEST",
            period="1y",
            interval="1d",
            cache_dir=tmp_path,
        )

    result = market_data.get_history(
        "TEST",
        period="1y",
        interval="1d",
        cache_dir=tmp_path,
    )

    assert result.empty
    assert len(calls) == 1
    assert list(tmp_path.glob("*.fail"))


def test_market_data_expired_failure_allows_retry_and_clears_marker(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AI_TRADER_NEGATIVE_CACHE_TTL_SECONDS", "60")

    def failing_download(*args, **kwargs):
        raise RuntimeError("Simulated Yahoo failure")

    monkeypatch.setattr(
        market_data.yf, "download", failing_download
    )

    with pytest.raises(RuntimeError):
        market_data.get_history(
            "TEST",
            period="1y",
            interval="1d",
            cache_dir=tmp_path,
        )

    markers = list(tmp_path.glob("*.fail"))
    assert len(markers) == 1

    # Simulate a failure marker older than its 60-second TTL.
    import os
    old_time = markers[0].stat().st_mtime - 120
    os.utime(markers[0], (old_time, old_time))

    expected = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1200],
        },
        index=pd.date_range("2026-01-01", periods=2, freq="B"),
    )

    calls = []

    def successful_download(*args, **kwargs):
        calls.append(1)
        return expected.copy()

    monkeypatch.setattr(
        market_data.yf, "download", successful_download
    )

    result = market_data.get_history(
        "TEST",
        period="1y",
        interval="1d",
        cache_dir=tmp_path,
    )

    assert not result.empty
    assert len(calls) == 1
    assert not list(tmp_path.glob("*.fail"))


def test_market_data_empty_download_records_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AI_TRADER_NEGATIVE_CACHE_TTL_SECONDS", "60")

    calls = []

    def empty_download(*args, **kwargs):
        calls.append(1)
        return pd.DataFrame()

    monkeypatch.setattr(
        market_data.yf, "download", empty_download
    )

    result = market_data.get_history(
        "TEST",
        period="1y",
        interval="1d",
        cache_dir=tmp_path,
    )

    assert result.empty
    assert len(calls) == 1
    assert list(tmp_path.glob("*.fail"))


# ============================================================
# WATCHLIST SCANNER - ENGINE_ERROR hard block
# ============================================================

def test_scanner_engine_error_is_a_hard_block():
    project_root = Path(__file__).resolve().parents[2]
    scanner_path = (
        project_root
        / "app"
        / "scanners"
        / "watchlist_scanner_v2_2_2.py"
    )

    source = scanner_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    scanner_function = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "scan_buy_opportunity"
    )

    hard_block = None

    for node in ast.walk(scanner_function):
        if not isinstance(node, ast.Assign):
            continue

        if any(
            isinstance(target, ast.Name)
            and target.id == "hard_block_signals"
            for target in node.targets
        ):
            hard_block = ast.literal_eval(node.value)
            break

    assert hard_block is not None
    assert "ENGINE_ERROR" in hard_block

    # Verify the BUY promotion condition consults the hard-block set.
    uses_hard_block = any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "current_signal"
        and any(
            isinstance(op, ast.NotIn)
            and isinstance(comparator, ast.Name)
            and comparator.id == "hard_block_signals"
            for op, comparator in zip(node.ops, node.comparators)
        )
        for node in ast.walk(scanner_function)
    )

    assert uses_hard_block
