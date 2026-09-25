"""
AI Trader - Prediction Tracker V1.3

Evaluates historical prediction snapshots stored by Prediction Logger.

V1.4:
- Preserves the V1.1 public contract and evaluation calculations.
- Keeps all market-data access behind Shared Market Data Layer V2.
- Reuses the persistent 5y Market Data V2 cache whenever it covers a ticker.
- Slices the cached historical frame to each prediction's required window.
- Falls back to the exact V1.1 request window only when the broad cache
  cannot cover an eligible prediction.
- Fixes today's-file detection to use UTC, matching Prediction Logger V1.1.
- Reuses one DataFrame per ticker across all snapshots.
- Preserves atomic JSONL replacement semantics.

Supported horizons:
    +1, +3, +5, +10, +20, +60 trading days
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from copy import deepcopy
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

from app.data.market_data import get_history

VERSION = "1.4"
MODULE_NAME = "Prediction Tracker"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = PROJECT_ROOT / "data" / "predictions"

HORIZONS_DAYS = (1, 3, 5, 10, 20, 60)

# V1.4 acquisition batch size is owned by Market Data V2; this tracker
# deliberately does not bypass or duplicate its cache implementation.


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: Optional[datetime] = None) -> str:
    value = dt or _utc_now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _normalize_symbol(symbol: Any) -> str:
    return str(symbol or "").strip().upper()


def prediction_file(
    prediction_date: Optional[date | str] = None,
    base_dir: Optional[str | Path] = None,
) -> Path:
    if prediction_date is None:
        day = _utc_now().date()
    elif isinstance(prediction_date, date):
        day = prediction_date
    else:
        day = date.fromisoformat(str(prediction_date))

    root = Path(base_dir) if base_dir is not None else DEFAULT_DIRECTORY
    return root / f"predictions_{day.isoformat()}.jsonl"


def _iter_files(base_dir: str | Path = DEFAULT_DIRECTORY) -> Iterable[Path]:
    root = Path(base_dir)
    if not root.exists():
        return []
    return sorted(root.glob("predictions_*.jsonl"))


def _is_today_file(path: Path) -> bool:
    # Prediction Logger V1.1 names files using UTC date.
    return path.name == f"predictions_{_utc_now().date().isoformat()}.jsonl"


def read_predictions(
    base_dir: str | Path = DEFAULT_DIRECTORY,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for path in _iter_files(base_dir):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        records.append(value)
        except OSError:
            continue

    return records


def _prediction_request_window(
    prediction_timestamp: str,
    max_horizon: int = 60,
) -> tuple[str, str]:
    """
    Return the same start/end calendar window used by V1.1.

    V1.1 requested:
        start = prediction date
        end   = start + max_horizon * 3 + 15 calendar days

    V1.2 keeps that exact request window; it only combines compatible
    windows for multiple snapshots of the same ticker.
    """
    ts = datetime.fromisoformat(
        prediction_timestamp.replace("Z", "+00:00")
    )
    start = ts.date()
    end = start + timedelta(days=max_horizon * 3 + 15)
    return start.isoformat(), end.isoformat()


def _slice_market_window(
    data: Any,
    start: str,
    end: str,
):
    """Return the requested calendar window from a broader daily frame."""
    if data is None or getattr(data, "empty", True):
        return None

    frame = data.copy(deep=True)

    if hasattr(frame.columns, "nlevels") and frame.columns.nlevels > 1:
        columns = frame.columns

        # Normalize common yfinance MultiIndex layouts to OHLCV columns.
        if columns.nlevels > 1:
            yahoo_level_values = {
                str(value).upper()
                for value in columns.get_level_values(-1)
            }
            yahoo_ticker = _normalize_symbol(
                frame.attrs.get("ticker", "")
            ).replace(".", "-")

            if yahoo_ticker and yahoo_ticker in yahoo_level_values:
                try:
                    frame = frame.xs(
                        yahoo_ticker,
                        axis=1,
                        level=-1,
                    )
                except Exception:
                    pass

            if (
                hasattr(frame.columns, "nlevels")
                and frame.columns.nlevels > 1
            ):
                frame.columns = frame.columns.get_level_values(0)

    frame = frame.sort_index()

    try:
        start_date = datetime.fromisoformat(start).date()
        end_date = datetime.fromisoformat(end).date()
    except (TypeError, ValueError):
        return None

    mask = []
    for idx in frame.index:
        try:
            idx_date = idx.date()
        except AttributeError:
            idx_date = idx

        mask.append(
            idx_date >= start_date
            and idx_date < end_date
        )

    if not any(mask):
        return None

    sliced = frame.loc[mask].copy(deep=True)
    if sliced.empty:
        return None

    return sliced


def _trading_days_since(
    prediction_timestamp: str,
    ticker: str,
    max_horizon: int = 60,
):
    """
    Compatibility helper matching the V1.1 function contract.

    V1.3 first requests a broad 5y Market Data V2 frame so an existing
    persistent 5y cache can be reused. The resulting frame is then sliced
    to the exact V1.1 calendar window. If the broad cache cannot cover the
    prediction date, the exact V1.1 request is used as a fallback.
    """
    start, end = _prediction_request_window(
        prediction_timestamp,
        max_horizon,
    )

    broad = get_history(
        ticker,
        period="5y",
        start=None,
        end=None,
        interval="1d",
        auto_adjust=False,
        actions=False,
        group_by="column",
        threads=False,
    )

    sliced = _slice_market_window(broad, start, end)
    if sliced is not None:
        return sliced

    # Compatibility fallback for predictions older than the broad cache.
    data = get_history(
        ticker,
        period=None,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        group_by="column",
        threads=False,
    )

    if data is None or data.empty:
        return None

    data = data.sort_index()
    return data


def _series_value(
    frame: Any,
    column: str,
    index: Any,
) -> Optional[float]:
    try:
        value = frame[column].loc[index]
    except Exception:
        return None

    if hasattr(value, "iloc"):
        try:
            value = value.iloc[0]
        except Exception:
            return None

    return _safe_float(value)


def _find_prediction_session(
    frame: Any,
    prediction_timestamp: str,
    price_at_prediction: Optional[float],
):
    if frame is None or len(frame.index) == 0:
        return None

    ts = datetime.fromisoformat(
        prediction_timestamp.replace("Z", "+00:00")
    )
    target_date = ts.date()

    candidates = []
    for idx in frame.index:
        try:
            idx_date = idx.date()
        except AttributeError:
            idx_date = idx
        if idx_date >= target_date:
            candidates.append(idx)

    if not candidates:
        return None

    return candidates[0]


def _horizon_observation(
    frame: Any,
    t0_index: Any,
    horizon: int,
    entry_price: float,
    direction: str,
    target: Optional[float],
    stop: Optional[float],
) -> Optional[dict[str, Any]]:
    try:
        pos = list(frame.index).index(t0_index)
    except ValueError:
        return None

    future_pos = pos + horizon
    if future_pos >= len(frame.index):
        return None

    idx = frame.index[future_pos]
    close = _series_value(frame, "Close", idx)
    high = _series_value(frame, "High", idx)
    low = _series_value(frame, "Low", idx)

    if close is None:
        return None

    direction = direction.upper()

    if direction in {"SELL", "REDUCE"}:
        return_pct = (entry_price - close) / entry_price * 100.0
        mfe_pct = (
            (entry_price - low) / entry_price * 100.0
            if low is not None else None
        )
        mae_pct = (
            (entry_price - high) / entry_price * 100.0
            if high is not None else None
        )
    else:
        return_pct = (close - entry_price) / entry_price * 100.0
        mfe_pct = (
            (high - entry_price) / entry_price * 100.0
            if high is not None else None
        )
        mae_pct = (
            (low - entry_price) / entry_price * 100.0
            if low is not None else None
        )

    target_hit = None
    stop_hit = None

    if target is not None:
        target_hit = (
            high >= target
            if direction not in {"SELL", "REDUCE"}
            else low <= target
        )

    if stop is not None:
        stop_hit = (
            low <= stop
            if direction not in {"SELL", "REDUCE"}
            else high >= stop
        )

    correctness: Optional[bool] = None
    if direction in {
        "BUY",
        "BUY_ON_CONFIRMATION",
        "BUY_ON_PULLBACK",
    }:
        correctness = return_pct > 0
    elif direction in {"SELL", "REDUCE"}:
        correctness = return_pct > 0

    return {
        "actual_price": close,
        "return_pct": round(return_pct, 6),
        "mfe_pct": round(mfe_pct, 6) if mfe_pct is not None else None,
        "mae_pct": round(mae_pct, 6) if mae_pct is not None else None,
        "target_hit": target_hit,
        "stop_hit": stop_hit,
        "correctness": correctness,
        "evaluated_at_utc": _iso_utc(),
    }


def evaluate_snapshot(
    snapshot: dict[str, Any],
    market_data: Any = None,
) -> dict[str, Any]:
    result = deepcopy(snapshot)

    status = result.get("evaluation", {}).get(
        "status", "PENDING"
    )
    if status == "COMPLETE":
        return result

    ticker = _normalize_symbol(result.get("ticker"))
    timestamp = result.get("prediction_timestamp_utc")
    entry = _safe_float(result.get("price_at_prediction"))

    if not ticker or not timestamp or entry is None or entry <= 0:
        return result

    frame = (
        market_data
        if market_data is not None
        else _trading_days_since(
            timestamp,
            ticker,
            max(HORIZONS_DAYS),
        )
    )

    t0_index = _find_prediction_session(
        frame,
        timestamp,
        entry,
    )

    if t0_index is None:
        return result

    levels = result.get("levels") or {}

    target = _safe_float(levels.get("target"))
    if target is None:
        targets = levels.get("targets")
        if isinstance(targets, list) and targets:
            target = _safe_float(targets[0])

    stop = _safe_float(levels.get("stop"))

    signal = str(
        result.get("final_signal")
        or result.get("signal")
        or result.get("raw_signal")
        or ""
    ).strip().upper()

    observations: dict[str, Any] = {}
    complete_count = 0

    for horizon in HORIZONS_DAYS:
        obs = _horizon_observation(
            frame,
            t0_index,
            horizon,
            entry,
            signal,
            target,
            stop,
        )
        key = f"+{horizon}d"
        observations[key] = obs
        if obs is not None:
            complete_count += 1

    result["evaluation"] = {
        "status": (
            "COMPLETE"
            if complete_count == len(HORIZONS_DAYS)
            else "PARTIAL"
        ),
        "evaluated_at_utc": _iso_utc(),
        "observations": observations,
    }

    return result


def _atomic_replace(
    path: Path,
    records: list[dict[str, Any]],
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_name, path)
        return True

    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        return False


def _load_file_records(
    file_path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        return []

    return records


def _eligible_snapshot_request(
    record: dict[str, Any],
) -> Optional[tuple[str, str, str]]:
    """
    Return (ticker, start, end) for a snapshot that can request market data.

    This deliberately mirrors the V1.1 validation gates in evaluate_snapshot:
    COMPLETE records and invalid ticker/timestamp/entry records do not trigger
    market-data requests.
    """
    status = record.get("evaluation", {}).get(
        "status", "PENDING"
    )
    if status == "COMPLETE":
        return None

    ticker = _normalize_symbol(record.get("ticker"))
    timestamp = record.get("prediction_timestamp_utc")
    entry = _safe_float(record.get("price_at_prediction"))

    if not ticker or not timestamp or entry is None or entry <= 0:
        return None

    start, end = _prediction_request_window(
        timestamp,
        max(HORIZONS_DAYS),
    )
    return ticker, start, end


def _build_market_data_cache(
    files: list[Path],
) -> dict[str, Any]:
    """
    Build one shared Market Data V2 frame per unique ticker.

    V1.4:
    - Preserves V1.3's persistent 5y/raw/1d cache strategy.
    - Separates cache hits from cache misses before acquisition.
    - Uses Market Data V2's multi-ticker acquisition API for misses so the
      Prediction Tracker has a single acquisition boundary and can reuse all
      returned frames across every prediction snapshot.
    - Keeps exact V1.1 evaluation calculations unchanged.
    - Does not bypass Market Data V2 or write its cache directly.
    """
    windows: dict[str, tuple[str, str]] = {}

    for path in files:
        for record in _load_file_records(path):
            request = _eligible_snapshot_request(record)
            if request is None:
                continue

            ticker, start, end = request
            existing = windows.get(ticker)

            if existing is None:
                windows[ticker] = (start, end)
            else:
                windows[ticker] = (
                    min(existing[0], start),
                    max(existing[1], end),
                )

    cache: dict[str, Any] = {}
    cache_hits = 0
    cache_misses = 0
    batch_calls = 0
    started_total = time.perf_counter()

    print(
        "[PERF-TRACKER] V1.4 cache strategy | "
        "persistent_broad_request=5y | auto_adjust=False | "
        "miss_acquisition=MarketDataV2-multi",
        flush=True,
    )

    # Import lazily so the module remains compatible with environments that
    # only need the single-ticker Market Data V2 API.
    from app.data.market_data import get_multiple_daily_history

    missing: list[str] = []

    # First pass: determine which persistent 5y cache entries already exist.
    # We intentionally use the same get_history contract as V1.3. A cache hit
    # is effectively free; a miss is retained for the multi-ticker acquisition
    # pass below.
    for ticker in sorted(windows):
        start, end = windows[ticker]
        started = time.perf_counter()

        try:
            frame = get_history(
                ticker,
                period="5y",
                start=None,
                end=None,
                interval="1d",
                auto_adjust=False,
                actions=False,
                group_by="column",
                threads=False,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(
                f"[PERF-TRACKER] Market Data PROBE ERROR | "
                f"ticker={ticker} | elapsed={elapsed:.2f}s | "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            cache[ticker] = None
            cache_misses += 1
            missing.append(ticker)
            continue

        elapsed = time.perf_counter() - started

        if frame is not None and not frame.empty:
            frame = frame.sort_index()
            cache[ticker] = frame
            cache_hits += 1
            print(
                f"[PERF-TRACKER] Market Data V1.4 | ticker={ticker} | "
                f"cache=AVAILABLE | broad=5y | rows={len(frame)} | "
                f"elapsed={elapsed:.2f}s",
                flush=True,
            )
        else:
            cache[ticker] = None
            cache_misses += 1
            missing.append(ticker)
            print(
                f"[PERF-TRACKER] Market Data V1.4 | ticker={ticker} | "
                f"cache=MISSING | broad=5y | elapsed={elapsed:.2f}s",
                flush=True,
            )

    # Second pass: ask Market Data V2 to acquire all missing symbols through
    # its multi-ticker API. Market Data V2 owns the actual Yahoo/cache logic.
    if missing:
        batch_calls = 1
        batch_started = time.perf_counter()

        try:
            acquired = get_multiple_daily_history(
                missing,
                period="5y",
                auto_adjust=False,
                actions=False,
                group_by="column",
            )
        except TypeError:
            # V2's public get_multiple_daily_history contract does not expose
            # threads; retry without that optional argument.
            acquired = get_multiple_daily_history(
                missing,
                period="5y",
                auto_adjust=False,
                actions=False,
                group_by="column",
            )
        except Exception as exc:
            print(
                f"[PERF-TRACKER] Market Data V1.4 BATCH ERROR | "
                f"tickers={len(missing)} | "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            acquired = {}

        batch_elapsed = time.perf_counter() - batch_started

        for ticker in missing:
            frame = acquired.get(ticker)
            if frame is not None and not frame.empty:
                cache[ticker] = frame.sort_index()
            else:
                cache[ticker] = None

        print(
            f"[PERF-TRACKER] Market Data V1.4 BATCH | "
            f"tickers={len(missing)} | calls={batch_calls} | "
            f"elapsed={batch_elapsed:.2f}s",
            flush=True,
        )

    elapsed_total = time.perf_counter() - started_total
    print(
        f"[PERF-TRACKER] Market Data requests={len(windows)} | "
        f"unique_tickers={len(windows)} | cache_hits={cache_hits} | "
        f"cache_misses={cache_misses} | yahoo_batches={batch_calls} | "
        f"elapsed={elapsed_total:.2f}s",
        flush=True,
    )

    return cache

def update_prediction_file(
    path: str | Path,
    market_data_cache: Optional[dict[str, Any]] = None,
) -> tuple[int, int]:
    file_path = Path(path)
    if not file_path.exists():
        return 0, 0

    records = _load_file_records(file_path)
    updated = 0
    unchanged = 0

    for record in records:
        before = json.dumps(
            record,
            sort_keys=True,
            default=str,
        )

        ticker = _normalize_symbol(record.get("ticker"))
        frame = (
            market_data_cache.get(ticker)
            if market_data_cache is not None
            else None
        )

        evaluated = evaluate_snapshot(
            record,
            market_data=frame,
        )

        after = json.dumps(
            evaluated,
            sort_keys=True,
            default=str,
        )

        if before != after:
            updated += 1
        else:
            unchanged += 1

        record.clear()
        record.update(evaluated)

    if not _atomic_replace(file_path, records):
        return 0, len(records)

    return updated, unchanged


def update_all_predictions(
    base_dir: str | Path = DEFAULT_DIRECTORY,
    *,
    include_today: bool = False,
) -> dict[str, int]:
    """
    Evaluate eligible historical prediction files.

    Public V1.1 contract preserved:
        include_today=False:
            Exclude today's prediction file.
        include_today=True:
            Evaluate all prediction files.

    Return contract preserved:
        {"files": int, "updated": int, "unchanged": int}
    """
    started = time.perf_counter()

    files = list(_iter_files(base_dir))

    if not include_today:
        files = [
            path for path in files
            if not _is_today_file(path)
        ]

    # Critical V1.4 optimization:
    # build the shared market-data cache before evaluating files.
    market_data_cache = _build_market_data_cache(files)

    updated = 0
    unchanged = 0

    for path in files:
        file_started = time.perf_counter()

        u, n = update_prediction_file(
            path,
            market_data_cache=market_data_cache,
        )
        updated += u
        unchanged += n

        elapsed = time.perf_counter() - file_started
        print(
            f"[PERF-TRACKER] File | file={path.name} | "
            f"updated={u} | unchanged={n} | elapsed={elapsed:.2f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    print(
        f"[PERF-TRACKER] TOTAL | files={len(files)} | "
        f"updated={updated} | unchanged={unchanged} | "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )

    return {
        "files": len(files),
        "updated": updated,
        "unchanged": unchanged,
    }


def _self_test() -> int:
    """
    Deterministic contract + calculation test.

    Uses an in-memory synthetic DataFrame and does not contact Yahoo.
    """
    import tempfile as _tempfile
    import pandas as pd

    index = pd.date_range(
        "2026-01-02",
        periods=70,
        freq="B",
    )

    frame = pd.DataFrame(
        {
            "Open": [100.0] * 70,
            "High": [101.0 + i * 0.1 for i in range(70)],
            "Low": [99.0 + i * 0.1 for i in range(70)],
            "Close": [100.0 + i * 0.1 for i in range(70)],
            "Adj Close": [100.0 + i * 0.1 for i in range(70)],
            "Volume": [1000] * 70,
        },
        index=index,
    )

    sample = {
        "ticker": "TEST",
        "prediction_timestamp_utc":
            "2026-01-02T15:00:00+00:00",
        "price_at_prediction": 100.0,
        "final_signal": "BUY",
        "levels": {
            "target": 101.0,
            "stop": 99.0,
        },
        "evaluation": {
            "status": "PENDING",
            "observations": {},
        },
    }

    evaluated = evaluate_snapshot(
        sample,
        market_data=frame,
    )

    evaluation = evaluated.get("evaluation", {})
    observations = evaluation.get("observations", {})

    if evaluation.get("status") != "COMPLETE":
        print("SELF-TEST FAIL: expected COMPLETE evaluation")
        return 1

    if set(observations) != {
        "+1d",
        "+3d",
        "+5d",
        "+10d",
        "+20d",
        "+60d",
    }:
        print("SELF-TEST FAIL: horizon contract changed")
        return 1

    one_day = observations["+1d"]

    # V1.4 cache-window slicing test.
    # The shared cache builder remains responsible for acquisition.

    sliced = _slice_market_window(
        frame,
        "2026-01-02",
        "2026-01-20",
    )
    if sliced is None or len(sliced) != 12:
        print(
            "SELF-TEST FAIL: V1.3 cache-window slicing failed: "
            f"{0 if sliced is None else len(sliced)}"
        )
        return 1

    # Exact V1.1 calculation:
    # close at T+1 = 100.1 => return = 0.1%
    if one_day["actual_price"] != 100.1:
        print(
            "SELF-TEST FAIL: actual_price calculation changed: "
            f"{one_day['actual_price']}"
        )
        return 1

    if one_day["return_pct"] != 0.1:
        print(
            "SELF-TEST FAIL: return_pct calculation changed: "
            f"{one_day['return_pct']}"
        )
        return 1

    if one_day["correctness"] is not True:
        print("SELF-TEST FAIL: correctness calculation changed")
        return 1

    # Public file/update contract test.
    with _tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        today_path = prediction_file(base_dir=root)
        old_path = prediction_file(
            "2026-01-02",
            base_dir=root,
        )

        today_path.write_text(
            json.dumps(sample) + "\n",
            encoding="utf-8",
        )
        old_path.write_text(
            json.dumps(sample) + "\n",
            encoding="utf-8",
        )

        # Prevent the contract test from contacting Yahoo.
        original_get_history = get_history

        def _fake_get_history(*args, **kwargs):
            return frame.copy(deep=True)

        globals()["get_history"] = _fake_get_history

        excluded = update_all_predictions(
            root,
            include_today=False,
        )
        if excluded["files"] != 1:
            print(
                "SELF-TEST FAIL: expected 1 eligible file, "
                f"got {excluded['files']}"
            )
            return 1

        included = update_all_predictions(
            root,
            include_today=True,
        )

        globals()["get_history"] = original_get_history

        if included["files"] != 2:
            print(
                "SELF-TEST FAIL: expected 2 files, "
                f"got {included['files']}"
            )
            return 1

    print(f"{MODULE_NAME} V{VERSION} SELF-TEST: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=MODULE_NAME
    )
    parser.add_argument(
        "--directory",
        default=str(DEFAULT_DIRECTORY),
        help="Prediction JSONL directory",
    )
    parser.add_argument(
        "--file",
        help="Evaluate one prediction JSONL file",
    )
    parser.add_argument(
        "--include-today",
        action="store_true",
        help="Include today's prediction file",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic interface/calculation self-test",
    )

    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if args.file:
        updated, unchanged = update_prediction_file(
            args.file,
        )
        print(f"{MODULE_NAME} V{VERSION}")
        print(f"File      : {args.file}")
        print(f"Updated   : {updated}")
        print(f"Unchanged : {unchanged}")
        return 0

    summary = update_all_predictions(
        args.directory,
        include_today=args.include_today,
    )

    print(f"{MODULE_NAME} V{VERSION}")
    print(f"Directory : {args.directory}")
    print(f"Files     : {summary['files']}")
    print(f"Updated   : {summary['updated']}")
    print(f"Unchanged : {summary['unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
