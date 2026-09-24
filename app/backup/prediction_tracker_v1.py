"""
AI Trader - Prediction Tracker V1.1

Evaluates historical prediction snapshots stored by Prediction Logger.

V1.1:
- update_all_predictions(..., include_today=False) excludes today's file
  by default so current T0 predictions remain PENDING.
- include_today=True restores processing of today's file.

Supported horizons:
    +1, +3, +5, +10, +20, +60 trading days
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

VERSION = "1.1"
MODULE_NAME = "Prediction Tracker"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = PROJECT_ROOT / "data" / "predictions"

HORIZONS_DAYS = (1, 3, 5, 10, 20, 60)


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
        day = date.today()
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
    return path.name == f"predictions_{date.today().isoformat()}.jsonl"


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


def _trading_days_since(
    prediction_timestamp: str,
    ticker: str,
    max_horizon: int = 60,
):
    import yfinance as yf

    ts = datetime.fromisoformat(
        prediction_timestamp.replace("Z", "+00:00")
    )
    start = ts.date()
    end = start.fromordinal(
        start.toordinal() + max_horizon * 3 + 15
    )

    data = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=False,
        actions=False,
    )

    if data is None or data.empty:
        return None

    if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
        if ticker in data.columns.get_level_values(-1):
            try:
                data = data.xs(ticker, axis=1, level=-1)
            except Exception:
                pass
        if (
            hasattr(data.columns, "nlevels")
            and data.columns.nlevels > 1
        ):
            data.columns = data.columns.get_level_values(0)

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
            timestamp, ticker, max(HORIZONS_DAYS)
        )
    )

    t0_index = _find_prediction_session(
        frame, timestamp, entry
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


def update_prediction_file(
    path: str | Path,
) -> tuple[int, int]:
    file_path = Path(path)
    if not file_path.exists():
        return 0, 0

    records: list[dict[str, Any]] = []
    updated = 0
    unchanged = 0

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(record, dict):
                continue

            before = json.dumps(
                record,
                sort_keys=True,
                default=str,
            )

            evaluated = evaluate_snapshot(record)

            after = json.dumps(
                evaluated,
                sort_keys=True,
                default=str,
            )

            if before != after:
                updated += 1
            else:
                unchanged += 1

            records.append(evaluated)

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

    include_today=False:
        Exclude today's prediction file so current T0 snapshots remain
        PENDING during the same scan.

    include_today=True:
        Evaluate all prediction files, including today's file.
    """
    files = list(_iter_files(base_dir))

    if not include_today:
        files = [
            path for path in files
            if not _is_today_file(path)
        ]

    updated = 0
    unchanged = 0

    for path in files:
        u, n = update_prediction_file(path)
        updated += u
        unchanged += n

    return {
        "files": len(files),
        "updated": updated,
        "unchanged": unchanged,
    }


def _self_test() -> int:
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        today_path = prediction_file(base_dir=root)
        old_path = prediction_file(
            "2026-01-02",
            base_dir=root,
        )

        sample = {
            "ticker": "TEST",
            "prediction_timestamp_utc":
                "2026-01-02T15:00:00+00:00",
            "price_at_prediction": 100.0,
            "final_signal": "BUY",
            "evaluation": {
                "status": "PENDING",
                "observations": {},
            },
        }

        today_path.write_text(
            json.dumps(sample) + "\n",
            encoding="utf-8",
        )
        old_path.write_text(
            json.dumps(sample) + "\n",
            encoding="utf-8",
        )

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
        help="Run deterministic interface self-test",
    )

    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if args.file:
        updated, unchanged = update_prediction_file(
            args.file
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
