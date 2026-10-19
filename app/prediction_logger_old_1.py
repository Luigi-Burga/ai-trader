"""
AI Trader - Prediction Logger V1.1

Purpose
-------
Persist every prediction snapshot produced by the trading pipeline so that
it can later be evaluated against real market prices.

Application code lives under app/.
Persistent prediction data remains outside app/ under:
    data/predictions/

The logger is independent from CCE, NLE, Decision Gate, Signal Engine
and Watchlist Scanner. It records the state at T0 only.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


VERSION = "1.1"
MODULE_NAME = "Prediction Logger"

# app/prediction_logger.py -> project root -> data/predictions
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = PROJECT_ROOT / "data" / "predictions"

HORIZONS_DAYS = (1, 3, 5, 10, 20, 60)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: Optional[datetime] = None) -> str:
    dt = value or _utc_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _normalize_symbol(symbol: Any) -> str:
    return str(symbol).strip().upper()


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _clean_value(value: Any) -> Any:
    """Convert common Python/numpy-like values into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {str(k): _clean_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_clean_value(v) for v in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _clean_value(item())
        except Exception:
            pass

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass

    return str(value)


def _default_directory(base_dir: Optional[str | Path] = None) -> Path:
    return Path(base_dir) if base_dir is not None else DEFAULT_DIRECTORY


def prediction_file(
    prediction_date: Optional[str] = None,
    *,
    base_dir: Optional[str | Path] = None,
) -> Path:
    """
    Return the daily JSONL file for a YYYY-MM-DD date.

    If prediction_date is omitted, today's UTC date is used.
    """
    if prediction_date is None:
        prediction_date = _utc_now().date().isoformat()

    directory = _default_directory(base_dir)
    return directory / f"predictions_{prediction_date}.jsonl"


def _extract_levels(result: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [
        result.get("levels"),
        result.get("dynamic_levels"),
        result.get("risk"),
        result.get("targets"),
    ]

    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return {
                "entry": _safe_float(
                    candidate.get("entry", candidate.get("entry_price"))
                ),
                "stop": _safe_float(
                    candidate.get("stop", candidate.get("stop_price"))
                ),
                "target_1": _safe_float(
                    candidate.get("target_1", candidate.get("target1"))
                ),
                "target_2": _safe_float(
                    candidate.get("target_2", candidate.get("target2"))
                ),
                "target_3": _safe_float(
                    candidate.get("target_3", candidate.get("target3"))
                ),
            }

    return {
        "entry": _safe_float(
            result.get("entry", result.get("entry_price"))
        ),
        "stop": _safe_float(
            result.get("stop", result.get("stop_price"))
        ),
        "target_1": _safe_float(
            result.get("target_1", result.get("target1"))
        ),
        "target_2": _safe_float(
            result.get("target_2", result.get("target2"))
        ),
        "target_3": _safe_float(
            result.get("target_3", result.get("target3"))
        ),
    }


def build_snapshot(
    ticker: str,
    result: Mapping[str, Any],
    *,
    prediction_timestamp: Optional[datetime] = None,
    source: str = "watchlist_scanner",
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    result = dict(result)

    engine_result = result.get("engine_result")
    if not isinstance(engine_result, Mapping):
        engine_result = {}

    engine_summary = result.get("engine_summary")
    if not isinstance(engine_summary, Mapping):
        engine_summary = {}

    def pick(name: str, *fallback_names: str) -> Any:
        for container in (result, engine_summary, engine_result):
            for key in (name, *fallback_names):
                value = container.get(key)
                if value is not None:
                    return value
        return None

    current_price = _safe_float(
        pick("current_price", "price", "current")
    )

    # Explicit signal semantics:
    # raw_signal = pre-gate scanner/engine signal
    # gated_signal = Decision Gate output
    # final_signal = authoritative final scanner signal
    raw_signal = pick("raw_signal")
    if raw_signal is None:
        raw_signal = result.get("signal")

    gated_signal = pick("gated_signal")

    final_signal = pick("final_signal")
    if final_signal is None:
        final_signal = result.get("signal")

    signal = final_signal
    action = pick("action", "decision")
    score = _safe_float(pick("score", "characteristic_score"))
    confidence = _safe_float(pick("confidence", "trading_confidence"))

    benchmark = pick("benchmark", "benchmark_symbol")
    benchmark_trust = pick("benchmark_trust", "trust")
    benchmark_grade = pick(
        "benchmark_trust_decision_grade",
        "benchmark_decision_grade",
    )
    regime = pick("regime", "cycle")
    route = pick("route")
    engine = pick("engine")
    engine_version = pick("engine_version", "version")

    age = result.get("asset_age")
    if not isinstance(age, Mapping):
        age = {}

    levels = _extract_levels(engine_result)
    if all(value is None for value in levels.values()):
        levels = _extract_levels(result)

    timestamp = _iso_utc(prediction_timestamp)

    snapshot = {
        "schema": "ai_trader_prediction_snapshot",
        "schema_version": "1.0",
        "logger_version": VERSION,

        "prediction_id": (
            f"{_normalize_symbol(ticker)}_"
            f"{timestamp.replace(':', '').replace('+00:00', 'Z')}"
        ),
        "run_id": run_id,
        "source": source,

        "prediction_timestamp_utc": timestamp,
        "prediction_date_utc": timestamp[:10],

        "ticker": _normalize_symbol(ticker),
        "price_at_prediction": current_price,

        "route": route,
        "engine": engine,
        "engine_version": engine_version,

        "raw_signal": raw_signal,
        "gated_signal": gated_signal,
        "signal": signal,
        "final_signal": signal,
        "action": action,

        "score": score,
        "confidence": confidence,
        "regime": regime,

        "benchmark": benchmark,
        "benchmark_trust": benchmark_trust,
        "benchmark_trust_decision_grade": benchmark_grade,

        "asset_age_days": age.get("listing_days", age.get("days")),
        "asset_age_route": age.get("route"),

        "levels": levels,

        "evaluation_horizons_days": list(HORIZONS_DAYS),

        "evaluation": {
            "status": "PENDING",
            "observations": {},
        },
    }

    return _clean_value(snapshot)


def append_snapshot(
    snapshot: Mapping[str, Any],
    *,
    base_dir: Optional[str | Path] = None,
    flush: bool = True,
) -> Path:
    ticker = _normalize_symbol(snapshot.get("ticker"))
    if not ticker:
        raise ValueError("snapshot must contain a non-empty ticker")

    date_value = str(
        snapshot.get("prediction_date_utc")
        or _utc_now().date().isoformat()
    )
    path = prediction_file(date_value, base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(
        _clean_value(dict(snapshot)),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )

    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        if flush:
            handle.flush()
            os.fsync(handle.fileno())

    return path


def log_prediction(
    ticker: str,
    result: Mapping[str, Any],
    *,
    prediction_timestamp: Optional[datetime] = None,
    source: str = "watchlist_scanner",
    run_id: Optional[str] = None,
    base_dir: Optional[str | Path] = None,
) -> tuple[dict[str, Any], Path]:
    snapshot = build_snapshot(
        ticker,
        result,
        prediction_timestamp=prediction_timestamp,
        source=source,
        run_id=run_id,
    )
    path = append_snapshot(snapshot, base_dir=base_dir)
    return snapshot, path


def read_predictions(
    prediction_date: Optional[str] = None,
    *,
    base_dir: Optional[str | Path] = None,
) -> list[dict[str, Any]]:
    path = prediction_file(prediction_date, base_dir=base_dir)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Prediction line {line_number} is not a JSON object"
                )
            records.append(value)

    return records


def _demo_result() -> dict[str, Any]:
    return {
        "route": "MATURE",
        "engine": "CHARACTERISTIC_CURVE_ENGINE V1.6.1",
        "engine_version": "1.6.1",
        "raw_signal": "WATCH",
        "signal": "WATCH",
        "action": "WATCH",
        "score": 66.11,
        "confidence": 53.41,
        "current_price": 71.38,
        "regime": "CONSOLIDATION",
        "benchmark": "QQQ",
        "benchmark_trust": "HIGH",
        "benchmark_trust_decision_grade": True,
        "asset_age": {
            "listing_days": 1254,
            "route": "ROUTE_TO_CCE",
        },
        "engine_result": {
            "dynamic_levels": {
                "entry": 65.0,
                "stop": 61.0,
                "target_1": 75.0,
                "target_2": 80.0,
            }
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI Trader Prediction Logger V1.1"
    )
    parser.add_argument("--ticker", default="TQQQ")
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_DIRECTORY),
        help="Prediction log directory.",
    )
    args = parser.parse_args()

    snapshot, path = log_prediction(
        args.ticker,
        _demo_result(),
        base_dir=args.base_dir,
        source="prediction_logger_self_test",
        run_id="SELFTEST",
    )

    print("=" * 70)
    print(f"{MODULE_NAME} V{VERSION}")
    print("=" * 70)
    print(f"File       : {path.resolve()}")
    print(f"Ticker     : {snapshot['ticker']}")
    print(f"Raw Signal : {snapshot['raw_signal']}")
    print(f"Final      : {snapshot['final_signal']}")
    print(f"Price      : {snapshot['price_at_prediction']}")
    print(f"Benchmark  : {snapshot['benchmark']}")
    print(f"Horizons   : {snapshot['evaluation_horizons_days']}")
    print("RESULT     : PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
