"""
AI Trader - Decision Audit V1.3
Research-only audit of historical prediction JSONL decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


VERSION = "1.3"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "data" / "predictions"
DEFAULT_OUTPUT = PROJECT_ROOT / "decision_audit_v1_3_results.json"

BUY_SIGNALS = {"BUY", "STRONG_BUY"}
CONFIRMATION_SIGNAL = "BUY_ON_CONFIRMATION"

ALIASES = {
    "symbol": ["symbol", "ticker"],
    "raw_signal": ["raw_signal", "signal", "raw_decision"],
    "final_signal": ["final_signal", "gated_signal", "decision"],
    "gated_signal": ["gated_signal"],
    "score": ["score", "characteristic_score", "signal_score"],
    "confidence": ["confidence", "confidence_score"],
    "appt": ["appt", "APPT", "appt_x", "APPT_X", "expected_value"],
    "price": ["price_at_prediction", "current_price", "price", "entry_price"],
    "timestamp": ["prediction_timestamp_utc", "timestamp", "prediction_timestamp", "as_of", "datetime", "date", "created_at"],
    "reason": ["reason", "gate_reason", "decision_reason", "final_reason", "gating_reason"],
    "benchmark": ["benchmark", "selected_benchmark"],
    "regime": ["regime", "cycle"],
    "engine": ["engine", "strategy", "source"],
}


class AuditError(RuntimeError):
    pass


def first(record: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        value = record.get(name)
        if value not in (None, ""):
            return value
    return None


def number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def signal(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    for parser in (
        lambda x: datetime.fromisoformat(x),
        lambda x: datetime.strptime(x, "%Y-%m-%d"),
        lambda x: datetime.strptime(x, "%Y/%m/%d"),
    ):
        try:
            dt = parser(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_files(directory: Path, explicit: list[str] | None, days: int | None) -> list[Path]:
    if explicit:
        paths = []
        for item in explicit:
            path = Path(item)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if not path.exists():
                raise FileNotFoundError(path)
            paths.append(path)
        return paths

    if not directory.exists():
        raise FileNotFoundError(f"Prediction directory not found: {directory}")

    paths = sorted(directory.glob("predictions_*.jsonl"))
    if days is None:
        return paths

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, days))
    selected = []
    for path in paths:
        try:
            date_value = datetime.strptime(
                path.stem.replace("predictions_", "")[:10],
                "%Y-%m-%d",
            ).replace(tzinfo=timezone.utc)
            if date_value >= cutoff:
                selected.append(path)
        except ValueError:
            selected.append(path)
    return selected


def load_records(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    metadata = []

    for path in paths:
        count = 0
        errors = 0
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError("JSON record is not an object")
                except Exception:
                    errors += 1
                    continue

                obj["_audit_file"] = str(path)
                obj["_audit_line"] = line_no
                records.append(obj)
                count += 1

        metadata.append({
            "file": str(path),
            "sha256": sha256(path),
            "records": count,
            "parse_errors": errors,
        })

    return records, metadata


def normalize(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": first(record, ALIASES["symbol"]),
        "raw_signal": signal(first(record, ALIASES["raw_signal"])),
        "gated_signal": signal(first(record, ALIASES["gated_signal"])),
        "final_signal": signal(first(record, ALIASES["final_signal"])),
        "score": number(first(record, ALIASES["score"])),
        "confidence": number(first(record, ALIASES["confidence"])),
        "appt": number(first(record, ALIASES["appt"])),
        "price": number(first(record, ALIASES["price"])),
        "timestamp": timestamp(first(record, ALIASES["timestamp"])),
        "reason": first(record, ALIASES["reason"]),
        "benchmark": first(record, ALIASES["benchmark"]),
        "regime": first(record, ALIASES["regime"]),
        "engine": first(record, ALIASES["engine"]),
        "source_file": record["_audit_file"],
        "source_line": record["_audit_line"],
        "outcomes": {},
    }


def classify(row: dict[str, Any]) -> str:
    raw = row["raw_signal"]
    final = row["final_signal"]
    if raw in BUY_SIGNALS and final in BUY_SIGNALS:
        return "BUY_EXECUTABLE"
    if raw in BUY_SIGNALS and final == CONFIRMATION_SIGNAL:
        return "BUY_ON_CONFIRMATION"
    if raw in BUY_SIGNALS:
        return "BUY_DOWNGRADED_OTHER"
    if raw == CONFIRMATION_SIGNAL:
        return "RAW_BUY_ON_CONFIRMATION"
    if raw is None:
        return "UNKNOWN"
    return "NON_BUY"


def load_market_data(symbols: list[str]) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        from app.data.market_data import get_daily_history
    except Exception as exc:
        return {}, {"IMPORT": f"{type(exc).__name__}: {exc}"}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    data = {}
    errors = {}

    def fetch(symbol: str):
        return symbol, get_daily_history(
            symbol,
            period="5y",
            auto_adjust=False,
            actions=False,
            group_by="column",
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, s): s for s in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                _, frame = future.result()
                data[symbol] = frame
            except Exception as exc:
                errors[symbol] = f"{type(exc).__name__}: {exc}"

    return data, errors


def close_series(frame):
    import pandas as pd

    if frame is None or getattr(frame, "empty", True):
        return None

    frame = frame.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        candidates = [
            col for col in frame.columns
            if "close" in [str(part).strip().lower() for part in col]
        ]
        if not candidates:
            return None
        series = frame[candidates[0]]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
    else:
        col = next(
            (c for c in frame.columns if str(c).strip().lower() == "close"),
            None,
        )
        if col is None:
            return None
        series = frame[col]

    return pd.to_numeric(series, errors="coerce").dropna()


def session_position(series, when: datetime | None) -> int | None:
    """Position of the first available market session on/after prediction date."""
    if series is None or when is None:
        return None
    target_date = when.date()
    for pos, idx in enumerate(series.index):
        try:
            idx_date = idx.date()
        except AttributeError:
            try:
                idx_date = idx.to_pydatetime().date()
            except Exception:
                continue
        if idx_date >= target_date:
            return pos
    return None


def future_close(series, when: datetime | None, horizon: int) -> float | None:
    """Return close exactly N trading sessions after the prediction session."""
    if series is None:
        return None
    pos = session_position(series, when)
    if pos is None:
        return None
    future_pos = pos + horizon
    if future_pos >= len(series):
        return None
    try:
        value = series.iloc[future_pos]
        if hasattr(value, "iloc"):
            value = value.iloc[0]
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, IndexError):
        return None


def close_at_session(series, when: datetime | None) -> float | None:
    pos = session_position(series, when)
    if series is None or pos is None:
        return None
    try:
        value = float(series.iloc[pos])
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, IndexError):
        return None


def add_outcomes(rows: list[dict[str, Any]], histories: dict[str, Any], horizons: list[int]) -> None:
    """Add asset returns and same-window benchmark-relative returns."""
    series_cache = {symbol: close_series(frame) for symbol, frame in histories.items()}
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        asset_series = series_cache.get(symbol)
        entry = row.get("price")
        benchmark_symbol = str(row.get("benchmark") or "").strip().upper()
        benchmark_series = series_cache.get(benchmark_symbol)
        benchmark_entry = close_at_session(benchmark_series, row.get("timestamp"))

        for horizon in horizons:
            future = future_close(asset_series, row.get("timestamp"), horizon)
            ret = None
            if entry is not None and future is not None and entry > 0:
                ret = (future / entry - 1.0) * 100.0

            benchmark_future = future_close(benchmark_series, row.get("timestamp"), horizon)
            benchmark_return = None
            if benchmark_entry is not None and benchmark_future is not None and benchmark_entry > 0:
                benchmark_return = (benchmark_future / benchmark_entry - 1.0) * 100.0

            row["outcomes"][f"{horizon}d"] = {
                "future_close": future,
                "return_pct": ret,
                "positive": None if ret is None else ret > 0,
                "benchmark_symbol": benchmark_symbol or None,
                "benchmark_return_pct": benchmark_return,
                "excess_return_pct": (
                    ret - benchmark_return
                    if ret is not None and benchmark_return is not None else None
                ),
            }


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile, without a numpy dependency."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def robust_stats(values: list[float]) -> dict[str, Any]:
    """Robust distribution metrics; all values are percentages where applicable."""
    if not values:
        return {
            "N": 0, "mean": None, "median": None, "stddev_population": None,
            "min": None, "p10": None, "p25": None, "p75": None, "p90": None,
            "max": None, "trimmed_mean_5pct": None, "winsorized_mean_5pct": None,
            "positive_pct": None,
        }
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return robust_stats([])
    n = len(clean)
    trim = int(n * 0.05)
    trimmed = clean[trim:n-trim] if trim and n - 2 * trim > 0 else clean
    p05, p95 = percentile(clean, 0.05), percentile(clean, 0.95)
    winsorized = [min(max(v, p05), p95) for v in clean]
    return {
        "N": n,
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "stddev_population": statistics.pstdev(clean) if n > 1 else 0.0,
        "min": clean[0],
        "p10": percentile(clean, 0.10),
        "p25": percentile(clean, 0.25),
        "p75": percentile(clean, 0.75),
        "p90": percentile(clean, 0.90),
        "max": clean[-1],
        "trimmed_mean_5pct": statistics.fmean(trimmed),
        "winsorized_mean_5pct": statistics.fmean(winsorized),
        "positive_pct": sum(v > 0 for v in clean) / n * 100.0,
    }


def group_stats(rows: list[dict[str, Any]], horizons: list[int]) -> dict[str, Any]:
    result = {
        "N": len(rows),
        "unique_symbols_N": len({str(r.get("symbol") or "").strip().upper() for r in rows if r.get("symbol")}),
        "confidence_mean": None,
        "score_mean": None,
        "appt_mean": None,
    }

    for field in ("confidence", "score", "appt"):
        values = [r[field] for r in rows if r[field] is not None]
        result[f"{field}_N"] = len(values)
        result[f"{field}_mean"] = sum(values) / len(values) if values else None
        result[f"{field}_median"] = statistics.median(values) if values else None

    for horizon in horizons:
        key = f"{horizon}d"
        outcomes = [r["outcomes"].get(key, {}) for r in rows]
        returns = [o["return_pct"] for o in outcomes if o.get("return_pct") is not None]
        benchmark_values = [o["benchmark_return_pct"] for o in outcomes if o.get("benchmark_return_pct") is not None]
        excess_values = [o["excess_return_pct"] for o in outcomes if o.get("excess_return_pct") is not None]
        asset_stats = robust_stats(returns)
        benchmark_stats = robust_stats(benchmark_values)
        excess_stats = robust_stats(excess_values)
        result[f"{key}_N"] = asset_stats["N"]
        result[f"{key}_mean_return_pct"] = asset_stats["mean"]
        result[f"{key}_positive_pct"] = asset_stats["positive_pct"]
        result[f"{key}_median_return_pct"] = asset_stats["median"]
        result[f"{key}_stddev_return_pct"] = asset_stats["stddev_population"]
        result[f"{key}_p10_return_pct"] = asset_stats["p10"]
        result[f"{key}_p90_return_pct"] = asset_stats["p90"]
        result[f"{key}_trimmed_mean_5pct_return_pct"] = asset_stats["trimmed_mean_5pct"]
        result[f"{key}_winsorized_mean_5pct_return_pct"] = asset_stats["winsorized_mean_5pct"]
        result[f"{key}_min_return_pct"] = asset_stats["min"]
        result[f"{key}_max_return_pct"] = asset_stats["max"]
        result[f"{key}_benchmark_N"] = benchmark_stats["N"]
        result[f"{key}_mean_benchmark_return_pct"] = benchmark_stats["mean"]
        result[f"{key}_median_benchmark_return_pct"] = benchmark_stats["median"]
        result[f"{key}_excess_N"] = excess_stats["N"]
        result[f"{key}_mean_excess_return_pct"] = excess_stats["mean"]
        result[f"{key}_median_excess_return_pct"] = excess_stats["median"]
        result[f"{key}_winsorized_mean_excess_return_pct"] = excess_stats["winsorized_mean_5pct"]
    return result

def aggregate(rows: list[dict[str, Any]], horizons: list[int]) -> dict[str, Any]:
    groups = {
        "all": rows,
        "raw_buy": [r for r in rows if r["raw_signal"] in BUY_SIGNALS],
        "raw_buy_only": [r for r in rows if r["raw_signal"] == "BUY"],
        "raw_strong_buy": [r for r in rows if r["raw_signal"] == "STRONG_BUY"],
        "final_buy": [r for r in rows if r["final_signal"] in BUY_SIGNALS],
        "buy_on_confirmation": [
            r for r in rows if r["final_signal"] == CONFIRMATION_SIGNAL
        ],
        "raw_buy_to_confirmation": [
            r for r in rows
            if r["raw_signal"] in BUY_SIGNALS
            and r["final_signal"] == CONFIRMATION_SIGNAL
        ],
        "raw_buy_to_other": [
            r for r in rows
            if r["raw_signal"] in BUY_SIGNALS
            and r["final_signal"] not in BUY_SIGNALS
            and r["final_signal"] != CONFIRMATION_SIGNAL
        ],
    }
    return {name: group_stats(items, horizons) for name, items in groups.items()}


def downgrade_reasons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        r for r in rows
        if r["raw_signal"] in BUY_SIGNALS
        and r["final_signal"] == CONFIRMATION_SIGNAL
    ]

    counts = Counter()
    samples = defaultdict(list)

    for row in selected:
        reason = str(row["reason"]).strip() if row["reason"] else "<reason unavailable>"
        counts[reason] += 1
        if len(samples[reason]) < 5:
            samples[reason].append({
                "symbol": row["symbol"],
                "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                "score": row["score"],
                "confidence": row["confidence"],
                "appt": row["appt"],
            })

    return {
        "count": len(selected),
        "reasons": [
            {
                "reason": reason,
                "count": count,
                "samples": samples[reason],
            }
            for reason, count in counts.most_common()
        ],
    }


def decision_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix = []
    for row in rows:
        if row["raw_signal"] not in BUY_SIGNALS:
            continue

        matrix.append({
            "symbol": row["symbol"],
            "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
            "raw_signal": row["raw_signal"],
            "final_signal": row["final_signal"],
            "classification": classify(row),
            "score": row["score"],
            "confidence": row["confidence"],
            "appt": row["appt"],
            "price": row["price"],
            "reason": row["reason"],
            "benchmark": row["benchmark"],
            "regime": row["regime"],
            "engine": row["engine"],
            "source_file": row["source_file"],
            "source_line": row["source_line"],
            "outcomes": row["outcomes"],
        })
    return matrix


def grouped_analysis(rows: list[dict[str, Any]], field: str, horizons: list[int]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        key = str(value).strip().upper() if value not in (None, "") else "<UNKNOWN>"
        buckets[key].append(row)
    return {
        key: group_stats(items, horizons)
        for key, items in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0]))
    }


def concentration_analysis(rows: list[dict[str, Any]], horizons: list[int]) -> dict[str, Any]:
    counts = Counter(str(r.get("symbol") or "<UNKNOWN>").strip().upper() for r in rows)
    horizon_data = {}
    for horizon in horizons:
        key = f"{horizon}d"
        eligible = [r for r in rows if r["outcomes"].get(key, {}).get("return_pct") is not None]
        by_symbol: dict[str, list[float]] = defaultdict(list)
        for row in eligible:
            by_symbol[str(row.get("symbol") or "<UNKNOWN>").strip().upper()].append(row["outcomes"][key]["return_pct"])
        symbol_means = {sym: statistics.fmean(vals) for sym, vals in by_symbol.items() if vals}
        horizon_data[key] = {
            "evaluated_records_N": len(eligible),
            "unique_symbols_N": len(by_symbol),
            "top_symbols_by_record_count": [{"symbol": sym, "records": n} for sym, n in counts.most_common(10)],
            "top_symbols_by_absolute_mean_return": [
                {"symbol": sym, "mean_return_pct": val, "N": len(by_symbol[sym])}
                for sym, val in sorted(symbol_means.items(), key=lambda item: abs(item[1]), reverse=True)[:10]
            ],
        }
    return {"unique_symbols_N": len(counts), "top_symbols_by_record_count": [{"symbol": s, "records": n} for s, n in counts.most_common(20)], "by_horizon": horizon_data}


def coverage_diagnostics(rows: list[dict[str, Any]], horizons: list[int], histories: dict[str, Any] | None = None) -> dict[str, Any]:
    total = len(rows)
    missing = Counter()
    for row in rows:
        if not row.get("symbol"):
            missing["missing_symbol"] += 1
        if row.get("timestamp") is None:
            missing["missing_or_unparseable_timestamp"] += 1
        if row.get("price") is None or row.get("price", 0) <= 0:
            missing["missing_or_invalid_entry_price"] += 1
        if not row.get("benchmark"):
            missing["missing_benchmark"] += 1
    by_horizon = {}
    for horizon in horizons:
        key = f"{horizon}d"
        n = sum(row["outcomes"].get(key, {}).get("return_pct") is not None for row in rows)
        by_horizon[key] = {"evaluated_N": n, "unevaluable_N": total - n, "coverage_pct": n / total * 100 if total else None}
    return {
        "records_N": total,
        "field_issues_record_counts": dict(missing),
        "horizon_coverage": by_horizon,
        "history_symbols_loaded_N": len(histories or {}),
    }


def audit_outliers(rows: list[dict[str, Any]], horizons: list[int], absolute_threshold_pct: float = 100.0) -> dict[str, Any]:
    """Flag unusual returns for review; never silently remove them from statistics."""
    observations = []
    by_horizon = {}
    for horizon in horizons:
        key = f"{horizon}d"
        values = [float(outcome["return_pct"]) for row in rows
                  for outcome in [row.get("outcomes", {}).get(key, {})]
                  if number(outcome.get("return_pct")) is not None]
        if not values:
            by_horizon[key] = {"evaluated_N": 0, "absolute_threshold_pct": absolute_threshold_pct,
                               "iqr_lower_fence_pct": None, "iqr_upper_fence_pct": None, "flagged_N": 0}
            continue
        q1, q3 = percentile(values, 0.25), percentile(values, 0.75)
        iqr = (q3 - q1) if q1 is not None and q3 is not None else 0.0
        lower = q1 - 3.0 * iqr if q1 is not None else None
        upper = q3 + 3.0 * iqr if q3 is not None else None
        count = 0
        for row in rows:
            outcome = row.get("outcomes", {}).get(key, {})
            ret = number(outcome.get("return_pct"))
            if ret is None:
                continue
            reasons = []
            if abs(ret) >= absolute_threshold_pct:
                reasons.append("ABS_RETURN_THRESHOLD")
            if (lower is not None and ret < lower) or (upper is not None and ret > upper):
                reasons.append("EXTREME_IQR_FENCE_3X")
            if not reasons:
                continue
            count += 1
            observations.append({
                "horizon": key, "symbol": row.get("symbol"),
                "timestamp": row.get("timestamp").isoformat() if row.get("timestamp") else None,
                "raw_signal": row.get("raw_signal"), "final_signal": row.get("final_signal"),
                "classification": classify(row), "entry_price_recorded": row.get("price"),
                "future_close": outcome.get("future_close"), "return_pct": ret,
                "benchmark": outcome.get("benchmark_symbol") or row.get("benchmark"),
                "benchmark_return_pct": outcome.get("benchmark_return_pct"),
                "excess_return_pct": outcome.get("excess_return_pct"),
                "engine": row.get("engine"), "regime": row.get("regime"),
                "reason": row.get("reason"), "flag_reasons": reasons,
                "source_file": row.get("source_file"), "source_line": row.get("source_line"),
            })
        by_horizon[key] = {"evaluated_N": len(values), "absolute_threshold_pct": absolute_threshold_pct,
                           "iqr_lower_fence_pct": lower, "iqr_upper_fence_pct": upper, "flagged_N": count}
    observations.sort(key=lambda item: abs(item["return_pct"]), reverse=True)
    return {"purpose": "Review flags only; flagged observations remain in all aggregate statistics.",
            "absolute_return_threshold_pct": absolute_threshold_pct,
            "iqr_rule": "outside Q1 - 3*IQR or Q3 + 3*IQR, separately per horizon",
            "by_horizon": by_horizon, "flagged_observations_N": len(observations),
            "flagged_observations": observations}


def buy_signal_audit(rows: list[dict[str, Any]], horizons: list[int]) -> dict[str, Any]:
    """Record-level trace for every raw BUY/STRONG_BUY signal."""
    details, reason_counts, classification_counts = [], Counter(), Counter()
    for row in rows:
        if row.get("raw_signal") not in BUY_SIGNALS:
            continue
        cls = classify(row)
        reason = str(row.get("reason") or "").strip() or None
        reason_counts[reason or "<REASON_UNAVAILABLE>"] += 1
        classification_counts[cls] += 1
        outcomes = {}
        for horizon in horizons:
            key = f"{horizon}d"
            item = row.get("outcomes", {}).get(key, {})
            outcomes[key] = {"future_close": item.get("future_close"), "return_pct": item.get("return_pct"),
                             "benchmark": item.get("benchmark_symbol") or row.get("benchmark"),
                             "benchmark_return_pct": item.get("benchmark_return_pct"),
                             "excess_return_pct": item.get("excess_return_pct")}
        details.append({"symbol": row.get("symbol"),
            "timestamp": row.get("timestamp").isoformat() if row.get("timestamp") else None,
            "raw_signal": row.get("raw_signal"), "gated_signal": row.get("gated_signal"),
            "final_signal": row.get("final_signal"), "classification": cls,
            "score": row.get("score"), "confidence": row.get("confidence"), "appt": row.get("appt"),
            "entry_price_recorded": row.get("price"), "reason": reason,
            "reason_status": "AVAILABLE" if reason else "UNAVAILABLE_IN_SOURCE_RECORD",
            "benchmark": row.get("benchmark"), "regime": row.get("regime"), "engine": row.get("engine"),
            "outcomes": outcomes, "source_file": row.get("source_file"), "source_line": row.get("source_line")})
    return {"raw_buy_records_N": len(details), "classification_counts": dict(sorted(classification_counts.items())),
            "reason_counts": [{"reason": k, "count": v} for k, v in reason_counts.most_common()],
            "reason_unavailable_N": reason_counts.get("<REASON_UNAVAILABLE>", 0), "records": details}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit AI Trader decision conservatism from prediction snapshots."
    )
    parser.add_argument("--prediction-dir", default=str(DEFAULT_PREDICTION_DIR))
    parser.add_argument("--file", action="append", dest="files")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10, 20, 60])
    parser.add_argument("--no-market-data", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    started = time.perf_counter()
    output = Path(args.output)
    prediction_dir = Path(args.prediction_dir)

    result = {
        "audit": "AI Trader Decision Audit",
        "version": VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changes": False,
        "orders_allowed": False,
        "alpaca_trading_endpoints_called": False,
        "inputs": {},
        "summary": {},
        "decision_rates": {},
        "downgrade_reasons": {},
        "decision_matrix": [],
        "outlier_audit": {},
        "buy_signal_audit": {},
        "by_engine": {},
        "by_regime": {},
        "by_raw_signal": {},
        "concentration": {},
        "coverage_diagnostics": {},
        "market_data": {},
        "methodology": {
            "horizons_unit": "trading_sessions_after_first_session_on_or_after_prediction_date",
            "asset_return_entry": "recorded price_at_prediction",
            "benchmark_return_entry": "benchmark close on prediction session",
            "excess_return": "asset_return_pct minus benchmark_return_pct",
            "input_files_modified": False,
            "outlier_policy": "flag absolute returns >= 100% or beyond a 3*IQR fence; never auto-remove observations",
            "buy_signal_audit": "raw BUY/STRONG_BUY row-level trace with source line, reason availability, and per-horizon outcomes",
            "trading_endpoints_called": False,
        },
    }

    try:
        paths = find_files(prediction_dir, args.files, args.days)
        if not paths:
            raise AuditError(f"No prediction JSONL files found in {prediction_dir}")

        raw, metadata = load_records(paths)
        rows = [normalize(record) for record in raw]

        result["inputs"] = {
            "prediction_dir": str(prediction_dir),
            "files": metadata,
            "total_records": len(rows),
            "horizons_days": args.horizons,
        }

        symbols = sorted({
            str(value).strip().upper()
            for row in rows
            for value in (row.get("symbol"), row.get("benchmark"))
            if value
        })

        histories = {}
        if args.no_market_data:
            result["market_data"] = {"status": "SKIPPED"}
        else:
            histories, errors = load_market_data(symbols)
            result["market_data"] = {
                "status": "OK" if not errors else "PARTIAL",
                "symbols_requested": symbols,
                "symbols_loaded": sorted(histories.keys()),
                "errors": errors,
            }
            add_outcomes(rows, histories, args.horizons)

        result["summary"] = aggregate(rows, args.horizons)
        result["downgrade_reasons"] = downgrade_reasons(rows)
        result["decision_matrix"] = decision_matrix(rows)
        result["outlier_audit"] = audit_outliers(rows, args.horizons)
        result["buy_signal_audit"] = buy_signal_audit(rows, args.horizons)
        result["by_engine"] = grouped_analysis(rows, "engine", args.horizons)
        result["by_regime"] = grouped_analysis(rows, "regime", args.horizons)
        result["by_raw_signal"] = grouped_analysis(rows, "raw_signal", args.horizons)
        result["concentration"] = concentration_analysis(rows, args.horizons)
        result["coverage_diagnostics"] = coverage_diagnostics(rows, args.horizons, histories if not args.no_market_data else {})

        raw_buy = result["summary"]["raw_buy"]["N"]
        final_buy = result["summary"]["final_buy"]["N"]
        confirmation = result["summary"]["raw_buy_to_confirmation"]["N"]
        confirmation_total = result["summary"]["buy_on_confirmation"]["N"]

        result["decision_rates"] = {
            "raw_buy_count": raw_buy,
            "final_buy_count": final_buy,
            "buy_on_confirmation_count": confirmation,
            "raw_buy_to_final_buy_pct": final_buy / raw_buy * 100 if raw_buy else None,
            "raw_buy_to_confirmation_pct": confirmation / raw_buy * 100 if raw_buy else None,
            "raw_buy_to_other_pct": (
                result["summary"]["raw_buy_to_other"]["N"] / raw_buy * 100
                if raw_buy else None
            ),
        }

        evaluated_total = sum(
            result["summary"]["all"].get(f"{h}d_N", 0) for h in args.horizons
        )
        market_status = result.get("market_data", {}).get("status")
        if args.no_market_data:
            result["status"] = "PARTIAL"
            result["status_reason"] = "Market data was explicitly skipped; returns are not evaluated."
        elif not histories:
            result["status"] = "ERROR"
            result["status_reason"] = "No market histories loaded; outcome analysis is invalid."
        elif result["market_data"].get("errors") or evaluated_total == 0:
            result["status"] = "PARTIAL"
            result["status_reason"] = "Market data errors or zero evaluable returns; inspect coverage and errors."
        else:
            result["status"] = "OK"

    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    result["elapsed_seconds"] = round(time.perf_counter() - started, 6)

    output.write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(f"AI TRADER - DECISION AUDIT V{VERSION}")
    print("=" * 72)
    print("Research-only | no production changes | no orders")
    print()

    if result["status"] == "ERROR":
        print("STATUS: ERROR")
        print(result.get("error", {}).get("message", result.get("status_reason", "Audit failed")))
        print(f"JSON: {output}")
        return 1
    if result["status"] == "PARTIAL":
        print("STATUS: PARTIAL")
        print(result.get("status_reason", "Some results are incomplete; inspect coverage diagnostics."))
    else:
        print("STATUS: OK")

    print(f"Prediction records: {result['inputs']['total_records']}")
    print(f"Files:              {len(result['inputs']['files'])}")
    print(f"Symbols:            {len(symbols)}")
    print()
    print("DECISION PIPELINE")
    print("-" * 72)
    print(f"RAW BUY / STRONG_BUY:    {raw_buy}")
    print(f"Final BUY / STRONG_BUY:  {final_buy}")
    print(f"BUY_ON_CONFIRMATION:     {confirmation_total}")
    print(f"RAW BUY -> CONFIRMATION: {confirmation}")

    if result["decision_rates"]["raw_buy_to_confirmation_pct"] is not None:
        print(
            "RAW BUY -> CONFIRMATION: "
            f"{result['decision_rates']['raw_buy_to_confirmation_pct']:.2f}%"
        )
        print(
            "RAW BUY -> FINAL BUY:    "
            f"{result['decision_rates']['raw_buy_to_final_buy_pct']:.2f}%"
        )

    print()
    print("BUY GROUP OUTCOMES")
    print("-" * 72)

    for name in (
        "raw_buy",
        "final_buy",
        "buy_on_confirmation",
        "raw_buy_to_other",
    ):
        item = result["summary"][name]
        print(
            f"{name:26s} N={item['N']:4d} "
            f"confidence={item['confidence_mean']} "
            f"score={item['score_mean']} "
            f"APPT={item['appt_mean']}"
        )
        for horizon in args.horizons:
            print(
                f"  {horizon:2d}d: "
                f"N={item[f'{horizon}d_N']:4d} "
                f"mean={item[f'{horizon}d_mean_return_pct']}% "
                f"positive={item[f'{horizon}d_positive_pct']}% "
                f"benchmark_mean={item[f'{horizon}d_mean_benchmark_return_pct']}% "
                f"excess_mean={item[f'{horizon}d_mean_excess_return_pct']}%"
            )

    print()
    print("BUY -> BUY_ON_CONFIRMATION REASONS")
    print("-" * 72)
    for item in result["downgrade_reasons"]["reasons"][:15]:
        print(f"{item['count']:4d} | {item['reason']}")

    print()
    print(f"JSON: {output}")
    print(f"Elapsed: {result['elapsed_seconds']:.2f}s")
    print("=" * 72)

    return 2 if result["status"] == "PARTIAL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
