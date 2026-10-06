"""
AI Trader - Decision Audit V1.0
Research-only audit of historical prediction JSONL decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "data" / "predictions"
DEFAULT_OUTPUT = PROJECT_ROOT / "decision_audit_v1_results.json"

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
    "timestamp": ["timestamp", "prediction_timestamp", "as_of", "datetime", "date", "created_at"],
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
            auto_adjust=True,
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


def future_close(series, when: datetime | None, horizon: int) -> float | None:
    if series is None or when is None:
        return None

    target = when.date() + timedelta(days=horizon)

    for idx in series.index:
        try:
            idx_date = idx.date()
        except AttributeError:
            try:
                idx_date = idx.to_pydatetime().date()
            except Exception:
                continue

        if idx_date >= target:
            try:
                value = series.loc[idx]
                if hasattr(value, "iloc"):
                    value = value.iloc[0]
                return float(value)
            except Exception:
                return None

    return None


def add_outcomes(rows: list[dict[str, Any]], histories: dict[str, Any], horizons: list[int]) -> None:
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        series = close_series(histories.get(symbol))

        for horizon in horizons:
            future = future_close(series, row.get("timestamp"), horizon)
            entry = row.get("price")
            ret = None
            if entry is not None and future is not None and entry > 0:
                ret = (future / entry - 1.0) * 100.0

            row["outcomes"][f"{horizon}d"] = {
                "future_close": future,
                "return_pct": ret,
                "positive": None if ret is None else ret > 0,
            }


def group_stats(rows: list[dict[str, Any]], horizons: list[int]) -> dict[str, Any]:
    result = {
        "N": len(rows),
        "confidence_mean": None,
        "score_mean": None,
        "appt_mean": None,
    }

    for field in ("confidence", "score", "appt"):
        values = [r[field] for r in rows if r[field] is not None]
        result[f"{field}_N"] = len(values)
        result[f"{field}_mean"] = sum(values) / len(values) if values else None

    for horizon in horizons:
        values = [
            r["outcomes"][f"{horizon}d"]["return_pct"]
            for r in rows
            if r["outcomes"].get(f"{horizon}d", {}).get("return_pct") is not None
        ]
        positive = [
            r["outcomes"][f"{horizon}d"]["positive"]
            for r in rows
            if r["outcomes"].get(f"{horizon}d", {}).get("positive") is not None
        ]

        result[f"{horizon}d_N"] = len(values)
        result[f"{horizon}d_mean_return_pct"] = (
            sum(values) / len(values) if values else None
        )
        result[f"{horizon}d_positive_pct"] = (
            sum(bool(x) for x in positive) / len(positive) * 100
            if positive else None
        )

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
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20])
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
        "market_data": {},
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
            str(row["symbol"]).strip().upper()
            for row in rows if row.get("symbol")
        })

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

        raw_buy = result["summary"]["raw_buy"]["N"]
        final_buy = result["summary"]["final_buy"]["N"]
        confirmation = result["summary"]["buy_on_confirmation"]["N"]

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

    if result["status"] != "OK":
        print("STATUS: ERROR")
        print(result["error"]["message"])
        print(f"JSON: {output}")
        return 1

    print(f"Prediction records: {result['inputs']['total_records']}")
    print(f"Files:              {len(result['inputs']['files'])}")
    print(f"Symbols:            {len(symbols)}")
    print()
    print("DECISION PIPELINE")
    print("-" * 72)
    print(f"RAW BUY / STRONG_BUY:    {raw_buy}")
    print(f"Final BUY / STRONG_BUY:  {final_buy}")
    print(f"BUY_ON_CONFIRMATION:     {confirmation}")

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
                f"positive={item[f'{horizon}d_positive_pct']}%"
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
