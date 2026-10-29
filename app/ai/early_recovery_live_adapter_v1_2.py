"""Live adapter for Early Recovery V1.2 active experimental path.

Evaluates the existing cached OHLCV history and records experimental recovery
recommendations. It never changes the production signal and never places orders.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from app.ai.recovery_entry_experiment_v1_2 import evaluate_recovery_entry
from app.data.market_data import get_history


def _flatten_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # yfinance can return (field, ticker) or (ticker, field).
        fields = {"open", "high", "low", "close", "volume"}
        flattened = []
        for col in out.columns:
            parts = [str(p) for p in col if str(p).strip()]
            match = next((p for p in parts if p.lower() in fields), parts[0] if parts else "")
            flattened.append(match)
        out.columns = flattened
    else:
        out.columns = [str(c) for c in out.columns]
    return out


def _cce_metrics(analysis: Mapping[str, Any], engine_result: Mapping[str, Any]) -> dict:
    cce = dict(engine_result)
    for key in ("score", "historical_probability_20d", "reward_risk_20d", "historical_pattern"):
        if cce.get(key) is None and analysis.get(key) is not None:
            cce[key] = analysis[key]
    # Preserve the orchestrator's authoritative score if present.
    if analysis.get("score") is not None:
        cce["score"] = analysis["score"]
    return cce


def _append_once_per_session(record: dict, session_date: str) -> bool:
    root = Path(os.getenv("AI_TRADER_EARLY_RECOVERY_LOG_DIR", "data/early_recovery_v1_2"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "observations.jsonl"
    key = (record.get("ticker"), session_date, record.get("action"), record.get("tier"))
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        old = json.loads(line)
                        old_key = (old.get("ticker"), old.get("session_date"), old.get("action"), old.get("tier"))
                        if old_key == key:
                            return False
                    except (json.JSONDecodeError, TypeError):
                        continue
        except OSError:
            # Fail-safe: the scan still continues; attempt append below.
            pass
    record["session_date"] = session_date
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return True


def evaluate_live_early_recovery(
    ticker: str,
    analysis: Mapping[str, Any],
    engine_result: Mapping[str, Any],
    *,
    period: str = "5y",
    interval: str = "1d",
) -> dict:
    """Return a V1.2 evaluation; experimental errors never raise into production scan."""
    symbol = str(ticker or "").strip().upper()
    try:
        if str(analysis.get("status", "ANALYZED")).upper() in {"ERROR", "REVIEW_DATA", "INVALID_DATA"}:
            return {"engine": "EARLY_RECOVERY_EXPERIMENT", "version": "1.2", "action": "NO_ENTRY", "tier": "NONE", "recommendation": "SKIPPED_INVALID_ANALYSIS", "safety": {"mode": "SHADOW_ONLY", "orders_submitted": False}}
        # Reuse the project's shared market cache. Do not force a refresh.
        history = get_history(
            symbol, period=period, interval=interval, auto_adjust=True,
            actions=False, group_by="column", threads=False,
        )
        history = _flatten_ohlcv_columns(history) if isinstance(history, pd.DataFrame) else pd.DataFrame()
        cce = _cce_metrics(analysis, engine_result)
        result = evaluate_recovery_entry(history, cce_result=cce)
        result["ticker"] = symbol
        result["source_signal"] = str(analysis.get("signal") or engine_result.get("signal") or "UNKNOWN").upper()
        result["production_signal_unchanged"] = True
        result["safety"] = {"mode": "SHADOW_ONLY", "orders_submitted": False, "production_integrated": False}
        if not history.empty:
            session_date = str(history.index[-1].date()) if hasattr(history.index[-1], "date") else str(history.index[-1])[:10]
        else:
            session_date = "unknown"
        try:
            wrote = _append_once_per_session(result, session_date)
            result["logged"] = wrote
        except Exception as log_exc:
            result["logged"] = False
            result["log_error"] = f"{type(log_exc).__name__}: {log_exc}"
        if result.get("action") == "BUY_CANDIDATE":
            print(
                f"EARLY RECOVERY V1.2 | {symbol} | {result.get('tier')} | "
                f"price={result.get('metrics', {}).get('price', '-') } | "
                f"source={result.get('source_signal')} | SHADOW ONLY | NO ORDER"
            )
        return result
    except Exception as exc:
        # Experimental failures must never interrupt or change production scan.
        return {
            "engine": "EARLY_RECOVERY_EXPERIMENT", "version": "1.2",
            "action": "ERROR", "tier": "NONE", "recommendation": "EXPERIMENT_ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "safety": {"mode": "SHADOW_ONLY", "orders_submitted": False},
        }
