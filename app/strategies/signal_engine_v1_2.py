"""
AI Trader - Signal Engine V1.2

V1.2 adds optional risk-based position sizing to the auditable decision
returned by the engine. Existing callers that do not supply account equity
receive the same signal semantics as V1.1.
"""

from __future__ import annotations

from typing import Any, Optional

import ta

from app.portfolio.position_sizer import calculate_position_size

VERSION = "1.2"

DECISION_GRADE_TRUST = {"HIGH", "MEDIUM", "LOW"}
BLOCKED_BUY_TRUST = {"CONTEXT_ONLY", "NONE", ""}


def _generate_standard_signal(df):
    if df is None or df.empty:
        return "NO_DATA"
    if len(df) < 30:
        return "INSUFFICIENT_DATA"

    data = df.copy()
    close_prices = data["Close"]

    rsi_indicator = ta.momentum.RSIIndicator(close=close_prices, window=14)
    data["rsi"] = rsi_indicator.rsi()

    macd_indicator = ta.trend.MACD(close_prices)
    data["macd"] = macd_indicator.macd()
    data["macd_signal"] = macd_indicator.macd_signal()

    rsi = data["rsi"].iloc[-1]
    macd = data["macd"].iloc[-1]
    macd_signal = data["macd_signal"].iloc[-1]

    if rsi < 30 and macd > macd_signal:
        return "BUY"
    if rsi > 70 and macd < macd_signal:
        return "SELL"
    return "HOLD"


def _norm(value: Any) -> str:
    return str(value or "").strip().upper()


def build_decision(
    engine_signal: str,
    analysis_result: Optional[dict] = None,
    benchmark_trust: Optional[str] = None,
    account_equity: Optional[float] = None,
    max_risk_pct: float = 1.0,
) -> dict:
    result = analysis_result if isinstance(analysis_result, dict) else {}

    trust = _norm(
        benchmark_trust
        if benchmark_trust is not None
        else result.get("benchmark_trust")
    )

    grade = result.get("benchmark_trust_decision_grade")
    metadata_present = (
        benchmark_trust is not None
        or "benchmark_trust" in result
        or "benchmark_trust_decision_grade" in result
    )

    if grade is None and metadata_present:
        grade = trust in DECISION_GRADE_TRUST
    elif grade is not None:
        grade = bool(grade)

    raw_signal = _norm(engine_signal or "NO_DATA")

    if not metadata_present:
        operational = raw_signal
        reason = "benchmark_trust_not_supplied_legacy_path"
    elif raw_signal in {"BUY", "STRONG_BUY"} and not grade:
        operational = "BUY_ON_CONFIRMATION"
        reason = "buy_blocked_by_benchmark_trust"
    else:
        operational = raw_signal
        reason = "benchmark_trust_accepted"

    risk_sizing = None
    levels = result.get("levels") or {}
    entry = levels.get("current_price") or result.get("current_price") or result.get("price")
    stop = (
        levels.get("historical_stop_reference")
        or levels.get("stop")
        or result.get("stop")
    )

    if account_equity is not None and entry is not None and stop is not None:
        risk_sizing = calculate_position_size(
            account_equity=account_equity,
            entry_price=entry,
            stop_price=stop,
            max_risk_pct=max_risk_pct,
        )

    return {
        "version": VERSION,
        "engine_signal": raw_signal,
        "operational_signal": operational,
        "signal": operational,
        "benchmark": result.get("benchmark") or result.get("selected_benchmark"),
        "benchmark_trust": trust or None,
        "benchmark_trust_decision_grade": grade,
        "gate_reason": reason,
        "risk_sizing": risk_sizing,
    }


def generate_signal(
    df=None,
    ticker=None,
    soxl_result=None,
    analysis_result=None,
    benchmark_trust=None,
    characteristic_result=None,
    account_equity=None,
    max_risk_pct=1.0,
):
    if analysis_result is None and characteristic_result is not None:
        analysis_result = characteristic_result

    if isinstance(analysis_result, dict):
        raw = _norm(
            analysis_result.get("signal")
            or analysis_result.get("engine_signal")
            or "NO_DATA"
        )
        return build_decision(
            raw,
            analysis_result=analysis_result,
            benchmark_trust=benchmark_trust,
            account_equity=account_equity,
            max_risk_pct=max_risk_pct,
        )

    if ticker and str(ticker).upper() == "SOXL":
        if soxl_result is None:
            from app.ai.soxl_cycle_analyzer import AnalyzerConfig, analyze
            soxl_result = analyze(
                AnalyzerConfig(
                    ticker="SOXL",
                    benchmark="SOXX",
                    period="2y",
                    interval="1d",
                )
            )
        engine_signal = soxl_result["signal"]
    else:
        engine_signal = _generate_standard_signal(df)

    return build_decision(
        engine_signal,
        analysis_result=analysis_result,
        benchmark_trust=benchmark_trust,
        account_equity=account_equity,
        max_risk_pct=max_risk_pct,
    )


def self_test():
    result = build_decision(
        "BUY",
        {
            "benchmark": "QQQ",
            "benchmark_trust": "HIGH",
            "benchmark_trust_decision_grade": True,
            "levels": {
                "current_price": 100,
                "historical_stop_reference": 95,
            },
        },
        account_equity=100_000,
        max_risk_pct=1.0,
    )
    assert result["signal"] == "BUY"
    assert result["risk_sizing"]["shares"] == 200
    assert result["risk_sizing"]["actual_risk"] == 1000.0


if __name__ == "__main__":
    self_test()
    print(f"Signal Engine V{VERSION} SELF-TEST: PASS")
