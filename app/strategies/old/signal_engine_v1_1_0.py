"""
Central signal engine V1.1.

V1.1 adds Benchmark Trust as a decision gate while preserving the
existing public generate_signal() string contract.

Policy:
    HIGH          -> decision-grade
    MEDIUM        -> decision-grade
    LOW           -> decision-grade (lower-quality context)
    CONTEXT_ONLY  -> cannot validate BUY/STRONG_BUY
    NONE          -> cannot validate BUY/STRONG_BUY

When a bullish engine signal is blocked by benchmark trust, the engine
preserves the underlying signal and downgrades the operational signal to
BUY_ON_CONFIRMATION. This avoids losing information while preventing an
unvalidated BUY from becoming executable.
"""

import ta

VERSION = "1.1"

DECISION_GRADE_TRUST = {"HIGH", "MEDIUM", "LOW"}
BLOCKED_BUY_TRUST = {"CONTEXT_ONLY", "NONE", ""}


def _generate_standard_signal(df):
    """Existing signal logic for non-SOXL instruments."""
    if df.empty:
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


def _normalise_trust(value):
    """Return a canonical Benchmark Trust value."""
    if value is None:
        return ""
    return str(value).strip().upper()


def benchmark_trust_allows_decision(benchmark_trust):
    """Return True when Benchmark Trust is decision-grade."""
    return _normalise_trust(benchmark_trust) in DECISION_GRADE_TRUST


def apply_benchmark_trust(engine_signal, benchmark_trust=None):
    """
    Apply Benchmark Trust to an engine signal.

    Only bullish BUY signals are gated. SELL/HOLD/WATCH/AVOID/etc. are
    preserved because a weak benchmark should not manufacture a bullish
    signal or suppress an existing risk signal.
    """
    signal = str(engine_signal or "NO_DATA").upper()
    trust = _normalise_trust(benchmark_trust)

    if signal in {"BUY", "STRONG_BUY"} and trust in BLOCKED_BUY_TRUST:
        return "BUY_ON_CONFIRMATION"

    return signal


def build_decision(engine_signal, analysis_result=None, benchmark_trust=None):
    """
    Build an auditable signal decision without changing the legacy API.

    Parameters
    ----------
    engine_signal : str
        Raw signal produced by the selected analysis engine.
    analysis_result : dict, optional
        Asset Analysis Orchestrator / CCE / NLE result. Benchmark Trust is
        read from this dictionary when benchmark_trust is not supplied.
    benchmark_trust : str, optional
        Explicit trust override.

    Returns
    -------
    dict
        Raw signal, operational signal, benchmark metadata and gate reason.
    """
    result = analysis_result if isinstance(analysis_result, dict) else {}

    trust = _normalise_trust(
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
        grade = benchmark_trust_allows_decision(trust)
    elif grade is not None:
        grade = bool(grade)

    raw_signal = str(engine_signal or "NO_DATA").upper()

    # Preserve the legacy behavior when no Benchmark Trust metadata is
    # supplied. Once metadata is supplied, its decision grade is authoritative.
    if not metadata_present:
        operational_signal = raw_signal
        gate_reason = "benchmark_trust_not_supplied_legacy_path"
    elif raw_signal in {"BUY", "STRONG_BUY"} and not grade:
        operational_signal = "BUY_ON_CONFIRMATION"
        gate_reason = "buy_blocked_by_benchmark_trust"
    else:
        operational_signal = raw_signal
        gate_reason = "benchmark_trust_accepted"

    return {
        "engine_signal": raw_signal,
        "operational_signal": operational_signal,
        "benchmark": result.get("benchmark")
        or result.get("selected_benchmark"),
        "benchmark_trust": trust or None,
        "benchmark_trust_decision_grade": grade,
        "benchmark_status": result.get("benchmark_status"),
        "relevance_class": result.get("relevance_class"),
        "gate_reason": gate_reason,
    }


def generate_signal(df, ticker=None, soxl_result=None, analysis_result=None,
                    benchmark_trust=None):
    """
    Generate the operational signal.

    Backward compatible with the original API. Existing callers that only
    pass df/ticker/soxl_result receive the same signal behavior. When
    Benchmark Trust metadata is supplied, bullish signals are gated.

    Returns
    -------
    str
        NO_DATA / INSUFFICIENT_DATA / BUY / STRONG_BUY / BUY_ON_CONFIRMATION /
        WATCH / HOLD / REDUCE / SELL
    """
    if ticker and ticker.upper() == "SOXL":
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
    )["operational_signal"]


if __name__ == "__main__":
    # Lightweight deterministic self-test: no market/network dependency.
    cases = [
        ("BUY", "HIGH", "BUY"),
        ("BUY", "MEDIUM", "BUY"),
        ("BUY", "LOW", "BUY"),
        ("BUY", "CONTEXT_ONLY", "BUY_ON_CONFIRMATION"),
        ("BUY", "NONE", "BUY_ON_CONFIRMATION"),
        ("STRONG_BUY", "CONTEXT_ONLY", "BUY_ON_CONFIRMATION"),
        ("SELL", "CONTEXT_ONLY", "SELL"),
        ("WATCH", "NONE", "WATCH"),
    ]

    for raw, trust, expected in cases:
        actual = apply_benchmark_trust(raw, trust)
        assert actual == expected, (raw, trust, actual, expected)

    print(f"Signal Engine V{VERSION} SELF-TEST: PASS")
