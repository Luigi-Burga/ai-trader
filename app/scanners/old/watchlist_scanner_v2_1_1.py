"""AI Trader - Watchlist Scanner V2.1

Compatibility layer:
    watchlist -> Orchestrator V1.3 -> Decision Gate V1.2 -> Signal Engine V1.1

No SOXL-specific analyzer. buy_target is informational only.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

from app.ai.asset_analysis_orchestrator_v1_4_2 import analyze_ticker
from app.strategies.decision_gate_v1_2 import apply_decision_gate
from app.strategies.signal_engine_v1_1 import generate_signal

VERSION = "2.1.1"
SCANNER = "Watchlist Scanner"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        n = float(value)
        if n == n and abs(n) != float("inf"):
            return n
    except (TypeError, ValueError):
        pass
    return default


def _engine_result(analysis: Mapping[str, Any]) -> Dict[str, Any]:
    result = analysis.get("engine_result")
    if isinstance(result, Mapping):
        return dict(result)
    return {
        "ticker": analysis.get("ticker"),
        "signal": analysis.get("signal", "REVIEW_DATA"),
        "score": analysis.get("score", 0.0),
        "confidence": analysis.get("confidence", 0.0),
    }


def _gate_input(
    engine_result: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> Dict[str, Any]:
    """Inject authoritative orchestrator benchmark metadata into gate input.

    The Orchestrator is the source of truth for benchmark resolution. The CCE
    receives the resolved ticker explicitly, which makes the CCE label it as
    a manual benchmark internally. That label is technically true from the
    CCE's perspective, but misleading at the pipeline level. Normalize the
    engine result here so the scanner's audit output reflects the real source
    of the decision-grade benchmark.
    """
    result = dict(engine_result)

    resolved_benchmark = analysis.get("benchmark")
    result["benchmark"] = resolved_benchmark
    result["selected_benchmark"] = resolved_benchmark
    result["benchmark_trust"] = analysis.get("benchmark_trust")
    result["benchmark_trust_decision_grade"] = analysis.get(
        "benchmark_trust_decision_grade"
    )

    # Normalize benchmark provenance for pipeline-level auditability.
    # The authoritative Orchestrator resolver result is nested inside the
    # engine_result returned by analyze_ticker(). The CCE also carries its
    # own internal benchmark_resolution, where an explicitly injected
    # benchmark is legitimately labelled as manual/custom.
    #
    # Resolution order: authoritative nested Orchestrator result first,
    # then top-level compatibility fields from older/partial responses.
    orchestrator_resolution = result.get(
        "orchestrator_benchmark_resolution"
    )
    if not isinstance(orchestrator_resolution, Mapping):
        orchestrator_resolution = analysis.get(
            "orchestrator_benchmark_resolution"
        )
    if not isinstance(orchestrator_resolution, Mapping):
        # Compatibility with an alternate orchestrator field name.
        orchestrator_resolution = analysis.get("benchmark_resolution")

    # Preserve CCE-internal provenance separately for diagnostics.
    # These fields must not override the pipeline-level provenance below.
    if "benchmark_source" in result:
        result["benchmark_internal_source"] = result["benchmark_source"]
    if "benchmark_role" in result:
        result["benchmark_internal_role"] = result["benchmark_role"]
    if "benchmark_is_manual" in result:
        result["benchmark_internal_is_manual"] = result[
            "benchmark_is_manual"
        ]
    internal_resolution = result.get("benchmark_resolution")
    if isinstance(internal_resolution, Mapping):
        result["benchmark_internal_resolution"] = dict(internal_resolution)
    if isinstance(orchestrator_resolution, Mapping):
        result["benchmark_source"] = orchestrator_resolution.get(
            "source", result.get("benchmark_source")
        )
        result["benchmark_role"] = orchestrator_resolution.get(
            "role", result.get("benchmark_role")
        )
        result["benchmark_selection_reason"] = orchestrator_resolution.get(
            "selection_reason", result.get("benchmark_selection_reason")
        )
        result["benchmark_selection_confidence"] = orchestrator_resolution.get(
            "selection_confidence", result.get("benchmark_selection_confidence")
        )
        result["benchmark_fallback_reason"] = orchestrator_resolution.get(
            "fallback_reason", result.get("benchmark_fallback_reason", "")
        )
        result["benchmark_is_manual"] = bool(
            orchestrator_resolution.get(
                "is_manual", result.get("benchmark_is_manual", False)
            )
        )
        result["benchmark_resolution"] = dict(orchestrator_resolution)
    else:
        # Fallback for older/partial orchestrator responses.
        result["benchmark_source"] = analysis.get(
            "benchmark_source", result.get("benchmark_source")
        )
        result["benchmark_role"] = analysis.get(
            "benchmark_role", result.get("benchmark_role")
        )

    return result


def _print_result(
    analysis: Mapping[str, Any],
    gate: Mapping[str, Any],
    final_signal: str,
    buy_target: float,
) -> None:
    ticker = str(analysis.get("ticker") or "UNKNOWN")
    age = analysis.get("asset_age") or {}
    print()
    print("=" * 88)
    print(f"WATCHLIST ANALYSIS | {ticker}")
    print("=" * 88)
    print(
        f"route={analysis.get('route', '-')} | "
        f"engine={analysis.get('engine', '-')} "
        f"V{analysis.get('engine_version') or '-'} | "
        f"raw={analysis.get('signal', '-')} | FINAL={final_signal}"
    )
    print(
        f"score={_safe_float(analysis.get('score')):.2f} | "
        f"confidence={_safe_float(analysis.get('confidence')):.2f} | "
        f"days={age.get('trading_days', '-')} | "
        f"features={age.get('valid_feature_rows', '-')}"
    )
    current = analysis.get("current_price")
    current_text = f"${_safe_float(current):.2f}" if current is not None else "-"
    print(
        f"benchmark={analysis.get('benchmark') or '-'} | "
        f"trust={analysis.get('benchmark_trust') or 'NONE'} | "
        f"grade={analysis.get('benchmark_trust_decision_grade')} | "
        f"current={current_text}"
    )
    if buy_target:
        print(
            f"buy_target=${buy_target:.2f} "
            "(informational only; never used as BUY trigger)"
        )

    print(
        f"gate={gate.get('gate_status', '-')} | "
        f"reason={gate.get('gate_reason', '-')}"
    )
    if gate.get("gate_status") == "BLOCKED":
        failed = gate.get("failed_gates") or []
        if failed:
            print("failed BUY gates: " + ", ".join(map(str, failed)))
    elif gate.get("gate_status") == "NOT_APPLICABLE":
        print("BUY gates: NOT_APPLICABLE for non-bullish signal")
    elif gate.get("gate_status") == "CONFIRMATION_REQUIRED":
        print("BUY gates: confirmation required by engine")

    print(f"regime={analysis.get('regime') or '-'}")

    engine = analysis.get("engine_result") or {}
    levels = engine.get("levels")
    if isinstance(levels, Mapping):
        low = levels.get("entry_zone_low")
        high = levels.get("entry_zone_high")
        stop = levels.get("dynamic_stop")
        tp1 = levels.get("take_profit_1")
        tp2 = levels.get("take_profit_2")
        tp3 = levels.get("take_profit_3")
        if low is not None and high is not None:
            print(
                f"dynamic entry zone="
                f"${_safe_float(low):.2f}-${_safe_float(high):.2f}"
            )
        if stop is not None:
            print(f"dynamic stop=${_safe_float(stop):.2f}")
        if tp1 is not None:
            print(f"TP1=${_safe_float(tp1):.2f}")
        if tp2 is not None:
            print(f"TP2=${_safe_float(tp2):.2f}")
        if tp3 is not None:
            print(f"TP3=${_safe_float(tp3):.2f}")
    print("=" * 88)


def _review_data_result(
    analysis: Mapping[str, Any],
    engine_result: Mapping[str, Any],
    buy_target: float,
) -> Dict[str, Any]:
    """Return a valid terminal result for data-insufficient assets.

    REVIEW_DATA is an intentional terminal route from the Asset Age Resolver.
    In this state the engine is not expected to run and benchmark/trust may be
    unavailable.  The scanner therefore preserves REVIEW_DATA without trying
    to manufacture a benchmark or an actionable trading signal.
    """
    return {
        "scanner": SCANNER,
        "version": VERSION,
        "ticker": str(analysis.get("ticker") or "").upper(),
        "status": analysis.get("status", "REVIEW_DATA"),
        "route": "REVIEW_DATA",
        "engine": analysis.get("engine"),
        "engine_version": analysis.get("engine_version"),
        "signal": "REVIEW_DATA",
        "gated_signal": "REVIEW_DATA",
        "final_signal": "REVIEW_DATA",
        "score": analysis.get("score", 0.0),
        "confidence": analysis.get("confidence", 0.0),
        "current_price": analysis.get("current_price"),
        "regime": analysis.get("regime"),
        "benchmark": analysis.get("benchmark"),
        "benchmark_trust": analysis.get("benchmark_trust"),
        "benchmark_trust_decision_grade": analysis.get(
            "benchmark_trust_decision_grade"
        ),
        "watchlist_buy_target": buy_target,
        "asset_age": analysis.get("asset_age"),
        "engine_result": dict(engine_result),
        "analysis": analysis,
        "decision_gate": {
            "gate_status": "NOT_APPLICABLE",
            "gate_reason": "REVIEW_DATA: insufficient data for engine analysis",
        },
        "signal_engine": {
            "signal": "REVIEW_DATA",
            "reason": "REVIEW_DATA is a terminal non-actionable route",
        },
    }


def scan_buy_opportunity(stock: Mapping[str, Any]) -> Dict[str, Any]:
    """Analyze one watchlist item through the complete validated pipeline."""
    ticker = str(stock.get("ticker") or "").strip().upper()
    if not ticker:
        return {
            "scanner": SCANNER,
            "version": VERSION,
            "ticker": "",
            "signal": "ERROR",
            "final_signal": "ERROR",
            "status": "ERROR",
            "error": "Watchlist item has no ticker.",
        }

    buy_target = _safe_float(stock.get("buy_target"), 0.0)
    period = str(stock.get("curve_period") or stock.get("period") or "5y")
    interval = str(stock.get("interval") or "1d")
    benchmark = stock.get("benchmark") or None
    listing_price = stock.get("listing_price")
    validate = bool(stock.get("validate_benchmark", True))

    try:
        analysis = analyze_ticker(
            ticker,
            period=period,
            interval=interval,
            benchmark=benchmark,
            listing_price=(
                _safe_float(listing_price)
                if listing_price is not None else None
            ),
            validate_benchmark=validate,
        )
    except Exception as exc:
        print(f"{ticker} => Orchestrator error: {exc}")
        return {
            "scanner": SCANNER,
            "version": VERSION,
            "ticker": ticker,
            "signal": "ERROR",
            "final_signal": "ERROR",
            "status": "ERROR",
            "watchlist_buy_target": buy_target,
            "error": str(exc),
        }

    # Normalize the engine result at the scanner boundary so the returned
    # audit tree uses the authoritative Orchestrator benchmark provenance.
    # This prevents the CCE's internal "manual/custom" label from leaking
    # into the pipeline-level result when the benchmark was auto-resolved.
    engine_result = _gate_input(_engine_result(analysis), analysis)

    # REVIEW_DATA is a valid terminal route.  The Asset Age Resolver uses it
    # when there is not enough usable history to run NLE/CCE.  Do not require
    # a benchmark/trust value and do not send this state through BUY gates.
    route = str(analysis.get("route") or "").upper()
    raw_signal = str(analysis.get("signal") or "").upper()
    if route == "REVIEW_DATA" or raw_signal == "REVIEW_DATA":
        _print_result(
            analysis,
            {
                "gate_status": "NOT_APPLICABLE",
                "gate_reason": "REVIEW_DATA: insufficient data for engine analysis",
            },
            "REVIEW_DATA",
            buy_target,
        )
        return _review_data_result(
            analysis,
            engine_result,
            buy_target,
        )

    gate_input = deepcopy(engine_result)

    # Decision Gate is deliberately upstream of Signal Engine.
    try:
        gate = apply_decision_gate(gate_input)
    except Exception as exc:
        print(f"{ticker} => Decision Gate error: {exc}")
        return {
            "scanner": SCANNER,
            "version": VERSION,
            "ticker": ticker,
            "signal": "ERROR",
            "final_signal": "ERROR",
            "status": "ERROR",
            "watchlist_buy_target": buy_target,
            "analysis": analysis,
            "engine_result": engine_result,
            "error": f"Decision Gate failed: {exc}",
        }

    gated_signal = str(
        gate.get("operational_signal")
        or gate.get("signal")
        or analysis.get("signal")
        or "REVIEW_DATA"
    ).upper()

    # Signal Engine receives the already-gated operational signal.
    signal_input = deepcopy(gate_input)
    signal_input["engine_signal"] = gated_signal
    signal_input["signal"] = gated_signal
    signal_input["operational_signal"] = gated_signal

    try:
        signal_result = generate_signal(
            analysis_result=signal_input,
            benchmark_trust=analysis.get("benchmark_trust"),
            characteristic_result=signal_input,
        )
    except TypeError:
        # Compatibility fallback for an equivalent V1.1 signature.
        signal_result = generate_signal(
            analysis_result=signal_input,
            benchmark_trust=analysis.get("benchmark_trust"),
        )
    except Exception as exc:
        print(f"{ticker} => Signal Engine error: {exc}")
        return {
            "scanner": SCANNER,
            "version": VERSION,
            "ticker": ticker,
            "signal": gated_signal,
            "final_signal": gated_signal,
            "status": "SIGNAL_ENGINE_ERROR",
            "watchlist_buy_target": buy_target,
            "analysis": analysis,
            "engine_result": engine_result,
            "decision_gate": dict(gate),
            "error": str(exc),
        }

    if not isinstance(signal_result, Mapping):
        signal_result = {"signal": gated_signal}

    final_signal = str(
        signal_result.get("signal")
        or signal_result.get("final_signal")
        or gated_signal
    ).upper()

    # Downstream safety invariants.
    non_bullish = {
        "SELL", "STRONG_SELL", "REDUCE", "AVOID", "WATCH",
        "NO_DATA", "REVIEW_DATA", "INSUFFICIENT_HISTORY", "ERROR",
    }
    if gated_signal in non_bullish and final_signal in {"BUY", "STRONG_BUY"}:
        final_signal = gated_signal

    if (
        final_signal in {"BUY", "STRONG_BUY"}
        and analysis.get("benchmark_trust") in {"CONTEXT_ONLY", "NONE", None, ""}
    ):
        final_signal = "BUY_ON_CONFIRMATION"

    _print_result(analysis, gate, final_signal, buy_target)

    if buy_target and analysis.get("current_price") is not None:
        if _safe_float(analysis["current_price"]) <= buy_target:
            print(
                f"INFO: {ticker} is at/below legacy target "
                f"${buy_target:.2f}; target is informational only."
            )

    return {
        "scanner": SCANNER,
        "version": VERSION,
        "ticker": ticker,
        "status": analysis.get("status", "ANALYZED"),
        "route": analysis.get("route"),
        "engine": analysis.get("engine"),
        "engine_version": analysis.get("engine_version"),
        "signal": analysis.get("signal"),
        "gated_signal": gated_signal,
        "final_signal": final_signal,
        "score": analysis.get("score", 0.0),
        "confidence": analysis.get("confidence", 0.0),
        "current_price": analysis.get("current_price"),
        "regime": analysis.get("regime"),
        "benchmark": analysis.get("benchmark"),
        "benchmark_trust": analysis.get("benchmark_trust"),
        "benchmark_trust_decision_grade": analysis.get(
            "benchmark_trust_decision_grade"
        ),
        "watchlist_buy_target": buy_target,
        "asset_age": analysis.get("asset_age"),
        "engine_result": engine_result,
        "analysis": analysis,
        "decision_gate": dict(gate),
        "signal_engine": dict(signal_result),
    }


scan_asset = scan_buy_opportunity

__all__ = ["VERSION", "SCANNER", "scan_buy_opportunity", "scan_asset"]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="AI Trader Watchlist Scanner V2.1"
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--buy-target", type=float, default=0.0)
    args = parser.parse_args()

    result = scan_buy_opportunity({
        "ticker": args.ticker,
        "period": args.period,
        "benchmark": args.benchmark,
        "buy_target": args.buy_target,
    })
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
