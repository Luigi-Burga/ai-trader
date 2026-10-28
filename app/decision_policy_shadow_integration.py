"""Pure shadow adapter used by app.main; never authorizes orders."""
from app.ai.decision_policy_v1 import DecisionInput, evaluate_decision

def _policy_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        number = float(value)
        if number != number or abs(number) == float("inf"):
            return default
        return number
    except (TypeError, ValueError):
        return default


def _decision_policy_shadow(symbol: str, result: dict) -> dict:
    """Evaluate policy beside the production signal; never mutate that signal."""
    engine = result.get("engine_result")
    if not isinstance(engine, dict):
        engine = {}
    levels = engine.get("levels")
    if not isinstance(levels, dict):
        levels = result.get("levels") if isinstance(result.get("levels"), dict) else {}
    historical = engine.get("historical_pattern")
    if not isinstance(historical, dict):
        historical = {}

    original = str(result.get("final_signal") or result.get("signal") or "WATCH").upper()
    # The policy accepts BUY/STRONG_BUY candidates. BUY_ON_CONFIRMATION is
    # evaluated as a clearly marked hypothetical BUY probe, never as a signal change.
    probe = "BUY" if original == "BUY_ON_CONFIRMATION" else original

    confidence = _policy_float(result.get("confidence", engine.get("confidence")))
    # Scanner confidence is commonly expressed as 54.09; policy expects 0.5409.
    if confidence is not None and confidence > 1.0:
        confidence /= 100.0

    status = str(result.get("status", "ANALYZED")).upper()
    entry = _policy_float(levels.get("entry_price", levels.get("entry_zone_high")))
    stop = _policy_float(levels.get("stop", levels.get("dynamic_stop")))
    price = _policy_float(result.get("current_price", engine.get("current_price")))
    data = DecisionInput(
        symbol=symbol,
        analytical_signal=probe,
        score=_policy_float(result.get("score", engine.get("score"))),
        probability_20d=_policy_float(engine.get("historical_probability_20d", historical.get("probability_positive_20d"))),
        reward_risk=_policy_float(engine.get("reward_risk_20d", historical.get("historical_excursion_ratio_20d"))),
        confidence=confidence,
        fallback_ratio=_policy_float(engine.get("fallback_ratio")),
        match_quality=engine.get("match_quality"),
        regime=result.get("regime", engine.get("cycle", engine.get("regime"))),
        appt=_policy_float(result.get("appt_x_20d", engine.get("appt_x_20d"))),
        benchmark_trust=result.get("benchmark_trust", engine.get("benchmark_trust")),
        benchmark_decision_grade=result.get("benchmark_trust_decision_grade", engine.get("benchmark_trust_decision_grade")),
        data_valid=status == "ANALYZED" and price is not None and price > 0,
        entry_valid=entry is not None and entry > 0,
        stop_valid=stop is not None and stop > 0 and entry is not None and stop < entry,
        source=str(result.get("engine", "CCE")),
    )
    decision = evaluate_decision(data).to_dict()
    decision.update({
        "mode": "SHADOW_ONLY",
        "original_signal": original,
        "analytical_signal_preserved": original,
        "eligibility_probe": "HYPOTHETICAL_ONLY" if original == "BUY_ON_CONFIRMATION" else "NONE",
        "policy_does_not_override_pipeline": True,
        "orders_submitted": False,
    })
    # Preserve the scanner's original signal. The policy result is audit-only.
    result["decision_policy_shadow"] = decision
    print(
        f"POLICY SHADOW | {symbol} | original={original} | "
        f"probe={probe} | policy={decision.get('operational_signal')} | "
        f"status={decision.get('decision_status')} | "
        f"failed={','.join(decision.get('failed_gates', [])) or '-'} | "
        "orders_submitted=False"
    )
    return decision
