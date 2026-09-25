"""
AI Trader - Risk in X Engine V1.0

Defines a normalized unit of initial trade risk:
    1X = absolute distance from entry to initial stop.

All percentage outcomes can then be expressed in X so trades with
different volatility/stops can be compared consistently.
"""

from __future__ import annotations

from typing import Any, Optional


VERSION = "1.0"


def _f(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        x = float(value)
        return x if x == x and abs(x) != float("inf") else None
    except (TypeError, ValueError):
        return None


def risk_percent(entry_price: Any, stop_price: Any) -> Optional[float]:
    entry = _f(entry_price)
    stop = _f(stop_price)
    if entry is None or stop is None or entry <= 0:
        return None
    return abs(entry - stop) / entry * 100.0


def x_from_percent(outcome_percent: Any, risk_pct: Any) -> Optional[float]:
    outcome = _f(outcome_percent)
    risk = _f(risk_pct)
    if outcome is None or risk is None or risk <= 0:
        return None
    return outcome / risk


def build_risk_context(
    entry_price: Any,
    stop_price: Any,
    target_price: Any = None,
) -> dict[str, Any]:
    entry = _f(entry_price)
    stop = _f(stop_price)
    target = _f(target_price)

    risk_pct = risk_percent(entry, stop)
    target_pct = (
        (target - entry) / entry * 100.0
        if entry is not None and target is not None and entry > 0
        else None
    )

    return {
        "version": VERSION,
        "entry_price": entry,
        "initial_stop": stop,
        "risk_pct": round(risk_pct, 6) if risk_pct is not None else None,
        "target_price": target,
        "target_pct": round(target_pct, 6) if target_pct is not None else None,
        "target_x": (
            round(x_from_percent(target_pct, risk_pct), 6)
            if target_pct is not None and risk_pct is not None
            else None
        ),
        "unit_x": "initial entry-to-stop risk",
    }


def enrich_observation_with_x(
    observation: dict[str, Any],
    risk_pct: Optional[float],
) -> dict[str, Any]:
    out = dict(observation)
    out["return_x"] = (
        round(x_from_percent(out.get("return_pct"), risk_pct), 6)
        if risk_pct is not None else None
    )
    out["mfe_x"] = (
        round(x_from_percent(out.get("mfe_pct"), risk_pct), 6)
        if risk_pct is not None else None
    )
    out["mae_x"] = (
        round(x_from_percent(out.get("mae_pct"), risk_pct), 6)
        if risk_pct is not None else None
    )
    return out


def self_test() -> None:
    ctx = build_risk_context(100, 95, 110)
    assert ctx["risk_pct"] == 5.0
    assert ctx["target_x"] == 2.0

    obs = enrich_observation_with_x(
        {"return_pct": 10.0, "mfe_pct": 12.0, "mae_pct": -4.0},
        5.0,
    )
    assert obs["return_x"] == 2.0
    assert obs["mfe_x"] == 2.4
    assert obs["mae_x"] == -0.8


if __name__ == "__main__":
    self_test()
    print(f"Risk X Engine V{VERSION} SELF-TEST: PASS")
