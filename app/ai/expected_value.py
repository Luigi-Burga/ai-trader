"""
AI Trader - Expected Value / APPT Engine V1.0

APPT is expressed in units of initial trade risk (X).

For a set of historical forward returns:
    EV = P(win)*AvgWin + P(loss)*AvgLoss

APPT_X = EV / InitialRisk%

This is an empirical expectancy metric, not a probability forecast.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


VERSION = "1.0"


def _finite(values: Iterable[Any]) -> list[float]:
    out = []
    for value in values:
        try:
            x = float(value)
            if x == x and abs(x) != float("inf"):
                out.append(x)
        except (TypeError, ValueError):
            pass
    return out


def calculate_appt(
    returns: Iterable[Any],
    risk_pct: Optional[float] = None,
) -> dict[str, Any]:
    vals = _finite(returns)

    if not vals:
        return {
            "version": VERSION,
            "status": "NO_DATA",
            "sample_size": 0,
            "expected_return_pct": None,
            "appt_x": None,
        }

    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x <= 0]

    p_win = len(wins) / len(vals)
    p_loss = len(losses) / len(vals)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    expected_return = p_win * avg_win + p_loss * avg_loss

    appt_x = (
        expected_return / float(risk_pct)
        if risk_pct is not None and float(risk_pct) > 0
        else None
    )

    return {
        "version": VERSION,
        "status": "COMPLETE",
        "sample_size": len(vals),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(p_win, 6),
        "avg_win_pct": round(avg_win, 6),
        "avg_loss_pct": round(avg_loss, 6),
        "expected_return": round(expected_return, 6),
        "risk_unit": (
            round(float(risk_pct), 6)
            if risk_pct is not None else None
        ),
        "appt_x": round(appt_x, 6) if appt_x is not None else None,
    }


def self_test() -> None:
    result = calculate_appt([10, 5, -5, -5], risk_pct=5)
    assert result["win_rate"] == 0.5
    assert result["expected_return"] == 1.25
    assert result["appt_x"] == 0.25


if __name__ == "__main__":
    self_test()
    print(f"Expected Value / APPT V{VERSION} SELF-TEST: PASS")
