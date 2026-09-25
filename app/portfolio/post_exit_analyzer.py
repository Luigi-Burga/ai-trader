"""
AI Trader - Post Exit Analyzer V1.0

Analyzes what happened after an actual trade exit.

This is deliberately separate from Prediction Tracker's forward prediction
evaluation. It answers:
    "After the exit, how much additional opportunity/risk was available?"

No look-ahead is used before the supplied exit index.
"""

from __future__ import annotations

from typing import Any, Optional

from app.portfolio.risk_x_engine import x_from_percent


VERSION = "1.0"


def _f(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        x = float(value)
        return x if x == x and abs(x) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _series_value(frame, column: str, index) -> Optional[float]:
    try:
        value = frame[column].loc[index]
        if hasattr(value, "iloc"):
            value = value.iloc[0]
        return _f(value)
    except Exception:
        return None


def analyze_post_exit(
    frame,
    exit_index,
    exit_price: float,
    direction: str = "BUY",
    risk_pct: Optional[float] = None,
    horizons=(1, 3, 5, 10, 20),
) -> dict[str, Any]:
    """
    Measure post-exit opportunity/risk from the exit fill.

    For LONG:
        opportunity = future high relative to exit
        risk        = future low relative to exit

    For SHORT/SELL:
        opportunity = future low relative to exit
        risk        = future high relative to exit
    """
    direction = str(direction or "BUY").upper()
    exit_price = _f(exit_price)

    if exit_price is None or exit_price <= 0:
        return {"status": "INVALID_EXIT"}

    try:
        pos = list(frame.index).index(exit_index)
    except Exception:
        return {"status": "EXIT_INDEX_NOT_FOUND"}

    observations = {}

    for horizon in horizons:
        end = min(pos + horizon, len(frame.index) - 1)
        if end <= pos:
            observations[f"+{horizon}d"] = None
            continue

        future = frame.iloc[pos + 1:end + 1]
        if future.empty:
            observations[f"+{horizon}d"] = None
            continue

        high = future["High"].astype(float)
        low = future["Low"].astype(float)

        if direction in {"SELL", "SHORT", "REDUCE"}:
            max_opportunity_pct = (exit_price - float(low.min())) / exit_price * 100.0
            max_risk_pct = (float(high.max()) - exit_price) / exit_price * 100.0
        else:
            max_opportunity_pct = (float(high.max()) - exit_price) / exit_price * 100.0
            max_risk_pct = (float(low.min()) - exit_price) / exit_price * 100.0

        observations[f"+{horizon}d"] = {
            "max_opportunity_pct": round(max_opportunity_pct, 6),
            "max_adverse_move_pct": round(max_risk_pct, 6),
            "max_opportunity_x": (
                round(x_from_percent(max_opportunity_pct, risk_pct), 6)
                if risk_pct else None
            ),
            "max_adverse_move_x": (
                round(x_from_percent(max_risk_pct, risk_pct), 6)
                if risk_pct else None
            ),
        }

    available = [
        x["max_opportunity_pct"]
        for x in observations.values()
        if isinstance(x, dict)
    ]
    max_opportunity = max(available) if available else None

    return {
        "version": VERSION,
        "status": "COMPLETE" if observations else "NO_DATA",
        "exit_price": exit_price,
        "direction": direction,
        "risk_pct": risk_pct,
        "observations": observations,
        "max_opportunity_after_exit_pct": (
            round(max_opportunity, 6) if max_opportunity is not None else None
        ),
        "interpretation": (
            "Potential early exit"
            if max_opportunity is not None and max_opportunity > 0
            else "No additional upside detected"
        ),
    }


def self_test() -> None:
    import pandas as pd

    idx = pd.date_range("2026-01-01", periods=8, freq="B")
    frame = pd.DataFrame(
        {
            "High": [100, 102, 104, 106, 105, 107, 109, 108],
            "Low": [98, 99, 101, 103, 102, 104, 106, 105],
            "Close": [99, 101, 103, 105, 104, 106, 108, 107],
        },
        index=idx,
    )

    result = analyze_post_exit(
        frame,
        idx[1],
        101.0,
        risk_pct=5.0,
        horizons=(1, 3, 5),
    )
    assert result["status"] == "COMPLETE"
    assert result["observations"]["+1d"]["max_opportunity_pct"] > 0


if __name__ == "__main__":
    self_test()
    print(f"Post Exit Analyzer V{VERSION} SELF-TEST: PASS")
