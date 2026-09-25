"""
AI Trader - Risk Based Position Sizer V1.0

Position size is determined from account risk and initial stop amplitude.

Formula:
    risk_budget = equity * max_risk_pct
    shares      = risk_budget / abs(entry - stop)

Optional caps prevent a risk-based calculation from consuming excessive
capital or exceeding a configured maximum position.
"""

from __future__ import annotations

from typing import Any, Optional


VERSION = "1.0"


def _f(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return default
        x = float(value)
        if x != x or abs(x) == float("inf"):
            return default
        return x
    except (TypeError, ValueError):
        return default


def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_price: float,
    max_risk_pct: float = 1.0,
    max_position_pct: Optional[float] = None,
    max_shares: Optional[float] = None,
) -> dict[str, Any]:
    equity = _f(account_equity)
    entry = _f(entry_price)
    stop = _f(stop_price)
    risk_pct = _f(max_risk_pct)

    if equity is None or entry is None or stop is None or risk_pct is None:
        return {"status": "INVALID_INPUT"}

    if equity <= 0 or entry <= 0 or stop <= 0 or risk_pct <= 0:
        return {"status": "INVALID_INPUT"}

    per_share_risk = abs(entry - stop)
    if per_share_risk <= 0:
        return {"status": "INVALID_STOP"}

    risk_budget = equity * risk_pct / 100.0
    shares = risk_budget / per_share_risk

    capital = shares * entry

    if max_position_pct is not None:
        cap = equity * _f(max_position_pct, 100.0) / 100.0
        shares = min(shares, cap / entry)
        capital = shares * entry

    if max_shares is not None:
        shares = min(shares, max(0.0, _f(max_shares, shares)))
        capital = shares * entry

    actual_risk = shares * per_share_risk
    actual_risk_pct = actual_risk / equity * 100.0

    return {
        "version": VERSION,
        "status": "OK",
        "account_equity": round(equity, 2),
        "entry_price": round(entry, 6),
        "stop_price": round(stop, 6),
        "risk_per_share": round(per_share_risk, 6),
        "max_risk_pct": round(risk_pct, 6),
        "risk_budget": round(risk_budget, 2),
        "shares": int(shares),
        "position_value": round(int(shares) * entry, 2),
        "actual_risk": round(int(shares) * per_share_risk, 2),
        "actual_risk_pct": round(
            int(shares) * per_share_risk / equity * 100.0, 6
        ),
        "capital_utilization_pct": round(
            int(shares) * entry / equity * 100.0, 6
        ),
    }


def enrich_position(
    position: dict[str, Any],
    account_equity: Optional[float] = None,
    max_risk_pct: float = 1.0,
) -> dict[str, Any]:
    """
    Non-destructive helper. If account_equity is absent, the position is
    returned with sizing status NOT_CONFIGURED.
    """
    out = dict(position)

    equity = account_equity
    if equity is None:
        equity = position.get("account_equity")

    entry = position.get("buy_price", position.get("entry_price"))
    stop = position.get("stop_price", position.get("stop"))

    if equity is None or entry is None or stop is None:
        out["risk_sizing"] = {
            "version": VERSION,
            "status": "NOT_CONFIGURED",
            "reason": "account_equity_and_stop_required",
        }
        return out

    out["risk_sizing"] = calculate_position_size(
        equity,
        entry,
        stop,
        max_risk_pct=max_risk_pct,
        max_position_pct=position.get("max_position_pct"),
        max_shares=position.get("max_shares"),
    )
    return out


def self_test() -> None:
    result = calculate_position_size(
        100_000,
        100,
        95,
        max_risk_pct=1.0,
    )
    assert result["shares"] == 200
    assert result["actual_risk"] == 1000.0


if __name__ == "__main__":
    self_test()
    print(f"Risk Based Position Sizer V{VERSION} SELF-TEST: PASS")
