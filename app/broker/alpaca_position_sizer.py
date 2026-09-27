"""
AI Trader - Capital-Aware Position Sizer V2.0

Responsibilities
----------------
1. Convert AI Trader confidence into a desired BUY quantity.
2. Adjust that desired quantity to the amount of strategy capital
   actually available at the current market price.

The module does not connect to Alpaca and never submits orders.

Sizing policy
-------------
<60%       -> 0
60-69.99%  -> 100
70-79.99%  -> 130
80-89.99%  -> 160
90-94.99%  -> 180
95-100%    -> 200
"""

from __future__ import annotations

import math


MIN_BUY_CONFIDENCE = 60.0

DESIRED_QUANTITY_BANDS = (
    (60.0, 100),
    (70.0, 130),
    (80.0, 160),
    (90.0, 180),
    (95.0, 200),
)


def _validate_confidence(confidence: float) -> float:
    confidence = float(confidence)
    if not 0.0 <= confidence <= 100.0:
        raise ValueError("confidence must be between 0 and 100.")
    return confidence


def calculate_buy_quantity(confidence: float) -> int:
    """Return the desired BUY quantity from AI Trader confidence."""
    confidence = _validate_confidence(confidence)

    if confidence < 60.0:
        return 0
    if confidence < 70.0:
        return 100
    if confidence < 80.0:
        return 130
    if confidence < 90.0:
        return 160
    if confidence < 95.0:
        return 180
    return 200


def sizing_band(confidence: float) -> str:
    """Return the confidence band associated with the desired quantity."""
    quantity = calculate_buy_quantity(confidence)

    if quantity == 0:
        return "NO_BUY"

    return {
        100: "60-69",
        130: "70-79",
        160: "80-89",
        180: "90-94",
        200: "95-100",
    }[quantity]


def calculate_executable_quantity(
    desired_quantity: int,
    available_capital: float,
    market_price: float,
) -> int:
    """
    Return the maximum whole-share quantity that can be funded.

    The executable quantity is never greater than desired_quantity.

    Formula:
        affordable_quantity = floor(available_capital / market_price)
        executable_quantity = min(desired_quantity, affordable_quantity)

    Returns 0 when available capital cannot fund one whole share.
    """
    desired_quantity = int(desired_quantity)
    available_capital = float(available_capital)
    market_price = float(market_price)

    if desired_quantity < 0:
        raise ValueError("desired_quantity must be >= 0.")

    if available_capital < 0:
        raise ValueError("available_capital must be >= 0.")

    if market_price <= 0:
        raise ValueError("market_price must be > 0.")

    if desired_quantity == 0 or available_capital == 0:
        return 0

    affordable_quantity = math.floor(available_capital / market_price)

    return min(desired_quantity, affordable_quantity)


def calculate_capital_aware_buy_quantity(
    confidence: float,
    available_capital: float,
    market_price: float,
) -> int:
    """
    Calculate the final executable BUY quantity.

    Step 1: confidence determines desired quantity.
    Step 2: available strategy capital and market price determine
            how many of those shares can actually be funded.
    """
    desired_quantity = calculate_buy_quantity(confidence)

    return calculate_executable_quantity(
        desired_quantity=desired_quantity,
        available_capital=available_capital,
        market_price=market_price,
    )


def sizing_summary(
    confidence: float,
    available_capital: float,
    market_price: float,
) -> dict:
    """Return a transparent sizing decision for logging/testing."""
    desired_quantity = calculate_buy_quantity(confidence)
    executable_quantity = calculate_executable_quantity(
        desired_quantity=desired_quantity,
        available_capital=available_capital,
        market_price=market_price,
    )

    return {
        "confidence": float(confidence),
        "band": sizing_band(confidence),
        "desired_quantity": desired_quantity,
        "available_capital": float(available_capital),
        "market_price": float(market_price),
        "executable_quantity": executable_quantity,
        "desired_notional": desired_quantity * float(market_price),
        "executable_notional": executable_quantity * float(market_price),
        "quantity_adjusted": executable_quantity < desired_quantity,
        "buy_allowed": executable_quantity > 0,
    }


if __name__ == "__main__":
    examples = (
        (59.99, 100_000, 100),
        (60.00, 100_000, 100),
        (80.00, 100_000, 100),
        (95.00, 100_000, 700),
        (95.00, 100_000, 500),
        (95.00, 60_000, 700),
        (95.00, 500, 700),
    )

    print("AI TRADER - CAPITAL-AWARE POSITION SIZER V2.0")
    print("=" * 60)

    for confidence, capital, price in examples:
        result = sizing_summary(confidence, capital, price)
        print(
            f"{confidence:6.2f}% | "
            f"desired={result['desired_quantity']:3d} | "
            f"capital=${capital:,.2f} | "
            f"price=${price:,.2f} | "
            f"executable={result['executable_quantity']:3d}"
        )
