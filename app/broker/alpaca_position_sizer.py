"""AI Trader - Confidence Position Sizer V1.0

Maps AI Trader confidence to the fixed 100-200 share BUY sizing policy.
This module does not connect to Alpaca and never submits orders.
"""

from __future__ import annotations


MIN_BUY_CONFIDENCE = 60.0


def calculate_buy_quantity(confidence: float) -> int:
    """Return BUY quantity from AI Trader confidence."""
    confidence = float(confidence)
    if not 0.0 <= confidence <= 100.0:
        raise ValueError("confidence must be between 0 and 100.")

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
    q = calculate_buy_quantity(confidence)
    if q == 0:
        return "NO_BUY"
    return {100:"60-69",130:"70-79",160:"80-89",180:"90-94",200:"95-100"}[q]


if __name__ == "__main__":
    for c in (59.99,60,69.99,70,79.99,80,89.99,90,94.99,95,100):
        print(f"{c:6.2f}% -> {calculate_buy_quantity(c):2d} shares ({sizing_band(c)})")
