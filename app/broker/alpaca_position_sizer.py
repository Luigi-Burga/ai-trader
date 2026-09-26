"""AI Trader - Confidence Position Sizer V1.0

Maps AI Trader confidence to the fixed 10-20 share BUY sizing policy.
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
        return 10
    if confidence < 80.0:
        return 13
    if confidence < 90.0:
        return 16
    if confidence < 95.0:
        return 18
    return 20


def sizing_band(confidence: float) -> str:
    q = calculate_buy_quantity(confidence)
    if q == 0:
        return "NO_BUY"
    return {10:"60-69",13:"70-79",16:"80-89",18:"90-94",20:"95-100"}[q]


if __name__ == "__main__":
    for c in (59.99,60,69.99,70,79.99,80,89.99,90,94.99,95,100):
        print(f"{c:6.2f}% -> {calculate_buy_quantity(c):2d} shares ({sizing_band(c)})")
