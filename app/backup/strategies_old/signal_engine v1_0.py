"""
Final Signal Engine.

The characteristic curve is the primary behavioral signal.
Legacy RSI/MACD logic is retained only as a fallback when the
characteristic result is unavailable.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import ta


def _legacy_signal(df: pd.DataFrame) -> Dict[str, object]:
    if df is None or df.empty:
        return {"signal": "NO_DATA", "score": 0, "confidence": "LOW", "reasons": []}
    if len(df) < 30:
        return {"signal": "INSUFFICIENT_DATA", "score": 0, "confidence": "LOW", "reasons": []}

    x = df.copy()
    close = x["Close"]
    x["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    x["macd"] = macd.macd()
    x["macd_signal"] = macd.macd_signal()

    rsi = float(x["rsi"].iloc[-1])
    macd_v = float(x["macd"].iloc[-1])
    macd_s = float(x["macd_signal"].iloc[-1])

    score = 0
    reasons = []
    if rsi < 30:
        score += 1
        reasons.append("RSI Oversold")
    if macd_v > macd_s:
        score += 1
        reasons.append("MACD Bullish")

    if score >= 2:
        signal, confidence = "BUY", "MEDIUM"
    elif score == 1:
        signal, confidence = "WATCH", "LOW"
    else:
        signal, confidence = "HOLD", "LOW"

    return {
        "signal": signal,
        "score": score,
        "confidence": confidence,
        "reasons": reasons,
        "rsi": round(rsi, 2),
        "macd": round(macd_v, 4),
        "macd_signal": round(macd_s, 4),
    }


def generate_signal(
    df: Optional[pd.DataFrame] = None,
    characteristic_result: Optional[Dict[str, object]] = None,
    technical_result: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Characteristic Curve is authoritative when valid.

    This function returns a dictionary so the scanner can preserve
    the complete analytical context.
    """
    if characteristic_result:
        signal = characteristic_result.get("signal")
        if signal not in {"NO_DATA", "INSUFFICIENT_HISTORY"}:
            return {
                "signal": signal,
                "score": characteristic_result.get("score", 0),
                "confidence": characteristic_result.get("confidence", 0),
                "source": "characteristic_curve",
                "reasons": [
                    f"Cycle={characteristic_result.get('cycle')}",
                    f"20D probability={characteristic_result.get('horizons', {}).get('20d', {}).get('prob_positive', 0)*100:.1f}%",
                    f"20D R/R={characteristic_result.get('reward_risk_20d', 0):.2f}",
                ],
            }

    if technical_result:
        return technical_result

    return _legacy_signal(df)
