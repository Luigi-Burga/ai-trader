"""Early recovery experiment V1.2: classify early vs confirmed setups; shadow-only."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class RecoveryConfig:
    min_score: float = 50.0
    min_probability_20d: float = 0.50
    min_reward_risk_20d: float = 0.95
    min_volume_ratio: float = 1.05
    min_rsi: float = 38.0
    max_atr_pct: float = 0.15
    min_technical_confirmations: int = 2
    confirmed_min_confirmations: int = 3


def _series(df: pd.DataFrame, name: str) -> pd.Series:
    cols = {str(c).lower(): c for c in df.columns}
    if name.lower() not in cols:
        raise ValueError(f"Missing required OHLCV column: {name}")
    value = df[cols[name.lower()]]
    if isinstance(value, pd.DataFrame): value = value.iloc[:, 0]
    return pd.to_numeric(value, errors="coerce")


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def evaluate_recovery_entry(df: pd.DataFrame, cce_result: Optional[Dict[str, Any]] = None,
                            config: RecoveryConfig = RecoveryConfig()) -> Dict[str, Any]:
    if df is None or len(df) < 60:
        return {"engine":"EARLY_RECOVERY_EXPERIMENT", "version":"1.2", "action":"NO_ENTRY", "tier":"NONE", "recommendation":"INSUFFICIENT_DATA", "reason":"insufficient_ohlcv_history", "checks":{}, "orders_submitted":False, "production_integrated":False}
    close, high, low, volume = (_series(df, n) for n in ("Close", "High", "Low", "Volume"))
    valid = pd.concat([close, high, low, volume], axis=1).dropna()
    if len(valid) < 60:
        return {"engine":"EARLY_RECOVERY_EXPERIMENT", "version":"1.2", "action":"NO_ENTRY", "tier":"NONE", "recommendation":"INSUFFICIENT_DATA", "reason":"insufficient_valid_ohlcv_history", "checks":{}, "orders_submitted":False, "production_integrated":False}
    close, high, low, volume = (valid.iloc[:, i].astype(float) for i in range(4))
    price = float(close.iloc[-1])
    prior_high = float(high.iloc[-6:-1].max())
    breakout = price > prior_high
    higher_low = bool(low.iloc[-1] > low.iloc[-2] > low.iloc[-3] and price > float(high.iloc[-2]))
    price_confirmation = bool(breakout or higher_low)
    vol_avg = float(volume.iloc[-21:-1].mean())
    vol_ratio = float(volume.iloc[-1] / vol_avg) if vol_avg > 0 else 0.0
    rsi = _rsi(close)
    rsi_now = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 0.0
    rsi_prev = float(rsi.iloc[-2]) if pd.notna(rsi.iloc[-2]) else 0.0
    rsi_recovering = bool(rsi_now >= config.min_rsi and rsi_now > rsi_prev)
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    macd_improving = bool(hist.iloc[-1] > hist.iloc[-2])
    volume_expansion = bool(vol_ratio >= config.min_volume_ratio)
    change5 = float(price / close.iloc[-6] - 1.0)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr_pct = float(tr.rolling(14).mean().iloc[-1] / price)
    cce = cce_result or {}
    score = float(cce.get("score", 0.0) or 0.0)
    hist_pattern = cce.get("historical_pattern") or {}
    prob = cce.get("historical_probability_20d", hist_pattern.get("probability_positive_20d", 0.0))
    rr = cce.get("reward_risk_20d", hist_pattern.get("historical_excursion_ratio_20d", 0.0))
    prob, rr = float(prob or 0.0), float(rr or 0.0)
    volatility_ok = bool(np.isfinite(atr_pct) and atr_pct <= config.max_atr_pct)
    historical_ok = bool(score >= config.min_score and prob >= config.min_probability_20d and rr >= config.min_reward_risk_20d)
    technical = {"price_confirmation":price_confirmation, "volume_expansion":volume_expansion,
                 "rsi_recovering":rsi_recovering, "macd_improving":macd_improving}
    count = sum(technical.values())
    eligible = historical_ok and volatility_ok and count >= config.min_technical_confirmations and (price_confirmation or (rsi_recovering and macd_improving))
    watch = historical_ok and volatility_ok and (rsi_recovering or macd_improving or price_confirmation)
    # V1.1 replay-informed tiers: avoid assuming more confirmations always means better entry.
    if eligible and price_confirmation and count == config.confirmed_min_confirmations:
        action, tier, recommendation, reason = "BUY_CANDIDATE", "CONFIRMED_RECOVERY", "CONFIRMED_RECOVERY_CANDIDATE", "price_confirmation_plus_three_technical_checks"
    elif eligible and not price_confirmation and count == config.min_technical_confirmations:
        action, tier, recommendation, reason = "BUY_CANDIDATE", "EARLY_RECOVERY", "EARLY_RECOVERY_CANDIDATE", "early_setup_with_two_checks_before_price_breakout"
    elif eligible:
        action, tier, recommendation, reason = "WATCH", "LATE_OR_UNCLEAR", "LATE_OR_UNCLEAR_RECOVERY", "setup_does_not_match_replay_supported_entry_tiers"
    elif watch:
        action, tier, recommendation, reason = "WATCH", "DEVELOPING", "EARLY_RECOVERY_WATCH", "recovery_evidence_emerging_but_entry_tier_not_met"
    else:
        action, tier, recommendation, reason = "NO_ENTRY", "NONE", "NO_RECOVERY_SETUP", "minimum_historical_or_volatility_or_momentum_conditions_not_met"
    checks = {**technical, "historical_evidence":historical_ok, "volatility_acceptable":volatility_ok,
              "technical_confirmations_met":count >= config.min_technical_confirmations}
    return {"engine":"EARLY_RECOVERY_EXPERIMENT", "version":"1.2", "action":action, "tier":tier,
        "recommendation":recommendation, "reason":reason, "checks":checks,
        "failed_checks":[k for k,v in checks.items() if not v],
        "metrics":{"price":round(price,4), "prior_5d_high":round(prior_high,4), "volume_ratio_20d":round(vol_ratio,3),
        "rsi14":round(rsi_now,2), "macd_histogram":round(float(hist.iloc[-1]),5), "macd_histogram_previous":round(float(hist.iloc[-2]),5),
        "change_5d":round(change5,4), "atr_pct":round(atr_pct,4), "cce_score":round(score,2),
        "historical_probability_20d":round(prob,4), "reward_risk_20d":round(rr,3), "technical_confirmations":count},
        "safety":{"mode":"SHADOW_ONLY", "orders_submitted":False, "production_integrated":False}}


def apply_experimental_decision_overlay(cce_result: Dict[str, Any], recovery_result: Dict[str, Any]) -> Dict[str, Any]:
    action = recovery_result.get("action")
    cce_signal = str(cce_result.get("signal", "WATCH")).upper()
    if action == "BUY_CANDIDATE": signal, path = recovery_result.get("tier", "EARLY_RECOVERY"), "EARLY_RECOVERY_EXPERIMENT"
    elif action == "WATCH": signal, path = recovery_result.get("recommendation", "EARLY_RECOVERY_WATCH"), "EARLY_RECOVERY_EXPERIMENT"
    else: signal, path = cce_signal, "EXISTING_CCE_PATH"
    return {"engine":"EXPERIMENTAL_DECISION_OVERLAY", "version":"1.2", "signal":signal,
        "operational_signal":signal, "source_signal":cce_signal, "decision_path":path,
        "reason":recovery_result.get("reason", "recovery_not_confirmed"),
        "recommendation":recovery_result.get("recommendation", "NO_RECOVERY_SETUP"),
        "orders_submitted":False, "production_integrated":False}
