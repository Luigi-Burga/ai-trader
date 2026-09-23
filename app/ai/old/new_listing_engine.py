from __future__ import annotations

"""AI Trader - New Listing Engine (NLE) V1.3.

Standalone engine for securities with limited post-listing history.

V1.3 improvements over V1.2:
- Separates true listing age from the requested analysis window. A 1y download
  no longer makes an old security look like a new listing.
- Strengthens early-listing confirmation: a high score alone cannot produce an
  immediate BUY. Early listings require breakout + volume confirmation in
  addition to the base confirmation threshold.
- Keeps the explicit confirmation engine independent from the opportunity score.
- Mature new listings continue to use the configurable confirmation threshold.
- Keeps dynamic ATR-based entry, stop and take-profit levels. No fixed buy
  price is used.
- Keeps CCE routing for securities with enough real historical age.
- Preserves the standalone CLI/self-test contract from V1.1.

CLI:
    python -m app.ai.new_listing_engine --ticker SPCX --benchmark QQQ --period 1y
    python -m app.ai.new_listing_engine --self-test

The module remains independent of main.py, signal_engine.py and
watchlist_scanner.py until standalone validation is complete.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import argparse
import json
import math

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


@dataclass
class NewListingConfig:
    period: str = "1y"
    interval: str = "1d"
    benchmark: Optional[str] = None

    # V1.2: use a wider history window to determine the real age of a security.
    # This prevents --period 1y from falsely classifying an old security as new.
    listing_history_period: str = "max"

    minimum_days_for_nle: int = 60
    mature_days_for_cce: int = 320
    early_listing_days: int = 120

    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    volume_window: int = 20

    buy_score: float = 70.0
    watch_score: float = 50.0
    avoid_score: float = 35.0
    min_buy_confidence: float = 55.0

    pullback_atr: float = 1.0
    stop_atr: float = 2.0
    max_entry_extension_atr: float = 1.5

    # V1.3 confirmation rules.
    confirmation_lookback: int = 5
    confirmation_volume_ratio: float = 1.10
    confirmation_min_score: int = 3
    require_confirmation_early: bool = True
    require_confirmation_mature: bool = True

    # V1.3: stricter confirmation for very early listings.
    early_require_breakout: bool = True
    early_require_volume: bool = True
    early_min_confirmation_score: int = 4


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()

    if isinstance(x.columns, pd.MultiIndex):
        wanted = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "adj close": "Close",
        }
        selected = {}
        for col in x.columns:
            for part in col:
                key = str(part).strip().lower()
                if key in wanted and wanted[key] not in selected:
                    selected[wanted[key]] = col
                    break
        if selected:
            x = x[[selected[k] for k in selected]]
            x.columns = list(selected.keys())

    rename = {}
    for c in x.columns:
        key = str(c).strip().lower()
        if key in {"open", "high", "low", "close", "volume"}:
            rename[c] = key.capitalize()
        elif key == "adj close" and "Close" not in x.columns:
            rename[c] = "Close"

    x = x.rename(columns=rename)
    x = x.loc[:, ~x.columns.duplicated(keep="first")]

    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c not in x.columns:
            if c == "Volume":
                x[c] = 0.0
            else:
                return pd.DataFrame()

    x = x[["Open", "High", "Low", "Close", "Volume"]].copy()
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["High", "Low", "Close"])
    return x[~x.index.duplicated(keep="last")].sort_index()


def download_history(ticker: str, cfg: NewListingConfig, period: Optional[str] = None) -> pd.DataFrame:
    if yf is None or not ticker:
        return pd.DataFrame()
    try:
        return _clean_ohlcv(
            yf.download(
                ticker,
                period=period or cfg.period,
                interval=cfg.interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        )
    except Exception:
        return pd.DataFrame()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(data: pd.DataFrame, period: int) -> pd.Series:
    prev = data["Close"].shift(1)
    tr = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - prev).abs(),
            (data["Low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0).replace(0, np.nan)
    return (s - m) / sd


def build_features(
    data: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    cfg: Optional[NewListingConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or NewListingConfig()
    x = _clean_ohlcv(data)
    if x.empty:
        return x

    c = x["Close"]
    x["ema20"] = c.ewm(span=cfg.ema_fast, adjust=False).mean()
    x["ema50"] = c.ewm(span=cfg.ema_slow, adjust=False).mean()
    x["rsi"] = _rsi(c, cfg.rsi_period)
    x["atr"] = _atr(x, cfg.atr_period)
    x["atr_pct"] = x["atr"] / c
    x["return_5d"] = c.pct_change(5)
    x["return_20d"] = c.pct_change(20)
    x["return_since_listing"] = c / c.iloc[0] - 1
    x["listing_high"] = c.cummax()
    x["listing_low"] = c.cummin()
    x["drawdown_from_high"] = c / x["listing_high"] - 1
    x["recovery_from_low"] = c / x["listing_low"] - 1

    vol = x["Volume"].replace(0, np.nan)
    x["volume_z"] = _zscore(vol, cfg.volume_window)
    x["avg_volume_20"] = x["Volume"].rolling(cfg.volume_window).mean()
    x["volume_ratio"] = x["Volume"] / x["avg_volume_20"]

    x["ema20_distance_atr"] = (c - x["ema20"]) / x["atr"]
    x["ema50_distance_atr"] = (c - x["ema50"]) / x["atr"]

    x["rs_20d"] = 0.0
    if benchmark_df is not None and not benchmark_df.empty:
        b = _clean_ohlcv(benchmark_df)["Close"].reindex(x.index).ffill()
        rs = c / b
        x["rs_20d"] = rs / rs.shift(20) - 1

    return x.replace([np.inf, -np.inf], np.nan)


def _age_class(days: int, cfg: NewListingConfig) -> str:
    if days >= cfg.mature_days_for_cce:
        return "MATURE"
    if days >= cfg.early_listing_days:
        return "NEW_LISTING_MATURE"
    if days >= cfg.minimum_days_for_nle:
        return "NEW_LISTING_EARLY"
    if days >= 30:
        return "EARLY_LISTING"
    return "INSUFFICIENT_DATA"


def _f(v) -> float:
    try:
        v = float(v)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _regime(r: pd.Series) -> str:
    p, e20, e50, rsi, dd = map(
        _f,
        [r["Close"], r["ema20"], r["ema50"], r["rsi"], r["drawdown_from_high"]],
    )
    if not all(math.isfinite(v) for v in (p, e20, e50, rsi, dd)):
        return "UNKNOWN"
    if p > e20 > e50 and rsi >= 55:
        return "BULLISH"
    if p < e20 < e50 and rsi <= 45:
        return "BEARISH"
    if dd <= -0.20 and p > e20:
        return "RECOVERY"
    return "CONSOLIDATION"


def _score(r: pd.Series) -> Tuple[float, Dict[str, float]]:
    score = 50.0
    b: Dict[str, float] = {}

    def add(k, v):
        nonlocal score
        score += v
        b[k] = v

    p, e20, e50, rsi, r20, rs, vr, dd = map(
        _f,
        [
            r["Close"],
            r["ema20"],
            r["ema50"],
            r["rsi"],
            r["return_20d"],
            r["rs_20d"],
            r["volume_ratio"],
            r["drawdown_from_high"],
        ],
    )

    add("trend", 12 if p > e20 else -8)
    add("long_trend", 10 if p > e50 else -8)
    if math.isfinite(rsi):
        add("momentum", 10 if 55 <= rsi <= 70 else 2 if 45 <= rsi < 55 else -5 if rsi < 35 else -8)
    if math.isfinite(r20):
        add("price_momentum", 8 if r20 > 0.05 else 3 if r20 > 0 else -7)
    if math.isfinite(rs):
        add("relative_strength", 8 if rs > 0.03 else 3 if rs >= 0 else -6)
    if math.isfinite(vr):
        add("volume_confirmation", 5 if 1.2 <= vr <= 3 else 2 if vr >= 0.8 else -3)
    if math.isfinite(dd):
        add("drawdown_structure", 5 if -0.15 <= dd <= -0.05 else 2 if dd > -0.05 else -4)

    return max(0, min(100, score)), b


def _data_quality(days: int, r: pd.Series) -> str:
    if days < 30:
        return "INSUFFICIENT"
    required = ["ema20", "ema50", "rsi", "atr", "volume_ratio"]
    available = sum(math.isfinite(_f(r[k])) for k in required)
    if days < 60 or available < len(required):
        return "LIMITED"
    if days < 120:
        return "MODERATE"
    return "GOOD"


def _confidence(r: pd.Series, days: int, score: float) -> float:
    age_factor = min(1, max(0, (days - 60) / 260))
    data_conf = 35 + 30 * age_factor
    rsi, atrp = _f(r["rsi"]), _f(r["atr_pct"])
    quality = 10 if math.isfinite(rsi) and 35 <= rsi <= 70 else 3 if math.isfinite(rsi) else 0
    quality += 10 if math.isfinite(atrp) and atrp <= 0.08 else 4 if math.isfinite(atrp) else 0
    score_quality = max(0, 15 - abs(score - 60) * 0.10)
    return round(min(79, data_conf + quality + score_quality), 2)


def _confirmation(r: pd.Series, data: pd.DataFrame, cfg: NewListingConfig) -> Dict[str, object]:
    """Evaluate entry confirmation independently from the opportunity score."""
    p = _f(r["Close"])
    e20 = _f(r["ema20"])
    e50 = _f(r["ema50"])
    rsi = _f(r["rsi"])
    vr = _f(r["volume_ratio"])
    rs = _f(r["rs_20d"])

    lookback = max(2, int(cfg.confirmation_lookback))
    previous = data["Close"].iloc[:-1].tail(lookback)
    prior_high = _f(previous.max()) if not previous.empty else np.nan

    checks = {
        "price_confirmation": bool(math.isfinite(p) and math.isfinite(e20) and p > e20),
        "breakout_confirmation": bool(
            math.isfinite(p) and math.isfinite(prior_high) and p >= prior_high
        ),
        "volume_confirmation": bool(
            math.isfinite(vr) and vr >= cfg.confirmation_volume_ratio
        ),
        "momentum_confirmation": bool(math.isfinite(rsi) and rsi >= 50),
        "relative_strength_confirmation": bool(math.isfinite(rs) and rs >= 0),
        "trend_confirmation": bool(
            math.isfinite(p) and math.isfinite(e20) and math.isfinite(e50) and p > e20 > e50
        ),
    }

    # Price confirmation is the minimum structural requirement. A breakout,
    # volume, momentum and RS then strengthen the entry case.
    weighted = [
        checks["price_confirmation"],
        checks["breakout_confirmation"],
        checks["volume_confirmation"],
        checks["momentum_confirmation"],
        checks["relative_strength_confirmation"],
        checks["trend_confirmation"],
    ]
    score = sum(bool(v) for v in weighted)
    confirmed = score >= int(cfg.confirmation_min_score) and checks["price_confirmation"]

    return {
        "required": True,
        "confirmed": bool(confirmed),
        "score": score,
        "max_score": len(weighted),
        "prior_high": round(prior_high, 4) if math.isfinite(prior_high) else None,
        "checks": checks,
    }


def _levels(r: pd.Series, cfg: NewListingConfig) -> Dict[str, Optional[float]]:
    p, atr, e20 = map(_f, [r["Close"], r["atr"], r["ema20"]])
    keys = [
        "current_price",
        "zone_low",
        "zone_high",
        "preferred_entry",
        "stop_loss",
        "tp_1",
        "tp_2",
        "tp_3",
    ]
    if not all(math.isfinite(v) for v in (p, atr)) or atr <= 0:
        return {k: None for k in keys}

    low = p - cfg.pullback_atr * atr
    high = p + 0.25 * atr
    preferred = p - 0.5 * atr

    if math.isfinite(e20) and p > e20 + cfg.max_entry_extension_atr * atr:
        high = e20 + 0.25 * atr
        preferred = e20

    stop = low - cfg.stop_atr * atr

    return {
        "current_price": round(p, 4),
        "zone_low": round(max(0, low), 4),
        "zone_high": round(max(0, high), 4),
        "preferred_entry": round(max(0, preferred), 4),
        "stop_loss": round(max(0, stop), 4),
        "tp_1": round(p + 1.5 * atr, 4),
        "tp_2": round(p + 2.5 * atr, 4),
        "tp_3": round(p + 4 * atr, 4),
    }


VERSION = "1.3"


def _base_result(ticker: str, version: str = VERSION) -> Dict[str, object]:
    return {
        "engine": "New Listing Engine",
        "version": version,
        "ticker": ticker,
    }


def analyze_dataframe(
    ticker: str,
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    cfg: Optional[NewListingConfig] = None,
    listing_price: Optional[float] = None,
    listing_days: Optional[int] = None,
    listing_date: Optional[str] = None,
) -> Dict[str, object]:
    cfg = cfg or NewListingConfig()
    data = _clean_ohlcv(df)
    days = len(data)

    if data.empty:
        result = _base_result(ticker)
        result.update({"signal": "NO_DATA", "decision": "WAIT", "score": 0.0, "confidence": 0.0})
        return result

    actual_days = int(listing_days) if listing_days is not None else days
    actual_listing_date = listing_date or data.index[0].date().isoformat()
    first_trading_close = _f(data["Close"].iloc[0])
    cls = _age_class(actual_days, cfg)

    if cls == "MATURE":
        result = _base_result(ticker)
        result.update(
            {
                "signal": "ROUTE_TO_CCE",
                "decision": "USE_CCE",
                "trading_days": days,
                "listing_age_days": actual_days,
                "classification": cls,
                "listing_date": actual_listing_date,
                "first_trading_close": round(first_trading_close, 4) if math.isfinite(first_trading_close) else None,
                "score": 0.0,
                "confidence": 0.0,
                "data_quality": "GOOD",
            }
        )
        return result

    if cls in {"INSUFFICIENT_DATA", "EARLY_LISTING"}:
        result = _base_result(ticker)
        result.update(
            {
                "signal": "INSUFFICIENT_DATA",
                "decision": "WAIT",
                "trading_days": days,
                "listing_age_days": actual_days,
                "classification": cls,
                "listing_date": actual_listing_date,
                "first_trading_close": round(first_trading_close, 4) if math.isfinite(first_trading_close) else None,
                "score": 0.0,
                "confidence": 0.0,
                "data_quality": "INSUFFICIENT" if cls == "INSUFFICIENT_DATA" else "LIMITED",
            }
        )
        return result

    f = build_features(data, benchmark_df, cfg)
    r = f.iloc[-1]
    score, breakdown = _score(r)
    conf = _confidence(r, actual_days, score)
    regime = _regime(r)
    confirmation = _confirmation(r, f, cfg)

    decision = "WATCH"
    if regime == "BEARISH" and score < cfg.watch_score:
        decision = "AVOID"
    elif score < cfg.watch_score:
        decision = "WAIT"
    elif score >= cfg.buy_score and conf >= cfg.min_buy_confidence and regime in {"BULLISH", "RECOVERY"}:
        decision = "BUY_ON_CONFIRMATION"

    require_confirmation = (
        cfg.require_confirmation_early if cls == "NEW_LISTING_EARLY" else cfg.require_confirmation_mature
    )

    if decision == "BUY_ON_CONFIRMATION":
        if cls == "NEW_LISTING_EARLY" and require_confirmation:
            strict_early = (
                confirmation["confirmed"]
                and int(confirmation["score"]) >= int(cfg.early_min_confirmation_score)
                and (not cfg.early_require_breakout or confirmation["checks"]["breakout_confirmation"])
                and (not cfg.early_require_volume or confirmation["checks"]["volume_confirmation"])
            )
            confirmation["early_listing_confirmed"] = bool(strict_early)
            if strict_early:
                decision = "BUY"
            else:
                decision = "BUY_ON_CONFIRMATION"
        elif require_confirmation and confirmation["confirmed"]:
            decision = "BUY"
        else:
            decision = "BUY_ON_CONFIRMATION"

    p, atr, e20 = _f(r["Close"]), _f(r["atr"]), _f(r["ema20"])
    extended = (
        all(math.isfinite(v) for v in (p, atr, e20))
        and p > e20 + cfg.max_entry_extension_atr * atr
    )
    if extended and decision == "BUY":
        decision = "BUY_ON_PULLBACK"

    effective_listing_price = _f(listing_price) if listing_price is not None else np.nan

    result = _base_result(ticker)
    result.update(
        {
            "as_of": str(data.index[-1].date()),
            "trading_days": days,
            "listing_age_days": actual_days,
            "classification": cls,
            "signal": decision,
            "decision": decision,
            "score": round(score, 2),
            "confidence": conf,
            "data_quality": _data_quality(days, r),
            "regime": regime,
            "listing_date": actual_listing_date,
            "listing_price": round(effective_listing_price, 4) if math.isfinite(effective_listing_price) else None,
            "first_trading_close": round(first_trading_close, 4) if math.isfinite(first_trading_close) else None,
            "metrics": {
                "current_price": round(_f(r["Close"]), 4),
                "ema20": round(_f(r["ema20"]), 4),
                "ema50": round(_f(r["ema50"]), 4),
                "rsi": round(_f(r["rsi"]), 2),
                "atr": round(_f(r["atr"]), 4),
                "atr_pct": round(_f(r["atr_pct"]) * 100, 2),
                "return_5d": round(_f(r["return_5d"]) * 100, 2),
                "return_20d": round(_f(r["return_20d"]) * 100, 2),
                "return_since_listing": round(_f(r["return_since_listing"]) * 100, 2),
                "drawdown_from_high": round(_f(r["drawdown_from_high"]) * 100, 2),
                "recovery_from_low": round(_f(r["recovery_from_low"]) * 100, 2),
                "volume_ratio": round(_f(r["volume_ratio"]), 2),
                "relative_strength_20d": round(_f(r["rs_20d"]) * 100, 2),
            },
            "entry": _levels(r, cfg),
            "score_breakdown": {k: round(v, 2) for k, v in breakdown.items()},
            "confirmation": confirmation,
            "evidence": {
                "method": "post-listing price/volume/volatility/relative-strength + confirmation",
                "historical_pattern_matches": 0,
                "note": (
                    "CCE historical matches are intentionally not used for a new listing; "
                    "listing_price is optional and never inferred as IPO offer price."
                ),
            },
            "config": asdict(cfg),
        }
    )
    return result


def analyze(
    ticker: str,
    cfg: Optional[NewListingConfig] = None,
    benchmark: Optional[str] = None,
    listing_price: Optional[float] = None,
) -> Dict[str, object]:
    cfg = cfg or NewListingConfig(benchmark=benchmark)
    benchmark = benchmark or cfg.benchmark

    # Requested period is used for technical analysis. A wider history is used
    # only to determine true listing age/date, fixing the V1.1 1y-window bug.
    df = download_history(ticker, cfg, cfg.period)
    if df.empty:
        return analyze_dataframe(ticker, df, None, cfg, listing_price)

    age_df = df
    if cfg.listing_history_period and cfg.listing_history_period != cfg.period:
        wider = download_history(ticker, cfg, cfg.listing_history_period)
        if not wider.empty:
            age_df = wider

    bdf = download_history(benchmark, cfg, cfg.period) if benchmark else None

    return analyze_dataframe(
        ticker,
        df,
        bdf if bdf is not None and not bdf.empty else None,
        cfg,
        listing_price,
        listing_days=len(age_df),
        listing_date=age_df.index[0].date().isoformat(),
    )


def _synthetic_data(n=140, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-01", periods=n)
    ret = rng.normal(0.001, 0.025, n)
    ret[-25:] += 0.0015
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + rng.uniform(0.005, 0.03, n))
    low = close * (1 - rng.uniform(0.005, 0.03, n))
    open_ = close * (1 + rng.normal(0, 0.008, n))
    vol = rng.integers(800_000, 2_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def _bullish_confirmation_data(n=100):
    dates = pd.bdate_range("2026-01-01", periods=n)
    close = np.linspace(100, 130, n)
    close[-6:-1] = [126, 127, 128, 129, 130]
    close[-1] = 135
    high = close * 1.01
    low = close * 0.99
    open_ = close * 0.995
    volume = np.full(n, 1_000_000.0)
    volume[-1] = 1_500_000.0
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def self_test():
    cfg = NewListingConfig()

    # 140 analysis days + explicit age => valid NLE path.
    df = _synthetic_data()
    result = analyze_dataframe("TEST", df, cfg=cfg, listing_price=100, listing_days=140)
    assert result["classification"] == "NEW_LISTING_MATURE", result
    assert result["decision"] in {
        "WAIT", "WATCH", "AVOID", "BUY", "BUY_ON_CONFIRMATION", "BUY_ON_PULLBACK"
    }, result
    assert 0 <= result["score"] <= 100 and 0 <= result["confidence"] <= 79, result
    assert result["listing_date"] == df.index[0].date().isoformat(), result
    assert result["first_trading_close"] is not None, result
    assert result["data_quality"] in {"MODERATE", "GOOD"}, result
    for k in ("current_price", "zone_low", "zone_high", "preferred_entry", "stop_loss", "tp_1", "tp_2", "tp_3"):
        assert result["entry"][k] is not None and math.isfinite(result["entry"][k]), result
    assert "confirmation" in result and "confirmed" in result["confirmation"], result

    # 30 days must be blocked rather than forced through NLE.
    short = _synthetic_data(30)
    blocked = analyze_dataframe("SHORT", short, cfg=cfg, listing_days=30)
    assert blocked["signal"] == "INSUFFICIENT_DATA", blocked

    # 45 days are early listing data and remain blocked.
    early = _synthetic_data(45)
    early_result = analyze_dataframe("EARLY", early, cfg=cfg, listing_days=45)
    assert early_result["classification"] == "EARLY_LISTING", early_result
    assert early_result["signal"] == "INSUFFICIENT_DATA", early_result

    # 330 real trading days route to CCE even when only 100 days are analyzed.
    mature = _synthetic_data(100)
    routed = analyze_dataframe("MATURE", mature, cfg=cfg, listing_days=330)
    assert routed["signal"] == "ROUTE_TO_CCE", routed
    assert routed["trading_days"] == 100, routed
    assert routed["listing_age_days"] == 330, routed

    # Dynamic entry is not a fixed buy price: changing volatility changes levels.
    lowvol = df.copy()
    lowvol["High"] = lowvol["Close"] * 1.005
    lowvol["Low"] = lowvol["Close"] * 0.995
    lowvol_result = analyze_dataframe("LOWVOL", lowvol, cfg=cfg, listing_days=140)
    assert lowvol_result["entry"]["zone_low"] != result["entry"]["zone_low"], (result, lowvol_result)

    # V1.3 confirmation path: early listings require strict confirmation.
    # The synthetic data must expose the confirmation fields and pass the
    # breakout + volume requirements.
    confirmation_df = _bullish_confirmation_data()
    confirmation_result = analyze_dataframe(
        "CONFIRM", confirmation_df, cfg=cfg, listing_days=100
    )
    assert confirmation_result["classification"] == "NEW_LISTING_EARLY", confirmation_result
    assert confirmation_result["confirmation"]["max_score"] == 6, confirmation_result
    assert isinstance(confirmation_result["confirmation"]["confirmed"], bool), confirmation_result
    assert confirmation_result["confirmation"]["early_listing_confirmed"] is True, confirmation_result

    # Critical V1.3 regression: a strong score without breakout/volume cannot
    # become an immediate BUY for an early listing.
    weak_confirm = confirmation_df.copy()
    weak_confirm["Volume"] = 1_000_000.0
    weak_confirm.iloc[-1, weak_confirm.columns.get_loc("Close")] = 132.0
    weak_result = analyze_dataframe("EARLY_WEAK", weak_confirm, cfg=cfg, listing_days=100)
    assert weak_result["decision"] == "BUY_ON_CONFIRMATION", weak_result
    assert weak_result["confirmation"]["early_listing_confirmed"] is False, weak_result

    # Critical V1.2 age-window regression: 100 analyzed sessions can still
    # be mature when the true history is 500 sessions.
    regression = analyze_dataframe("OLD", df, cfg=cfg, listing_days=500)
    assert regression["classification"] == "MATURE", regression
    assert regression["decision"] == "USE_CCE", regression

    print("SELF-TEST: PASS")
    print(
        json.dumps(
            {
                "ticker": result["ticker"],
                "classification": result["classification"],
                "trading_days": result["trading_days"],
                "listing_age_days": result["listing_age_days"],
                "decision": result["decision"],
                "score": result["score"],
                "confidence": result["confidence"],
                "regime": result["regime"],
                "confirmation": result["confirmation"],
                "entry": result["entry"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Trader New Listing Engine V1.3")
    parser.add_argument("--ticker", default="SPCX")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--period", default="1y")
    parser.add_argument("--listing-price", type=float, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        result = analyze(
            args.ticker,
            NewListingConfig(period=args.period, benchmark=args.benchmark),
            args.benchmark,
            args.listing_price,
        )
        print(json.dumps(result, indent=2, default=str))
