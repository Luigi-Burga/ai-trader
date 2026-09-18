"""
Asset Age Resolver V1.3

Purpose
-------
Align CCE routing with the CCE's actual statistical-history requirements.

CCE V1.6.1 currently uses:
    min_history = 320
    min_matches = 20
    min_gap_days = 20
    max forward horizon = 60 days

320 valid feature rows are therefore enough to pass the CCE warm-up,
but NOT enough to guarantee 20 independent historical states separated
by 20 sessions while retaining a complete 60-session forward window.

The theoretical minimum is:

    (min_matches - 1) * min_gap_days + 1 + max_horizon
    = 19 * 20 + 1 + 60
    = 441 feature rows

V1.3 therefore distinguishes:
    >= 441 valid feature rows -> CCE-ready for independent matches
    320..440 valid feature rows -> NLE_EXTENDED_HISTORY
    < 320 -> NLE_EXTENDED_HISTORY

This is a routing-policy correction, not a change to CCE scoring.
Production orchestrator is NOT modified by this standalone file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf


VERSION = "1.3"

CCE_MIN_HISTORY = 320
CCE_MIN_MATCHES = 20
CCE_MIN_GAP_DAYS = 20
CCE_MAX_FORWARD_HORIZON = 60

CCE_MIN_INDEPENDENT_FEATURE_ROWS = (
    (CCE_MIN_MATCHES - 1) * CCE_MIN_GAP_DAYS
    + 1
    + CCE_MAX_FORWARD_HORIZON
)


@dataclass
class ResolverConfig:
    period: str = "5y"
    interval: str = "1d"

    early_listing_days: int = 126
    mature_days: int = 252

    cce_min_feature_rows: int = CCE_MIN_HISTORY
    cce_min_independent_feature_rows: int = CCE_MIN_INDEPENDENT_FEATURE_ROWS

    feature_warmup_days: int = 320

    minimum_price_rows: int = 20
    minimum_valid_close_ratio: float = 0.98


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    result = df.copy()

    if isinstance(result.columns, pd.MultiIndex):
        if "Close" in result.columns.get_level_values(0):
            result.columns = result.columns.get_level_values(0)
        elif "Close" in result.columns.get_level_values(-1):
            result.columns = result.columns.get_level_values(-1)
        else:
            result.columns = [
                "_".join(str(x) for x in col if str(x) != "")
                for col in result.columns
            ]

    result.columns = [str(c) for c in result.columns]
    return result


def _download_history(ticker: str, cfg: ResolverConfig) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=cfg.period,
        interval=cfg.interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )
    return _flatten_yfinance_columns(df)


def _quality_from_data(
    df: pd.DataFrame,
    cfg: ResolverConfig,
) -> Dict[str, Any]:
    if df is None or df.empty or "Close" not in df.columns:
        return {
            "data_quality": "NO_DATA",
            "price_rows": 0,
            "valid_close_rows": 0,
            "valid_close_ratio": 0.0,
        }

    close = pd.to_numeric(df["Close"], errors="coerce")
    price_rows = int(len(df))
    valid_close_rows = int(close.notna().sum())
    ratio = valid_close_rows / price_rows if price_rows else 0.0

    if price_rows < cfg.minimum_price_rows or valid_close_rows == 0:
        quality = "INSUFFICIENT"
    elif ratio >= cfg.minimum_valid_close_ratio:
        quality = "EXCELLENT"
    else:
        quality = "INSUFFICIENT"

    return {
        "data_quality": quality,
        "price_rows": price_rows,
        "valid_close_rows": valid_close_rows,
        "valid_close_ratio": round(ratio, 6),
    }


def _calculate_feature_rows(
    df: pd.DataFrame,
    cfg: ResolverConfig,
) -> int:
    if df is None or df.empty or "Close" not in df.columns:
        return 0

    close = pd.to_numeric(df["Close"], errors="coerce")
    valid_rows = int(close.notna().sum())

    return max(
        0,
        valid_rows - cfg.feature_warmup_days + 1,
    )


def resolve_asset_age(
    ticker: str,
    df: Optional[pd.DataFrame] = None,
    config: Optional[ResolverConfig] = None,
) -> Dict[str, Any]:
    cfg = config or ResolverConfig()
    ticker = ticker.upper().strip()

    if df is None:
        try:
            df = _download_history(ticker, cfg)
        except Exception as exc:
            return {
                "engine": "UNKNOWN",
                "route": "NO_DATA",
                "ticker": ticker,
                "version": VERSION,
                "as_of": date.today().isoformat(),
                "trading_days": 0,
                "valid_feature_rows": 0,
                "data_quality": "NO_DATA",
                "reason": f"Unable to download usable history: {exc}",
                "recommendation": "REVIEW_DATA",
                "cce_eligible": False,
                "nle_eligible": False,
                "config": asdict(cfg),
            }

    df = _flatten_yfinance_columns(df)
    quality = _quality_from_data(df, cfg)

    if quality["data_quality"] == "NO_DATA":
        return {
            "engine": "UNKNOWN",
            "route": "NO_DATA",
            "ticker": ticker,
            "version": VERSION,
            "as_of": date.today().isoformat(),
            "trading_days": 0,
            "valid_feature_rows": 0,
            "data_quality": "NO_DATA",
            "reason": "No usable historical Close data was returned.",
            "recommendation": "REVIEW_DATA",
            "cce_eligible": False,
            "nle_eligible": False,
            "config": asdict(cfg),
        }

    trading_days = quality["valid_close_rows"]
    valid_feature_rows = _calculate_feature_rows(df, cfg)

    if quality["data_quality"] != "EXCELLENT":
        route = "REVIEW_DATA"
        engine = "UNKNOWN"
        recommendation = "REVIEW_DATA"
        cce_eligible = False
        nle_eligible = False
        reason = "Historical price data does not satisfy minimum quality."
    elif trading_days < cfg.early_listing_days:
        route = "NEW_LISTING"
        engine = "NEW_LISTING_ENGINE"
        recommendation = "USE_NLE"
        cce_eligible = False
        nle_eligible = True
        reason = (
            f"{trading_days} trading days (< {cfg.early_listing_days}); "
            "early-history phase."
        )
    elif valid_feature_rows < cfg.cce_min_independent_feature_rows:
        route = "NLE_EXTENDED_HISTORY"
        engine = "NEW_LISTING_ENGINE"
        recommendation = "USE_NLE"
        cce_eligible = False
        nle_eligible = True
        reason = (
            f"{valid_feature_rows} valid feature rows are below the "
            f"{cfg.cce_min_independent_feature_rows}-row CCE "
            "independent-match requirement."
        )
    else:
        route = "MATURE"
        engine = "CHARACTERISTIC_CURVE_ENGINE"
        recommendation = "USE_CCE"
        cce_eligible = True
        nle_eligible = False
        reason = (
            f"{valid_feature_rows} valid feature rows satisfy the "
            f"{cfg.cce_min_independent_feature_rows}-row CCE requirement."
        )

    return {
        "engine": engine,
        "route": route,
        "ticker": ticker,
        "version": VERSION,
        "as_of": date.today().isoformat(),
        "trading_days": trading_days,
        "valid_feature_rows": valid_feature_rows,
        "data_quality": quality["data_quality"],
        "reason": reason,
        "recommendation": recommendation,
        "cce_eligible": cce_eligible,
        "nle_eligible": nle_eligible,
        "config": asdict(cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Asset Age Resolver V1.3"
    )
    parser.add_argument("--ticker", default="ANRO")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--batch", nargs="*")
    args = parser.parse_args()

    cfg = ResolverConfig(
        period=args.period,
        interval=args.interval,
    )

    tickers = args.batch if args.batch else [args.ticker]

    print("=" * 100)
    print("ASSET AGE RESOLVER V1.3")
    print("=" * 100)
    print(
        f"CCE min_history={CCE_MIN_HISTORY} | "
        f"min_matches={CCE_MIN_MATCHES} | "
        f"min_gap={CCE_MIN_GAP_DAYS} | "
        f"max_horizon={CCE_MAX_FORWARD_HORIZON} | "
        f"independent_feature_rows={CCE_MIN_INDEPENDENT_FEATURE_ROWS}"
    )
    print("-" * 100)

    for ticker in tickers:
        result = resolve_asset_age(ticker, config=cfg)
        print(
            f"{ticker.upper():<6} | "
            f"route={result['route']:<22} | "
            f"engine={result['engine']:<30} | "
            f"days={result['trading_days']:<5} | "
            f"features={result['valid_feature_rows']:<5} | "
            f"CCE={result['cce_eligible']}"
        )
        print(f"       reason={result['reason']}")

    print("-" * 100)
    print(
        "V1.3 routing correction is standalone; "
        "production orchestrator is unchanged."
    )


if __name__ == "__main__":
    main()
