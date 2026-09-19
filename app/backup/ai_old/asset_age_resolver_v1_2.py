"""
Asset Age Resolver V1.2
=======================

Routes a ticker to the appropriate analysis engine based on:
- trading history
- valid feature-row depth
- data quality

Routing model:

    < 126 trading days
        -> NEW_LISTING

    >= 126 trading days but < 320 valid feature rows
        -> NLE_EXTENDED_HISTORY

    >= 320 valid feature rows
        -> MATURE / CCE

    insufficient data quality
        -> REVIEW_DATA

    no usable data
        -> NO_DATA

V1.2 semantic correction:
- NLE_EXTENDED_HISTORY is explicitly NLE-eligible.
- nle_eligible is therefore True for both NEW_LISTING and
  NLE_EXTENDED_HISTORY.
- cce_eligible is True only when the actual number of valid
  feature rows is >= cce_min_feature_rows and data quality is acceptable.

This module is intentionally standalone. It does not call CCE or NLE;
it only determines which engine should be used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf


VERSION = "1.2"


@dataclass
class ResolverConfig:
    period: str = "5y"
    interval: str = "1d"

    # Asset-age routing thresholds.
    early_listing_days: int = 126
    mature_days: int = 252

    # CCE requires this many usable feature rows after its warm-up.
    cce_min_feature_rows: int = 320

    # Kept explicit because feature construction may have a warm-up
    # different from simple trading-day age.
    feature_warmup_days: int = 320

    # Basic data-quality requirements.
    minimum_price_rows: int = 20
    minimum_valid_close_ratio: float = 0.98


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance single/multi-level columns."""
    if df is None or df.empty:
        return df

    result = df.copy()

    if isinstance(result.columns, pd.MultiIndex):
        # Prefer the OHLCV field level when yfinance returns
        # columns such as ('Close', 'SPCX').
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


def _download_history(
    ticker: str,
    cfg: ResolverConfig,
) -> pd.DataFrame:
    """Download historical OHLCV data."""
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
    """Calculate basic data-quality metrics."""
    if df is None or df.empty:
        return {
            "data_quality": "NO_DATA",
            "price_rows": 0,
            "valid_close_rows": 0,
            "valid_close_ratio": 0.0,
        }

    if "Close" not in df.columns:
        return {
            "data_quality": "NO_DATA",
            "price_rows": len(df),
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
    """
    Estimate the number of rows available after the CCE feature warm-up.

    The resolver uses the same conceptual rule validated in V1.1:
    actual usable feature rows are the valid close rows after the
    feature warm-up, rather than simply using trading_days.

    For a normal daily OHLCV series this is:

        valid_close_rows - feature_warmup_days + 1

    with a floor of zero.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return 0

    close = pd.to_numeric(df["Close"], errors="coerce")
    valid_rows = int(close.notna().sum())

    return max(
        0,
        valid_rows - cfg.feature_warmup_days + 1,
    )


def _date_value(value: Any) -> Optional[str]:
    """Convert a pandas Timestamp/date-like value to ISO text."""
    if value is None:
        return None

    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return None


def _trading_days(df: pd.DataFrame) -> int:
    if df is None or df.empty or "Close" not in df.columns:
        return 0

    close = pd.to_numeric(df["Close"], errors="coerce")
    return int(close.notna().sum())


def resolve_asset_age(
    ticker: str,
    df: Optional[pd.DataFrame] = None,
    config: Optional[ResolverConfig] = None,
) -> Dict[str, Any]:
    """
    Resolve the correct analysis engine for a ticker.

    Returns a JSON-serializable dictionary.
    """
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
                "first_valid_date": None,
                "last_valid_date": None,
                "calendar_age_days": None,
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
            "first_valid_date": None,
            "last_valid_date": None,
            "calendar_age_days": None,
            "trading_days": 0,
            "valid_feature_rows": 0,
            "data_quality": "NO_DATA",
            "reason": "No usable historical Close data was returned.",
            "recommendation": "REVIEW_DATA",
            "cce_eligible": False,
            "nle_eligible": False,
            "config": asdict(cfg),
        }

    if quality["data_quality"] != "EXCELLENT":
        return {
            "engine": "UNKNOWN",
            "route": "REVIEW_DATA",
            "ticker": ticker,
            "version": VERSION,
            "as_of": date.today().isoformat(),
            "first_valid_date": None,
            "last_valid_date": None,
            "calendar_age_days": None,
            "trading_days": quality["valid_close_rows"],
            "valid_feature_rows": 0,
            "data_quality": quality["data_quality"],
            "reason": (
                "Historical price data does not satisfy the minimum "
                "data-quality requirements."
            ),
            "recommendation": "REVIEW_DATA",
            "cce_eligible": False,
            "nle_eligible": False,
            "config": asdict(cfg),
        }

    close = pd.to_numeric(df["Close"], errors="coerce")
    valid = close.dropna()

    if valid.empty:
        return {
            "engine": "UNKNOWN",
            "route": "NO_DATA",
            "ticker": ticker,
            "version": VERSION,
            "as_of": date.today().isoformat(),
            "first_valid_date": None,
            "last_valid_date": None,
            "calendar_age_days": None,
            "trading_days": 0,
            "valid_feature_rows": 0,
            "data_quality": "NO_DATA",
            "reason": "No valid closing prices are available.",
            "recommendation": "REVIEW_DATA",
            "cce_eligible": False,
            "nle_eligible": False,
            "config": asdict(cfg),
        }

    first_date = valid.index[0]
    last_date = valid.index[-1]

    first_ts = pd.Timestamp(first_date)
    last_ts = pd.Timestamp(last_date)
    calendar_age_days = int((last_ts.normalize() - first_ts.normalize()).days)

    trading_days = int(len(valid))
    valid_feature_rows = _calculate_feature_rows(df, cfg)

    # ------------------------------------------------------------------
    # V1.2 ROUTING
    # ------------------------------------------------------------------

    # Early history: always use NLE.
    if trading_days < cfg.early_listing_days:
        return {
            "engine": "NEW_LISTING_ENGINE",
            "route": "NEW_LISTING",
            "ticker": ticker,
            "version": VERSION,
            "as_of": date.today().isoformat(),
            "first_valid_date": _date_value(first_date),
            "last_valid_date": _date_value(last_date),
            "calendar_age_days": calendar_age_days,
            "trading_days": trading_days,
            "valid_feature_rows": valid_feature_rows,
            "data_quality": quality["data_quality"],
            "reason": (
                f"{trading_days} valid trading days "
                f"(< {cfg.early_listing_days}); "
                "asset is in the early-history phase."
            ),
            "recommendation": "USE_NLE",
            "cce_eligible": False,
            "nle_eligible": True,
            "config": asdict(cfg),
        }

    # Enough chronological age, but not enough CCE feature depth.
    # V1.2 explicitly marks this route as NLE eligible.
    if valid_feature_rows < cfg.cce_min_feature_rows:
        return {
            "engine": "NEW_LISTING_ENGINE",
            "route": "NLE_EXTENDED_HISTORY",
            "ticker": ticker,
            "version": VERSION,
            "as_of": date.today().isoformat(),
            "first_valid_date": _date_value(first_date),
            "last_valid_date": _date_value(last_date),
            "calendar_age_days": calendar_age_days,
            "trading_days": trading_days,
            "valid_feature_rows": valid_feature_rows,
            "data_quality": quality["data_quality"],
            "reason": (
                f"{trading_days} valid trading days but only "
                f"{valid_feature_rows} valid feature rows; "
                f"CCE requires {cfg.cce_min_feature_rows}. "
                "NLE is eligible for extended-history assets."
            ),
            "recommendation": "USE_NLE",
            "cce_eligible": False,
            "nle_eligible": True,
            "config": asdict(cfg),
        }

    # Mature enough for CCE. The actual gate is feature depth, not
    # simply calendar age.
    return {
        "engine": "CHARACTERISTIC_CURVE_ENGINE",
        "route": "MATURE",
        "ticker": ticker,
        "version": VERSION,
        "as_of": date.today().isoformat(),
        "first_valid_date": _date_value(first_date),
        "last_valid_date": _date_value(last_date),
        "calendar_age_days": calendar_age_days,
        "trading_days": trading_days,
        "valid_feature_rows": valid_feature_rows,
        "data_quality": quality["data_quality"],
        "reason": (
            f"{trading_days} valid trading days and "
            f"{valid_feature_rows} valid feature rows; "
            f"CCE minimum of {cfg.cce_min_feature_rows} is satisfied."
        ),
        "recommendation": "USE_CCE",
        "cce_eligible": True,
        "nle_eligible": False,
        "config": asdict(cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Asset Age Resolver V1.2"
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol, e.g. PLTR, SPCX, SOXL",
    )
    parser.add_argument(
        "--period",
        default="5y",
        help="yfinance period (default: 5y)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="yfinance interval (default: 1d)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    args = parser.parse_args()

    cfg = ResolverConfig(
        period=args.period,
        interval=args.interval,
    )

    result = resolve_asset_age(args.ticker, config=cfg)

    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
