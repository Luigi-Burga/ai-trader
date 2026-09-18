"""
Asset Age Resolver V1.0
-----------------------

Routes an asset to the appropriate analysis engine based on the amount
of usable historical market data available.

Design:
    ticker
      |
      +--> usable trading history
              |
              +--> < 6 months      -> NEW_LISTING_ENGINE
              +--> 6-12 months     -> NEW_LISTING_ENGINE / CCE
              +--> >= 12 months    -> CCE only if valid feature rows >= 320

The resolver is intentionally conservative:
- It uses actual downloaded price history rather than ticker age inferred
  from external metadata.
- It counts valid trading sessions, not calendar days.
- It recommends CCE only when the usable feature-row depth satisfies the
  current CCE minimum; otherwise it keeps the asset on NLE.
- It does not itself generate a BUY/SELL signal.
- It is standalone and does not modify main.py, signal_engine.py, or the
  existing engines.

Dependencies:
    pip install yfinance pandas numpy

CLI examples:
    python -m app.ai.asset_age_resolver --ticker SPCX
    python -m app.ai.asset_age_resolver --ticker PLTU
    python -m app.ai.asset_age_resolver --ticker MSTU
    python -m app.ai.asset_age_resolver --ticker ASMG
    python -m app.ai.asset_age_resolver --ticker PLTR
    python -m app.ai.asset_age_resolver --ticker TSM
    python -m app.ai.asset_age_resolver --ticker SOXL
    python -m app.ai.asset_age_resolver --ticker TQQQ
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


VERSION = "1.1"

# Trading-session thresholds.
TRADING_DAYS_6_MONTHS = 126
TRADING_DAYS_12_MONTHS = 252

# CCE currently requires at least 320 valid feature rows.
CCE_MIN_FEATURE_ROWS = 320

# NLE is the preferred engine while the asset is still in its first year.
NLE_MAX_TRADING_DAYS = TRADING_DAYS_12_MONTHS


@dataclass
class ResolverConfig:
    """Configuration for the asset-age routing policy."""

    period: str = "5y"
    interval: str = "1d"

    early_listing_days: int = TRADING_DAYS_6_MONTHS
    mature_days: int = TRADING_DAYS_12_MONTHS
    cce_min_feature_rows: int = CCE_MIN_FEATURE_ROWS

    # CCE requires enough history for its feature vector plus forward windows.
    # Keeping this separate makes the policy explicit and easy to tune later.
    feature_warmup_days: int = 320

    # Reject obviously unusable price histories.
    minimum_price_rows: int = 20
    minimum_valid_close_ratio: float = 0.98


@dataclass
class AssetAgeResolution:
    """Serializable result returned by the resolver."""

    engine: str
    route: str
    ticker: str

    version: str

    as_of: str
    first_valid_date: Optional[str]
    last_valid_date: Optional[str]

    calendar_age_days: Optional[int]
    trading_days: int
    valid_feature_rows: int

    data_quality: str

    reason: str
    recommendation: str

    cce_eligible: bool
    nle_eligible: bool

    config: dict


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance output.

    yfinance can return a MultiIndex even for a single ticker depending on
    version/options. This resolver only needs the OHLCV columns and therefore
    safely collapses a single-ticker MultiIndex.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        # Single ticker: ('Close', 'SPCX') etc.
        if out.columns.nlevels == 2:
            first_level = list(out.columns.get_level_values(0))
            ohlcv = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

            selected = {}
            for col in ohlcv:
                matches = [
                    idx for idx, value in enumerate(first_level)
                    if str(value) == col
                ]
                if matches:
                    selected[col] = out.iloc[:, matches[0]]

            if selected:
                out = pd.DataFrame(selected, index=out.index)
            else:
                out.columns = [
                    "_".join(str(x) for x in col if str(x) != "nan")
                    for col in out.columns
                ]

        else:
            out.columns = [
                "_".join(str(x) for x in col if str(x) != "nan")
                for col in out.columns
            ]

    # Remove accidental duplicate columns while preserving the first one.
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def _safe_float(value: object) -> Optional[float]:
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return None


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sort and normalize the datetime index."""
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")

    out = out.loc[~out.index.isna()].sort_index()

    # Convert timezone-aware indexes to timezone-naive dates. The resolver
    # operates on daily trading sessions, so timezone is not decision-critical.
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)

    return out


def _valid_close_mask(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        return pd.Series(False, index=df.index)

    close = pd.to_numeric(df["Close"], errors="coerce")
    return close.notna() & np.isfinite(close) & (close > 0)


def _estimate_calendar_age_days(first_date: pd.Timestamp,
                                last_date: pd.Timestamp) -> Optional[int]:
    if pd.isna(first_date) or pd.isna(last_date):
        return None
    return max(0, int((last_date - first_date).days))


def _calculate_feature_rows(df: pd.DataFrame) -> int:
    """
    Estimate the number of rows that are usable by the current CCE.

    The current CCE has a hard minimum of 320 valid feature rows. We keep this
    calculation intentionally conservative: the resolver only considers rows
    with valid Close values and enough warm-up history.

    This is a routing estimate, not a replacement for running CCE itself.
    """
    valid = _valid_close_mask(df)
    valid_rows = int(valid.sum())

    if valid_rows < CCE_MIN_FEATURE_ROWS:
        return valid_rows

    # Current CCE's effective feature warm-up is represented by the configured
    # warmup threshold. Forward windows are not included here because the
    # resolver is deciding whether the asset has sufficient historical depth.
    return max(0, valid_rows - CCE_MIN_FEATURE_ROWS + 1)


def _classify_quality(
    total_rows: int,
    valid_rows: int,
    close_ratio: float,
    volume_ratio: float,
    cfg: ResolverConfig,
) -> str:
    if total_rows == 0 or valid_rows < cfg.minimum_price_rows:
        return "INVALID"

    if close_ratio < cfg.minimum_valid_close_ratio:
        return "POOR"

    if volume_ratio < 0.90:
        return "MODERATE"

    if close_ratio >= 0.995 and volume_ratio >= 0.98:
        return "EXCELLENT"

    return "GOOD"


def _build_resolution(
    ticker: str,
    df: pd.DataFrame,
    cfg: ResolverConfig,
) -> AssetAgeResolution:
    ticker = ticker.upper().strip()

    if df is None or df.empty:
        return AssetAgeResolution(
            engine="NONE",
            route="NO_DATA",
            ticker=ticker,
            version=VERSION,
            as_of=date.today().isoformat(),
            first_valid_date=None,
            last_valid_date=None,
            calendar_age_days=None,
            trading_days=0,
            valid_feature_rows=0,
            data_quality="INVALID",
            reason="No market data returned for ticker.",
            recommendation="NO_DATA",
            cce_eligible=False,
            nle_eligible=False,
            config=asdict(cfg),
        )

    df = _normalize_index(_flatten_columns(df))

    total_rows = len(df)
    valid_close = _valid_close_mask(df)
    valid_rows = int(valid_close.sum())

    if valid_rows:
        valid_dates = df.index[valid_close]
        first_valid = pd.Timestamp(valid_dates[0])
        last_valid = pd.Timestamp(valid_dates[-1])
        first_valid_date = first_valid.date().isoformat()
        last_valid_date = last_valid.date().isoformat()
        calendar_age = _estimate_calendar_age_days(first_valid, last_valid)
    else:
        first_valid_date = None
        last_valid_date = None
        calendar_age = None

    close_ratio = valid_rows / total_rows if total_rows else 0.0

    if "Volume" in df.columns:
        volume = pd.to_numeric(df["Volume"], errors="coerce")
        volume_valid = volume.notna() & np.isfinite(volume) & (volume >= 0)
        volume_ratio = float(volume_valid.sum() / total_rows) if total_rows else 0.0
    else:
        volume_ratio = 0.0

    quality = _classify_quality(
        total_rows=total_rows,
        valid_rows=valid_rows,
        close_ratio=close_ratio,
        volume_ratio=volume_ratio,
        cfg=cfg,
    )

    trading_days = valid_rows
    valid_feature_rows = _calculate_feature_rows(df)

    # NLE is designed for assets with less than one year of usable history.
    # It can also be used for intermediate-age assets when the CCE history
    # requirement is not yet met.
    nle_eligible = (
        quality != "INVALID"
        and trading_days >= cfg.minimum_price_rows
        and trading_days < cfg.mature_days
    )

    # V1.1: CCE eligibility must be based on the number of usable feature
    # rows, not merely on total trading days. The current CCE itself rejects
    # assets when valid feature rows are below 320.
    cce_eligible = (
        quality in {"EXCELLENT", "GOOD", "MODERATE"}
        and valid_feature_rows >= cfg.cce_min_feature_rows
    )

    if quality == "INVALID":
        engine = "NONE"
        route = "NO_DATA"
        reason = "Insufficient or invalid price history."
        recommendation = "NO_DATA"

    elif trading_days < cfg.early_listing_days:
        engine = "NEW_LISTING_ENGINE"
        route = "NEW_LISTING"
        reason = (
            f"{trading_days} valid trading days (< {cfg.early_listing_days}); "
            "asset is in the early-history phase."
        )
        recommendation = "USE_NLE"

    elif trading_days < cfg.mature_days:
        if cce_eligible:
            engine = "CHARACTERISTIC_CURVE_ENGINE"
            route = "CCE_INTERMEDIATE"
            reason = (
                f"{trading_days} valid trading days and "
                f"{valid_feature_rows} valid feature rows; CCE depth is "
                "satisfied."
            )
            recommendation = "USE_CCE"
        else:
            engine = "NEW_LISTING_ENGINE"
            route = "NEW_LISTING_INTERMEDIATE"
            reason = (
                f"{trading_days} valid trading days and only "
                f"{valid_feature_rows} valid feature rows; CCE requires "
                f"{cfg.cce_min_feature_rows}."
            )
            recommendation = "USE_NLE"

    else:
        if cce_eligible:
            engine = "CHARACTERISTIC_CURVE_ENGINE"
            route = "MATURE"
            reason = (
                f"{trading_days} valid trading days and "
                f"{valid_feature_rows} valid feature rows; CCE minimum of "
                f"{cfg.cce_min_feature_rows} is satisfied."
            )
            recommendation = "USE_CCE"
        else:
            # Mature by calendar/trading history does not automatically mean
            # mature for CCE. If the feature matrix is still too short, route
            # to NLE rather than falsely declaring CCE eligibility.
            engine = "NEW_LISTING_ENGINE"
            route = "NLE_EXTENDED_HISTORY"
            reason = (
                f"{trading_days} valid trading days but only "
                f"{valid_feature_rows} valid feature rows; CCE requires "
                f"{cfg.cce_min_feature_rows}."
            )
            recommendation = "USE_NLE"

    return AssetAgeResolution(
        engine=engine,
        route=route,
        ticker=ticker,
        version=VERSION,
        as_of=(
            last_valid_date
            or date.today().isoformat()
        ),
        first_valid_date=first_valid_date,
        last_valid_date=last_valid_date,
        calendar_age_days=calendar_age,
        trading_days=trading_days,
        valid_feature_rows=valid_feature_rows,
        data_quality=quality,
        reason=reason,
        recommendation=recommendation,
        cce_eligible=cce_eligible,
        nle_eligible=nle_eligible,
        config=asdict(cfg),
    )


def download_history(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download daily market history through yfinance."""
    ticker = ticker.upper().strip()

    return yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )


def resolve_asset_age(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    config: Optional[ResolverConfig] = None,
) -> AssetAgeResolution:
    """
    Resolve the correct historical-analysis engine for a ticker.

    This function performs no signal analysis. It only determines whether
    enough usable history exists for NLE or CCE.
    """
    cfg = config or ResolverConfig(
        period=period,
        interval=interval,
    )

    # If a custom config was supplied, its period/interval are authoritative.
    if config is None:
        cfg.period = period
        cfg.interval = interval

    try:
        data = download_history(
            ticker=ticker,
            period=cfg.period,
            interval=cfg.interval,
        )
    except Exception as exc:
        return AssetAgeResolution(
            engine="NONE",
            route="DATA_ERROR",
            ticker=ticker.upper().strip(),
            version=VERSION,
            as_of=date.today().isoformat(),
            first_valid_date=None,
            last_valid_date=None,
            calendar_age_days=None,
            trading_days=0,
            valid_feature_rows=0,
            data_quality="INVALID",
            reason=f"Market data download failed: {exc}",
            recommendation="NO_DATA",
            cce_eligible=False,
            nle_eligible=False,
            config=asdict(cfg),
        )

    return _build_resolution(
        ticker=ticker,
        df=data,
        cfg=cfg,
    )


def resolve_from_dataframe(
    ticker: str,
    dataframe: pd.DataFrame,
    config: Optional[ResolverConfig] = None,
) -> AssetAgeResolution:
    """Resolve routing from an already downloaded DataFrame."""
    cfg = config or ResolverConfig()
    return _build_resolution(
        ticker=ticker,
        df=dataframe,
        cfg=cfg,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Asset Age Resolver V1.1"
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol, e.g. SPCX, PLTR, TSM, SOXL",
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
        help="Pretty-print JSON output.",
    )

    args = parser.parse_args()

    result = resolve_asset_age(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
    )

    if args.pretty:
        print(json.dumps(asdict(result), indent=2, default=str))
    else:
        print(json.dumps(asdict(result), default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
