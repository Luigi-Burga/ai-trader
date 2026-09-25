"""
Asset Age Resolver V1.4
=======================

Purpose
-------
Routes an asset to the appropriate historical-analysis engine based on
the ACTUAL feature matrix used by Characteristic Curve Engine V1.6.1.

V1.4 architectural correction
-----------------------------
V1.3 estimated ``valid_feature_rows`` with a simple Close warm-up formula:

    valid_close_rows - feature_warmup_days + 1

That estimate could diverge from CCE because CCE constructs 20 real features
and then applies:

    features[FEATURES].dropna()

V1.4 removes that duplicated approximation. It reuses the exact CCE V1.6.1
``_clean_ohlcv()`` and ``build_features()`` functions when available.

Therefore:
    Asset Age Resolver V1.4
              |
              +--> exact CCE feature builder
              |
              +--> exact valid feature-row count
              |
              +--> forward-complete feature states
              |
              +--> structural independent-match capacity
              |
              +--> routing decision

Important
---------
- This version does NOT modify CCE V1.6.1.
- This version does NOT guarantee that CCE will find 20 similar historical
  matches. Similarity matching is still a CCE responsibility.
- ``cce_eligible=True`` means the asset has enough STRUCTURAL history for CCE's
  configured minimum sample requirements. CCE can still return
  INSUFFICIENT_HISTORY if similarity selection cannot find enough independent
  matches.
- ``buy_target`` is not used.
- No trading signal is generated.

Compatibility
-------------
The public functions remain compatible with the V1.3-style resolver:

    resolve_asset_age(ticker, period="5y", interval="1d", config=None)
    resolve_from_dataframe(ticker, dataframe, config=None)

Optional:
    resolve_from_dataframe(..., benchmark_df=...)

CLI:
    python asset_age_resolver_v1_4.py --ticker TQQQ --pretty
    python asset_age_resolver_v1_4.py --ticker TQQQ --benchmark QQQ --pretty

Dependencies
------------
pip install yfinance pandas numpy
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd
from app.data.market_data import get_history


def get_history_compat(ticker: str, *, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
    return get_history(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=False,
        group_by="column",
    )


VERSION = "1.4"

# ---------------------------------------------------------------------------
# Routing constants
# ---------------------------------------------------------------------------

TRADING_DAYS_6_MONTHS = 126

# CCE V1.6.1 configuration.
CCE_MIN_HISTORY = 320
CCE_MIN_MATCHES = 20
CCE_MIN_GAP_DAYS = 20
CCE_MAX_FORWARD_HORIZON = 60

# Data-quality guardrails.
MINIMUM_PRICE_ROWS = 20
MINIMUM_VALID_CLOSE_RATIO = 0.98


# ---------------------------------------------------------------------------
# CCE feature definitions are intentionally duplicated as a fallback only.
# The primary path imports the exact CCE V1.6.1 builder.
# ---------------------------------------------------------------------------

CCE_FEATURES = [
    "trend20",
    "trend50",
    "trend200",
    "slope50",
    "roc5",
    "roc20",
    "roc60",
    "rsi_norm",
    "atr_pct",
    "atr_z",
    "volatility_z",
    "drawdown20",
    "drawdown50",
    "drawdown252",
    "recovery50",
    "position50",
    "position252",
    "breakout20",
    "volume_z",
    "rs_momentum",
]


@dataclass
class ResolverConfig:
    """Configuration for Asset Age Resolver V1.4."""

    period: str = "5y"
    interval: str = "1d"

    # Early-history route.
    early_listing_days: int = TRADING_DAYS_6_MONTHS

    # Exact structural CCE requirements.
    cce_min_history: int = CCE_MIN_HISTORY
    cce_min_matches: int = CCE_MIN_MATCHES
    cce_min_gap_days: int = CCE_MIN_GAP_DAYS
    cce_max_forward_horizon: int = CCE_MAX_FORWARD_HORIZON

    # Basic market-data quality.
    minimum_price_rows: int = MINIMUM_PRICE_ROWS
    minimum_valid_close_ratio: float = MINIMUM_VALID_CLOSE_RATIO

    # If True, failure to import the exact CCE builder is fatal rather than
    # silently reverting to an approximate implementation.
    require_exact_cce_builder: bool = True


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance single- and multi-level OHLCV columns."""

    if df is None or df.empty:
        return pd.DataFrame()

    try:
        # Prefer the exact CCE V1.6.1 cleaner.
        from app.ai.characteristic_curve_v1_6_1 import _clean_ohlcv

        cleaned = _clean_ohlcv(df)
        if not cleaned.empty:
            return cleaned
    except Exception:
        pass

    # Defensive fallback used only when the CCE module is unavailable.
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        wanted = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "adj close": "Close",
        }

        selected: dict[str, Any] = {}

        for col in out.columns:
            parts = [
                str(x).strip()
                for x in col
                if str(x).strip() not in {"", "None"}
            ]

            for part in parts:
                key = part.lower()
                if key in wanted and wanted[key] not in selected:
                    selected[wanted[key]] = col
                    break

        if selected:
            out = out[[selected[k] for k in selected]]
            out.columns = list(selected.keys())

    rename: dict[Any, str] = {}

    for col in out.columns:
        key = str(col).strip().lower()

        if key in {"open", "high", "low", "close", "volume"}:
            rename[col] = key.capitalize()
        elif key == "adj close":
            rename[col] = "Close"

    out = out.rename(columns=rename)

    needed = ["Open", "High", "Low", "Close", "Volume"]

    for col in needed:
        if col not in out.columns:
            if col == "Volume":
                out[col] = 0.0
            else:
                return pd.DataFrame()

    out = out[needed].copy()

    for col in needed:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Close", "High", "Low"]).sort_index()

    return out


def _date_value(value: Any) -> Optional[str]:
    if value is None:
        return None

    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return str(value)


def _quality_from_data(
    df: pd.DataFrame,
    cfg: ResolverConfig,
) -> dict[str, Any]:
    """Evaluate basic OHLCV usability before feature construction."""

    if df is None or df.empty:
        return {
            "data_quality": "NO_DATA",
            "raw_rows": 0,
            "valid_close_rows": 0,
            "valid_close_ratio": 0.0,
        }

    close = pd.to_numeric(df.get("Close"), errors="coerce")

    raw_rows = int(len(df))
    valid_close_rows = int(close.notna().sum())

    ratio = (
        valid_close_rows / raw_rows
        if raw_rows > 0
        else 0.0
    )

    if raw_rows < cfg.minimum_price_rows:
        quality = "INSUFFICIENT"
    elif ratio < cfg.minimum_valid_close_ratio:
        quality = "INSUFFICIENT"
    else:
        quality = "EXCELLENT"

    return {
        "data_quality": quality,
        "raw_rows": raw_rows,
        "valid_close_rows": valid_close_rows,
        "valid_close_ratio": round(float(ratio), 6),
    }


def _load_exact_cce_builder():
    """
    Return the exact CCE V1.6.1 cleaner and feature builder.

    V1.4 deliberately fails closed when exact reuse is required. This prevents
    the resolver from quietly drifting away from CCE again.
    """

    try:
        from app.ai.characteristic_curve_v1_6_1 import (
            FEATURES,
            _clean_ohlcv,
            build_features,
        )

        return _clean_ohlcv, build_features, list(FEATURES), None

    except Exception as exc:
        return None, None, None, exc


def _build_exact_features(
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame],
    cfg: ResolverConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Construct the feature matrix using the exact CCE V1.6.1 implementation.

    The benchmark is optional. When absent, CCE V1.6.1 itself sets
    rs_momentum to 0.0, which is still the exact CCE behavior.
    """

    clean_fn, build_fn, feature_names, import_error = _load_exact_cce_builder()

    if clean_fn is None or build_fn is None or feature_names is None:
        if cfg.require_exact_cce_builder:
            raise RuntimeError(
                "CCE V1.6.1 exact feature builder is unavailable: "
                f"{import_error}"
            )

        # This fallback is intentionally conservative and only exists for
        # environments where exact CCE reuse has explicitly been disabled.
        return pd.DataFrame(), {
            "builder": "fallback_unavailable",
            "builder_version": "unknown",
            "exact_builder": False,
            "error": str(import_error),
        }

    clean_asset = clean_fn(df)

    if clean_asset.empty:
        return pd.DataFrame(), {
            "builder": "characteristic_curve_v1_6_1.build_features",
            "builder_version": "1.6.1",
            "exact_builder": True,
            "error": "CCE cleaner returned no usable OHLCV rows.",
        }

    clean_benchmark = None

    if benchmark_df is not None and not benchmark_df.empty:
        clean_benchmark = clean_fn(benchmark_df)

    features = build_fn(clean_asset, clean_benchmark)

    if features is None or features.empty:
        return pd.DataFrame(), {
            "builder": "characteristic_curve_v1_6_1.build_features",
            "builder_version": "1.6.1",
            "exact_builder": True,
            "error": "CCE build_features returned no rows.",
        }

    missing_features = [
        feature
        for feature in feature_names
        if feature not in features.columns
    ]

    if missing_features:
        raise RuntimeError(
            "CCE V1.6.1 feature matrix is missing expected features: "
            + ", ".join(missing_features)
        )

    valid = features[feature_names].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    return features, {
        "builder": "characteristic_curve_v1_6_1.build_features",
        "builder_version": "1.6.1",
        "exact_builder": True,
        "feature_count": int(len(feature_names)),
        "feature_names": list(feature_names),
        "valid_feature_rows": int(len(valid)),
        "first_valid_feature_date": (
            _date_value(valid.index[0])
            if not valid.empty
            else None
        ),
        "last_valid_feature_date": (
            _date_value(valid.index[-1])
            if not valid.empty
            else None
        ),
        "error": "",
    }


def _forward_complete_positions(
    features: pd.DataFrame,
    data_length: int,
    feature_names: list[str],
    max_forward_horizon: int,
) -> list[int]:
    """Return positions that have complete CCE feature rows and future data."""

    valid = features[feature_names].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    positions: list[int] = []

    position_map = {
        timestamp: position
        for position, timestamp in enumerate(features.index)
    }

    for timestamp in valid.index:
        position = position_map.get(timestamp)

        if position is None:
            continue

        if position + max_forward_horizon < data_length:
            positions.append(int(position))

    return positions


def _max_independent_matches(
    positions: list[int],
    min_gap_days: int,
) -> int:
    """
    Compute the maximum number of chronologically separated observations.

    This is a structural capacity test only. It does NOT use Mahalanobis
    similarity, so it cannot guarantee that CCE will actually select the
    required number of similar states.
    """

    if not positions:
        return 0

    selected = 0
    last_selected = None

    for position in sorted(set(positions)):
        if last_selected is None or position - last_selected >= min_gap_days:
            selected += 1
            last_selected = position

    return int(selected)


def _base_result(
    ticker: str,
    *,
    engine: str,
    route: str,
    first_valid_date: Optional[str],
    last_valid_date: Optional[str],
    calendar_age_days: Optional[int],
    trading_days: int,
    valid_feature_rows: int,
    forward_complete_feature_rows: int,
    max_independent_matches: int,
    data_quality: str,
    reason: str,
    recommendation: str,
    cce_eligible: bool,
    nle_eligible: bool,
    feature_builder: dict[str, Any],
    config: ResolverConfig,
) -> dict[str, Any]:

    return {
        "engine": engine,
        "route": route,
        "ticker": ticker.upper().strip(),
        "version": VERSION,
        "as_of": (
            last_valid_date
            or date.today().isoformat()
        ),
        "first_valid_date": first_valid_date,
        "last_valid_date": last_valid_date,
        "calendar_age_days": calendar_age_days,
        "trading_days": int(trading_days),

        # V1.4: this is now the ACTUAL CCE feature count.
        "valid_feature_rows": int(valid_feature_rows),

        # Additional structural diagnostics.
        "forward_complete_feature_rows": int(
            forward_complete_feature_rows
        ),
        "max_independent_matches": int(
            max_independent_matches
        ),
        "cce_min_history": int(config.cce_min_history),
        "cce_min_matches": int(config.cce_min_matches),
        "cce_min_gap_days": int(config.cce_min_gap_days),
        "cce_max_forward_horizon": int(
            config.cce_max_forward_horizon
        ),

        "data_quality": data_quality,
        "reason": reason,
        "recommendation": recommendation,
        "cce_eligible": bool(cce_eligible),
        "nle_eligible": bool(nle_eligible),

        "feature_builder": feature_builder,
        "config": asdict(config),
    }


def _build_resolution(
    ticker: str,
    df: pd.DataFrame,
    cfg: ResolverConfig,
    benchmark_df: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Build a V1.4 routing decision from an already downloaded DataFrame."""

    ticker = ticker.upper().strip()

    normalized = _flatten_yfinance_columns(df)
    quality = _quality_from_data(normalized, cfg)

    if quality["data_quality"] == "NO_DATA":
        return _base_result(
            ticker,
            engine="UNKNOWN",
            route="NO_DATA",
            first_valid_date=None,
            last_valid_date=None,
            calendar_age_days=None,
            trading_days=0,
            valid_feature_rows=0,
            forward_complete_feature_rows=0,
            max_independent_matches=0,
            data_quality="NO_DATA",
            reason="No usable historical OHLCV data was returned.",
            recommendation="REVIEW_DATA",
            cce_eligible=False,
            nle_eligible=False,
            feature_builder={
                "builder": "not_run",
                "builder_version": "1.6.1",
                "exact_builder": False,
                "error": "No usable OHLCV data.",
            },
            config=cfg,
        )

    close = pd.to_numeric(
        normalized["Close"],
        errors="coerce",
    )

    valid_close = close.dropna()

    if valid_close.empty:
        return _base_result(
            ticker,
            engine="UNKNOWN",
            route="NO_DATA",
            first_valid_date=None,
            last_valid_date=None,
            calendar_age_days=None,
            trading_days=0,
            valid_feature_rows=0,
            forward_complete_feature_rows=0,
            max_independent_matches=0,
            data_quality="NO_DATA",
            reason="No valid closing prices are available.",
            recommendation="REVIEW_DATA",
            cce_eligible=False,
            nle_eligible=False,
            feature_builder={
                "builder": "not_run",
                "builder_version": "1.6.1",
                "exact_builder": False,
                "error": "No valid Close values.",
            },
            config=cfg,
        )

    first_date = valid_close.index[0]
    last_date = valid_close.index[-1]

    first_ts = pd.Timestamp(first_date)
    last_ts = pd.Timestamp(last_date)

    calendar_age_days = int(
        (last_ts.normalize() - first_ts.normalize()).days
    )

    trading_days = int(len(valid_close))

    if quality["data_quality"] != "EXCELLENT":
        return _base_result(
            ticker,
            engine="UNKNOWN",
            route="REVIEW_DATA",
            first_valid_date=_date_value(first_date),
            last_valid_date=_date_value(last_date),
            calendar_age_days=calendar_age_days,
            trading_days=trading_days,
            valid_feature_rows=0,
            forward_complete_feature_rows=0,
            max_independent_matches=0,
            data_quality=quality["data_quality"],
            reason=(
                "Historical price data does not satisfy the minimum "
                "data-quality requirements."
            ),
            recommendation="REVIEW_DATA",
            cce_eligible=False,
            nle_eligible=False,
            feature_builder={
                "builder": "not_run",
                "builder_version": "1.6.1",
                "exact_builder": False,
                "error": "Basic data-quality gate failed.",
            },
            config=cfg,
        )

    # ------------------------------------------------------------------
    # V1.4: early-history route remains unchanged.
    # ------------------------------------------------------------------

    if trading_days < cfg.early_listing_days:
        return _base_result(
            ticker,
            engine="NEW_LISTING_ENGINE",
            route="NEW_LISTING",
            first_valid_date=_date_value(first_date),
            last_valid_date=_date_value(last_date),
            calendar_age_days=calendar_age_days,
            trading_days=trading_days,
            valid_feature_rows=0,
            forward_complete_feature_rows=0,
            max_independent_matches=0,
            data_quality=quality["data_quality"],
            reason=(
                f"{trading_days} valid trading days "
                f"(< {cfg.early_listing_days}); "
                "asset is in the early-history phase."
            ),
            recommendation="USE_NLE",
            cce_eligible=False,
            nle_eligible=True,
            feature_builder={
                "builder": "not_required",
                "builder_version": "1.6.1",
                "exact_builder": True,
                "error": "",
            },
            config=cfg,
        )

    # ------------------------------------------------------------------
    # V1.4 exact CCE feature construction.
    # ------------------------------------------------------------------

    try:
        features, feature_builder = _build_exact_features(
            normalized,
            benchmark_df,
            cfg,
        )

    except Exception as exc:
        return _base_result(
            ticker,
            engine="UNKNOWN",
            route="REVIEW_DATA",
            first_valid_date=_date_value(first_date),
            last_valid_date=_date_value(last_date),
            calendar_age_days=calendar_age_days,
            trading_days=trading_days,
            valid_feature_rows=0,
            forward_complete_feature_rows=0,
            max_independent_matches=0,
            data_quality="FEATURE_BUILD_ERROR",
            reason=(
                "Exact CCE V1.6.1 feature construction failed. "
                f"{exc}"
            ),
            recommendation="REVIEW_DATA",
            cce_eligible=False,
            nle_eligible=False,
            feature_builder={
                "builder": (
                    "characteristic_curve_v1_6_1.build_features"
                ),
                "builder_version": "1.6.1",
                "exact_builder": False,
                "error": str(exc),
            },
            config=cfg,
        )

    valid_feature_rows = int(
        feature_builder.get("valid_feature_rows", 0)
    )

    if features.empty or valid_feature_rows <= 0:
        return _base_result(
            ticker,
            engine="NEW_LISTING_ENGINE",
            route="NLE_EXTENDED_HISTORY",
            first_valid_date=_date_value(first_date),
            last_valid_date=_date_value(last_date),
            calendar_age_days=calendar_age_days,
            trading_days=trading_days,
            valid_feature_rows=valid_feature_rows,
            forward_complete_feature_rows=0,
            max_independent_matches=0,
            data_quality=quality["data_quality"],
            reason=(
                f"{trading_days} valid trading days but the exact "
                "CCE feature matrix has no usable rows; "
                "NLE is eligible for extended-history assets."
            ),
            recommendation="USE_NLE",
            cce_eligible=False,
            nle_eligible=True,
            feature_builder=feature_builder,
            config=cfg,
        )

    feature_names = feature_builder.get(
        "feature_names",
        CCE_FEATURES,
    )

    complete_positions = _forward_complete_positions(
        features,
        len(normalized),
        feature_names,
        cfg.cce_max_forward_horizon,
    )

    forward_complete_feature_rows = len(complete_positions)

    max_independent_matches = _max_independent_matches(
        complete_positions,
        cfg.cce_min_gap_days,
    )

    # ------------------------------------------------------------------
    # V1.4 structural CCE eligibility
    #
    # All three conditions are required:
    #   1. CCE min feature history
    #   2. enough states with complete forward windows
    #   3. enough chronologically independent states
    #
    # This does NOT evaluate similarity quality.
    # ------------------------------------------------------------------

    cce_eligible = (
        valid_feature_rows >= cfg.cce_min_history
        and forward_complete_feature_rows >= cfg.cce_min_matches
        and max_independent_matches >= cfg.cce_min_matches
    )

    if not cce_eligible:
        reasons: list[str] = []

        if valid_feature_rows < cfg.cce_min_history:
            reasons.append(
                f"{valid_feature_rows} valid feature rows "
                f"< {cfg.cce_min_history}"
            )

        if forward_complete_feature_rows < cfg.cce_min_matches:
            reasons.append(
                f"{forward_complete_feature_rows} forward-complete "
                f"feature states < {cfg.cce_min_matches}"
            )

        if max_independent_matches < cfg.cce_min_matches:
            reasons.append(
                f"{max_independent_matches} maximum independent "
                f"states < {cfg.cce_min_matches}"
            )

        return _base_result(
            ticker,
            engine="NEW_LISTING_ENGINE",
            route="NLE_EXTENDED_HISTORY",
            first_valid_date=_date_value(first_date),
            last_valid_date=_date_value(last_date),
            calendar_age_days=calendar_age_days,
            trading_days=trading_days,
            valid_feature_rows=valid_feature_rows,
            forward_complete_feature_rows=forward_complete_feature_rows,
            max_independent_matches=max_independent_matches,
            data_quality=quality["data_quality"],
            reason=(
                "Structural CCE history is insufficient: "
                + "; ".join(reasons)
                + ". NLE is eligible for extended-history assets."
            ),
            recommendation="USE_NLE",
            cce_eligible=False,
            nle_eligible=True,
            feature_builder=feature_builder,
            config=cfg,
        )

    return _base_result(
        ticker,
        engine="CHARACTERISTIC_CURVE_ENGINE",
        route="MATURE",
        first_valid_date=_date_value(first_date),
        last_valid_date=_date_value(last_date),
        calendar_age_days=calendar_age_days,
        trading_days=trading_days,
        valid_feature_rows=valid_feature_rows,
        forward_complete_feature_rows=forward_complete_feature_rows,
        max_independent_matches=max_independent_matches,
        data_quality=quality["data_quality"],
        reason=(
            f"{trading_days} valid trading days, "
            f"{valid_feature_rows} exact CCE feature rows, "
            f"{forward_complete_feature_rows} forward-complete states, "
            f"and capacity for {max_independent_matches} independent "
            f"states; CCE structural minimums are satisfied."
        ),
        recommendation="USE_CCE",
        cce_eligible=True,
        nle_eligible=False,
        feature_builder=feature_builder,
        config=cfg,
    )


def download_history(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download daily market history through yfinance."""

    return get_history_compat(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
    )


def download_benchmark_history(
    benchmark: str,
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download optional benchmark history for exact rs_momentum construction."""

    return get_history_compat(
        benchmark,
        period=period,
        interval=interval,
        auto_adjust=False,
    )


def resolve_asset_age(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    config: Optional[ResolverConfig] = None,
    benchmark: Optional[str] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """
    Resolve the historical-analysis engine for a ticker.

    ``benchmark`` / ``benchmark_df`` are optional because CCE V1.6.1 can
    construct its rs_momentum feature with 0.0 when no benchmark is supplied.
    Supplying the same benchmark used by the orchestrator gives the resolver
    the exact same relative-strength input as CCE.
    """

    cfg = config or ResolverConfig(
        period=period,
        interval=interval,
    )

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
        return _base_result(
            ticker,
            engine="NONE",
            route="DATA_ERROR",
            first_valid_date=None,
            last_valid_date=None,
            calendar_age_days=None,
            trading_days=0,
            valid_feature_rows=0,
            forward_complete_feature_rows=0,
            max_independent_matches=0,
            data_quality="INVALID",
            reason=f"Market data download failed: {exc}",
            recommendation="NO_DATA",
            cce_eligible=False,
            nle_eligible=False,
            feature_builder={
                "builder": "not_run",
                "builder_version": "1.6.1",
                "exact_builder": False,
                "error": str(exc),
            },
            config=cfg,
        )

    if benchmark_df is None and benchmark:
        try:
            benchmark_df = download_benchmark_history(
                benchmark,
                period=cfg.period,
                interval=cfg.interval,
            )
        except Exception:
            # The asset itself can still be evaluated exactly because CCE
            # permits benchmark=None and sets rs_momentum to 0.0.
            benchmark_df = None

    return _build_resolution(
        ticker=ticker,
        df=data,
        cfg=cfg,
        benchmark_df=benchmark_df,
    )


def resolve_from_dataframe(
    ticker: str,
    dataframe: pd.DataFrame,
    config: Optional[ResolverConfig] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Resolve routing from an already downloaded DataFrame."""

    cfg = config or ResolverConfig()

    return _build_resolution(
        ticker=ticker,
        df=dataframe,
        cfg=cfg,
        benchmark_df=benchmark_df,
    )


def _self_test() -> int:
    """Deterministic structural tests using synthetic OHLCV data."""

    dates = pd.bdate_range("2018-01-01", periods=900)

    base = np.linspace(50.0, 140.0, len(dates))
    noise = np.sin(np.arange(len(dates)) / 13.0) * 1.5

    close = pd.Series(base + noise, index=dates)
    high = close * 1.01
    low = close * 0.99
    open_ = close * (1.0 + np.sin(np.arange(len(dates))) * 0.001)
    volume = pd.Series(
        1_000_000.0 + np.abs(np.sin(np.arange(len(dates)) / 7.0)) * 250_000,
        index=dates,
    )

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )

    result = resolve_from_dataframe(
        "TEST",
        df,
        ResolverConfig(require_exact_cce_builder=True),
    )

    assert result["version"] == "1.4"
    assert result["data_quality"] == "EXCELLENT"
    assert result["valid_feature_rows"] > 0
    assert result["forward_complete_feature_rows"] > 0
    assert result["max_independent_matches"] >= 20
    assert result["cce_eligible"] is True
    assert result["route"] == "MATURE"

    # Verify that the resolver's count is exactly the count returned by the
    # CCE builder itself, not a warm-up estimate.
    from app.ai.characteristic_curve_v1_6_1 import (
        FEATURES,
        _clean_ohlcv,
        build_features,
    )

    clean = _clean_ohlcv(df)
    features = build_features(clean, None)
    expected = len(features[FEATURES].dropna())

    assert result["valid_feature_rows"] == expected

    # Early-history test.
    short_df = df.iloc[:100].copy()

    early = resolve_from_dataframe(
        "EARLY",
        short_df,
        ResolverConfig(require_exact_cce_builder=True),
    )

    assert early["route"] == "NEW_LISTING"
    assert early["engine"] == "NEW_LISTING_ENGINE"
    assert early["nle_eligible"] is True
    assert early["cce_eligible"] is False

    print("Asset Age Resolver V1.4 self-test: PASS")
    print(
        "  Exact CCE feature rows:",
        result["valid_feature_rows"],
    )
    print(
        "  Forward-complete rows:",
        result["forward_complete_feature_rows"],
    )
    print(
        "  Max independent matches:",
        result["max_independent_matches"],
    )
    print("  Mature route: PASS")
    print("  Early route: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Asset Age Resolver V1.4"
    )

    parser.add_argument(
        "--ticker",
        required=False,
        help="Ticker symbol, e.g. TQQQ, SOXL, NUAI, ANRO",
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
        "--benchmark",
        default=None,
        help="Optional benchmark used by the exact CCE feature builder.",
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic self-tests and exit",
    )

    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if not args.ticker:
        parser.error("--ticker is required unless --self-test is used")

    result = resolve_asset_age(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        benchmark=args.benchmark,
    )

    if args.pretty:
        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    else:
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
