"""
Benchmark Resolver V1.4
-----------------------

Listing-aware benchmark resolution for AI Trader.

V1.4 keeps the V1.3 architecture and fixes the main limitation found with
very recent listings: a benchmark was required to have 252 overlapping
sessions even when the asset itself had only a few dozen sessions.

Resolution priority:
    1. Manual benchmark override
    2. Dynamic candidate universe
    3. Configured sector benchmark
    4. Broad-market fallback (SPY)
    5. No benchmark

V1.4 validation is adaptive for young assets:
    - Mature assets: strict 252-day history/overlap requirements.
    - Young assets (<252 asset sessions): required benchmark history and
      overlap scale from 30 to 60 sessions.
    - Data quality, positive volume, liquidity, close-overlap and statistical
      relevance checks remain unchanged.
    - Manual benchmark choices are never silently replaced.
    - The resolver reports the effective adaptive thresholds.

CLI examples:
    python -m app.ai.benchmark_resolver_v1_4 --ticker SPCX --validate
    python -m app.ai.benchmark_resolver_v1_4 --ticker PLTR --validate
    python -m app.ai.benchmark_resolver_v1_4 --ticker DFEN --validate
    python -m app.ai.benchmark_resolver_v1_4 --ticker SPCX --benchmark SPY --validate

Optional listing age:
    resolve_benchmarks("SPCX", listing_days=64, validate=True)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


VERSION = "1.4"
MARKET_BENCHMARK = "SPY"

BENCHMARK_MAP: Dict[str, str] = {
    "TQQQ": "QQQ",
    "NVDA": "SMH",
    "SOXL": "SMH",
    "META": "QQQ",
    "GDXU": "GDX",
    "CCJ": "URNM",
    "AGQ": "SLV",
}

BENCHMARK_CANDIDATES: Dict[str, List[str]] = {
    "DFEN": ["ITA", "PPA", "XAR"],
    "PLTR": ["QQQ", "IGV", "XLK"],
}

BENCHMARK_ROLE: Dict[str, str] = {
    "QQQ": "technology_growth",
    "SMH": "semiconductors",
    "GDX": "gold_miners",
    "URNM": "uranium_nuclear",
    "SLV": "silver",
    "SPY": "broad_market",
    "ITA": "aerospace_defense",
    "PPA": "aerospace_defense",
    "XAR": "aerospace_defense",
    "IGV": "software_technology",
    "XLK": "technology",
}


@dataclass(frozen=True)
class ValidationConfig:
    # Mature/default requirements.
    min_history_days: int = 252
    min_overlap_days: int = 252

    # Young-listing adaptive bounds.
    young_asset_threshold_days: int = 252
    young_min_history_days: int = 30
    young_min_overlap_days: int = 30
    young_max_overlap_days: int = 60

    min_data_quality_ratio: float = 0.98
    min_positive_volume_ratio: float = 0.95
    min_median_dollar_volume: float = 1_000_000.0
    min_close_overlap_ratio: float = 0.98

    relevance_window_days: int = 252
    short_relevance_window_days: int = 60
    rolling_correlation_window: int = 60
    rolling_observations: int = 6

    min_relevance_score: float = 55.0
    min_return_correlation: float = 0.30


@dataclass
class BenchmarkValidation:
    requested_benchmark: str
    available: bool = False
    sufficient_history: bool = False
    overlap_ok: bool = False
    ohlcv_ok: bool = False
    liquidity_ok: bool = False
    valid: bool = False

    data_quality: str = "UNKNOWN"
    benchmark_rows: int = 0
    asset_rows: int = 0
    overlap_rows: int = 0
    median_dollar_volume: float = 0.0
    quality_ratio: float = 0.0
    positive_volume_ratio: float = 0.0
    overlap_close_ratio: float = 0.0

    required_history_days: int = 252
    required_overlap_days: int = 252
    validation_mode: str = "MATURE"

    reason: str = ""


@dataclass
class BenchmarkRelevance:
    benchmark: str
    relevance_score: float = 0.0
    daily_return_correlation: float = 0.0
    short_return_correlation: float = 0.0
    rolling_correlation: float = 0.0
    beta: float = 0.0
    observations: int = 0
    role: str = "unknown"

    quality_valid: bool = False
    valid: bool = False
    reason: str = ""


@dataclass
class CandidateEvaluation:
    benchmark: str
    validation: BenchmarkValidation
    relevance: BenchmarkRelevance
    eligible: bool = False


@dataclass
class BenchmarkResolution:
    ticker: str
    sector_benchmark: Optional[str]
    market_benchmark: str
    selected_benchmark: Optional[str]
    source: str
    role: str
    is_manual: bool

    fallback_reason: str = ""
    selection_reason: str = ""
    selection_confidence: float = 0.0

    listing_days: Optional[int] = None
    validation_mode: str = "MATURE"
    required_history_days: int = 252
    required_overlap_days: int = 252

    validation: Optional[BenchmarkValidation] = None
    relevance: Optional[BenchmarkRelevance] = None
    candidate_evaluations: Optional[List[CandidateEvaluation]] = None

    resolver_version: str = VERSION


def _clean_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def _download_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame:
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        flattened = []
        for col in data.columns:
            flattened.append(col[0] if isinstance(col, tuple) else col)
        data.columns = flattened

    data = data.loc[:, ~data.columns.duplicated()].copy()

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        return pd.DataFrame()

    data = data[required].copy()
    data.index = pd.to_datetime(data.index)

    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_localize(None)

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.sort_index()


def _adaptive_requirements(
    asset_rows: int,
    cfg: ValidationConfig,
    listing_days: Optional[int] = None,
) -> tuple[int, int, str]:
    """
    Return effective benchmark-history and overlap requirements.

    Young assets use a minimum of 30 overlapping sessions and scale toward
    60 sessions. Mature assets retain the original 252-session requirement.

    If listing_days is explicitly supplied, it is authoritative for the
    young/mature classification. Otherwise asset_rows is used as a safe
    proxy for the available analysis age.
    """
    age = listing_days if listing_days is not None else asset_rows
    try:
        age = max(0, int(age))
    except (TypeError, ValueError):
        age = asset_rows

    if age >= cfg.young_asset_threshold_days:
        return (
            cfg.min_history_days,
            cfg.min_overlap_days,
            "MATURE",
        )

    required_overlap = int(
        min(
            cfg.young_max_overlap_days,
            max(
                cfg.young_min_overlap_days,
                math.ceil(age * 0.50),
            ),
        )
    )
    required_history = max(
        cfg.young_min_history_days,
        required_overlap,
    )

    return required_history, required_overlap, "LISTING_AWARE"


def _validation_from_data(
    asset: pd.DataFrame,
    benchmark: pd.DataFrame,
    benchmark_ticker: str,
    cfg: ValidationConfig,
    listing_days: Optional[int] = None,
) -> BenchmarkValidation:
    required_history, required_overlap, mode = _adaptive_requirements(
        len(asset), cfg, listing_days
    )

    result = BenchmarkValidation(
        requested_benchmark=benchmark_ticker,
        asset_rows=len(asset),
        required_history_days=required_history,
        required_overlap_days=required_overlap,
        validation_mode=mode,
    )

    if benchmark.empty:
        result.reason = "benchmark_data_unavailable"
        return result

    result.available = True
    result.benchmark_rows = len(benchmark)
    result.sufficient_history = len(benchmark) >= required_history

    if asset.empty:
        result.reason = "asset_data_unavailable"
        return result

    overlap = pd.concat(
        [
            asset["Close"].rename("asset"),
            benchmark["Close"].rename("benchmark"),
        ],
        axis=1,
        join="inner",
    )

    result.overlap_rows = len(overlap)
    result.overlap_ok = len(overlap) >= required_overlap

    required = ["Open", "High", "Low", "Close", "Volume"]
    quality = benchmark[required].notna().all(axis=1)

    result.quality_ratio = float(quality.mean()) if len(quality) else 0.0
    result.positive_volume_ratio = float(
        (benchmark["Volume"].fillna(0) > 0).mean()
    ) if len(benchmark) else 0.0

    dollar_volume = benchmark["Close"] * benchmark["Volume"]
    result.median_dollar_volume = float(
        dollar_volume.replace([np.inf, -np.inf], np.nan).median()
    )

    # Because overlap is constructed from both close series, this measures
    # how complete the asset close is across the actual overlap.
    result.overlap_close_ratio = (
        float(overlap["asset"].notna().mean())
        if len(overlap)
        else 0.0
    )

    result.ohlcv_ok = (
        result.quality_ratio >= cfg.min_data_quality_ratio
        and result.positive_volume_ratio >= cfg.min_positive_volume_ratio
    )

    result.liquidity_ok = (
        math.isfinite(result.median_dollar_volume)
        and result.median_dollar_volume >= cfg.min_median_dollar_volume
    )

    result.valid = (
        result.available
        and result.sufficient_history
        and result.overlap_ok
        and result.ohlcv_ok
        and result.liquidity_ok
        and result.overlap_close_ratio >= cfg.min_close_overlap_ratio
    )

    if result.valid:
        if result.quality_ratio >= 0.995:
            result.data_quality = "EXCELLENT"
        elif result.quality_ratio >= 0.98:
            result.data_quality = "GOOD"
        else:
            result.data_quality = "MODERATE"
        result.reason = "validation_passed"
    else:
        reasons = []
        if not result.sufficient_history:
            reasons.append("insufficient_history")
        if not result.overlap_ok:
            reasons.append("insufficient_overlap")
        if not result.ohlcv_ok:
            reasons.append("ohlcv_quality_failed")
        if not result.liquidity_ok:
            reasons.append("liquidity_failed")
        if result.overlap_close_ratio < cfg.min_close_overlap_ratio:
            reasons.append("close_overlap_failed")

        result.data_quality = "WEAK" if reasons else "UNKNOWN"
        result.reason = ";".join(reasons) if reasons else "validation_failed"

    return result


def validate_benchmark(
    ticker: str,
    benchmark: str,
    cfg: Optional[ValidationConfig] = None,
    period: str = "5y",
    listing_days: Optional[int] = None,
) -> BenchmarkValidation:
    cfg = cfg or ValidationConfig()

    asset_ticker = _clean_ticker(ticker)
    benchmark_ticker = _clean_ticker(benchmark)

    asset = _download_ohlcv(asset_ticker, period=period)
    bench = _download_ohlcv(benchmark_ticker, period=period)

    return _validation_from_data(
        asset,
        bench,
        benchmark_ticker,
        cfg,
        listing_days=listing_days,
    )


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 10:
        return 0.0

    value = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    return 0.0 if pd.isna(value) else float(value)


def _compute_relevance(
    asset: pd.DataFrame,
    benchmark: pd.DataFrame,
    benchmark_ticker: str,
    validation: BenchmarkValidation,
    cfg: ValidationConfig,
) -> BenchmarkRelevance:
    result = BenchmarkRelevance(
        benchmark=benchmark_ticker,
        role=BENCHMARK_ROLE.get(benchmark_ticker, "unknown"),
        quality_valid=validation.valid,
    )

    if asset.empty or benchmark.empty:
        result.reason = "data_unavailable"
        return result

    merged = pd.concat(
        [
            asset["Close"].rename("asset"),
            benchmark["Close"].rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    # Listing-aware relevance is allowed to operate with the same effective
    # minimum overlap as validation, but never with fewer than 20 returns.
    minimum_observations = max(20, validation.required_overlap_days - 1)

    if len(merged) < minimum_observations:
        result.reason = "insufficient_overlap_for_relevance"
        return result

    returns = merged.pct_change().dropna()
    result.observations = len(returns)

    if len(returns) < 20:
        result.reason = "insufficient_return_observations"
        return result

    long_returns = returns.tail(
        min(cfg.relevance_window_days, len(returns))
    )
    short_returns = returns.tail(
        min(cfg.short_relevance_window_days, len(returns))
    )

    result.daily_return_correlation = _safe_corr(
        long_returns["asset"],
        long_returns["benchmark"],
    )
    result.short_return_correlation = _safe_corr(
        short_returns["asset"],
        short_returns["benchmark"],
    )

    rolling_window = min(
        cfg.rolling_correlation_window,
        max(20, len(returns)),
    )

    rolling = (
        returns["asset"]
        .rolling(rolling_window)
        .corr(returns["benchmark"])
        .dropna()
    )

    if len(rolling):
        result.rolling_correlation = float(
            rolling.tail(cfg.rolling_observations).median()
        )
    else:
        result.rolling_correlation = result.daily_return_correlation

    benchmark_var = float(long_returns["benchmark"].var())
    if benchmark_var > 0:
        result.beta = float(
            long_returns["asset"].cov(long_returns["benchmark"])
            / benchmark_var
        )

    long_component = max(0.0, result.daily_return_correlation) * 100.0
    short_component = max(0.0, result.short_return_correlation) * 100.0
    rolling_component = max(0.0, result.rolling_correlation) * 100.0

    score = (
        0.55 * long_component
        + 0.25 * short_component
        + 0.20 * rolling_component
    )

    if validation.data_quality == "EXCELLENT":
        score += 2.0
    elif validation.data_quality == "GOOD":
        score += 1.0

    result.relevance_score = round(min(100.0, score), 2)

    result.valid = (
        validation.valid
        and result.relevance_score >= cfg.min_relevance_score
        and result.daily_return_correlation >= cfg.min_return_correlation
    )

    if result.valid:
        result.reason = "relevance_passed"
    elif not validation.valid:
        result.reason = "benchmark_data_validation_failed"
    elif result.relevance_score < cfg.min_relevance_score:
        result.reason = "low_statistical_relevance"
    elif result.daily_return_correlation < cfg.min_return_correlation:
        result.reason = "low_return_correlation"
    else:
        result.reason = "relevance_failed"

    return result


def _evaluate_candidate(
    asset: pd.DataFrame,
    benchmark_ticker: str,
    cfg: ValidationConfig,
    period: str,
    listing_days: Optional[int],
) -> CandidateEvaluation:
    benchmark = _download_ohlcv(benchmark_ticker, period=period)
    validation = _validation_from_data(
        asset,
        benchmark,
        benchmark_ticker,
        cfg,
        listing_days=listing_days,
    )
    relevance = _compute_relevance(
        asset,
        benchmark,
        benchmark_ticker,
        validation,
        cfg,
    )

    return CandidateEvaluation(
        benchmark=benchmark_ticker,
        validation=validation,
        relevance=relevance,
        eligible=bool(validation.valid and relevance.valid),
    )


def _evaluate_candidates(
    asset: pd.DataFrame,
    candidates: List[str],
    cfg: ValidationConfig,
    period: str,
    listing_days: Optional[int],
) -> List[CandidateEvaluation]:
    evaluations = [
        _evaluate_candidate(
            asset,
            candidate,
            cfg,
            period,
            listing_days,
        )
        for candidate in candidates
    ]

    evaluations.sort(
        key=lambda item: (
            item.eligible,
            item.relevance.relevance_score,
            item.validation.median_dollar_volume,
        ),
        reverse=True,
    )
    return evaluations


def _selection_confidence(
    relevance: Optional[BenchmarkRelevance],
    source: str,
) -> float:
    if relevance is None:
        return 0.0

    base = relevance.relevance_score

    if source == "sector_candidate":
        return round(base, 2)

    if source == "sector":
        return round(min(100.0, 0.70 * base + 30.0), 2)

    if source == "market":
        return round(min(100.0, 0.60 * base + 40.0), 2)

    if source == "manual":
        return 100.0

    return round(base, 2)


def _new_resolution(
    ticker: str,
    sector_benchmark: Optional[str],
    *,
    selected_benchmark: Optional[str],
    source: str,
    role: str,
    is_manual: bool,
    cfg: ValidationConfig,
    asset_rows: int,
    listing_days: Optional[int],
    **kwargs,
) -> BenchmarkResolution:
    required_history, required_overlap, mode = _adaptive_requirements(
        asset_rows,
        cfg,
        listing_days,
    )

    return BenchmarkResolution(
        ticker=ticker,
        sector_benchmark=sector_benchmark,
        market_benchmark=MARKET_BENCHMARK,
        selected_benchmark=selected_benchmark,
        source=source,
        role=role,
        is_manual=is_manual,
        listing_days=listing_days,
        validation_mode=mode,
        required_history_days=required_history,
        required_overlap_days=required_overlap,
        **kwargs,
    )


def resolve_benchmarks(
    ticker: str,
    manual_benchmark: Optional[str] = None,
    allow_market_fallback: bool = True,
    validate: bool = False,
    cfg: Optional[ValidationConfig] = None,
    period: str = "5y",
    listing_days: Optional[int] = None,
) -> BenchmarkResolution:
    cfg = cfg or ValidationConfig()
    symbol = _clean_ticker(ticker)

    if not symbol:
        raise ValueError("ticker is required")

    manual = _clean_ticker(manual_benchmark)
    sector_benchmark = BENCHMARK_MAP.get(symbol)

    # ---------------------------------------------------------------
    # 1. Manual override: NEVER silently replace the user's choice.
    # ---------------------------------------------------------------
    if manual:
        resolution = _new_resolution(
            symbol,
            sector_benchmark,
            selected_benchmark=manual,
            source="manual",
            role=BENCHMARK_ROLE.get(manual, "custom"),
            is_manual=True,
            cfg=cfg,
            asset_rows=0,
            listing_days=listing_days,
            selection_reason="manual_benchmark_override",
            selection_confidence=100.0,
        )

        if validate:
            asset = _download_ohlcv(symbol, period=period)
            benchmark = _download_ohlcv(manual, period=period)

            resolution = _new_resolution(
                symbol,
                sector_benchmark,
                selected_benchmark=manual,
                source="manual",
                role=BENCHMARK_ROLE.get(manual, "custom"),
                is_manual=True,
                cfg=cfg,
                asset_rows=len(asset),
                listing_days=listing_days,
                selection_reason="manual_benchmark_override",
                selection_confidence=100.0,
                validation=_validation_from_data(
                    asset,
                    benchmark,
                    manual,
                    cfg,
                    listing_days=listing_days,
                ),
                relevance=None,
            )

            resolution.relevance = _compute_relevance(
                asset,
                benchmark,
                manual,
                resolution.validation,
                cfg,
            )

            if not resolution.validation.valid:
                resolution.fallback_reason = (
                    "manual_benchmark_validation_failed"
                )
            elif not resolution.relevance.valid:
                resolution.fallback_reason = (
                    "manual_benchmark_low_statistical_relevance"
                )

        return resolution

    # Download the asset once when validation is enabled. This is reused for
    # every candidate and fallback, avoiding repeated asset downloads.
    asset = _download_ohlcv(symbol, period=period) if validate else pd.DataFrame()

    # Derive effective age from real listing_days when supplied; otherwise
    # from the actual downloaded asset history.
    effective_listing_days = listing_days
    asset_rows = len(asset)

    required_history, required_overlap, mode = _adaptive_requirements(
        asset_rows,
        cfg,
        effective_listing_days,
    )

    # ---------------------------------------------------------------
    # 2. Dynamic candidate universe.
    # ---------------------------------------------------------------
    candidates = BENCHMARK_CANDIDATES.get(symbol)
    if candidates:
        if validate:
            evaluations = _evaluate_candidates(
                asset,
                candidates,
                cfg,
                period,
                listing_days,
            )

            eligible = [item for item in evaluations if item.eligible]

            if eligible:
                best = eligible[0]
                return _new_resolution(
                    symbol,
                    sector_benchmark,
                    selected_benchmark=best.benchmark,
                    source="sector_candidate",
                    role=BENCHMARK_ROLE.get(best.benchmark, "unknown"),
                    is_manual=False,
                    cfg=cfg,
                    asset_rows=asset_rows,
                    listing_days=listing_days,
                    selection_reason=(
                        "best_valid_candidate_by_statistical_relevance"
                    ),
                    selection_confidence=best.relevance.relevance_score,
                    validation=best.validation,
                    relevance=best.relevance,
                    candidate_evaluations=evaluations,
                )

            # No valid candidate: continue to market fallback.
            fallback_reason = "no_valid_sector_candidate"
        else:
            if allow_market_fallback:
                return _new_resolution(
                    symbol,
                    sector_benchmark,
                    selected_benchmark=MARKET_BENCHMARK,
                    source="market",
                    role=BENCHMARK_ROLE.get(
                        MARKET_BENCHMARK,
                        "broad_market",
                    ),
                    is_manual=False,
                    cfg=cfg,
                    asset_rows=0,
                    listing_days=listing_days,
                    fallback_reason=(
                        "sector_candidates_require_validation_for_dynamic_selection"
                    ),
                    selection_reason="market_fallback_without_validation",
                )
            fallback_reason = "market_fallback_disabled"
    else:
        fallback_reason = ""

    # ---------------------------------------------------------------
    # 3. Configured sector benchmark.
    # ---------------------------------------------------------------
    if sector_benchmark:
        if validate:
            benchmark = _download_ohlcv(
                sector_benchmark,
                period=period,
            )
            validation = _validation_from_data(
                asset,
                benchmark,
                sector_benchmark,
                cfg,
                listing_days=listing_days,
            )
            relevance = _compute_relevance(
                asset,
                benchmark,
                sector_benchmark,
                validation,
                cfg,
            )

            if validation.valid and relevance.valid:
                return _new_resolution(
                    symbol,
                    sector_benchmark,
                    selected_benchmark=sector_benchmark,
                    source="sector",
                    role=BENCHMARK_ROLE.get(
                        sector_benchmark,
                        "unknown",
                    ),
                    is_manual=False,
                    cfg=cfg,
                    asset_rows=asset_rows,
                    listing_days=listing_days,
                    selection_reason=(
                        "configured_sector_benchmark_passed_validation"
                    ),
                    selection_confidence=_selection_confidence(
                        relevance,
                        "sector",
                    ),
                    validation=validation,
                    relevance=relevance,
                    candidate_evaluations=None,
                )

            reason = (
                "sector_benchmark_validation_failed"
                if not validation.valid
                else "sector_benchmark_low_statistical_relevance"
            )

            if allow_market_fallback:
                market = _resolve_market(
                    symbol,
                    sector_benchmark,
                    asset,
                    cfg,
                    period,
                    listing_days,
                )
                if market is not None:
                    market.fallback_reason = reason
                    market.selection_reason = (
                        "sector_benchmark_rejected_market_fallback_selected"
                    )
                    return market

            return _new_resolution(
                symbol,
                sector_benchmark,
                selected_benchmark=None,
                source="none",
                role="unknown",
                is_manual=False,
                cfg=cfg,
                asset_rows=asset_rows,
                listing_days=listing_days,
                fallback_reason=reason,
                selection_reason="no_valid_benchmark",
                validation=validation,
                relevance=relevance,
            )

        return _new_resolution(
            symbol,
            sector_benchmark,
            selected_benchmark=sector_benchmark,
            source="sector",
            role=BENCHMARK_ROLE.get(sector_benchmark, "unknown"),
            is_manual=False,
            cfg=cfg,
            asset_rows=asset_rows,
            listing_days=listing_days,
            selection_reason="configured_sector_benchmark",
        )

    # ---------------------------------------------------------------
    # 4. Broad-market fallback.
    # ---------------------------------------------------------------
    if allow_market_fallback:
        if validate:
            market = _resolve_market(
                symbol,
                sector_benchmark,
                asset,
                cfg,
                period,
                listing_days,
            )
            if market is not None:
                return market

            return _new_resolution(
                symbol,
                sector_benchmark,
                selected_benchmark=None,
                source="none",
                role="unknown",
                is_manual=False,
                cfg=cfg,
                asset_rows=asset_rows,
                listing_days=listing_days,
                fallback_reason="market_benchmark_validation_failed",
                selection_reason="no_valid_benchmark",
            )

        return _new_resolution(
            symbol,
            sector_benchmark,
            selected_benchmark=MARKET_BENCHMARK,
            source="market",
            role=BENCHMARK_ROLE.get(
                MARKET_BENCHMARK,
                "broad_market",
            ),
            is_manual=False,
            cfg=cfg,
            asset_rows=asset_rows,
            listing_days=listing_days,
            selection_reason="broad_market_fallback",
        )

    # ---------------------------------------------------------------
    # 5. No fallback allowed.
    # ---------------------------------------------------------------
    return _new_resolution(
        symbol,
        sector_benchmark,
        selected_benchmark=None,
        source="none",
        role="unknown",
        is_manual=False,
        cfg=cfg,
        asset_rows=asset_rows,
        listing_days=listing_days,
        fallback_reason="market_fallback_disabled",
        selection_reason="no_valid_benchmark",
    )


def _resolve_market(
    symbol: str,
    sector_benchmark: Optional[str],
    asset: pd.DataFrame,
    cfg: ValidationConfig,
    period: str,
    listing_days: Optional[int],
) -> Optional[BenchmarkResolution]:
    benchmark = _download_ohlcv(
        MARKET_BENCHMARK,
        period=period,
    )

    validation = _validation_from_data(
        asset,
        benchmark,
        MARKET_BENCHMARK,
        cfg,
        listing_days=listing_days,
    )

    relevance = _compute_relevance(
        asset,
        benchmark,
        MARKET_BENCHMARK,
        validation,
        cfg,
    )

    if not validation.valid:
        return None

    return _new_resolution(
        symbol,
        sector_benchmark,
        selected_benchmark=MARKET_BENCHMARK,
        source="market",
        role=BENCHMARK_ROLE.get(
            MARKET_BENCHMARK,
            "broad_market",
        ),
        is_manual=False,
        cfg=cfg,
        asset_rows=len(asset),
        listing_days=listing_days,
        selection_reason="broad_market_fallback",
        selection_confidence=_selection_confidence(
            relevance,
            "market",
        ),
        validation=validation,
        relevance=relevance,
    )


def get_benchmark(
    ticker: str,
    manual_benchmark: Optional[str] = None,
    validate: bool = False,
    listing_days: Optional[int] = None,
) -> Optional[str]:
    return resolve_benchmarks(
        ticker,
        manual_benchmark=manual_benchmark,
        validate=validate,
        listing_days=listing_days,
    ).selected_benchmark


def get_sector_benchmark(ticker: str) -> Optional[str]:
    return BENCHMARK_MAP.get(_clean_ticker(ticker))


def validate_resolution(
    ticker: str,
    manual_benchmark: Optional[str] = None,
    cfg: Optional[ValidationConfig] = None,
    period: str = "5y",
    listing_days: Optional[int] = None,
) -> Dict:
    result = resolve_benchmarks(
        ticker,
        manual_benchmark=manual_benchmark,
        validate=True,
        cfg=cfg,
        period=period,
        listing_days=listing_days,
    )
    return asdict(result)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Resolver V1.4"
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Asset ticker, e.g. SPCX, SOXL, DFEN, PLTR",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Manual benchmark override",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Download data and validate benchmark quality/relevance",
    )
    parser.add_argument(
        "--period",
        default="5y",
        help="yfinance period, default: 5y",
    )
    parser.add_argument(
        "--listing-days",
        type=int,
        default=None,
        help="Optional real asset age in trading days. When omitted, asset history is used.",
    )

    args = parser.parse_args()

    result = resolve_benchmarks(
        args.ticker,
        manual_benchmark=args.benchmark,
        validate=args.validate,
        period=args.period,
        listing_days=args.listing_days,
    )

    print(
        json.dumps(
            asdict(result),
            indent=2,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
