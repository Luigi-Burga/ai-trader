"""
Benchmark Resolver V1.3
----------------------

Resolves the most appropriate benchmark for an asset using:
1. Manual benchmark override.
2. Configured sector benchmark.
3. Dynamic candidate universe with validation + statistical relevance.
4. Broad-market fallback (SPY).
5. No benchmark if the market fallback itself is invalid.

V1.3 fixes the V1.2 inconsistency where a configured sector benchmark
could remain selected even when its statistical relevance was invalid.

It also adds dynamic candidate selection for PLTR and DFEN, selection
confidence/reason, and explicit separation between data-quality validity
and statistical relevance.

CLI examples:
    python -m app.ai.benchmark_resolver_v1_3 --ticker SOXL --validate
    python -m app.ai.benchmark_resolver_v1_3 --ticker DFEN --validate
    python -m app.ai.benchmark_resolver_v1_3 --ticker PLTR --validate
    python -m app.ai.benchmark_resolver_v1_3 --ticker PLTR --benchmark QQQ
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


VERSION = "1.3"
MARKET_BENCHMARK = "SPY"


# Direct sector/context relationships that are considered structurally valid.
# V1.3 intentionally does NOT force a benchmark when statistical relevance
# is weak. Those cases are handled through candidate selection or SPY fallback.
BENCHMARK_MAP: Dict[str, str] = {
    "TQQQ": "QQQ",
    "NVDA": "SMH",
    "SOXL": "SMH",
    "META": "QQQ",
    "GDXU": "GDX",
    "CCJ": "URNM",
    "AGQ": "SLV",
}


# Candidate universes are evaluated dynamically when --validate is used.
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
    min_history_days: int = 252
    min_overlap_days: int = 252
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
    overlap_rows: int = 0
    median_dollar_volume: float = 0.0
    quality_ratio: float = 0.0
    positive_volume_ratio: float = 0.0
    overlap_close_ratio: float = 0.0

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

    # yfinance can return MultiIndex columns for a single ticker.
    if isinstance(data.columns, pd.MultiIndex):
        flattened = []
        for col in data.columns:
            if isinstance(col, tuple):
                flattened.append(col[0])
            else:
                flattened.append(col)
        data.columns = flattened

    # Keep first occurrence if duplicate columns appear.
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


def _validation_from_data(
    asset: pd.DataFrame,
    benchmark: pd.DataFrame,
    benchmark_ticker: str,
    cfg: ValidationConfig,
) -> BenchmarkValidation:
    result = BenchmarkValidation(requested_benchmark=benchmark_ticker)

    if benchmark.empty:
        result.reason = "benchmark_data_unavailable"
        return result

    result.available = True
    result.benchmark_rows = len(benchmark)
    result.sufficient_history = len(benchmark) >= cfg.min_history_days

    if asset.empty:
        result.reason = "asset_data_unavailable"
        return result

    asset_close = asset["Close"]
    benchmark_close = benchmark["Close"]

    overlap = pd.concat(
        [asset_close.rename("asset"), benchmark_close.rename("benchmark")],
        axis=1,
        join="inner",
    )

    result.overlap_rows = len(overlap)

    if len(overlap) > 0:
        result.overlap_ok = len(overlap) >= cfg.min_overlap_days

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
        result.median_dollar_volume >= cfg.min_median_dollar_volume
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
) -> BenchmarkValidation:
    cfg = cfg or ValidationConfig()

    asset_ticker = _clean_ticker(ticker)
    benchmark_ticker = _clean_ticker(benchmark)

    asset = _download_ohlcv(asset_ticker, period=period)
    bench = _download_ohlcv(benchmark_ticker, period=period)

    return _validation_from_data(asset, bench, benchmark_ticker, cfg)


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 10:
        return 0.0

    value = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    if pd.isna(value):
        return 0.0
    return float(value)


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

    if len(merged) < 30:
        result.reason = "insufficient_overlap_for_relevance"
        return result

    returns = merged.pct_change().dropna()
    result.observations = len(returns)

    if len(returns) < 20:
        result.reason = "insufficient_return_observations"
        return result

    long_returns = returns.tail(cfg.relevance_window_days)
    short_returns = returns.tail(cfg.short_relevance_window_days)

    result.daily_return_correlation = _safe_corr(
        long_returns["asset"],
        long_returns["benchmark"],
    )
    result.short_return_correlation = _safe_corr(
        short_returns["asset"],
        short_returns["benchmark"],
    )

    rolling = (
        returns["asset"]
        .rolling(cfg.rolling_correlation_window)
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

    # Convert correlations from [-1,1] to a relevance contribution [0,100].
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
) -> CandidateEvaluation:
    benchmark = _download_ohlcv(benchmark_ticker, period=period)
    validation = _validation_from_data(
        asset,
        benchmark,
        benchmark_ticker,
        cfg,
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
) -> List[CandidateEvaluation]:
    evaluations = [
        _evaluate_candidate(asset, candidate, cfg, period)
        for candidate in candidates
    ]

    # Best statistical relevance first; liquidity is the tie-breaker.
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
        # A configured benchmark has structural confidence, but statistical
        # relevance is still reflected in the score.
        return round(min(100.0, 0.70 * base + 30.0), 2)

    if source == "market":
        return round(min(100.0, 0.60 * base + 40.0), 2)

    if source == "manual":
        return 100.0

    return round(base, 2)


def resolve_benchmarks(
    ticker: str,
    manual_benchmark: Optional[str] = None,
    allow_market_fallback: bool = True,
    validate: bool = False,
    cfg: Optional[ValidationConfig] = None,
    period: str = "5y",
) -> BenchmarkResolution:
    cfg = cfg or ValidationConfig()
    symbol = _clean_ticker(ticker)

    manual = _clean_ticker(manual_benchmark)
    sector_benchmark = BENCHMARK_MAP.get(symbol)

    # ---------------------------------------------------------------
    # 1. Manual override: NEVER silently replace the user's choice.
    # ---------------------------------------------------------------
    if manual:
        resolution = BenchmarkResolution(
            ticker=symbol,
            sector_benchmark=sector_benchmark,
            market_benchmark=MARKET_BENCHMARK,
            selected_benchmark=manual,
            source="manual",
            role=BENCHMARK_ROLE.get(manual, "custom"),
            is_manual=True,
            selection_reason="manual_benchmark_override",
            selection_confidence=100.0,
        )

        if validate:
            asset = _download_ohlcv(symbol, period=period)
            benchmark = _download_ohlcv(manual, period=period)
            validation = _validation_from_data(
                asset, benchmark, manual, cfg
            )
            relevance = _compute_relevance(
                asset, benchmark, manual, validation, cfg
            )

            resolution.validation = validation
            resolution.relevance = relevance

            if not validation.valid:
                resolution.fallback_reason = (
                    "manual_benchmark_validation_failed"
                )
            elif not relevance.valid:
                resolution.fallback_reason = (
                    "manual_benchmark_low_statistical_relevance"
                )

        return resolution

    # ---------------------------------------------------------------
    # 2. Dynamic candidate universe.
    # ---------------------------------------------------------------
    candidates = BENCHMARK_CANDIDATES.get(symbol)
    if candidates:
        if validate:
            asset = _download_ohlcv(symbol, period=period)
            evaluations = _evaluate_candidates(
                asset,
                candidates,
                cfg,
                period,
            )

            resolution = BenchmarkResolution(
                ticker=symbol,
                sector_benchmark=sector_benchmark,
                market_benchmark=MARKET_BENCHMARK,
                selected_benchmark=None,
                source="",
                role="unknown",
                is_manual=False,
                candidate_evaluations=evaluations,
            )

            eligible = [item for item in evaluations if item.eligible]

            if eligible:
                best = eligible[0]
                resolution.selected_benchmark = best.benchmark
                resolution.source = "sector_candidate"
                resolution.role = BENCHMARK_ROLE.get(
                    best.benchmark,
                    "unknown",
                )
                resolution.validation = best.validation
                resolution.relevance = best.relevance
                resolution.selection_confidence = (
                    best.relevance.relevance_score
                )
                resolution.selection_reason = (
                    "best_valid_candidate_by_statistical_relevance"
                )
                return resolution

            # No valid candidate: continue to market fallback.
            resolution.fallback_reason = (
                "no_valid_sector_candidate"
            )
        else:
            # Without validation, do not make a data-dependent dynamic
            # selection. This keeps the resolver deterministic and honest.
            if allow_market_fallback:
                return BenchmarkResolution(
                    ticker=symbol,
                    sector_benchmark=sector_benchmark,
                    market_benchmark=MARKET_BENCHMARK,
                    selected_benchmark=MARKET_BENCHMARK,
                    source="market",
                    role=BENCHMARK_ROLE.get(
                        MARKET_BENCHMARK,
                        "broad_market",
                    ),
                    is_manual=False,
                    fallback_reason=(
                        "sector_candidates_require_validation_for_dynamic_selection"
                    ),
                    selection_reason="market_fallback_without_validation",
                )

    # ---------------------------------------------------------------
    # 3. Configured sector benchmark.
    #
    # V1.3 FIX:
    # If validation is requested and the configured benchmark is
    # statistically irrelevant, it is NOT left selected.
    # It falls back to SPY.
    # ---------------------------------------------------------------
    if sector_benchmark:
        if validate:
            asset = _download_ohlcv(symbol, period=period)
            benchmark = _download_ohlcv(
                sector_benchmark,
                period=period,
            )

            validation = _validation_from_data(
                asset,
                benchmark,
                sector_benchmark,
                cfg,
            )
            relevance = _compute_relevance(
                asset,
                benchmark,
                sector_benchmark,
                validation,
                cfg,
            )

            if validation.valid and relevance.valid:
                return BenchmarkResolution(
                    ticker=symbol,
                    sector_benchmark=sector_benchmark,
                    market_benchmark=MARKET_BENCHMARK,
                    selected_benchmark=sector_benchmark,
                    source="sector",
                    role=BENCHMARK_ROLE.get(
                        sector_benchmark,
                        "unknown",
                    ),
                    is_manual=False,
                    selection_reason="configured_sector_benchmark_passed_validation",
                    selection_confidence=_selection_confidence(
                        relevance,
                        "sector",
                    ),
                    validation=validation,
                    relevance=relevance,
                )

            # IMPORTANT: do not keep an invalid configured benchmark.
            reason = (
                "sector_benchmark_validation_failed"
                if not validation.valid
                else "sector_benchmark_low_statistical_relevance"
            )

            if allow_market_fallback:
                market_resolution = resolve_benchmarks(
                    symbol,
                    manual_benchmark=None,
                    allow_market_fallback=False,
                    validate=validate,
                    cfg=cfg,
                    period=period,
                )

                if market_resolution.selected_benchmark:
                    market_resolution.sector_benchmark = sector_benchmark
                    market_resolution.fallback_reason = reason
                    market_resolution.selection_reason = (
                        "sector_benchmark_rejected_market_fallback_selected"
                    )
                    return market_resolution

            return BenchmarkResolution(
                ticker=symbol,
                sector_benchmark=sector_benchmark,
                market_benchmark=MARKET_BENCHMARK,
                selected_benchmark=None,
                source="none",
                role="unknown",
                is_manual=False,
                fallback_reason=reason,
                selection_reason="no_valid_benchmark",
                validation=validation,
                relevance=relevance,
            )

        # No validation requested: use deterministic configured mapping.
        return BenchmarkResolution(
            ticker=symbol,
            sector_benchmark=sector_benchmark,
            market_benchmark=MARKET_BENCHMARK,
            selected_benchmark=sector_benchmark,
            source="sector",
            role=BENCHMARK_ROLE.get(sector_benchmark, "unknown"),
            is_manual=False,
            selection_reason="configured_sector_benchmark",
        )

    # ---------------------------------------------------------------
    # 4. Broad market fallback.
    # ---------------------------------------------------------------
    if allow_market_fallback:
        if validate:
            asset = _download_ohlcv(symbol, period=period)
            benchmark = _download_ohlcv(
                MARKET_BENCHMARK,
                period=period,
            )

            validation = _validation_from_data(
                asset,
                benchmark,
                MARKET_BENCHMARK,
                cfg,
            )
            relevance = _compute_relevance(
                asset,
                benchmark,
                MARKET_BENCHMARK,
                validation,
                cfg,
            )

            if validation.valid:
                return BenchmarkResolution(
                    ticker=symbol,
                    sector_benchmark=sector_benchmark,
                    market_benchmark=MARKET_BENCHMARK,
                    selected_benchmark=MARKET_BENCHMARK,
                    source="market",
                    role=BENCHMARK_ROLE.get(
                        MARKET_BENCHMARK,
                        "broad_market",
                    ),
                    is_manual=False,
                    selection_reason="broad_market_fallback",
                    selection_confidence=_selection_confidence(
                        relevance,
                        "market",
                    ),
                    validation=validation,
                    relevance=relevance,
                )

            return BenchmarkResolution(
                ticker=symbol,
                sector_benchmark=sector_benchmark,
                market_benchmark=MARKET_BENCHMARK,
                selected_benchmark=None,
                source="none",
                role="unknown",
                is_manual=False,
                fallback_reason="market_benchmark_validation_failed",
                selection_reason="no_valid_benchmark",
                validation=validation,
                relevance=relevance,
            )

        return BenchmarkResolution(
            ticker=symbol,
            sector_benchmark=sector_benchmark,
            market_benchmark=MARKET_BENCHMARK,
            selected_benchmark=MARKET_BENCHMARK,
            source="market",
            role=BENCHMARK_ROLE.get(
                MARKET_BENCHMARK,
                "broad_market",
            ),
            is_manual=False,
            selection_reason="broad_market_fallback",
            selection_confidence=0.0,
        )

    # ---------------------------------------------------------------
    # 5. No fallback allowed.
    # ---------------------------------------------------------------
    return BenchmarkResolution(
        ticker=symbol,
        sector_benchmark=sector_benchmark,
        market_benchmark=MARKET_BENCHMARK,
        selected_benchmark=None,
        source="none",
        role="unknown",
        is_manual=False,
        fallback_reason="market_fallback_disabled",
        selection_reason="no_valid_benchmark",
    )


def get_benchmark(
    ticker: str,
    manual_benchmark: Optional[str] = None,
    validate: bool = False,
) -> Optional[str]:
    return resolve_benchmarks(
        ticker,
        manual_benchmark=manual_benchmark,
        validate=validate,
    ).selected_benchmark


def get_sector_benchmark(ticker: str) -> Optional[str]:
    return BENCHMARK_MAP.get(_clean_ticker(ticker))


def validate_resolution(
    ticker: str,
    manual_benchmark: Optional[str] = None,
    cfg: Optional[ValidationConfig] = None,
    period: str = "5y",
) -> Dict:
    result = resolve_benchmarks(
        ticker,
        manual_benchmark=manual_benchmark,
        validate=True,
        cfg=cfg,
        period=period,
    )
    return asdict(result)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Resolver V1.3"
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Asset ticker, e.g. SOXL, DFEN, PLTR",
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
        "--show-validation",
        action="store_true",
        help="Alias for --validate",
    )

    args = parser.parse_args()

    validate = args.validate or args.show_validation

    result = resolve_benchmarks(
        args.ticker,
        manual_benchmark=args.benchmark,
        validate=validate,
        period=args.period,
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
