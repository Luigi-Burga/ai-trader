"""Benchmark Resolver V1.2 for AI Trader.

V1.2 extends V1.1 with benchmark relevance scoring and candidate selection.
Selection order:
    manual -> configured sector benchmark -> sector candidate search -> SPY -> none

The resolver distinguishes DATA QUALITY from BENCHMARK RELEVANCE. A benchmark can
have excellent market data while still being a poor contextual benchmark for an
asset. For assets without a configured sector benchmark, V1.2 can evaluate a
small candidate universe and select the best valid contextual benchmark.

Current specialized candidate universe includes DFEN (aerospace/defense):
    XAR, ITA, PPA

All relevance calculations use historical daily returns available at runtime.
No future data relative to the current observation is used by the resolver.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

VERSION = "1.2"
MARKET_BENCHMARK = "SPY"

# True contextual/sector relationships only. SPY is the broad-market fallback.
BENCHMARK_MAP: Dict[str, str] = {
    "TQQQ": "QQQ",
    "NVDA": "SMH",
    "SOXL": "SMH",
    "META": "QQQ",
    "PLTR": "QQQ",
    "GDXU": "GDX",
    "CCJ": "URNM",
    "AGQ": "SLV",
}

# Candidate universe for assets where a direct sector benchmark is not yet
# configured. Candidates are ranked dynamically by data quality + relevance.
BENCHMARK_CANDIDATES: Dict[str, List[str]] = {
    "DFEN": ["XAR", "ITA", "PPA"],
}

BENCHMARK_ROLE: Dict[str, str] = {
    "QQQ": "technology_growth",
    "SMH": "semiconductors",
    "GDX": "gold_miners",
    "URNM": "uranium_nuclear",
    "SLV": "silver",
    "XAR": "aerospace_defense",
    "ITA": "aerospace_defense",
    "PPA": "aerospace_defense",
    "SPY": "broad_market",
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
    min_relevance_score: float = 55.0
    min_return_correlation: float = 0.30


DEFAULT_VALIDATION_CONFIG = ValidationConfig()


@dataclass(frozen=True)
class BenchmarkValidation:
    requested_benchmark: Optional[str]
    available: bool
    sufficient_history: bool
    overlap_ok: bool
    ohlcv_ok: bool
    liquidity_ok: bool
    valid: bool
    data_quality: str
    benchmark_rows: int
    overlap_rows: int
    median_dollar_volume: Optional[float]
    quality_ratio: Optional[float]
    positive_volume_ratio: Optional[float]
    overlap_close_ratio: Optional[float]
    reason: str


@dataclass(frozen=True)
class BenchmarkRelevance:
    benchmark: str
    relevance_score: float
    daily_return_correlation: Optional[float]
    short_return_correlation: Optional[float]
    rolling_correlation: Optional[float]
    beta: Optional[float]
    observations: int
    role: str
    quality_valid: bool
    valid: bool
    reason: str


@dataclass(frozen=True)
class BenchmarkResolution:
    ticker: str
    sector_benchmark: Optional[str]
    market_benchmark: str
    selected_benchmark: Optional[str]
    source: str
    role: str
    is_manual: bool
    fallback_reason: Optional[str]
    validation: Optional[dict]
    relevance: Optional[dict]
    candidate_evaluations: Optional[List[dict]]
    resolver_version: str = VERSION


def _clean_ticker(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().upper()
    return value or None


def _import_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required for validation. Install it with: pip install yfinance"
        ) from exc
    return yf


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    result = df.copy()
    if isinstance(result.columns, pd.MultiIndex):
        known = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        cols = []
        for col in result.columns:
            parts = [str(x) for x in col if str(x) not in {"None", ""}]
            field = next((p for p in parts if p in known), None)
            cols.append(field if field else ("_".join(parts) if parts else ""))
        result.columns = cols
    return result.loc[:, ~result.columns.duplicated(keep="first")]


def _download_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    yf = _import_yfinance()
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to download data for {ticker}: {exc}") from exc
    return _flatten_yfinance_columns(df)


def _normalize_close(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(df["Close"], errors="coerce").dropna().copy()
    close.index = pd.to_datetime(close.index).normalize()
    close = close[~close.index.duplicated(keep="last")]
    return close.sort_index()


def _quality_label(
    available,
    sufficient_history,
    overlap_ok,
    ohlcv_ok,
    liquidity_ok,
    quality_ratio,
):
    if not available:
        return "UNAVAILABLE"
    if not sufficient_history:
        return "INSUFFICIENT_HISTORY"
    if not overlap_ok:
        return "INSUFFICIENT_OVERLAP"
    if not ohlcv_ok:
        return "POOR"
    if not liquidity_ok:
        return "LOW_LIQUIDITY"
    if quality_ratio is not None and quality_ratio >= 0.995:
        return "EXCELLENT"
    if quality_ratio is not None and quality_ratio >= 0.985:
        return "GOOD"
    return "MODERATE"


def _validate_benchmark_data(
    ticker: str,
    benchmark: str,
    *,
    validation_config: ValidationConfig,
    period: str,
    interval: str,
) -> BenchmarkValidation:
    asset_df = _download_history(ticker, period, interval)
    benchmark_df = _download_history(benchmark, period, interval)

    if benchmark_df.empty:
        return BenchmarkValidation(
            benchmark, False, False, False, False, False, False,
            "UNAVAILABLE", 0, 0, None, None, None, None,
            f"No market data returned for benchmark {benchmark}.",
        )

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in benchmark_df.columns]
    if missing:
        return BenchmarkValidation(
            benchmark, True, False, False, False, False, False,
            "POOR", len(benchmark_df), 0, None, None, None, None,
            "Missing OHLCV columns: " + ", ".join(missing),
        )

    work = benchmark_df[required].copy()
    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    rows = len(work)
    valid_ohlcv = work[required].notna().all(axis=1)
    quality_ratio = float(valid_ohlcv.mean()) if rows else 0.0
    positive_volume = work["Volume"].notna() & (work["Volume"] > 0)
    positive_volume_ratio = float(positive_volume.mean()) if rows else 0.0
    dollar_volume = work["Close"].abs() * work["Volume"]
    median_dollar_volume = (
        float(dollar_volume[positive_volume].median())
        if positive_volume.any()
        else None
    )

    ohlcv_ok = (
        quality_ratio >= validation_config.min_data_quality_ratio
        and positive_volume_ratio >= validation_config.min_positive_volume_ratio
    )
    liquidity_ok = (
        median_dollar_volume is not None
        and math.isfinite(median_dollar_volume)
        and median_dollar_volume >= validation_config.min_median_dollar_volume
    )

    bclose = _normalize_close(work)
    aclose = _normalize_close(asset_df)
    overlap_rows = len(bclose.index.intersection(aclose.index))
    overlap_close_ratio = overlap_rows / max(1, min(len(aclose), len(bclose)))
    sufficient_history = rows >= validation_config.min_history_days
    overlap_ok = (
        overlap_rows >= validation_config.min_overlap_days
        and overlap_close_ratio >= validation_config.min_close_overlap_ratio
    )
    available = rows > 0
    valid = available and sufficient_history and overlap_ok and ohlcv_ok and liquidity_ok
    quality = _quality_label(
        available,
        sufficient_history,
        overlap_ok,
        ohlcv_ok,
        liquidity_ok,
        quality_ratio,
    )

    if valid:
        reason = "Benchmark passed all data-quality checks."
    else:
        failures = []
        if not sufficient_history:
            failures.append("insufficient history")
        if not overlap_ok:
            failures.append("insufficient asset/benchmark overlap")
        if not ohlcv_ok:
            failures.append("OHLCV quality below threshold")
        if not liquidity_ok:
            failures.append("liquidity below threshold")
        reason = "; ".join(failures) or "Benchmark failed validation."

    return BenchmarkValidation(
        benchmark,
        available,
        sufficient_history,
        overlap_ok,
        ohlcv_ok,
        liquidity_ok,
        valid,
        quality,
        rows,
        overlap_rows,
        median_dollar_volume,
        quality_ratio,
        positive_volume_ratio,
        overlap_close_ratio,
        reason,
    )


def _correlation_score(corr: Optional[float]) -> float:
    if corr is None or not math.isfinite(corr):
        return 0.0
    # Map correlation 0..1 to 0..100. Negative correlation is not useful for
    # contextual benchmark selection, while very high correlation is preferred.
    return float(np.clip(corr, 0.0, 1.0) * 100.0)


def _compute_relevance(
    ticker: str,
    benchmark: str,
    *,
    validation: BenchmarkValidation,
    validation_config: ValidationConfig,
    period: str,
    interval: str,
) -> BenchmarkRelevance:
    if not validation.valid:
        return BenchmarkRelevance(
            benchmark=benchmark,
            relevance_score=0.0,
            daily_return_correlation=None,
            short_return_correlation=None,
            rolling_correlation=None,
            beta=None,
            observations=0,
            role=BENCHMARK_ROLE.get(benchmark, "custom"),
            quality_valid=False,
            valid=False,
            reason="Benchmark data-quality validation failed before relevance analysis.",
        )

    asset_df = _download_history(ticker, period, interval)
    benchmark_df = _download_history(benchmark, period, interval)
    asset_close = _normalize_close(asset_df)
    bench_close = _normalize_close(benchmark_df)

    merged = pd.concat(
        [asset_close.rename("asset"), bench_close.rename("benchmark")], axis=1, join="inner"
    ).dropna()
    if len(merged) < validation_config.min_overlap_days:
        return BenchmarkRelevance(
            benchmark, 0.0, None, None, None, None, len(merged),
            BENCHMARK_ROLE.get(benchmark, "custom"), True, False,
            "Not enough overlapping observations for relevance analysis.",
        )

    returns = merged.pct_change().dropna()
    long_window = min(validation_config.relevance_window_days, len(returns))
    short_window = min(validation_config.short_relevance_window_days, len(returns))
    long = returns.tail(long_window)
    short = returns.tail(short_window)

    corr_long = float(long["asset"].corr(long["benchmark"])) if len(long) >= 30 else None
    corr_short = float(short["asset"].corr(short["benchmark"])) if len(short) >= 30 else None

    # Rolling 60-day correlation stability: use the median of available rolling
    # correlations rather than the last value alone.
    rolling = returns["asset"].rolling(60).corr(returns["benchmark"]).dropna()
    rolling_corr = float(rolling.tail(6).median()) if len(rolling) >= 60 else corr_long

    variance = float(long["benchmark"].var()) if len(long) else 0.0
    covariance = float(long["asset"].cov(long["benchmark"])) if len(long) else 0.0
    beta = covariance / variance if variance > 1e-12 else None

    # Relevance score emphasizes long-term co-movement, then recent co-movement,
    # then correlation stability. Data quality contributes a small confidence term.
    score = (
        0.55 * _correlation_score(corr_long)
        + 0.25 * _correlation_score(corr_short)
        + 0.20 * _correlation_score(rolling_corr)
    )
    if validation.data_quality == "EXCELLENT":
        score = min(100.0, score + 2.0)
    elif validation.data_quality == "GOOD":
        score = min(100.0, score + 1.0)

    valid = (
        score >= validation_config.min_relevance_score
        and corr_long is not None
        and corr_long >= validation_config.min_return_correlation
    )

    if valid:
        reason = "Benchmark passed data-quality and relevance checks."
    else:
        reason = (
            f"Relevance below threshold: score={score:.2f}, "
            f"long_corr={corr_long if corr_long is not None else 'n/a'}."
        )

    return BenchmarkRelevance(
        benchmark=benchmark,
        relevance_score=round(float(score), 2),
        daily_return_correlation=round(corr_long, 4) if corr_long is not None else None,
        short_return_correlation=round(corr_short, 4) if corr_short is not None else None,
        rolling_correlation=round(rolling_corr, 4) if rolling_corr is not None else None,
        beta=round(float(beta), 4) if beta is not None and math.isfinite(beta) else None,
        observations=len(returns),
        role=BENCHMARK_ROLE.get(benchmark, "custom"),
        quality_valid=True,
        valid=valid,
        reason=reason,
    )


def validate_benchmark(
    ticker: str,
    benchmark: str,
    *,
    validation_config=DEFAULT_VALIDATION_CONFIG,
    period="5y",
    interval="1d",
    include_relevance: bool = False,
) -> dict:
    ticker, benchmark = _clean_ticker(ticker), _clean_ticker(benchmark)
    if not ticker:
        raise ValueError("ticker is required")
    if not benchmark:
        raise ValueError("benchmark is required")

    validation = _validate_benchmark_data(
        ticker,
        benchmark,
        validation_config=validation_config,
        period=period,
        interval=interval,
    )
    result = asdict(validation)
    if include_relevance:
        result["relevance"] = asdict(
            _compute_relevance(
                ticker,
                benchmark,
                validation=validation,
                validation_config=validation_config,
                period=period,
                interval=interval,
            )
        )
    return result


def _evaluate_candidates(
    ticker: str,
    candidates: List[str],
    *,
    validation_config: ValidationConfig,
    period: str,
    interval: str,
) -> List[dict]:
    evaluations: List[dict] = []
    for candidate in candidates:
        candidate = _clean_ticker(candidate)
        if not candidate:
            continue
        validation = _validate_benchmark_data(
            ticker,
            candidate,
            validation_config=validation_config,
            period=period,
            interval=interval,
        )
        relevance = _compute_relevance(
            ticker,
            candidate,
            validation=validation,
            validation_config=validation_config,
            period=period,
            interval=interval,
        )
        evaluations.append(
            {
                "benchmark": candidate,
                "validation": asdict(validation),
                "relevance": asdict(relevance),
                "eligible": bool(validation.valid and relevance.valid),
            }
        )
    evaluations.sort(
        key=lambda x: (
            x["eligible"],
            x["relevance"]["relevance_score"],
            x["validation"]["median_dollar_volume"] or 0.0,
        ),
        reverse=True,
    )
    return evaluations


def resolve_benchmarks(
    ticker: str,
    manual_benchmark: Optional[str] = None,
    *,
    allow_market_fallback=True,
    validate_data=False,
    validation_config=DEFAULT_VALIDATION_CONFIG,
    period="5y",
    interval="1d",
) -> dict:
    ticker = _clean_ticker(ticker)
    if not ticker:
        raise ValueError("ticker is required")

    manual = _clean_ticker(manual_benchmark)
    sector = BENCHMARK_MAP.get(ticker)
    candidates = BENCHMARK_CANDIDATES.get(ticker, [])
    selected = None
    source = "none"
    is_manual = False
    fallback_reason = None
    validation = None
    relevance = None
    candidate_evaluations = None

    if manual:
        selected, source, is_manual = manual, "manual", True
        if validate_data:
            validation = validate_benchmark(
                ticker,
                selected,
                validation_config=validation_config,
                period=period,
                interval=interval,
                include_relevance=True,
            )
            relevance = validation.pop("relevance", None)
            if not validation["valid"]:
                # Manual choice is preserved; never silently replace it.
                fallback_reason = "manual_benchmark_failed_validation"
            elif relevance is not None and not relevance["valid"]:
                fallback_reason = "manual_benchmark_low_relevance"

    elif sector:
        selected, source = sector, "sector"
        if validate_data:
            validation = validate_benchmark(
                ticker,
                selected,
                validation_config=validation_config,
                period=period,
                interval=interval,
                include_relevance=True,
            )
            relevance = validation.pop("relevance", None)
            if not validation["valid"] and allow_market_fallback:
                fallback_reason = f"sector_benchmark_failed_validation:{validation['reason']}"
                selected, source = MARKET_BENCHMARK, "market"
                validation = validate_benchmark(
                    ticker,
                    selected,
                    validation_config=validation_config,
                    period=period,
                    interval=interval,
                    include_relevance=True,
                )
                relevance = validation.pop("relevance", None)
                if not validation["valid"]:
                    fallback_reason += ";market_benchmark_failed_validation"
                    selected, source = None, "none"
            elif validation["valid"] and relevance is not None and not relevance["valid"]:
                # Do not immediately discard a configured sector relationship.
                # A low statistical correlation is reported, but the explicit
                # sector mapping remains authoritative unless market fallback
                # is needed for data-quality failure.
                fallback_reason = "sector_benchmark_low_statistical_relevance"

    elif candidates and validate_data:
        candidate_evaluations = _evaluate_candidates(
            ticker,
            candidates,
            validation_config=validation_config,
            period=period,
            interval=interval,
        )
        eligible = [x for x in candidate_evaluations if x["eligible"]]
        if eligible:
            best = eligible[0]
            selected = best["benchmark"]
            source = "sector_candidate"
            validation = best["validation"]
            relevance = best["relevance"]
        elif allow_market_fallback:
            fallback_reason = "no_sector_candidate_passed_validation_or_relevance"
            selected, source = MARKET_BENCHMARK, "market"
            validation = validate_benchmark(
                ticker,
                selected,
                validation_config=validation_config,
                period=period,
                interval=interval,
                include_relevance=True,
            )
            relevance = validation.pop("relevance", None)
            if not validation["valid"]:
                fallback_reason += ";market_benchmark_failed_validation"
                selected, source = None, "none"
        else:
            fallback_reason = "No sector candidate passed validation/relevance and market fallback disabled."

    elif candidates and not validate_data:
        # Without market data we cannot rank candidates responsibly. Expose the
        # candidate universe but use SPY as the deterministic non-validation path.
        selected, source = MARKET_BENCHMARK, "market"
        fallback_reason = "sector_candidates_require_validation_for_dynamic_selection"

    elif allow_market_fallback:
        selected, source = MARKET_BENCHMARK, "market"
        if validate_data:
            validation = validate_benchmark(
                ticker,
                selected,
                validation_config=validation_config,
                period=period,
                interval=interval,
                include_relevance=True,
            )
            relevance = validation.pop("relevance", None)
            if not validation["valid"]:
                fallback_reason = f"market_benchmark_failed_validation:{validation['reason']}"
                selected, source = None, "none"
    else:
        fallback_reason = "No configured sector benchmark and market fallback disabled."

    role = BENCHMARK_ROLE.get(selected, "custom" if selected else "none")
    return asdict(
        BenchmarkResolution(
            ticker,
            sector,
            MARKET_BENCHMARK,
            selected,
            source,
            role,
            is_manual,
            fallback_reason,
            validation,
            relevance,
            candidate_evaluations,
        )
    )


def get_benchmark(
    ticker: str,
    manual_benchmark: Optional[str] = None,
    *,
    validate_data=False,
):
    return resolve_benchmarks(
        ticker,
        manual_benchmark=manual_benchmark,
        validate_data=validate_data,
    )["selected_benchmark"]


def get_sector_benchmark(ticker: str):
    ticker = _clean_ticker(ticker)
    if not ticker:
        raise ValueError("ticker is required")
    return BENCHMARK_MAP.get(ticker)


def get_benchmark_candidates(ticker: str) -> List[str]:
    ticker = _clean_ticker(ticker)
    if not ticker:
        raise ValueError("ticker is required")
    return list(BENCHMARK_CANDIDATES.get(ticker, []))


def validate_resolution(result: dict) -> dict:
    required = {
        "ticker",
        "sector_benchmark",
        "market_benchmark",
        "selected_benchmark",
        "source",
        "role",
        "is_manual",
        "fallback_reason",
        "validation",
        "relevance",
        "candidate_evaluations",
        "resolver_version",
    }
    errors = []
    missing = sorted(required - set(result))
    if missing:
        errors.append("Missing fields: " + ", ".join(missing))

    source = result.get("source")
    selected = result.get("selected_benchmark")
    manual = result.get("is_manual")

    if source == "manual" and (not manual or not selected):
        errors.append("Manual source requires manual selection")
    if source in {"sector", "sector_candidate"} and not selected:
        errors.append("Sector source requires selected_benchmark")
    if source == "market" and selected != MARKET_BENCHMARK:
        errors.append("Market source must select SPY")
    if source == "none" and selected is not None:
        errors.append("None source must not select a benchmark")
    if selected == MARKET_BENCHMARK and source in {"sector", "sector_candidate"}:
        errors.append("SPY must be classified as market, not sector")

    validation = result.get("validation")
    if validation is not None and validation.get("requested_benchmark") != selected:
        errors.append("Validation benchmark does not match selected benchmark")

    relevance = result.get("relevance")
    if relevance is not None and relevance.get("benchmark") != selected:
        errors.append("Relevance benchmark does not match selected benchmark")

    return {"valid": not errors, "errors": errors}


def _build_parser():
    p = argparse.ArgumentParser(
        description="Resolve, validate and rank an AI Trader benchmark."
    )
    p.add_argument("--ticker", required=True)
    p.add_argument("--benchmark", "--manual-benchmark", dest="benchmark", default=None)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--period", default="5y")
    p.add_argument("--interval", default="1d")
    p.add_argument("--no-market-fallback", action="store_true")
    p.add_argument("--show-validation", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    result = resolve_benchmarks(
        args.ticker,
        manual_benchmark=args.benchmark,
        allow_market_fallback=not args.no_market_fallback,
        validate_data=args.validate,
        period=args.period,
        interval=args.interval,
    )
    if args.show_validation:
        result["resolver_validation"] = validate_resolution(result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
