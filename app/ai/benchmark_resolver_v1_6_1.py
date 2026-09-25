"""
Benchmark Resolver V1.6.1
=======================

Listing-aware benchmark selection and validation for AI Stock Trader.

V1.6.1 changes
------------
- Separates technical benchmark validity from statistical relevance.
- Adds explicit relevance_class:
    HIGH / MODERATE / LOW / UNAVAILABLE
- Adds benchmark_status:
    VALID_HIGH_RELEVANCE
    VALID_MODERATE_RELEVANCE
    VALID_LOW_RELEVANCE
    INVALID
    UNAVAILABLE
- Keeps low-relevance SPY as a contextual fallback instead of treating it
  as a strong benchmark.
- Preserves adaptive listing-aware validation.
- Preserves manual benchmark behavior: explicit benchmarks are never
  silently replaced.
- Keeps the V1.4 output fields for compatibility.

CLI examples
-------------
python -m app.ai.benchmark_resolver_v1_5 --ticker SPCX --validate --listing-days 64
python -m app.ai.benchmark_resolver_v1_5 --ticker PLTR --validate
python -m app.ai.benchmark_resolver_v1_5 --ticker DFEN --validate
python -m app.ai.benchmark_resolver_v1_5 --ticker SPCX --benchmark SPY --validate
"""

from __future__ import annotations



# ---------------------------------------------------------------------------
# Benchmark Trust Layer - V1.6
# ---------------------------------------------------------------------------

BENCHMARK_TRUST_LEVELS = {
    "HIGH",
    "MEDIUM",
    "LOW",
    "CONTEXT_ONLY",
    "NONE",
}


def classify_benchmark_trust(
    benchmark_status=None,
    relevance_class=None,
    selection_confidence=None,
    source=None,
    role=None,
):
    """Classify how strongly downstream engines should rely on the benchmark."""
    status = str(benchmark_status or "").upper()
    relevance = str(relevance_class or "").upper()
    src = str(source or "").lower()
    benchmark_role = str(role or "").lower()

    if status in {"INVALID", "UNAVAILABLE", ""}:
        return "NONE"
    if relevance == "HIGH":
        return "HIGH"
    if relevance == "MODERATE":
        return "MEDIUM"
    if relevance == "LOW":
        # Weak market fallback remains useful as context, but not as
        # decision-grade peer evidence.
        if src == "market" or benchmark_role == "broad_market":
            return "CONTEXT_ONLY"
        return "LOW"
    return "CONTEXT_ONLY" if src == "market" else "NONE"


def benchmark_trust_allows_decision(trust):
    """Whether benchmark evidence is acceptable for downstream decisions."""
    return str(trust or "").upper() in {"HIGH", "MEDIUM", "LOW"}


def enrich_with_benchmark_trust(result):
    """Add V1.6 trust metadata while preserving the V1.5 output contract."""
    if not isinstance(result, dict):
        return result

    result["benchmark_trust"] = classify_benchmark_trust(
        benchmark_status=result.get("benchmark_status"),
        relevance_class=result.get("relevance_class"),
        selection_confidence=result.get("selection_confidence"),
        source=result.get("source"),
        role=result.get("role"),
    )
    result["benchmark_trust_decision_grade"] = benchmark_trust_allows_decision(
        result["benchmark_trust"]
    )
    return result


import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf


VERSION = "1.6.1"

BENCHMARK_MAP = {
    "TQQQ": "QQQ",
    "NVDA": "SMH",
    "SOXL": "SMH",
    "META": "QQQ",
    "GDXU": "GDX",
    "CCJ": "URNM",
    "AGQ": "SLV",
}

BENCHMARK_CANDIDATES = {
    "DFEN": ["ITA", "PPA", "XAR"],
    "PLTR": ["QQQ", "IGV", "XLK"],
}

MARKET_BENCHMARK = "SPY"


@dataclass
class ValidationConfig:
    mature_min_history_days: int = 252
    mature_min_overlap_days: int = 252

    young_asset_threshold_days: int = 252
    young_min_history_days: int = 30
    young_min_overlap_days: int = 30
    young_max_overlap_days: int = 60

    min_data_quality_ratio: float = 0.98
    min_positive_volume_ratio: float = 0.95
    min_median_dollar_volume: float = 1_000_000.0
    min_overlap_close_ratio: float = 0.98

    relevance_window_days: int = 252
    short_relevance_window_days: int = 60
    rolling_correlation_window_days: int = 60

    min_relevance_score: float = 55.0
    min_return_correlation: float = 0.30

    min_relevance_observations: int = 20

    # V1.5 relevance classes.
    high_relevance_score: float = 70.0
    moderate_relevance_score: float = 55.0
    low_relevance_floor: float = 0.0


@dataclass
class ValidationRequirements:
    required_history_days: int
    required_overlap_days: int
    validation_mode: str


@dataclass
class BenchmarkValidation:
    requested_benchmark: str
    available: bool
    sufficient_history: bool
    overlap_ok: bool
    ohlcv_ok: bool
    liquidity_ok: bool
    valid: bool
    data_quality: str
    benchmark_rows: int
    asset_rows: int
    overlap_rows: int
    median_dollar_volume: float
    quality_ratio: float
    positive_volume_ratio: float
    overlap_close_ratio: float
    required_history_days: int
    required_overlap_days: int
    validation_mode: str
    reason: str


@dataclass
class BenchmarkRelevance:
    benchmark: str
    relevance_score: float
    relevance_class: str
    daily_return_correlation: float
    short_return_correlation: float
    rolling_correlation: float
    beta: float
    observations: int
    role: str
    quality_valid: bool
    valid: bool
    reason: str


def _clean_ticker(value: str) -> str:
    return str(value).strip().upper()


def _download_close_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download OHLCV and normalize yfinance's possible MultiIndex output."""
    ticker = _clean_ticker(ticker)

    try:
        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        # Try to extract the requested ticker from either MultiIndex level.
        extracted = None
        for level in range(df.columns.nlevels):
            values = [str(v).upper() for v in df.columns.get_level_values(level)]
            if ticker in values:
                try:
                    extracted = df.xs(
                        ticker,
                        axis=1,
                        level=level,
                        drop_level=True,
                    )
                    break
                except Exception:
                    pass
        if extracted is not None:
            df = extracted

    if isinstance(df.columns, pd.MultiIndex):
        flattened = []
        for col in df.columns:
            parts = [str(x) for x in col]
            match = next(
                (
                    p.capitalize()
                    for p in parts
                    if p.lower() in {"open", "high", "low", "close", "volume"}
                ),
                None,
            )
            flattened.append(match or parts[0])
        df.columns = flattened

    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"open", "high", "low", "close", "volume"}:
            rename[col] = key.capitalize()
    df = df.rename(columns=rename)

    if "Close" not in df.columns:
        return pd.DataFrame()

    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"]).copy()
    df.index = pd.to_datetime(df.index)

    try:
        df.index = df.index.tz_localize(None)
    except TypeError:
        pass

    return df[~df.index.duplicated(keep="last")].sort_index()


def _requirements(
    asset_rows: int,
    listing_days: Optional[int],
    config: ValidationConfig,
) -> ValidationRequirements:
    age = listing_days if listing_days is not None else asset_rows

    try:
        age = max(0, int(age))
    except (TypeError, ValueError):
        age = asset_rows

    if age >= config.young_asset_threshold_days:
        return ValidationRequirements(
            required_history_days=config.mature_min_history_days,
            required_overlap_days=config.mature_min_overlap_days,
            validation_mode="MATURE",
        )

    overlap = int(math.ceil(age * 0.50))
    overlap = max(config.young_min_overlap_days, overlap)
    overlap = min(config.young_max_overlap_days, overlap)

    return ValidationRequirements(
        required_history_days=max(config.young_min_history_days, overlap),
        required_overlap_days=overlap,
        validation_mode="LISTING_AWARE",
    )


def _quality_label(ratio: float) -> str:
    if ratio >= 0.995:
        return "EXCELLENT"
    if ratio >= 0.98:
        return "GOOD"
    if ratio >= 0.95:
        return "MODERATE"
    return "POOR"


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    paired = pd.concat([a, b], axis=1).dropna()
    if len(paired) < 2:
        return float("nan")
    corr = paired.iloc[:, 0].corr(paired.iloc[:, 1])
    return float(corr) if pd.notna(corr) else float("nan")


def _safe_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    paired = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(paired) < 2:
        return float("nan")
    benchmark_var = float(paired.iloc[:, 1].var())
    if benchmark_var <= 0:
        return float("nan")
    covariance = float(paired.iloc[:, 0].cov(paired.iloc[:, 1]))
    return covariance / benchmark_var


def _relevance_score(
    daily_corr: float,
    short_corr: float,
    rolling_corr: float,
) -> float:
    if any(not np.isfinite(v) for v in (daily_corr, short_corr, rolling_corr)):
        return 0.0

    positive = [
        max(0.0, min(1.0, float(daily_corr))),
        max(0.0, min(1.0, float(short_corr))),
        max(0.0, min(1.0, float(rolling_corr))),
    ]

    score = (
        positive[0] * 40.0
        + positive[1] * 30.0
        + positive[2] * 30.0
    )
    return round(float(score), 2)


def _classify_relevance(
    score: float,
    observations: int,
    config: ValidationConfig,
) -> str:
    if observations < config.min_relevance_observations:
        return "UNAVAILABLE"
    if score >= config.high_relevance_score:
        return "HIGH"
    if score >= config.moderate_relevance_score:
        return "MODERATE"
    return "LOW"


def _benchmark_status(
    technical_valid: bool,
    relevance: Optional[BenchmarkRelevance],
) -> str:
    if not technical_valid:
        return "INVALID"
    if relevance is None or relevance.relevance_class == "UNAVAILABLE":
        return "VALID_LOW_RELEVANCE"
    return f"VALID_{relevance.relevance_class}_RELEVANCE"


def _role_for_benchmark(
    asset: str,
    benchmark: str,
    source: Optional[str] = None,
) -> str:
    benchmark = _clean_ticker(benchmark)

    roles = {
        "QQQ": "technology_growth",
        "SMH": "semiconductors",
        "IGV": "software_technology",
        "XLK": "technology",
        "ITA": "aerospace_defense",
        "PPA": "aerospace_defense",
        "XAR": "aerospace_defense",
        "GDX": "gold_miners",
        "URNM": "uranium",
        "SLV": "silver",
        "SPY": "broad_market",
    }

    if benchmark in roles:
        return roles[benchmark]
    return "market" if source == "market" else "sector"


def _prepare_overlap(
    asset_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:
    asset = asset_df[["Close"]].rename(columns={"Close": "asset_close"})
    bench = benchmark_df[["Close"]].rename(
        columns={"Close": "benchmark_close"}
    )
    return asset.join(bench, how="inner").dropna()


def validate_benchmark(
    ticker: str,
    benchmark: str,
    *,
    period: str = "5y",
    listing_days: Optional[int] = None,
    asset_df: Optional[pd.DataFrame] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
    config: Optional[ValidationConfig] = None,
) -> tuple[BenchmarkValidation, pd.DataFrame]:
    """Validate technical benchmark suitability."""
    config = config or ValidationConfig()
    ticker = _clean_ticker(ticker)
    benchmark = _clean_ticker(benchmark)

    if asset_df is None:
        asset_df = _download_close_ohlcv(ticker, period)

    if benchmark_df is None:
        benchmark_df = _download_close_ohlcv(benchmark, period)

    asset_rows = int(len(asset_df))
    benchmark_rows = int(len(benchmark_df))
    req = _requirements(asset_rows, listing_days, config)

    if benchmark_rows == 0:
        return (
            BenchmarkValidation(
                requested_benchmark=benchmark,
                available=False,
                sufficient_history=False,
                overlap_ok=False,
                ohlcv_ok=False,
                liquidity_ok=False,
                valid=False,
                data_quality="POOR",
                benchmark_rows=0,
                asset_rows=asset_rows,
                overlap_rows=0,
                median_dollar_volume=0.0,
                quality_ratio=0.0,
                positive_volume_ratio=0.0,
                overlap_close_ratio=0.0,
                required_history_days=req.required_history_days,
                required_overlap_days=req.required_overlap_days,
                validation_mode=req.validation_mode,
                reason="benchmark_unavailable",
            ),
            pd.DataFrame(),
        )

    sufficient_history = benchmark_rows >= req.required_history_days

    overlap = _prepare_overlap(asset_df, benchmark_df)
    overlap_rows = int(len(overlap))
    overlap_ok = overlap_rows >= req.required_overlap_days

    close = pd.to_numeric(benchmark_df["Close"], errors="coerce")
    volume = pd.to_numeric(benchmark_df["Volume"], errors="coerce")

    valid_ohlcv = (
        close.notna()
        & volume.notna()
        & np.isfinite(close)
        & np.isfinite(volume)
        & (volume >= 0)
    )

    quality_ratio = (
        float(valid_ohlcv.mean()) if len(valid_ohlcv) else 0.0
    )
    positive_volume_ratio = (
        float((volume[valid_ohlcv] > 0).mean())
        if valid_ohlcv.any()
        else 0.0
    )

    dollar_volume = (close * volume).where(valid_ohlcv)
    median_dollar_volume = (
        float(dollar_volume.median())
        if dollar_volume.notna().any()
        else 0.0
    )

    ohlcv_ok = quality_ratio >= config.min_data_quality_ratio
    data_quality = _quality_label(quality_ratio)

    overlap_close_ratio = 0.0
    if overlap_rows:
        overlap_close_ratio = float(
            (
                np.isfinite(overlap["asset_close"])
                & np.isfinite(overlap["benchmark_close"])
            ).mean()
        )

    liquidity_ok = (
        positive_volume_ratio >= config.min_positive_volume_ratio
        and median_dollar_volume >= config.min_median_dollar_volume
    )

    valid = (
        sufficient_history
        and overlap_ok
        and ohlcv_ok
        and liquidity_ok
        and overlap_close_ratio >= config.min_overlap_close_ratio
    )

    if not sufficient_history:
        reason = "insufficient_benchmark_history"
    elif not overlap_ok:
        reason = "insufficient_overlap"
    elif not ohlcv_ok:
        reason = "invalid_ohlcv_quality"
    elif not liquidity_ok:
        reason = "insufficient_liquidity"
    elif overlap_close_ratio < config.min_overlap_close_ratio:
        reason = "insufficient_close_overlap"
    else:
        reason = "validation_passed"

    return (
        BenchmarkValidation(
            requested_benchmark=benchmark,
            available=True,
            sufficient_history=sufficient_history,
            overlap_ok=overlap_ok,
            ohlcv_ok=ohlcv_ok,
            liquidity_ok=liquidity_ok,
            valid=valid,
            data_quality=data_quality,
            benchmark_rows=benchmark_rows,
            asset_rows=asset_rows,
            overlap_rows=overlap_rows,
            median_dollar_volume=median_dollar_volume,
            quality_ratio=quality_ratio,
            positive_volume_ratio=positive_volume_ratio,
            overlap_close_ratio=overlap_close_ratio,
            required_history_days=req.required_history_days,
            required_overlap_days=req.required_overlap_days,
            validation_mode=req.validation_mode,
            reason=reason,
        ),
        overlap,
    )


def calculate_relevance(
    ticker: str,
    benchmark: str,
    overlap: pd.DataFrame,
    *,
    role: str,
    quality_valid: bool,
    config: Optional[ValidationConfig] = None,
) -> BenchmarkRelevance:
    """Calculate statistical relevance and classify its strength."""
    config = config or ValidationConfig()

    if overlap.empty:
        return BenchmarkRelevance(
            benchmark=_clean_ticker(benchmark),
            relevance_score=0.0,
            relevance_class="UNAVAILABLE",
            daily_return_correlation=0.0,
            short_return_correlation=0.0,
            rolling_correlation=0.0,
            beta=0.0,
            observations=0,
            role=role,
            quality_valid=quality_valid,
            valid=False,
            reason="no_overlap_data",
        )

    asset_close = pd.to_numeric(overlap["asset_close"], errors="coerce")
    benchmark_close = pd.to_numeric(
        overlap["benchmark_close"], errors="coerce"
    )

    returns = pd.concat(
        [
            asset_close.pct_change().rename("asset"),
            benchmark_close.pct_change().rename("benchmark"),
        ],
        axis=1,
    ).dropna()

    observations = int(len(returns))

    if observations < config.min_relevance_observations:
        return BenchmarkRelevance(
            benchmark=_clean_ticker(benchmark),
            relevance_score=0.0,
            relevance_class="UNAVAILABLE",
            daily_return_correlation=0.0,
            short_return_correlation=0.0,
            rolling_correlation=0.0,
            beta=0.0,
            observations=observations,
            role=role,
            quality_valid=quality_valid,
            valid=False,
            reason="insufficient_relevance_observations",
        )

    daily_corr = _safe_corr(returns["asset"], returns["benchmark"])

    short = returns.tail(
        min(config.short_relevance_window_days, observations)
    )
    short_corr = _safe_corr(short["asset"], short["benchmark"])

    rolling_window = min(
        config.rolling_correlation_window_days,
        observations,
    )

    rolling_series = (
        returns["asset"]
        .rolling(rolling_window)
        .corr(returns["benchmark"])
        .dropna()
    )

    rolling_corr = (
        float(rolling_series.iloc[-1])
        if not rolling_series.empty
        else float("nan")
    )

    beta = _safe_beta(returns["asset"], returns["benchmark"])

    score = _relevance_score(
        daily_corr,
        short_corr,
        rolling_corr,
    )

    relevance_class = _classify_relevance(
        score,
        observations,
        config,
    )

    correlations_valid = all(
        np.isfinite(v)
        for v in (daily_corr, short_corr, rolling_corr)
    )

    valid = (
        quality_valid
        and correlations_valid
        and score >= config.min_relevance_score
        and daily_corr >= config.min_return_correlation
        and short_corr >= config.min_return_correlation
    )

    return BenchmarkRelevance(
        benchmark=_clean_ticker(benchmark),
        relevance_score=score,
        relevance_class=relevance_class,
        daily_return_correlation=round(float(daily_corr), 6)
        if np.isfinite(daily_corr)
        else 0.0,
        short_return_correlation=round(float(short_corr), 6)
        if np.isfinite(short_corr)
        else 0.0,
        rolling_correlation=round(float(rolling_corr), 6)
        if np.isfinite(rolling_corr)
        else 0.0,
        beta=round(float(beta), 6)
        if np.isfinite(beta)
        else 0.0,
        observations=observations,
        role=role,
        quality_valid=quality_valid,
        valid=valid,
        reason="relevance_passed" if valid else "low_statistical_relevance",
    )


def _result_dict(
    ticker: str,
    sector_benchmark: Optional[str],
    market_benchmark: str,
    selected_benchmark: str,
    source: str,
    role: str,
    is_manual: bool,
    fallback_reason: str,
    selection_reason: str,
    selection_confidence: float,
    listing_days: Optional[int],
    requirements: ValidationRequirements,
    validation: BenchmarkValidation,
    relevance: Optional[BenchmarkRelevance],
    candidate_evaluations: Optional[list[dict[str, Any]]],
) -> dict[str, Any]:
    status = _benchmark_status(
        validation.valid,
        relevance,
    )

    relevance_class = (
        relevance.relevance_class
        if relevance is not None
        else "UNAVAILABLE"
    )

    return {
        "ticker": ticker,
        "sector_benchmark": sector_benchmark,
        "market_benchmark": market_benchmark,
        "selected_benchmark": selected_benchmark,
        "source": source,
        "role": role,
        "is_manual": is_manual,
        "fallback_reason": fallback_reason,
        "selection_reason": selection_reason,
        "selection_confidence": round(float(selection_confidence), 2),
        "benchmark_status": status,
        "relevance_class": relevance_class,
        "listing_days": listing_days,
        "validation_mode": requirements.validation_mode,
        "required_history_days": requirements.required_history_days,
        "required_overlap_days": requirements.required_overlap_days,
        "validation": asdict(validation),
        "relevance": asdict(relevance) if relevance else None,
        "candidate_evaluations": candidate_evaluations,
        "resolver_version": VERSION,
        # V1.6.1: benchmark trust is part of the public resolver contract.
        "benchmark_trust": classify_benchmark_trust(
            benchmark_status=status,
            relevance_class=relevance_class,
            selection_confidence=selection_confidence,
            source=source,
            role=role,
        ),
        "benchmark_trust_decision_grade": benchmark_trust_allows_decision(
            classify_benchmark_trust(
                benchmark_status=status,
                relevance_class=relevance_class,
                selection_confidence=selection_confidence,
                source=source,
                role=role,
            )
        ),
    }


def _attach_selected_benchmark_data(
    result: dict[str, Any],
    *,
    benchmark: str,
    period: str,
    return_selected_data: bool,
) -> dict[str, Any]:
    """Attach selected benchmark data for internal orchestrator transport only."""
    if not return_selected_data or not benchmark:
        return result
    try:
        selected_df = _download_close_ohlcv(benchmark, period)
    except Exception:
        selected_df = pd.DataFrame()
    if selected_df is not None and not selected_df.empty:
        result["_selected_benchmark_data"] = selected_df.copy()
    return result


def resolve_benchmarks(
    ticker: str,
    *,
    benchmark: Optional[str] = None,
    period: str = "5y",
    listing_days: Optional[int] = None,
    config: Optional[ValidationConfig] = None,
    asset_df: Optional[pd.DataFrame] = None,
    return_selected_data: bool = False,
) -> dict[str, Any]:
    """
    Resolve the best benchmark.

    Priority:
      1. Explicit/manual benchmark.
      2. Mapped/candidate sector benchmarks.
      3. SPY broad-market fallback.

    Sector selection requires both technical validity and statistical
    relevance. A low-relevance SPY fallback may still be selected for
    context, but its LOW relevance is explicitly exposed.
    """
    config = config or ValidationConfig()
    ticker = _clean_ticker(ticker)

    if asset_df is None:
        asset_df = _download_close_ohlcv(ticker, period)
    else:
        asset_df = asset_df.copy()

    if asset_df.empty:
        req = _requirements(0, listing_days, config)
        validation = BenchmarkValidation(
            requested_benchmark=benchmark or "",
            available=False,
            sufficient_history=False,
            overlap_ok=False,
            ohlcv_ok=False,
            liquidity_ok=False,
            valid=False,
            data_quality="POOR",
            benchmark_rows=0,
            asset_rows=0,
            overlap_rows=0,
            median_dollar_volume=0.0,
            quality_ratio=0.0,
            positive_volume_ratio=0.0,
            overlap_close_ratio=0.0,
            required_history_days=req.required_history_days,
            required_overlap_days=req.required_overlap_days,
            validation_mode=req.validation_mode,
            reason="asset_unavailable",
        )

        result = _result_dict(
            ticker=ticker,
            sector_benchmark=None,
            market_benchmark=MARKET_BENCHMARK,
            selected_benchmark="",
            source="none",
            role="none",
            is_manual=benchmark is not None,
            fallback_reason="asset_unavailable",
            selection_reason="no_data",
            selection_confidence=0.0,
            listing_days=listing_days,
            requirements=req,
            validation=validation,
            relevance=None,
            candidate_evaluations=None,
        )
        return _attach_selected_benchmark_data(
            result,
            benchmark=market if market_validation.available else "",
            period=period,
            return_selected_data=return_selected_data,
        )

    req = _requirements(len(asset_df), listing_days, config)

    # ---------------------------------------------------------------
    # Explicit/manual benchmark
    # ---------------------------------------------------------------
    if benchmark:
        manual = _clean_ticker(benchmark)
        validation, overlap = validate_benchmark(
            ticker,
            manual,
            period=period,
            listing_days=listing_days,
            asset_df=asset_df,
            config=config,
        )

        role = _role_for_benchmark(ticker, manual, source="manual")

        relevance = calculate_relevance(
            ticker,
            manual,
            overlap,
            role=role,
            quality_valid=validation.valid,
            config=config,
        )

        confidence = relevance.relevance_score

        if validation.valid:
            selection_reason = (
                "manual_benchmark_valid_and_relevant"
                if relevance.valid
                else "manual_benchmark_valid_low_relevance"
            )
            fallback_reason = ""
        else:
            selection_reason = "manual_benchmark_failed_validation"
            fallback_reason = validation.reason

        result = _result_dict(
            ticker=ticker,
            sector_benchmark=manual,
            market_benchmark=MARKET_BENCHMARK,
            selected_benchmark=manual,
            source="manual",
            role=role,
            is_manual=True,
            fallback_reason=fallback_reason,
            selection_reason=selection_reason,
            selection_confidence=confidence,
            listing_days=listing_days,
            requirements=req,
            validation=validation,
            relevance=relevance,
            candidate_evaluations=None,
        )
        return _attach_selected_benchmark_data(
            result,
            benchmark=manual,
            period=period,
            return_selected_data=return_selected_data,
        )

    # ---------------------------------------------------------------
    # Sector candidates
    # ---------------------------------------------------------------
    mapped = BENCHMARK_MAP.get(ticker)
    candidates = list(BENCHMARK_CANDIDATES.get(ticker, []))

    if mapped and mapped not in candidates:
        candidates.insert(0, mapped)

    candidate_evaluations: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate = _clean_ticker(candidate)

        validation, overlap = validate_benchmark(
            ticker,
            candidate,
            period=period,
            listing_days=listing_days,
            asset_df=asset_df,
            config=config,
        )

        role = _role_for_benchmark(
            ticker,
            candidate,
            source="sector_candidate",
        )

        relevance = calculate_relevance(
            ticker,
            candidate,
            overlap,
            role=role,
            quality_valid=validation.valid,
            config=config,
        )

        eligible = bool(validation.valid and relevance.valid)

        candidate_evaluations.append(
            {
                "benchmark": candidate,
                "validation": asdict(validation),
                "relevance": asdict(relevance),
                "eligible": eligible,
            }
        )

    eligible = [
        item for item in candidate_evaluations
        if item["eligible"]
    ]

    if eligible:
        eligible.sort(
            key=lambda item: (
                item["relevance"]["relevance_score"],
                item["validation"]["median_dollar_volume"],
            ),
            reverse=True,
        )

        best = eligible[0]
        selected = best["benchmark"]
        validation = BenchmarkValidation(**best["validation"])
        relevance = BenchmarkRelevance(**best["relevance"])

        result = _result_dict(
            ticker=ticker,
            sector_benchmark=selected,
            market_benchmark=MARKET_BENCHMARK,
            selected_benchmark=selected,
            source="sector_candidate",
            role=relevance.role,
            is_manual=False,
            fallback_reason="",
            selection_reason="best_valid_candidate_by_statistical_relevance",
            selection_confidence=relevance.relevance_score,
            listing_days=listing_days,
            requirements=req,
            validation=validation,
            relevance=relevance,
            candidate_evaluations=candidate_evaluations,
        )
        return _attach_selected_benchmark_data(
            result,
            benchmark=selected,
            period=period,
            return_selected_data=return_selected_data,
        )

    # ---------------------------------------------------------------
    # Broad-market fallback
    # ---------------------------------------------------------------
    market = MARKET_BENCHMARK

    market_validation, market_overlap = validate_benchmark(
        ticker,
        market,
        period=period,
        listing_days=listing_days,
        asset_df=asset_df,
        config=config,
    )

    market_role = _role_for_benchmark(
        ticker,
        market,
        source="market",
    )

    market_relevance = calculate_relevance(
        ticker,
        market,
        market_overlap,
        role=market_role,
        quality_valid=market_validation.valid,
        config=config,
    )

    if market_validation.valid:
        selection_reason = "broad_market_fallback"
        fallback_reason = ""
        confidence = market_relevance.relevance_score
    else:
        selection_reason = "no_valid_sector_or_market_benchmark"
        fallback_reason = market_validation.reason
        confidence = 0.0

    result = _result_dict(
        ticker=ticker,
        sector_benchmark=None,
        market_benchmark=market,
        selected_benchmark=market
        if market_validation.available
        else "",
        source="market" if market_validation.available else "none",
        role=market_role if market_validation.available else "none",
        is_manual=False,
        fallback_reason=fallback_reason,
        selection_reason=selection_reason,
        selection_confidence=confidence,
        listing_days=listing_days,
        requirements=req,
        validation=market_validation,
        relevance=market_relevance,
        candidate_evaluations=candidate_evaluations or None,
    )
    return _attach_selected_benchmark_data(
        result,
        benchmark=market if market_validation.available else "",
        period=period,
        return_selected_data=return_selected_data,
    )


def get_benchmark(
    ticker: str,
    benchmark: Optional[str] = None,
    *,
    period: str = "5y",
    listing_days: Optional[int] = None,
    config: Optional[ValidationConfig] = None,
) -> str:
    """Return only the selected benchmark symbol."""
    result = resolve_benchmarks(
        ticker,
        benchmark=benchmark,
        period=period,
        listing_days=listing_days,
        config=config,
    )
    return str(result.get("selected_benchmark", ""))


def validate_resolution(
    ticker: str,
    benchmark: Optional[str] = None,
    *,
    period: str = "5y",
    listing_days: Optional[int] = None,
    config: Optional[ValidationConfig] = None,
) -> dict[str, Any]:
    """Return the complete resolver result."""
    return resolve_benchmarks(
        ticker,
        benchmark=benchmark,
        period=period,
        listing_days=listing_days,
        config=config,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Resolver V1.6.1"
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Asset ticker, e.g. SPCX or PLTR",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Explicit/manual benchmark. Never silently replaced.",
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
        help="Known asset trading-session age for listing-aware validation.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run full validation and relevance analysis.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result = resolve_benchmarks(
        args.ticker,
        benchmark=args.benchmark,
        period=args.period,
        listing_days=args.listing_days,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
