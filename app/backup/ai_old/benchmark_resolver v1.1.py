"""Benchmark Resolver V1.1 for AI Trader.

Selects a benchmark in this order: manual -> sector -> SPY -> none.
V1.1 treats SPY strictly as the market benchmark and optionally validates
market-data availability, history, overlap, OHLCV quality and liquidity.
"""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional
import pandas as pd

VERSION = "1.1"
MARKET_BENCHMARK = "SPY"

# True contextual/sector relationships only. SPY is the market fallback.
BENCHMARK_MAP: Dict[str, str] = {
    "TQQQ": "QQQ", "NVDA": "SMH", "SOXL": "SMH", "META": "QQQ",
    "PLTR": "QQQ", "GDXU": "GDX", "CCJ": "URNM", "AGQ": "SLV",
}
BENCHMARK_ROLE: Dict[str, str] = {
    "QQQ": "technology_growth", "SMH": "semiconductors",
    "GDX": "gold_miners", "URNM": "uranium_nuclear", "SLV": "silver",
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
    resolver_version: str = VERSION

def _clean_ticker(value: Optional[str]) -> Optional[str]:
    if value is None: return None
    value = str(value).strip().upper()
    return value or None

def _import_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for validation. Install it with: pip install yfinance") from exc
    return yf

def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
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
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=False, progress=False, threads=False)
    except Exception as exc:
        raise RuntimeError(f"Unable to download data for {ticker}: {exc}") from exc
    return _flatten_yfinance_columns(df)

def _quality_label(available, sufficient_history, overlap_ok, ohlcv_ok, liquidity_ok, quality_ratio):
    if not available: return "UNAVAILABLE"
    if not sufficient_history: return "INSUFFICIENT_HISTORY"
    if not overlap_ok: return "INSUFFICIENT_OVERLAP"
    if not ohlcv_ok: return "POOR"
    if not liquidity_ok: return "LOW_LIQUIDITY"
    if quality_ratio is not None and quality_ratio >= .995: return "EXCELLENT"
    if quality_ratio is not None and quality_ratio >= .985: return "GOOD"
    return "MODERATE"

def _validate_benchmark_data(ticker: str, benchmark: str, *, validation_config: ValidationConfig,
                             period: str, interval: str) -> BenchmarkValidation:
    asset_df = _download_history(ticker, period, interval)
    benchmark_df = _download_history(benchmark, period, interval)
    if benchmark_df.empty:
        return BenchmarkValidation(benchmark, False, False, False, False, False, False,
                                   "UNAVAILABLE", 0, 0, None, None, None, None,
                                   f"No market data returned for benchmark {benchmark}.")
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in benchmark_df.columns]
    if missing:
        return BenchmarkValidation(benchmark, True, False, False, False, False, False,
                                   "POOR", len(benchmark_df), 0, None, None, None, None,
                                   "Missing OHLCV columns: " + ", ".join(missing))
    work = benchmark_df[required].copy()
    for col in required: work[col] = pd.to_numeric(work[col], errors="coerce")
    rows = len(work)
    valid_ohlcv = work[required].notna().all(axis=1)
    quality_ratio = float(valid_ohlcv.mean()) if rows else 0.0
    positive_volume = work["Volume"].notna() & (work["Volume"] > 0)
    positive_volume_ratio = float(positive_volume.mean()) if rows else 0.0
    dollar_volume = work["Close"].abs() * work["Volume"]
    median_dollar_volume = float(dollar_volume[positive_volume].median()) if positive_volume.any() else None
    ohlcv_ok = quality_ratio >= validation_config.min_data_quality_ratio and positive_volume_ratio >= validation_config.min_positive_volume_ratio
    liquidity_ok = (median_dollar_volume is not None and math.isfinite(median_dollar_volume)
                    and median_dollar_volume >= validation_config.min_median_dollar_volume)
    bclose = pd.to_numeric(work["Close"], errors="coerce").dropna()
    aclose = pd.to_numeric(asset_df["Close"], errors="coerce").dropna() if "Close" in asset_df.columns else pd.Series(dtype=float)
    bclose.index = pd.to_datetime(bclose.index).normalize()
    aclose.index = pd.to_datetime(aclose.index).normalize()
    bclose = bclose[~bclose.index.duplicated(keep="last")]
    aclose = aclose[~aclose.index.duplicated(keep="last")]
    overlap_rows = len(bclose.index.intersection(aclose.index))
    overlap_close_ratio = overlap_rows / max(1, min(len(aclose), len(bclose)))
    sufficient_history = rows >= validation_config.min_history_days
    overlap_ok = overlap_rows >= validation_config.min_overlap_days and overlap_close_ratio >= validation_config.min_close_overlap_ratio
    available = rows > 0
    valid = available and sufficient_history and overlap_ok and ohlcv_ok and liquidity_ok
    quality = _quality_label(available, sufficient_history, overlap_ok, ohlcv_ok, liquidity_ok, quality_ratio)
    if valid: reason = "Benchmark passed all data-quality checks."
    else:
        failures = []
        if not sufficient_history: failures.append("insufficient history")
        if not overlap_ok: failures.append("insufficient asset/benchmark overlap")
        if not ohlcv_ok: failures.append("OHLCV quality below threshold")
        if not liquidity_ok: failures.append("liquidity below threshold")
        reason = "; ".join(failures) or "Benchmark failed validation."
    return BenchmarkValidation(benchmark, available, sufficient_history, overlap_ok, ohlcv_ok,
                               liquidity_ok, valid, quality, rows, overlap_rows,
                               median_dollar_volume, quality_ratio, positive_volume_ratio,
                               overlap_close_ratio, reason)

def validate_benchmark(ticker: str, benchmark: str, *, validation_config=DEFAULT_VALIDATION_CONFIG,
                       period="5y", interval="1d") -> dict:
    ticker, benchmark = _clean_ticker(ticker), _clean_ticker(benchmark)
    if not ticker: raise ValueError("ticker is required")
    if not benchmark: raise ValueError("benchmark is required")
    return asdict(_validate_benchmark_data(ticker, benchmark, validation_config=validation_config,
                                           period=period, interval=interval))

def resolve_benchmarks(ticker: str, manual_benchmark: Optional[str] = None, *,
                       allow_market_fallback=True, validate_data=False,
                       validation_config=DEFAULT_VALIDATION_CONFIG, period="5y", interval="1d") -> dict:
    ticker = _clean_ticker(ticker)
    if not ticker: raise ValueError("ticker is required")
    manual = _clean_ticker(manual_benchmark)
    sector = BENCHMARK_MAP.get(ticker)
    selected = None
    source = "none"
    is_manual = False
    fallback_reason = None
    validation = None

    if manual:
        selected, source, is_manual = manual, "manual", True
        if validate_data:
            validation = validate_benchmark(ticker, selected, validation_config=validation_config, period=period, interval=interval)
            if not validation["valid"]:
                # Manual choice is preserved; never silently replace it.
                fallback_reason = "manual_benchmark_failed_validation"
    elif sector:
        selected, source = sector, "sector"
        if validate_data:
            validation = validate_benchmark(ticker, selected, validation_config=validation_config, period=period, interval=interval)
            if not validation["valid"] and allow_market_fallback:
                fallback_reason = f"sector_benchmark_failed_validation:{validation['reason']}"
                selected, source = MARKET_BENCHMARK, "market"
                validation = validate_benchmark(ticker, selected, validation_config=validation_config, period=period, interval=interval)
                if not validation["valid"]:
                    fallback_reason += ";market_benchmark_failed_validation"
                    selected, source = None, "none"
    elif allow_market_fallback:
        selected, source = MARKET_BENCHMARK, "market"
        if validate_data:
            validation = validate_benchmark(ticker, selected, validation_config=validation_config, period=period, interval=interval)
            if not validation["valid"]:
                fallback_reason = f"market_benchmark_failed_validation:{validation['reason']}"
                selected, source = None, "none"
    else:
        fallback_reason = "No configured sector benchmark and market fallback disabled."

    role = BENCHMARK_ROLE.get(selected, "custom" if selected else "none")
    return asdict(BenchmarkResolution(ticker, sector, MARKET_BENCHMARK, selected, source, role,
                                      is_manual, fallback_reason, validation))

def get_benchmark(ticker: str, manual_benchmark: Optional[str] = None, *, validate_data=False):
    return resolve_benchmarks(ticker, manual_benchmark=manual_benchmark, validate_data=validate_data)["selected_benchmark"]

def get_sector_benchmark(ticker: str):
    ticker = _clean_ticker(ticker)
    if not ticker: raise ValueError("ticker is required")
    return BENCHMARK_MAP.get(ticker)

def validate_resolution(result: dict) -> dict:
    required = {"ticker","sector_benchmark","market_benchmark","selected_benchmark","source","role","is_manual","fallback_reason","validation","resolver_version"}
    errors = []
    missing = sorted(required - set(result))
    if missing: errors.append("Missing fields: " + ", ".join(missing))
    source, selected, manual = result.get("source"), result.get("selected_benchmark"), result.get("is_manual")
    if source == "manual" and (not manual or not selected): errors.append("Manual source requires manual selection")
    if source == "sector" and not selected: errors.append("Sector source requires selected_benchmark")
    if source == "market" and selected != MARKET_BENCHMARK: errors.append("Market source must select SPY")
    if source == "none" and selected is not None: errors.append("None source must not select a benchmark")
    if selected == MARKET_BENCHMARK and source == "sector": errors.append("SPY must be classified as market, not sector")
    if result.get("validation") is not None and result["validation"].get("requested_benchmark") != selected:
        errors.append("Validation benchmark does not match selected benchmark")
    return {"valid": not errors, "errors": errors}

def _build_parser():
    p = argparse.ArgumentParser(description="Resolve and optionally validate an AI Trader benchmark.")
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
    result = resolve_benchmarks(args.ticker, manual_benchmark=args.benchmark,
                                allow_market_fallback=not args.no_market_fallback,
                                validate_data=args.validate, period=args.period, interval=args.interval)
    if args.show_validation: result["resolver_validation"] = validate_resolution(result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__": main()
