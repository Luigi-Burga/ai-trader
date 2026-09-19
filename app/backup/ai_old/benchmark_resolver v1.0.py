"""Standalone benchmark resolver for AI Trader."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, Optional

VERSION = "1.0"
MARKET_BENCHMARK = "SPY"

BENCHMARK_MAP: Dict[str, str] = {
    "TQQQ": "QQQ", "NVDA": "SMH", "SOXL": "SMH", "META": "QQQ", "PLTR": "QQQ",
    "GDXU": "GDX", "CCJ": "URNM", "AGQ": "SLV", "UPRO": "SPY", "VOO": "SPY",
}

BENCHMARK_ROLE: Dict[str, str] = {
    "QQQ": "technology_growth", "SMH": "semiconductors", "GDX": "gold_miners",
    "URNM": "uranium_nuclear", "SLV": "silver", "SPY": "broad_market",
}

@dataclass(frozen=True)
class BenchmarkResolution:
    ticker: str
    sector_benchmark: Optional[str]
    market_benchmark: str
    selected_benchmark: Optional[str]
    source: str
    role: str
    is_manual: bool
    resolver_version: str = VERSION

def _clean_ticker(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().upper()
    return value or None

def resolve_benchmarks(ticker: str, manual_benchmark: Optional[str] = None, *, allow_market_fallback: bool = True) -> dict:
    """Resolve benchmark using priority: manual -> sector -> SPY -> none."""
    normalized_ticker = _clean_ticker(ticker)
    if normalized_ticker is None:
        raise ValueError("ticker is required")
    manual = _clean_ticker(manual_benchmark)
    sector = BENCHMARK_MAP.get(normalized_ticker)
    if manual:
        selected, source, is_manual = manual, "manual", True
    elif sector:
        selected, source, is_manual = sector, "sector", False
    elif allow_market_fallback:
        selected, source, is_manual = MARKET_BENCHMARK, "market", False
    else:
        selected, source, is_manual = None, "none", False
    return asdict(BenchmarkResolution(
        ticker=normalized_ticker, sector_benchmark=sector,
        market_benchmark=MARKET_BENCHMARK, selected_benchmark=selected,
        source=source, role=BENCHMARK_ROLE.get(selected, "custom" if selected else "none"),
        is_manual=is_manual,
    ))

def get_benchmark(ticker: str, manual_benchmark: Optional[str] = None) -> Optional[str]:
    return resolve_benchmarks(ticker, manual_benchmark=manual_benchmark)["selected_benchmark"]

def get_sector_benchmark(ticker: str) -> Optional[str]:
    normalized = _clean_ticker(ticker)
    if normalized is None:
        raise ValueError("ticker is required")
    return BENCHMARK_MAP.get(normalized)

def validate_resolution(result: dict) -> dict:
    required = {"ticker", "sector_benchmark", "market_benchmark", "selected_benchmark", "source", "role", "is_manual", "resolver_version"}
    errors = []
    missing = sorted(required - result.keys())
    if missing:
        errors.append(f"Missing fields: {', '.join(missing)}")
    source, selected, manual = result.get("source"), result.get("selected_benchmark"), result.get("is_manual")
    if source == "manual" and (not manual or not selected): errors.append("Manual source requires is_manual=True and a selected benchmark")
    if source == "sector" and not selected: errors.append("Sector source requires selected_benchmark")
    if source == "market" and selected != MARKET_BENCHMARK: errors.append("Market source must select SPY")
    if source == "none" and selected is not None: errors.append("None source must not select a benchmark")
    return {"valid": not errors, "errors": errors}

def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve the preferred benchmark for an AI Trader ticker.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--benchmark", "--manual-benchmark", dest="benchmark")
    parser.add_argument("--no-market-fallback", action="store_true")
    parser.add_argument("--show-validation", action="store_true")
    args = parser.parse_args()
    result = resolve_benchmarks(args.ticker, args.benchmark, allow_market_fallback=not args.no_market_fallback)
    if args.show_validation:
        result["validation"] = validate_resolution(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
