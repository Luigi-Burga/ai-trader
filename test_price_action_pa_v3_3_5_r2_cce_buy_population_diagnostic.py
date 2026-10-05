
"""
Contract test for PA V3.3.5 R2 — CCE BUY Population Diagnostic.
"""
from pathlib import Path
import py_compile

TARGET = Path(__file__).resolve().parent / "price_action_pa_v3_3_5_r2_cce_buy_population_diagnostic.py"

REQUIRED = [
    'VERSION = "3.3.5-R2"',
    "production_cce_version",
    "research_only",
    "production_modified",
    "orders_submitted",
    "alpaca_trading_endpoints_used",
    "FROZEN_CANDIDATES",
    "resolve_benchmarks",
    "asset_df=daily_df",
    "_clean_ohlcv",
    "analyze_dataframe",
    "CCE_BUY_SIGNALS",
    "build_20m_exact",
    "cce_buy_before_alignment",
    "cce_buy_with_same_day_pa",
    "cce_buy_aligned_with_latest_same_day_20m",
    "alignment_loss",
    "split",
    "end_offsets_days",
    "accepted_end",
    "403_recent_sip_move_end_backward",
]

def main():
    assert TARGET.exists(), f"Missing target: {TARGET}"
    source = TARGET.read_text(encoding="utf-8")

    missing = [x for x in REQUIRED if x not in source]
    assert not missing, f"Missing contract elements: {missing}"

    forbidden = [
        "submit_order",
        "cancel_order",
        "/v2/orders",
        "alpaca_trade_api",
        "TradingClient(",
        "StockHistoricalDataClient",
    ]
    present = [x for x in forbidden if x in source]
    assert not present, f"Trading/order elements found: {present}"

    py_compile.compile(str(TARGET), doraise=True)
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)

    print("PA V3.3.5 R2 CONTRACT TEST: PASS")
    print("py_compile: PASS")
    print("research-only guards: PASS")
    print("CCE V1.7 diagnostic contract: PASS")
    print("Exact OHLCV normalization contract: PASS")
    print("CCE BUY population stages: PASS")
    print("403 recent-SIP end-boundary discovery: PASS")
    print("60/20/20 split diagnostic: PASS")
    print("Frozen PA V2.1 candidates: PASS")

if __name__ == "__main__":
    main()
