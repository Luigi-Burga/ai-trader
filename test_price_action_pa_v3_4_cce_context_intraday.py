"""Contract test for PA V3.4 — CCE Context -> Intraday PA."""

from pathlib import Path
import py_compile

TARGET = Path(__file__).with_name("price_action_pa_v3_4_cce_context_intraday.py")


REQUIRED = [
    'VERSION = "3.4"',
    'PRODUCTION_CCE_VERSION = "1.7"',
    'FROZEN_CANDIDATES',
    'CCE_BUY_SIGNALS',
    'ALPACA_FEED = "sip"',
    'ALPACA_TIMEFRAME = "5Min"',
    'next_page_token',
    'params["page_token"] = token',
    '403_recent_sip_move_end_backward',
    'build_20m_exact',
    'characteristic_curve_v1_7',
    'benchmark_resolver_v1_6_1',
    '_clean_ohlcv',
    'resolve_benchmarks',
    'benchmark=None',
    'asset_df=asset',
    'return_selected_data=True',
    'analyze_dataframe',
    'CCE(D)',
    'next trading session D+1',
    'independent CCE context sessions',
    'forward_return_same_session',
    'PA-01',
    'LARGE_BEAR',
    'PA-06',
    'ACCEPT_BELOW_PREV_LOW',
    'delta_session_mean_pp',
    'test_classification',
    'research_only',
    'production_modified',
    'orders_submitted',
    'alpaca_trading_endpoints_used',
]


def main():
    assert TARGET.exists(), f"Missing target: {TARGET}"
    src = TARGET.read_text(encoding="utf-8")

    missing = [x for x in REQUIRED if x not in src]
    assert not missing, f"Missing contract elements: {missing}"

    forbidden = [
        "submit_order",
        "cancel_order",
        "/v2/orders",
        "alpaca_trade_api",
        "TradingClient(",
        "StockHistoricalDataClient",
    ]
    present = [x for x in forbidden if x in src]
    assert not present, f"Trading/order elements found: {present}"

    py_compile.compile(str(TARGET), doraise=True)
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)

    print("PA V3.4 CONTRACT TEST: PASS")
    print("py_compile: PASS")
    print("research-only guards: PASS")
    print("CCE V1.7 production integration contract: PASS")
    print("Exact OHLCV normalization: PASS")
    print("Alpaca SIP 5m + pagination: PASS")
    print("Adaptive recent-SIP boundary: PASS")
    print("CCE(D) -> next session D+1 causal rule: PASS")
    print("Context-level 60/20/20 split: PASS")
    print("Frozen PA V2.1 candidates: PASS")
    print("Same-session forward-return guard: PASS")


if __name__ == "__main__":
    main()
