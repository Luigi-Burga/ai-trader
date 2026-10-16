AI Trader Decision Audit V1.1 — research-only

Files
- decision_audit_v1_1.py: independent audit script; does not modify prediction JSONL inputs and does not call Alpaca trading endpoints.
- test_decision_audit_v1_1.py: deterministic unit tests for timestamp parsing, trading-session horizons, benchmark-relative returns, and conversion-rate grouping.

Install/run from the AI Trader repository root
1. Copy decision_audit_v1_1.py to the repository root (same level as app/ and data/).
2. Run the tests:
   python test_decision_audit_v1_1.py
3. Run the audit, writing a new output file (not the V1.0 result):
   python decision_audit_v1_1.py --prediction-dir data/predictions --horizons 1 3 5 10 20 60 --output decision_audit_v1_1_results.json

The audit reads prediction JSONL files and writes only the specified report JSON. It uses app.data.market_data.get_daily_history to obtain daily histories for both assets and recorded benchmarks. No Alpaca order/trading API is used.

Method notes / limitations
- `prediction_timestamp_utc` is now recognized.
- Horizons are trading sessions: horizon 1 means the next available daily session after the prediction session, not one calendar day.
- Benchmark-relative return is asset return minus benchmark close-to-close return over the corresponding session horizon. The asset return starts from `price_at_prediction`; if that price was recorded intraday, the benchmark daily-close baseline is an approximation because daily bars cannot reconstruct an exact historical intraday benchmark price.
- Missing timestamps, entry prices, symbols, delisted/unavailable history, and horizons beyond available history remain unevaluable and are reported as null; they are not fabricated.
- The report's default horizons are 1, 3, 5, 10, 20, 60 sessions. `--horizons` can override them.
- Output report includes SHA-256 hashes of source JSONL files for reproducibility.
- This is a research artifact, not production integration. Review the JSON's `market_data.errors`, counts, and coverage before interpreting aggregate performance.
