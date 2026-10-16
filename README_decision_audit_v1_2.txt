AI Trader Decision Audit V1.2 — RESEARCH ONLY

Purpose
- Independently audit historical prediction JSONL snapshots.
- Add robust return-distribution statistics and breakouts by engine, regime, and raw signal.
- Highlight ticker concentration and per-horizon evaluation coverage.
- Report PARTIAL/ERROR rather than claiming a complete audit when market data is missing or evaluation coverage is absent.

Files
- decision_audit_v1_2.py: standalone audit script.
- test_decision_audit_v1_2.py: deterministic tests; does not need pandas.

Safety and scope
- Does not edit source prediction JSONL files.
- Does not import or call Alpaca trading endpoints.
- Does not submit, cancel, or modify orders.
- Writes only the JSON report specified by --output.
- Intended to be copied to the AI Trader repository root and run manually.

Install / run in the Raspberry Pi Docker environment
1. Copy decision_audit_v1_2.py and test_decision_audit_v1_2.py to /home/luigi/ai-trader/.
2. Run unit tests:
   docker compose run --rm ai-trader python /app/test_decision_audit_v1_2.py
3. Run audit with persistent host output directory (from repository root):
   mkdir -p audit_results
   docker compose run --rm \
     -v /home/luigi/ai-trader/audit_results:/audit_results \
     ai-trader python /app/decision_audit_v1_2.py \
     --prediction-dir /app/data/predictions \
     --horizons 1 3 5 10 20 60 \
     --output /audit_results/decision_audit_v1_2_results.json
4. Report location on host:
   /home/luigi/ai-trader/audit_results/decision_audit_v1_2_results.json

Metrics added in V1.2
- Mean and median returns.
- Population standard deviation, minimum, maximum, p10/p90, p25/p75.
- 5% trimmed mean and 5% winsorized mean to assess outlier sensitivity.
- Median/mean benchmark and excess returns.
- Evaluation coverage by horizon.
- Breakdowns by engine, regime, and raw signal.
- Unique ticker counts and ticker concentration diagnostics.

Interpretation cautions
- Repeated snapshots of the same ticker are not independent observations; use concentration output and avoid treating row count as independent sample size.
- Mean/median returns are based on recorded price_at_prediction and daily close histories. Intraday prediction price versus daily benchmark close remains an approximation.
- 20/60-session horizons may remain unevaluable for recent predictions.
- Exit code 0 means OK; 2 means PARTIAL; 1 means ERROR. Always inspect market_data.errors, coverage_diagnostics, and status_reason.
- This is a research report, not a recommendation to alter production thresholds.
