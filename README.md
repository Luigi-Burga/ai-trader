# Decision Policy Shadow Adapter V1

This package is a standalone shadow comparison tool. It does not modify production modules, call `main.py`, or submit orders. It calls the existing Watchlist Scanner per ticker, then compares its final signal with the proposed isolated decision policy.

## Install into your working copy
Copy these files into the repository root:
- `decision_policy_v1.py`
- `decision_policy_shadow_v1.py`

Keep your existing `app/` directory unchanged. The test file can stay in `tests/`.

## Run from PowerShell, at the repository root
```powershell
python -m unittest discover -s tests -v
python decision_policy_shadow_v1.py --tickers NVDA,MSFT,GOOGL,AMZN
```
Use `--tickers` with your watchlist symbols for a controlled first run. If omitted, the script attempts to read `app/config/watchlist.json`.

Reports are written to `artifacts/decision_policy_shadow/` as JSON and CSV. This script performs market-data analysis and may take time; it does not place orders.

## Important interpretation
The adapter is deliberately conservative about missing fields and dynamic entry/stop geometry. A `DATOS_INSUFICIENTES` result means the policy could not safely evaluate that symbol with the available output contract; it does not prove the asset is a bad opportunity. Inspect `failed_gates` and `reasons` before interpreting a changed signal.

The adapter is not production integration. Review the shadow output and reconcile any mismatches before changing the scanner.
