AI Trader — Main SHADOW integration V1.0

Contents:
- app/main.py: V2.14 main flow with policy shadow evaluation and hard no-order interlock.
- app/decision_policy_v1.py: centralized policy module.
- app/decision_policy_shadow_integration.py: mapping from scanner results to policy.
- tests/test_main_decision_policy_shadow.py: integration-focused tests.

Install into the existing repository by copying app/main.py over the existing app/main.py,
copying app/decision_policy_v1.py into app/, and copying the test into tests/.
Do not enable autonomous execution. SHADOW_ONLY is hard-coded True and blocks all watchlist
orders in _execute_watchlist_signal, even if AI_TRADER_AUTONOMOUS_EXECUTION=1.

Validate after merging into the complete repository:
python -m py_compile app/main.py app/decision_policy_v1.py
python -m unittest tests.test_main_decision_policy_shadow -v
python -m unittest discover -s tests -v

This package was built from the available AI Trader V2.14 main.py snapshot. Check your current
app/main.py diff before replacing it if you have local changes newer than that snapshot.
