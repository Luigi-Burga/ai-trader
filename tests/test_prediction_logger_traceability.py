import json

from app.prediction_logger import append_snapshot, build_snapshot


def test_build_snapshot_preserves_signal_and_gate_trace():
    result = {
        "raw_signal": "BUY",
        "gated_signal": "BUY",
        "final_signal": "BUY_ON_CONFIRMATION",
        "signal": "BUY_ON_CONFIRMATION",
        "decision_gate": {
            "operational_signal": "BUY",
            "gate_status": "PASSED",
            "gate_reason": "nle_all_buy_gates_passed",
            "failed_gates": [],
            "hard_failed_gates": [],
        },
    }

    snapshot = build_snapshot("PLTU", result)

    assert snapshot["raw_signal"] == "BUY"
    assert snapshot["gated_signal"] == "BUY"
    assert snapshot["final_signal"] == "BUY_ON_CONFIRMATION"
    assert snapshot["signal"] == "BUY_ON_CONFIRMATION"
    assert snapshot["decision_gate"]["gate_status"] == "PASSED"
    assert snapshot["decision_gate"]["gate_reason"] == (
        "nle_all_buy_gates_passed"
    )
    assert snapshot["decision_gate"]["failed_gates"] == []


def test_snapshot_without_gate_remains_backward_compatible():
    snapshot = build_snapshot(
        "PLTU",
        {
            "raw_signal": "BUY",
            "signal": "BUY_ON_CONFIRMATION",
        },
    )

    assert snapshot["raw_signal"] == "BUY"
    assert snapshot["signal"] == "BUY_ON_CONFIRMATION"
    assert snapshot["final_signal"] == "BUY_ON_CONFIRMATION"
    assert snapshot["decision_gate"] is None


def test_decision_gate_survives_jsonl_persistence(tmp_path):
    snapshot = build_snapshot(
        "PLTU",
        {
            "raw_signal": "BUY",
            "gated_signal": "BUY",
            "final_signal": "BUY_ON_CONFIRMATION",
            "signal": "BUY_ON_CONFIRMATION",
            "decision_gate": {
                "gate_status": "BLOCKED",
                "gate_reason": "nle_buy_gate_failed",
                "failed_gates": ["benchmark_trust"],
            },
        },
    )

    path = append_snapshot(snapshot, base_dir=tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8").strip())

    assert saved["raw_signal"] == "BUY"
    assert saved["gated_signal"] == "BUY"
    assert saved["final_signal"] == "BUY_ON_CONFIRMATION"
    assert saved["decision_gate"]["gate_status"] == "BLOCKED"
    assert saved["decision_gate"]["failed_gates"] == ["benchmark_trust"]
