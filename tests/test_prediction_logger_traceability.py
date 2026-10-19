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


def test_signal_engine_traceability_survives_jsonl_persistence(tmp_path):
    result = {
        "raw_signal": "BUY",
        "gated_signal": "BUY",
        "signal": "BUY_ON_CONFIRMATION",
        "final_signal": "BUY_ON_CONFIRMATION",
        "signal_engine": {
            "signal": "BUY",
            "reason": "simulated test",
        },
        "signal_engine_input_signal": "BUY",
        "signal_engine_output_signal": "BUY",
        "pre_safety_signal": "BUY",
        "signal_transformations": [
            {
                "stage": "post_signal_engine_safety",
                "rule": "benchmark_trust_confirmation",
                "from_signal": "BUY",
                "to_signal": "BUY_ON_CONFIRMATION",
                "reason": "Simulated benchmark trust restriction",
            }
        ],
        "safety_override_applied": True,
        "safety_override_reason": "Simulated benchmark trust restriction",
    }

    snapshot = build_snapshot("TEST", result)
    path = append_snapshot(snapshot, base_dir=tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8").strip())

    assert saved["signal_engine_input_signal"] == "BUY"
    assert saved["signal_engine_output_signal"] == "BUY"
    assert saved["pre_safety_signal"] == "BUY"
    assert saved["final_signal"] == "BUY_ON_CONFIRMATION"

    assert saved["signal_transformations"] == [
        {
            "stage": "post_signal_engine_safety",
            "rule": "benchmark_trust_confirmation",
            "from_signal": "BUY",
            "to_signal": "BUY_ON_CONFIRMATION",
            "reason": "Simulated benchmark trust restriction",
        }
    ]

    assert saved["safety_override_applied"] is True
    assert saved["safety_override_reason"] == (
        "Simulated benchmark trust restriction"
    )
