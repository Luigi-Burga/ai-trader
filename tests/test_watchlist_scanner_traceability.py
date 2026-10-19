"""Isolated tests for scanner signal traceability; all pipeline stages are mocked."""

from app.scanners import watchlist_scanner_v2_2_2 as scanner


def _analysis(signal="BUY", benchmark_trust="NONE"):
    return {
        "ticker": "TEST",
        "status": "ANALYZED",
        "route": "ROUTE_TO_CCE",
        "engine": "TEST_ENGINE",
        "engine_version": "TEST",
        "signal": signal,
        "score": 70.0,
        "confidence": 80.0,
        "current_price": 100.0,
        "regime": "TEST",
        "benchmark": "SPY",
        "benchmark_trust": benchmark_trust,
        "benchmark_trust_decision_grade": False,
        "asset_age": {"trading_days": 500, "valid_feature_rows": 250},
        "engine_result": {"signal": signal, "score": 70.0, "confidence": 80.0},
        "levels": {},
    }


def _mock_common(monkeypatch, analysis, gate, engine_result):
    monkeypatch.setattr(scanner, "analyze_ticker", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(scanner, "apply_decision_gate", lambda _input: gate)
    monkeypatch.setattr(scanner, "generate_signal", lambda **kwargs: engine_result)
    monkeypatch.setattr(scanner, "_print_result", lambda *args, **kwargs: None)


def test_benchmark_trust_downgrade_is_recorded(monkeypatch):
    _mock_common(
        monkeypatch,
        _analysis(signal="BUY", benchmark_trust="NONE"),
        {"operational_signal": "BUY", "signal": "BUY", "gate_status": "PASSED"},
        {"signal": "BUY", "reason": "simulated test"},
    )

    result = scanner.scan_buy_opportunity({"ticker": "TEST"})

    assert result["signal_engine_input_signal"] == "BUY"
    assert result["signal_engine_output_signal"] == "BUY"
    assert result["pre_safety_signal"] == "BUY"
    assert result["final_signal"] == "BUY_ON_CONFIRMATION"
    assert result["safety_override_applied"] is True
    assert result["safety_override_reason"]
    assert result["signal_transformations"] == [
        {
            "stage": "post_signal_engine_safety",
            "rule": "benchmark_trust_confirmation",
            "from_signal": "BUY",
            "to_signal": "BUY_ON_CONFIRMATION",
            "reason": (
                "Benchmark trust is CONTEXT_ONLY, NONE, or unavailable; "
                "BUY/STRONG_BUY is downgraded to BUY_ON_CONFIRMATION."
            ),
        }
    ]


def test_non_bullish_gate_override_is_recorded(monkeypatch):
    _mock_common(
        monkeypatch,
        _analysis(signal="WATCH", benchmark_trust="HIGH"),
        {"operational_signal": "WATCH", "signal": "WATCH", "gate_status": "NOT_APPLICABLE"},
        {"signal": "BUY", "reason": "simulated test"},
    )

    result = scanner.scan_buy_opportunity({"ticker": "TEST"})

    assert result["signal_engine_input_signal"] == "WATCH"
    assert result["signal_engine_output_signal"] == "BUY"
    assert result["pre_safety_signal"] == "BUY"
    assert result["final_signal"] == "WATCH"
    assert result["safety_override_applied"] is True
    assert result["signal_transformations"][0]["rule"] == "non_bullish_gate_override"
    assert result["signal_transformations"][0]["from_signal"] == "BUY"
    assert result["signal_transformations"][0]["to_signal"] == "WATCH"


def test_no_safety_transformation_is_recorded_when_not_needed(monkeypatch):
    _mock_common(
        monkeypatch,
        _analysis(signal="WATCH", benchmark_trust="HIGH"),
        {"operational_signal": "WATCH", "signal": "WATCH", "gate_status": "NOT_APPLICABLE"},
        {"signal": "WATCH", "reason": "simulated test"},
    )

    result = scanner.scan_buy_opportunity({"ticker": "TEST"})

    assert result["signal_engine_input_signal"] == "WATCH"
    assert result["signal_engine_output_signal"] == "WATCH"
    assert result["pre_safety_signal"] == "WATCH"
    assert result["final_signal"] == "WATCH"
    assert result["signal_transformations"] == []
    assert result["safety_override_applied"] is False
    assert result["safety_override_reason"] is None
