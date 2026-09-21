"""AI Trader - Telegram Formatter V1.0."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any

def _num(value: Any, digits: int = 2) -> str:
    try: return f"{float(value):.{digits}f}"
    except (TypeError, ValueError): return "-"

def _money(value: Any) -> str:
    try: return f"${float(value):,.2f}"
    except (TypeError, ValueError): return "-"

def format_analysis(result: Mapping[str, Any]) -> str:
    ticker = str(result.get("ticker") or "-").upper()
    signal = str(result.get("final_signal") or result.get("gated_signal") or result.get("signal") or "ERROR").upper()
    gate = result.get("decision_gate") if isinstance(result.get("decision_gate"), Mapping) else {}
    engine_result = result.get("engine_result") if isinstance(result.get("engine_result"), Mapping) else {}
    levels = engine_result.get("levels") if isinstance(engine_result.get("levels"), Mapping) else {}
    lines = [
        f"📊 <b>AI TRADER — {ticker}</b>", "", f"<b>FINAL:</b> {signal}",
        f"<b>Price:</b> {_money(result.get('current_price'))}",
        f"<b>Score:</b> {_num(result.get('score'))} / 100",
        f"<b>Confidence:</b> {_num(result.get('confidence'))}%",
        f"<b>Regime:</b> {result.get('regime') or '-'}", "",
        f"<b>Benchmark:</b> {result.get('benchmark') or '-'}",
        f"<b>Trust:</b> {result.get('benchmark_trust') or 'NONE'}",
        f"<b>Route:</b> {result.get('route') or '-'}",
        f"<b>Engine:</b> {result.get('engine') or '-'} V{result.get('engine_version') or '-'}", "",
        f"<b>Decision Gate:</b> {gate.get('gate_status') or '-'}",
        f"<b>Gate reason:</b> {gate.get('gate_reason') or '-'}",
    ]
    low, high = levels.get("entry_zone_low"), levels.get("entry_zone_high")
    if low is not None and high is not None:
        lines += ["", "<b>Dynamic Levels</b>", f"Entry zone: {_money(low)} — {_money(high)}"]
    elif low is not None or high is not None:
        lines += ["", "<b>Dynamic Levels</b>", f"Entry: {_money(low if low is not None else high)}"]
    for key, label in [("dynamic_stop", "Stop"), ("take_profit_1", "TP1"), ("take_profit_2", "TP2"), ("take_profit_3", "TP3")]:
        if levels.get(key) is not None: lines.append(f"{label}: {_money(levels[key])}")
    if result.get("error"): lines += ["", f"⚠️ {result['error']}"]
    return "\n".join(lines)

def format_error(message: str) -> str:
    return f"⚠️ <b>AI Trader</b>\n\n{message}"
