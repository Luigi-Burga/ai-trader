"""
AI Trader Decision Policy Shadow Adapter V1.2.
Standalone diagnostic only: calls the existing scanner, then evaluates its
returned analysis using the isolated proposed policy. It never invokes main.py
or the Alpaca executor and never submits orders.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from decision_policy_v1 import DecisionInput, evaluate_decision


def first(mapping: Mapping[str, Any], *keys: str):
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def as_float(value):
    try:
        if value is None or isinstance(value, bool):
            return None
        parsed = float(value)
        return parsed if parsed == parsed and abs(parsed) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _walk_mappings(value: Any, path: str = ""):
    """Yield (path, mapping) recursively, preserving paths for audit evidence."""
    if isinstance(value, Mapping):
        yield path, value
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if isinstance(child, Mapping):
                yield from _walk_mappings(child, child_path)


def _find_value(root: Mapping[str, Any], *keys: str):
    """Find first non-empty key across nested output contracts; return value and path."""
    wanted = {key.lower() for key in keys}
    for path, mapping in _walk_mappings(root):
        for key, value in mapping.items():
            if str(key).lower() in wanted and value is not None and value != "":
                return value, (f"{path}.{key}" if path else str(key))
    return None, None


def _find_float(root: Mapping[str, Any], *keys: str):
    value, path = _find_value(root, *keys)
    return as_float(value), path


def normalize_confidence(value):
    value = as_float(value)
    if value is None:
        return None
    # CCE V1.7 reports confidence in percentage points (e.g. 58.93);
    # some normalized outputs may already be fractions (0.5893).
    return value / 100.0 if value > 1.0 else value


def _nested(root: Mapping[str, Any], key: str):
    value = root.get(key)
    return value if isinstance(value, Mapping) else {}


def map_scanner_result(result: Mapping[str, Any]) -> tuple[DecisionInput, dict]:
    analysis = _nested(result, "analysis")
    engine = _nested(result, "engine_result")
    gate = _nested(result, "decision_gate")
    signal_engine = _nested(result, "signal_engine")
    source_tree = {
        "result": result,
        "analysis": analysis,
        "engine_result": engine,
        "decision_gate": gate,
        "signal_engine": signal_engine,
    }

    # Search only recognized result branches and their descendants, not arbitrary
    # application metadata. The paths are emitted to prove where each field came from.
    signal = first(analysis, "signal", "operational_signal", "decision")
    if signal is None:
        signal = first(engine, "operational_signal", "signal", "decision")
    if signal is None:
        signal = first(result, "signal", "final_signal")
    signal = str(signal or "REVIEW_DATA").upper()

    trust, trust_path = _find_value(source_tree, "benchmark_trust", "benchmark_trust_level")
    grade, grade_path = _find_value(source_tree, "benchmark_trust_decision_grade")
    if not isinstance(grade, bool):
        grade = None

    score, score_path = _find_float(source_tree, "score", "characteristic_score")
    probability, probability_path = _find_float(
        source_tree, "historical_probability_20d", "probability_20d",
        "prob_positive_20d", "prob_positive"
    )
    rr, rr_path = _find_float(
        source_tree, "reward_risk_20d", "reward_risk", "risk_reward_20d",
        "reward_to_risk", "rr_ratio"
    )
    confidence_raw, confidence_path = _find_value(
        source_tree, "trading_confidence", "confidence"
    )
    confidence = normalize_confidence(confidence_raw)
    fallback, fallback_path = _find_float(source_tree, "fallback_ratio")
    quality, quality_path = _find_value(source_tree, "match_quality", "quality")
    regime, regime_path = _find_value(source_tree, "cycle", "regime")
    appt, appt_path = _find_float(source_tree, "appt_x_20d", "appt", "expected_value_x_20d")

    current, current_path = _find_float(source_tree, "current_price", "price")
    entry, entry_path = _find_float(source_tree, "entry_price", "preferred_entry")
    low, low_path = _find_float(source_tree, "entry_zone_low")
    high, high_path = _find_float(source_tree, "entry_zone_high")
    stop, stop_path = _find_float(source_tree, "stop", "dynamic_stop", "stop_loss")

    data_valid = str(result.get("status", "ANALYZED")).upper() not in {"ERROR", "NO_DATA"}
    entry_valid = all(v is not None and v > 0 for v in (current, entry, low, high))
    stop_valid = stop is not None and stop > 0 and entry is not None and stop < entry
    if entry_valid:
        entry_valid = low <= high and low <= entry <= high and low <= current <= high

    # Required metrics for a full candidate audit. Missing fields are reported even
    # when the current signal is WATCH, because otherwise the shadow comparison
    # could falsely appear complete while the policy never evaluated its thresholds.
    metric_values = {
        "score": score,
        "probability_20d": probability,
        "reward_risk": rr,
        "confidence": confidence,
        "fallback_ratio": fallback,
        "match_quality": quality,
        "regime": regime,
        "benchmark_trust": trust,
        "appt": appt,
    }
    missing = [key for key, value in metric_values.items() if value is None]

    decision_input = DecisionInput(
        symbol=str(result.get("ticker") or analysis.get("ticker") or "UNKNOWN"),
        analytical_signal=signal,
        score=score,
        probability_20d=probability,
        reward_risk=rr,
        confidence=confidence,
        fallback_ratio=fallback,
        match_quality=str(quality).upper() if quality is not None else None,
        regime=str(regime).upper() if regime is not None else None,
        appt=appt,
        benchmark_trust=str(trust).upper() if trust is not None else None,
        benchmark_decision_grade=grade,
        data_valid=data_valid,
        entry_valid=entry_valid,
        stop_valid=stop_valid,
        source=str(analysis.get("engine") or result.get("engine") or "scanner"),
    )
    paths = {
        "score": score_path, "probability_20d": probability_path,
        "reward_risk": rr_path, "confidence": confidence_path,
        "fallback_ratio": fallback_path, "match_quality": quality_path,
        "regime": regime_path, "appt": appt_path,
        "benchmark_trust": trust_path, "benchmark_decision_grade": grade_path,
        "current_price": current_path, "entry_price": entry_path,
        "entry_zone_low": low_path, "entry_zone_high": high_path, "stop": stop_path,
    }
    return decision_input, {"missing_required_metrics": missing, "field_paths": paths}


def summarize(result: Mapping[str, Any]) -> dict:
    decision_input, extraction = map_scanner_result(result)
    proposed = evaluate_decision(decision_input)

    gate = _nested(result, "decision_gate")
    engine = _nested(result, "engine_result")
    analysis = _nested(result, "analysis")
    signal_engine = _nested(result, "signal_engine")

    current_signal = str(result.get("final_signal") or result.get("signal") or "UNKNOWN").upper()
    gate_signal = str(gate.get("operational_signal") or gate.get("signal") or result.get("gated_signal") or "")
    engine_operational = str(
        analysis.get("operational_signal")
        or engine.get("operational_signal")
        or analysis.get("signal")
        or ""
    ).upper()
    signal_engine_output = str(
        signal_engine.get("signal") or signal_engine.get("final_signal")
        or result.get("signal_engine_output_signal") or ""
    ).upper()

    missing = extraction["missing_required_metrics"]

    # Shadow-only eligibility probe: evaluate numeric gates even when CCE's current
    # analytical signal is WATCH. Preserve actual signal; never authorizes an order.
    probe_input = decision_input
    if decision_input.analytical_signal not in ("BUY", "STRONG_BUY"):
        from dataclasses import replace
        probe_input = replace(decision_input, analytical_signal="BUY")
    eligibility = evaluate_decision(probe_input)
    proposed_signal = eligibility.operational_signal
    eligibility_reasons = " | ".join(eligibility.reasons)
    eligibility_failed_gates = " | ".join(eligibility.failed_gates)

    if missing:
        comparison = "DATOS_INSUFICIENTES"
        audit_status = "INCOMPLETE_METRICS"
    else:
        comparison = "SIN_CAMBIOS" if current_signal == proposed_signal else "CAMBIO_DE_SEÑAL"
        audit_status = "COMPLETE"
    if result.get("status") in {"ERROR", "NO_DATA"}:
        comparison = "DATOS_INSUFICIENTES"
        audit_status = "SCANNER_ERROR"

    return {
        "ticker": decision_input.symbol,
        "analytical_signal": decision_input.analytical_signal,
        "engine_operational_signal": engine_operational,
        "current_gate_signal": gate_signal,
        "signal_engine_output_signal": signal_engine_output,
        "current_final_signal": current_signal,
        "proposed_signal": proposed_signal,
        "comparison": comparison,
        "audit_status": audit_status,
        "decision_status": eligibility.decision_status,
        "reasons": eligibility_reasons,
        "failed_gates": eligibility_failed_gates,
        "eligibility_probe": "HYPOTHETICAL_ONLY",
        "analytical_signal_preserved": decision_input.analytical_signal,
        "missing_required_metrics": " | ".join(missing),
        "field_paths_json": json.dumps(extraction["field_paths"], ensure_ascii=False),
        "score": decision_input.score,
        "probability_20d": decision_input.probability_20d,
        "reward_risk": decision_input.reward_risk,
        "confidence_fraction": decision_input.confidence,
        "fallback_ratio": decision_input.fallback_ratio,
        "match_quality": decision_input.match_quality,
        "regime": decision_input.regime,
        "appt": decision_input.appt,
        "benchmark_trust": decision_input.benchmark_trust,
        "benchmark_decision_grade": decision_input.benchmark_decision_grade,
        "data_valid": decision_input.data_valid,
        "entry_valid": decision_input.entry_valid,
        "stop_valid": decision_input.stop_valid,
        "current_price": result.get("current_price") or analysis.get("current_price"),
        "scanner_status": result.get("status"),
        "scanner_error": result.get("error"),
    }


def load_tickers(repo_root: Path, explicit: str | None):
    if explicit:
        return [x.strip().upper() for x in explicit.split(",") if x.strip()]
    path = repo_root / "app" / "config" / "watchlist.json"
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("watchlist", data.get("items", data.get("tickers", [])))
    if not isinstance(data, list):
        raise ValueError(f"Unsupported watchlist structure: {type(data).__name__}")
    tickers=[]
    for item in data:
        if isinstance(item, str):
            tickers.append(item.strip().upper())
        elif isinstance(item, dict) and item.get("ticker"):
            tickers.append(str(item["ticker"]).strip().upper())
    if not tickers:
        raise ValueError(f"No tickers found in {path}; use --tickers AAPL,MSFT")
    return list(dict.fromkeys(tickers))


def main():
    parser=argparse.ArgumentParser(description="AI Trader decision policy shadow comparison (no orders)")
    parser.add_argument("--tickers", help="Comma-separated symbols. Default: read app/config/watchlist.json")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--output-dir", default="artifacts/decision_policy_shadow")
    args=parser.parse_args()

    repo_root=Path.cwd().resolve()
    if not (repo_root/"app").is_dir():
        parser.error("Run this script from the AI Trader repository root (the folder containing app/).")
    symbols=load_tickers(repo_root,args.tickers)

    # Import scanner only after validating repository root. This path calls the
    # scanner's analysis pipeline but not main.py or any broker/executor module.
    from app.scanners.watchlist_scanner_v2_2_2 import scan_buy_opportunity

    rows=[]
    for symbol in symbols:
        print(f"[SHADOW] {symbol}: analyzing (no order execution)", flush=True)
        try:
            scan=scan_buy_opportunity({"ticker":symbol,"period":args.period,"benchmark":args.benchmark})
            rows.append(summarize(scan))
        except Exception as exc:
            rows.append({
                "ticker":symbol, "analytical_signal":"ERROR",
                "current_gate_signal":"", "current_final_signal":"ERROR",
                "proposed_signal":"REVIEW_DATA", "comparison":"DATOS_INSUFICIENTES",
                "decision_status":"ERROR", "reasons":str(exc), "failed_gates":"ADAPTER_EXCEPTION"
            })

    outdir=repo_root/args.output_dir
    outdir.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path=outdir/f"decision_policy_shadow_{stamp}.json"
    csv_path=outdir/f"decision_policy_shadow_{stamp}.csv"
    payload={
        "mode":"SHADOW_ONLY",
        "orders_submitted":False,
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "repository_root":str(repo_root),
        "policy_version":"decision_policy_v1; shadow_adapter_v1_2",
        "count":len(rows),
        "summary":{
            "sin_cambios":sum(r.get("comparison")=="SIN_CAMBIOS" for r in rows),
            "cambio_de_senal":sum(r.get("comparison")=="CAMBIO_DE_SEÑAL" for r in rows),
            "datos_insuficientes":sum(r.get("comparison")=="DATOS_INSUFICIENTES" for r in rows),
        },
        "results":rows,
    }
    json_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with csv_path.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(payload["summary"],indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print("SAFETY: SHADOW_ONLY; orders_submitted=false")


if __name__=="__main__":
    main()
