"""
Decision Policy V1 — isolated reference implementation.

This module is intentionally not wired into production modules yet.
It centralizes operational decision thresholds without replacing CCE analytics
or the executor's independent risk/authorization protections.

Integration must map real CCE/Decision Gate outputs into DecisionInput and
be validated against historical snapshots before enabling in production.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Sequence


@dataclass(frozen=True)
class DecisionPolicyConfig:
    buy_score_min: float = 70.0
    strong_buy_score_min: float = 85.0
    buy_probability_min: float = 0.60
    strong_buy_probability_min: float = 0.65
    buy_rr_min: float = 1.25
    strong_buy_rr_min: float = 2.00
    confidence_min: float = 0.60
    confirmation_confidence_min: float = 0.50
    fallback_max_auto: float = 0.25
    fallback_max_confirmation: float = 0.50
    confirmation_probability_min: float = 0.65
    confirmation_rr_min: float = 1.50
    confirmation_appt_min: float = 0.50
    allowed_match_quality_auto: tuple = ("GOOD", "HIGH", "VERY_GOOD", "EXCELLENT")
    allowed_match_quality_confirmation: tuple = ("MODERATE", "GOOD", "HIGH", "VERY_GOOD", "EXCELLENT")
    blocked_regimes: tuple = (
        "CAPITULATION", "DOWNTREND", "DISTRIBUTION", "EXTREME_MOMENTUM"
    )


@dataclass(frozen=True)
class DecisionInput:
    symbol: str
    analytical_signal: str
    score: Optional[float] = None
    probability_20d: Optional[float] = None
    reward_risk: Optional[float] = None
    confidence: Optional[float] = None
    fallback_ratio: Optional[float] = None
    match_quality: Optional[str] = None
    regime: Optional[str] = None
    appt: Optional[float] = None
    benchmark_trust: Optional[str] = None
    benchmark_decision_grade: Optional[bool] = None
    data_valid: bool = True
    entry_valid: bool = True
    stop_valid: bool = True
    source: str = "CCE"


@dataclass(frozen=True)
class DecisionResult:
    symbol: str
    analytical_signal: str
    operational_signal: str
    decision_status: str
    reasons: tuple
    failed_gates: tuple
    metrics: dict

    def to_dict(self):
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        result["failed_gates"] = list(self.failed_gates)
        return result


def _missing(value):
    return value is None


def evaluate_decision(data: DecisionInput,
                      config: DecisionPolicyConfig = DecisionPolicyConfig()) -> DecisionResult:
    """
    Pure decision function. No network, orders, files, or side effects.
    BUY/STRONG_BUY means eligible for downstream executor checks, not execution.
    """
    sig = (data.analytical_signal or "WATCH").upper()
    reasons, failed = [], []

    # Non-bullish analytical outcomes are preserved; this policy does not
    # manufacture buys or reverse SELL/REDUCE/WATCH.
    if sig not in ("BUY", "STRONG_BUY"):
        return DecisionResult(
            data.symbol, sig, sig, "NOT_BUY_CANDIDATE",
            (f"Analytical signal is {sig}; policy does not promote non-buy signals.",),
            (), _metrics(data)
        )

    required = {
        "score": data.score,
        "probability_20d": data.probability_20d,
        "reward_risk": data.reward_risk,
        "confidence": data.confidence,
        "fallback_ratio": data.fallback_ratio,
        "match_quality": data.match_quality,
        "regime": data.regime,
        "benchmark_trust": data.benchmark_trust,
    }
    missing = [name for name, value in required.items() if _missing(value)]
    if missing:
        failed.append("MISSING_REQUIRED_METRICS:" + ",".join(missing))

    if not data.data_valid:
        failed.append("INVALID_DATA")
    if not data.entry_valid or not data.stop_valid:
        failed.append("INVALID_ENTRY_STOP")
    if data.benchmark_decision_grade is False:
        failed.append("BENCHMARK_DECISION_GRADE_FALSE")
    if (data.benchmark_trust or "").upper() in ("NONE", "CONTEXT_ONLY", "UNTRUSTED"):
        failed.append("BENCHMARK_NOT_DECISION_GRADE")
    if (data.regime or "").upper() in config.blocked_regimes:
        failed.append("BLOCKED_REGIME")

    # Hard safety/data failures cannot be converted into a confirmation candidate.
    hard_blockers = (
        not data.data_valid
        or not data.entry_valid
        or not data.stop_valid
        or data.benchmark_decision_grade is False
        or (data.benchmark_trust or "").upper() in ("NONE", "CONTEXT_ONLY", "UNTRUSTED")
        or (data.regime or "").upper() in config.blocked_regimes
    )
    if missing or hard_blockers:
        return _result(data, sig, "REVIEW_DATA" if missing or not data.data_valid else "WATCH",
                       "BLOCKED_OR_REVIEW",
                       ["Mandatory data, benchmark, regime, or entry/stop gate failed."],
                       failed)

    score = float(data.score)
    prob = float(data.probability_20d)
    rr = float(data.reward_risk)
    conf = float(data.confidence)
    fallback = float(data.fallback_ratio)
    quality = str(data.match_quality).upper()
    appt = float(data.appt) if data.appt is not None else None

    # Existing strict route.
    strong = (
        score >= config.strong_buy_score_min
        and prob >= config.strong_buy_probability_min
        and rr >= config.strong_buy_rr_min
    )
    primary = (
        score >= config.buy_score_min
        and prob >= config.buy_probability_min
        and rr >= config.buy_rr_min
        and conf >= config.confidence_min
        and fallback <= config.fallback_max_auto
        and quality in config.allowed_match_quality_auto
    )
    if strong and primary:
        return _result(data, "STRONG_BUY", "STRONG_BUY", "AUTO_ELIGIBLE",
                       ["All centralized automatic-buy criteria passed."], [])

    if primary:
        return _result(data, "BUY", "BUY", "AUTO_ELIGIBLE",
                       ["All centralized automatic-buy criteria passed."], [])

    # Confirmation route: never automatically executable.
    confirmation = (
        score >= config.buy_score_min
        and prob >= config.confirmation_probability_min
        and rr >= config.confirmation_rr_min
        and conf >= config.confirmation_confidence_min
        and fallback <= config.fallback_max_confirmation
        and quality in config.allowed_match_quality_confirmation
        and appt is not None and appt >= config.confirmation_appt_min
    )
    if confirmation:
        # Report the actual failed gates rather than the opaque placeholder
        # PRIMARY_ROUTE_NOT_MET. Also document why STRONG_BUY is not eligible.
        blockers = []
        if score < config.buy_score_min: blockers.append("SCORE_BELOW_BUY_MIN")
        if prob < config.buy_probability_min: blockers.append("PROBABILITY_BELOW_BUY_MIN")
        if rr < config.buy_rr_min: blockers.append("REWARD_RISK_BELOW_BUY_MIN")
        if conf < config.confidence_min: blockers.append("CONFIDENCE_BELOW_AUTO_MIN")
        if fallback > config.fallback_max_auto: blockers.append("FALLBACK_ABOVE_AUTO_MAX")
        if quality not in config.allowed_match_quality_auto: blockers.append("MATCH_QUALITY_BELOW_AUTO_MIN")
        if score < config.strong_buy_score_min: blockers.append("SCORE_BELOW_STRONG_BUY_MIN")
        if prob < config.strong_buy_probability_min: blockers.append("PROBABILITY_BELOW_STRONG_BUY_MIN")
        if rr < config.strong_buy_rr_min: blockers.append("REWARD_RISK_BELOW_STRONG_BUY_MIN")
        return _result(
            data, sig, "BUY_ON_CONFIRMATION", "REQUIRES_CONFIRMATION",
            ["Confirmation route passed; no automatic order is authorized.",
             "Automatic BUY blockers: " + ", ".join(
                 item for item in blockers if item not in {
                     "SCORE_BELOW_STRONG_BUY_MIN",
                     "PROBABILITY_BELOW_STRONG_BUY_MIN",
                     "REWARD_RISK_BELOW_STRONG_BUY_MIN",
                 }
             ),
             "STRONG_BUY blockers: " + ", ".join(
                 item for item in blockers if item in {
                     "SCORE_BELOW_STRONG_BUY_MIN",
                     "PROBABILITY_BELOW_STRONG_BUY_MIN",
                     "REWARD_RISK_BELOW_STRONG_BUY_MIN",
                 }
             )],
            blockers,
        )

    # Explain every primary criterion that failed.
    if score < config.buy_score_min: failed.append("SCORE_BELOW_BUY_MIN")
    if prob < config.buy_probability_min: failed.append("PROBABILITY_BELOW_BUY_MIN")
    if rr < config.buy_rr_min: failed.append("REWARD_RISK_BELOW_BUY_MIN")
    if conf < config.confidence_min: failed.append("CONFIDENCE_BELOW_AUTO_MIN")
    if fallback > config.fallback_max_auto: failed.append("FALLBACK_ABOVE_AUTO_MAX")
    if quality not in config.allowed_match_quality_auto: failed.append("MATCH_QUALITY_BELOW_AUTO_MIN")
    if not confirmation:
        return _result(data, sig, "WATCH", "NOT_ELIGIBLE",
                       ["Neither automatic-buy nor confirmation criteria passed."], failed)
    return _result(data, sig, "BUY_ON_CONFIRMATION", "REQUIRES_CONFIRMATION",
                   ["Automatic-buy criteria failed; confirmation is required."], failed)


def _metrics(data):
    return {
        "score": data.score,
        "probability_20d": data.probability_20d,
        "reward_risk": data.reward_risk,
        "confidence": data.confidence,
        "fallback_ratio": data.fallback_ratio,
        "match_quality": data.match_quality,
        "regime": data.regime,
        "appt": data.appt,
        "benchmark_trust": data.benchmark_trust,
        "benchmark_decision_grade": data.benchmark_decision_grade,
        "data_valid": data.data_valid,
        "entry_valid": data.entry_valid,
        "stop_valid": data.stop_valid,
    }


def _result(data, analytical, operational, status, reasons, failed):
    return DecisionResult(
        data.symbol, analytical, operational, status,
        tuple(reasons), tuple(dict.fromkeys(failed)), _metrics(data)
    )
