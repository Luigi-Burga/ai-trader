"""
AI Trader - Fundamental Alert Engine V1
=======================================

Purpose
-------
Convert Fundamental Analysis results into actionable Telegram alerts
without sending the same fundamental condition on every main.py run.

Design goals
------------
1. Fundamental scoring continues to run normally.
2. Telegram is triggered by meaningful CHANGE, not by a static score.
3. State is persisted between process/container executions.
4. Duplicate alerts are suppressed.
5. Rating transitions can trigger alerts.
6. Large score deterioration/improvement can trigger alerts.
7. Cooldown prevents alert storms.
8. Critical transitions bypass the normal cooldown.
9. DRY_RUN mode allows validation without Telegram.
10. No investment decision is made by this module; it only determines
    whether a change is significant enough to notify the user.

Default state file
------------------
<project_root>/data/alerts/fundamental_alert_state.json

Environment variables
---------------------
AI_TRADER_FUNDAMENTAL_ALERT_STATE
    Optional custom state-file path.

AI_TRADER_FUNDAMENTAL_ALERT_COOLDOWN_SECONDS
    Default: 3600 (1 hour).

AI_TRADER_FUNDAMENTAL_ALERT_SCORE_DELTA
    Default: 5 points.

AI_TRADER_FUNDAMENTAL_ALERT_DRY_RUN
    Default: false.
    true/1/yes/on -> do not send Telegram.

Alert classes
-------------
ACTION
    Important rating deterioration or significant transition.

IMPORTANT
    Significant score/rating change that merits review.

WATCH
    Significant change below the immediate-alert threshold.

SILENT
    No meaningful change.

Important
---------
This engine deliberately does NOT change calculate_fundamental_score()
or build_fundamental_message(). It sits above the existing Fundamental
Analysis layer and controls notification policy.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


VERSION = "1.0"
MODULE_NAME = "Fundamental Alert Engine V1"

DEFAULT_COOLDOWN_SECONDS = 3600
DEFAULT_SCORE_DELTA = 5.0

# Rating order is used only to identify transitions.
# It is not an investment ranking.
_RATING_INDEX = {
    "SELL": 0,
    "REDUCE": 1,
    "HOLD": 2,
    "BUY": 3,
    "STRONG BUY": 4,
}


@dataclass
class FundamentalState:
    """Persisted state for one ticker."""

    symbol: str
    score: float
    rating: str
    revenue: float
    margins: float
    debt: float
    updated_at: str
    last_alert_at: Optional[str] = None
    last_alert_class: Optional[str] = None
    last_alert_score: Optional[float] = None
    last_alert_rating: Optional[str] = None


@dataclass
class AlertDecision:
    """Result returned by evaluate()."""

    symbol: str
    alert: bool
    alert_class: str
    reason: str
    score: float
    previous_score: Optional[float]
    score_delta: Optional[float]
    rating: str
    previous_rating: Optional[str]
    rating_changed: bool
    critical_transition: bool
    cooldown_active: bool
    state_updated: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _project_root() -> Path:
    """
    Resolve the project root.

    This module is expected under:
        <project_root>/app/alerts/
    """
    return Path(__file__).resolve().parents[2]


def default_state_path() -> Path:
    """Return the default persistent state path."""
    configured = os.getenv(
        "AI_TRADER_FUNDAMENTAL_ALERT_STATE",
        "",
    ).strip()

    if configured:
        return Path(configured).expanduser()

    return (
        _project_root()
        / "data"
        / "alerts"
        / "fundamental_alert_state.json"
    )


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()

    if not raw:
        return default

    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()

    if not raw:
        return default

    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()

    if not raw:
        return default

    return raw in {
        "1",
        "true",
        "yes",
        "on",
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value: Optional[str]) -> Optional[float]:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.timestamp()

    except (TypeError, ValueError):
        return None


def _normalize_rating(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _state_to_dict(state: FundamentalState) -> Dict[str, Any]:
    return asdict(state)


def _state_from_dict(data: Dict[str, Any]) -> FundamentalState:
    return FundamentalState(
        symbol=_normalize_symbol(data.get("symbol")),
        score=_safe_number(data.get("score")),
        rating=_normalize_rating(data.get("rating")),
        revenue=_safe_number(data.get("revenue")),
        margins=_safe_number(data.get("margins")),
        debt=_safe_number(data.get("debt")),
        updated_at=str(data.get("updated_at") or ""),
        last_alert_at=data.get("last_alert_at"),
        last_alert_class=data.get("last_alert_class"),
        last_alert_score=(
            None
            if data.get("last_alert_score") is None
            else _safe_number(data.get("last_alert_score"))
        ),
        last_alert_rating=(
            None
            if data.get("last_alert_rating") is None
            else _normalize_rating(data.get("last_alert_rating"))
        ),
    )


def load_state(
    path: Optional[os.PathLike[str] | str] = None,
) -> Dict[str, FundamentalState]:
    """
    Load persisted fundamental alert state.

    Corrupt/missing state is treated as empty state so an alert problem
    never stops the trading pipeline.
    """
    state_path = Path(path) if path else default_state_path()

    try:
        if not state_path.exists():
            return {}

        with state_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            return {}

        result: Dict[str, FundamentalState] = {}

        for symbol, raw_state in payload.items():
            if not isinstance(raw_state, dict):
                continue

            state = _state_from_dict(raw_state)

            if not state.symbol:
                state.symbol = _normalize_symbol(symbol)

            if state.symbol:
                result[state.symbol] = state

        return result

    except Exception as exc:
        print(
            f"[FUNDAMENTAL-ALERT] State load error: "
            f"{type(exc).__name__}: {exc}"
        )
        return {}


def save_state(
    state: Dict[str, FundamentalState],
    path: Optional[os.PathLike[str] | str] = None,
) -> bool:
    """
    Persist state atomically.

    Returns True on success and False on failure.
    """
    state_path = Path(path) if path else default_state_path()

    try:
        state_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            symbol: _state_to_dict(item)
            for symbol, item in sorted(state.items())
        }

        temporary = state_path.with_suffix(
            state_path.suffix + ".tmp"
        )

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write("\n")

        os.replace(temporary, state_path)
        return True

    except Exception as exc:
        print(
            f"[FUNDAMENTAL-ALERT] State save error: "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def _rating_changed(
    previous_rating: Optional[str],
    current_rating: str,
) -> bool:
    if previous_rating is None:
        return False

    return (
        _normalize_rating(previous_rating)
        != _normalize_rating(current_rating)
    )


def _rating_direction(
    previous_rating: Optional[str],
    current_rating: str,
) -> int:
    """
    Return:
        +1 = movement toward higher score category
        -1 = movement toward lower score category
         0 = no recognized movement
    """
    if previous_rating is None:
        return 0

    previous = _RATING_INDEX.get(
        _normalize_rating(previous_rating)
    )
    current = _RATING_INDEX.get(
        _normalize_rating(current_rating)
    )

    if previous is None or current is None:
        return 0

    if current > previous:
        return 1

    if current < previous:
        return -1

    return 0


def _critical_transition(
    previous_rating: Optional[str],
    current_rating: str,
) -> bool:
    """
    Identify transitions that should bypass cooldown.

    Critical examples:
        BUY/STRONG BUY -> HOLD/REDUCE/SELL
        HOLD -> REDUCE/SELL
        REDUCE -> SELL

    Improvement transitions are deliberately not treated as critical
    here; they can still produce IMPORTANT alerts.
    """
    if previous_rating is None:
        return False

    previous = _normalize_rating(previous_rating)
    current = _normalize_rating(current_rating)

    if previous == current:
        return False

    critical_pairs = {
        ("STRONG BUY", "HOLD"),
        ("STRONG BUY", "REDUCE"),
        ("STRONG BUY", "SELL"),
        ("BUY", "HOLD"),
        ("BUY", "REDUCE"),
        ("BUY", "SELL"),
        ("HOLD", "REDUCE"),
        ("HOLD", "SELL"),
        ("REDUCE", "SELL"),
    }

    return (previous, current) in critical_pairs


def _classify_change(
    previous_score: Optional[float],
    current_score: float,
    previous_rating: Optional[str],
    current_rating: str,
) -> Tuple[str, str, bool]:
    """
    Classify a fundamental change.

    Returns:
        alert_class, reason, critical_transition
    """
    if previous_score is None or previous_rating is None:
        return (
            "SILENT",
            "INITIAL_STATE",
            False,
        )

    score_delta = current_score - previous_score
    abs_delta = abs(score_delta)

    rating_changed = _rating_changed(
        previous_rating,
        current_rating,
    )

    critical = _critical_transition(
        previous_rating,
        current_rating,
    )

    if critical:
        return (
            "ACTION",
            (
                f"CRITICAL_RATING_CHANGE:"
                f"{previous_rating}->{current_rating}"
            ),
            True,
        )

    # Any rating transition is more meaningful than a small score drift.
    if rating_changed:
        direction = _rating_direction(
            previous_rating,
            current_rating,
        )

        if direction < 0:
            return (
                "ACTION",
                (
                    f"RATING_DETERIORATION:"
                    f"{previous_rating}->{current_rating}"
                ),
                False,
            )

        return (
            "IMPORTANT",
            (
                f"RATING_IMPROVEMENT:"
                f"{previous_rating}->{current_rating}"
            ),
            False,
        )

    if abs_delta >= DEFAULT_SCORE_DELTA:
        if score_delta < 0:
            return (
                "IMPORTANT",
                f"SCORE_DETERIORATION:{score_delta:+.1f}",
                False,
            )

        return (
            "IMPORTANT",
            f"SCORE_IMPROVEMENT:{score_delta:+.1f}",
            False,
        )

    return (
        "SILENT",
        "NO_SIGNIFICANT_CHANGE",
        False,
    )


def _cooldown_active(
    state: FundamentalState,
    cooldown_seconds: int,
    now_timestamp: Optional[float] = None,
) -> bool:
    if cooldown_seconds <= 0:
        return False

    last_alert = _parse_timestamp(
        state.last_alert_at
    )

    if last_alert is None:
        return False

    current_time = (
        time.time()
        if now_timestamp is None
        else now_timestamp
    )

    return (
        current_time - last_alert
        < cooldown_seconds
    )


def _effective_score_delta() -> float:
    return _env_float(
        "AI_TRADER_FUNDAMENTAL_ALERT_SCORE_DELTA",
        DEFAULT_SCORE_DELTA,
    )


def _effective_cooldown() -> int:
    return max(
        0,
        _env_int(
            "AI_TRADER_FUNDAMENTAL_ALERT_COOLDOWN_SECONDS",
            DEFAULT_COOLDOWN_SECONDS,
        ),
    )


def _dry_run() -> bool:
    return _env_bool(
        "AI_TRADER_FUNDAMENTAL_ALERT_DRY_RUN",
        False,
    )


def _classify_with_config(
    previous_score: Optional[float],
    current_score: float,
    previous_rating: Optional[str],
    current_rating: str,
) -> Tuple[str, str, bool]:
    """
    Same classifier as _classify_change(), but uses the configured
    score-delta threshold.
    """
    if previous_score is None or previous_rating is None:
        return (
            "SILENT",
            "INITIAL_STATE",
            False,
        )

    score_delta = current_score - previous_score
    abs_delta = abs(score_delta)

    rating_changed = _rating_changed(
        previous_rating,
        current_rating,
    )

    critical = _critical_transition(
        previous_rating,
        current_rating,
    )

    if critical:
        return (
            "ACTION",
            (
                f"CRITICAL_RATING_CHANGE:"
                f"{previous_rating}->{current_rating}"
            ),
            True,
        )

    if rating_changed:
        direction = _rating_direction(
            previous_rating,
            current_rating,
        )

        if direction < 0:
            return (
                "ACTION",
                (
                    f"RATING_DETERIORATION:"
                    f"{previous_rating}->{current_rating}"
                ),
                False,
            )

        return (
            "IMPORTANT",
            (
                f"RATING_IMPROVEMENT:"
                f"{previous_rating}->{current_rating}"
            ),
            False,
        )

    if abs_delta >= _effective_score_delta():
        if score_delta < 0:
            return (
                "IMPORTANT",
                f"SCORE_DETERIORATION:{score_delta:+.1f}",
                False,
            )

        return (
            "IMPORTANT",
            f"SCORE_IMPROVEMENT:{score_delta:+.1f}",
            False,
        )

    return (
        "SILENT",
        "NO_SIGNIFICANT_CHANGE",
        False,
    )


def evaluate(
    result: Dict[str, Any],
    state: Optional[Dict[str, FundamentalState]] = None,
    path: Optional[os.PathLike[str] | str] = None,
    now_timestamp: Optional[float] = None,
    persist: bool = True,
) -> AlertDecision:
    """
    Evaluate one Fundamental Analysis result.

    Parameters
    ----------
    result:
        Output from calculate_fundamental_score().

    state:
        Optional already-loaded state. If omitted, state is loaded from disk.

    path:
        Optional custom state path.

    now_timestamp:
        Optional timestamp used by tests.

    persist:
        Persist state when True.

    Returns
    -------
    AlertDecision
    """
    if state is None:
        state = load_state(path)

    symbol = _normalize_symbol(
        result.get("symbol")
    )

    current_score = _safe_number(
        result.get("total")
    )

    current_rating = _normalize_rating(
        result.get("rating")
    )

    previous = state.get(symbol)

    previous_score = (
        None
        if previous is None
        else previous.score
    )

    previous_rating = (
        None
        if previous is None
        else previous.rating
    )

    score_delta = (
        None
        if previous_score is None
        else current_score - previous_score
    )

    rating_changed = _rating_changed(
        previous_rating,
        current_rating,
    )

    alert_class, reason, critical = _classify_with_config(
        previous_score,
        current_score,
        previous_rating,
        current_rating,
    )

    cooldown = False

    if previous is not None:
        cooldown = _cooldown_active(
            previous,
            _effective_cooldown(),
            now_timestamp,
        )

    # Initial state establishes baseline but never alerts.
    should_alert = (
        alert_class != "SILENT"
        and (
            not cooldown
            or critical
        )
    )

    if cooldown and not critical:
        reason = (
            f"{reason}:COOLDOWN"
        )
        should_alert = False

    current_time = (
        _now_iso()
        if now_timestamp is None
        else datetime.fromtimestamp(
            now_timestamp,
            tz=timezone.utc,
        ).isoformat()
    )

    new_state = FundamentalState(
        symbol=symbol,
        score=current_score,
        rating=current_rating,
        revenue=_safe_number(
            result.get("revenue")
        ),
        margins=_safe_number(
            result.get("margins")
        ),
        debt=_safe_number(
            result.get("debt")
        ),
        updated_at=current_time,
        last_alert_at=(
            previous.last_alert_at
            if previous is not None
            else None
        ),
        last_alert_class=(
            previous.last_alert_class
            if previous is not None
            else None
        ),
        last_alert_score=(
            previous.last_alert_score
            if previous is not None
            else None
        ),
        last_alert_rating=(
            previous.last_alert_rating
            if previous is not None
            else None
        ),
    )

    if should_alert:
        new_state.last_alert_at = current_time
        new_state.last_alert_class = alert_class
        new_state.last_alert_score = current_score
        new_state.last_alert_rating = current_rating

    state[symbol] = new_state

    state_updated = False

    if persist:
        state_updated = save_state(
            state,
            path,
        )

    return AlertDecision(
        symbol=symbol,
        alert=should_alert,
        alert_class=alert_class,
        reason=reason,
        score=current_score,
        previous_score=previous_score,
        score_delta=score_delta,
        rating=current_rating,
        previous_rating=previous_rating,
        rating_changed=rating_changed,
        critical_transition=critical,
        cooldown_active=cooldown,
        state_updated=state_updated,
    )


def build_alert_message(
    result: Dict[str, Any],
    decision: AlertDecision,
) -> str:
    """
    Build a concise human-readable Telegram message.

    The message describes the detected change; it does not create a
    separate trading recommendation.
    """
    current_score = _safe_number(
        result.get("total")
    )

    current_rating = _normalize_rating(
        result.get("rating")
    )

    previous_score = decision.previous_score
    previous_rating = decision.previous_rating

    if previous_score is None:
        score_line = (
            f"Score: {current_score:.1f}/55"
        )
    else:
        score_line = (
            f"Score: {previous_score:.1f} "
            f"→ {current_score:.1f}/55 "
            f"({decision.score_delta:+.1f})"
        )

    if previous_rating is None:
        rating_line = (
            f"Rating: {current_rating}"
        )
    else:
        rating_line = (
            f"Rating: {previous_rating} "
            f"→ {current_rating}"
        )

    icon = {
        "ACTION": "🚨",
        "IMPORTANT": "🟠",
        "WATCH": "🟡",
        "SILENT": "⚪",
    }.get(
        decision.alert_class,
        "ℹ️",
    )

    return (
        f"{icon} FUNDAMENTAL CHANGE\n\n"
        f"Ticker: {decision.symbol}\n\n"
        f"{score_line}\n"
        f"{rating_line}\n\n"
        f"Revenue: {result.get('revenue', 0)}/25\n"
        f"Margins: {result.get('margins', 0)}/15\n"
        f"Debt: {result.get('debt', 0)}/15\n\n"
        f"Reason: {decision.reason}"
    )


def process_result(
    result: Dict[str, Any],
    send: bool = False,
    state: Optional[Dict[str, FundamentalState]] = None,
    path: Optional[os.PathLike[str] | str] = None,
) -> AlertDecision:
    """
    Convenience API.

    When send=False, only evaluate and persist state.

    When send=True and the decision is actionable, send through the
    existing Telegram API.
    """
    decision = evaluate(
        result,
        state=state,
        path=path,
    )

    if not decision.alert:
        print(
            f"[FUNDAMENTAL-ALERT] "
            f"{decision.symbol} | SILENT | "
            f"{decision.reason}"
        )
        return decision

    message = build_alert_message(
        result,
        decision,
    )

    if _dry_run() or not send:
        print(
            f"[FUNDAMENTAL-ALERT] "
            f"DRY-RUN | {decision.alert_class} | "
            f"{message.replace(chr(10), ' | ')}"
        )
        return decision

    try:
        from app.alerts.telegram_alert import (
            send_telegram,
        )

        telegram_ok = send_telegram(
            message
        )

        if telegram_ok:
            print(
                f"[FUNDAMENTAL-ALERT] "
                f"Telegram sent | "
                f"{decision.symbol} | "
                f"{decision.alert_class}"
            )
        else:
            print(
                f"[FUNDAMENTAL-ALERT] "
                f"Telegram failed | "
                f"{decision.symbol}"
            )

    except Exception as exc:
        print(
            f"[FUNDAMENTAL-ALERT] "
            f"Telegram error | "
            f"{decision.symbol} | "
            f"{type(exc).__name__}: {exc}"
        )

    return decision


def reset_state(
    path: Optional[os.PathLike[str] | str] = None,
) -> bool:
    """Delete the persisted state file."""
    state_path = Path(path) if path else default_state_path()

    try:
        if state_path.exists():
            state_path.unlink()

        return True

    except Exception as exc:
        print(
            f"[FUNDAMENTAL-ALERT] "
            f"State reset error: "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def config_summary() -> Dict[str, Any]:
    """Return effective configuration."""
    return {
        "version": VERSION,
        "state_path": str(
            default_state_path()
        ),
        "cooldown_seconds": _effective_cooldown(),
        "score_delta": _effective_score_delta(),
        "dry_run": _dry_run(),
    }


def _self_test() -> int:
    """
    Deterministic self-test.

    Uses a temporary state file and never sends Telegram.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        state_path = (
            Path(temp_dir)
            / "fundamental_alert_state.json"
        )

        base = {
            "symbol": "NVDA",
            "type": "STOCK",
            "revenue": 20,
            "margins": 14,
            "debt": 11,
            "total": 45,
            "rating": "STRONG BUY",
        }

        state = {}

        first = evaluate(
            base,
            state=state,
            path=state_path,
        )

        assert first.alert is False
        assert first.reason == "INITIAL_STATE"

        unchanged = dict(base)

        second = evaluate(
            unchanged,
            state=state,
            path=state_path,
        )

        assert second.alert is False
        assert second.score_delta == 0

        improved = dict(base)
        improved["total"] = 50

        third = evaluate(
            improved,
            state=state,
            path=state_path,
        )

        assert third.alert is True
        assert third.alert_class == "IMPORTANT"
        assert third.score_delta == 5

        cooldown_state = load_state(
            state_path
        )

        fourth = evaluate(
            {
                **improved,
                "total": 51,
            },
            state=cooldown_state,
            path=state_path,
        )

        assert fourth.alert is False
        assert fourth.cooldown_active is True

        deteriorated = {
            **improved,
            "total": 32,
            "rating": "HOLD",
        }

        fifth = evaluate(
            deteriorated,
            state=cooldown_state,
            path=state_path,
        )

        assert fifth.alert is True
        assert fifth.alert_class == "ACTION"
        assert fifth.critical_transition is True

        print(
            "Fundamental Alert Engine V1 self-test: PASS"
        )
        print(
            f"Version: {VERSION}"
        )
        print(
            f"Config: {config_summary()}"
        )

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=MODULE_NAME
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic self-test.",
    )

    parser.add_argument(
        "--config",
        action="store_true",
        help="Print effective configuration.",
    )

    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Reset persisted alert state.",
    )

    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(
            _self_test()
        )

    if args.config:
        print(
            json.dumps(
                config_summary(),
                indent=2,
                ensure_ascii=False,
            )
        )

    if args.reset_state:
        print(
            "State reset:",
            reset_state(),
        )
