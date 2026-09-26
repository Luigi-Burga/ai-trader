"""
AI Trader - Alpaca Executor
Version: 2.1
Mode: AUTONOMOUS PAPER

Purpose
-------
Controlled execution layer for Alpaca Paper Trading.

Execution flow
--------------
Decision Gate
    |
    v
Trade Intent
    |
    v
Risk Gate
    |
    v
Alpaca Executor
    |
    v
Telegram Notification          <-- informational only
    |
    v
Alpaca PAPER

Safety principles
-----------------
1. PAPER trading is mandatory.
2. A TradeIntent is required.
3. A RiskDecision must be approved.
4. Telegram is notification-only and never authorizes/rejects execution.
5. Duplicate/open-order protection is checked again immediately before submit.
6. This module never uses Alpaca live trading.
7. No order is submitted during the self-test.
8. Telegram delivery failure never changes an already-submitted order result.

Telegram configuration
---------------------
Set these environment variables:

    TELEGRAM_BOT_TOKEN=<your bot token>
    TELEGRAM_CHAT_ID=<your chat id>

The existing AI Trader Telegram alert module is reused. The executor
does not create a second Bot, token configuration, chat configuration, or
asyncio event loop.

Autonomous execution
--------------------
In AUTONOMOUS PAPER mode, an approved RiskDecision is sufficient to
execute after all execution-time safety checks pass. Telegram is used only
to notify the result after submission.

"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
)

from app.broker.alpaca_client import AlpacaClient, create_alpaca_client
from app.broker.alpaca_orders import AlpacaOrders
from app.broker.alpaca_reconciliation import AlpacaReconciliation
from app.broker.alpaca_risk_gate import RiskDecision
from app.broker.alpaca_trade_intent import TradeIntent


# ---------------------------------------------------------------------------
# Executor configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutorConfig:
    """Safety configuration for the execution layer."""

    # AUTONOMOUS PAPER: Telegram never authorizes execution.
    require_telegram_approval: bool = False
    approval_timeout_seconds: int = 300
    poll_interval_seconds: int = 2
    reject_duplicate_open_order: bool = True
    paper_only: bool = True

    # Legacy human-confirmation switches are retained for compatibility,
    # but are disabled in AUTONOMOUS PAPER mode.
    test_mode: bool = False
    first_order_confirmation_required: bool = False

    # Informational Telegram notification after an execution attempt.
    telegram_notifications: bool = True

    def __post_init__(self) -> None:
        if self.approval_timeout_seconds <= 0:
            raise ValueError(
                "approval_timeout_seconds must be greater than zero."
            )
        if self.poll_interval_seconds <= 0:
            raise ValueError(
                "poll_interval_seconds must be greater than zero."
            )


@dataclass(frozen=True)
class ExecutionResult:
    """Auditable result of an execution attempt."""

    submitted: bool
    status: str
    message: str
    proposal_token: Optional[str] = None
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    quantity: Optional[float] = None
    order_type: Optional[str] = None
    submitted_at: Optional[str] = None

    def summary(self) -> str:
        lines = [
            "ALPACA EXECUTOR",
            "=" * 60,
            "",
            f"Status:             {self.status}",
            f"Submitted:          {self.submitted}",
            f"Proposal Token:     {self.proposal_token or '-'}",
            f"Order ID:           {self.order_id or '-'}",
            f"Symbol:             {self.symbol or '-'}",
            f"Side:               {self.side.upper() if self.side else '-'}",
            f"Quantity:           {self.quantity:g}" if self.quantity is not None else "Quantity:           -",
            f"Order Type:         {self.order_type.upper() if self.order_type else '-'}",
            f"Submitted At:       {self.submitted_at or '-'}",
            "",
            self.message,
        ]
        return "\n".join(lines)


class TelegramApproval:
    """
    Adapter to AI Trader's existing app.alerts.telegram_alert module.

    No second bot, token, chat ID, or Telegram event loop is created.
    """

    def __init__(self) -> None:
        from app.alerts.telegram_alert import telegram_configured

        self._telegram_configured = telegram_configured

    @property
    def configured(self) -> bool:
        return bool(self._telegram_configured())

    def send_message(self, text: str) -> bool:
        from app.alerts.telegram_alert import send_telegram

        return bool(send_telegram(text))

    def wait_for_approval(
        self,
        token: str,
        timeout_seconds: int,
        poll_interval_seconds: int,
    ) -> str:
        from app.alerts.telegram_alert import telegram_wait_for_approval

        return telegram_wait_for_approval(
            token=token,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def wait_for_first_order_confirmation(
        self,
        token: str,
        timeout_seconds: int,
        poll_interval_seconds: int,
    ) -> str:
        from app.alerts.telegram_alert import telegram_wait_for_confirmation

        return telegram_wait_for_confirmation(
            token=token,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )


class AlpacaExecutor:
    """
    Controlled Alpaca Paper Trading executor.

    This is the first module in the project that can submit an order.
    It will never submit without an approved RiskDecision and Telegram
    approval when configured to require it.
    """

    def __init__(
        self,
        client: Optional[AlpacaClient] = None,
        orders: Optional[AlpacaOrders] = None,
        reconciliation: Optional[AlpacaReconciliation] = None,
        telegram: Optional[TelegramApproval] = None,
        config: Optional[ExecutorConfig] = None,
    ) -> None:
        self.client = client or create_alpaca_client()
        self.orders = orders or AlpacaOrders(client=self.client)
        self.reconciliation = (
            reconciliation
            or AlpacaReconciliation(client=self.client)
        )
        self.telegram = telegram or TelegramApproval()
        self.config = config or ExecutorConfig()

    def execute(
        self,
        intent: TradeIntent,
        risk_decision: RiskDecision,
        *,
        wait_for_telegram: bool = False,
    ) -> ExecutionResult:
        """Execute an approved TradeIntent in AUTONOMOUS PAPER mode.

        Telegram is notification-only. It never blocks, approves, or rejects
        execution. The Alpaca submit_order() call is reached only after every
        execution-time safety gate passes.
        """

        token = self._create_proposal_token(intent)

        # 1. Risk Decision
        if not risk_decision.approved:
            result = self._rejected(
                "RISK_REJECTED",
                "Execution blocked because the Risk Gate did not approve the TradeIntent.",
                token,
                intent,
            )
            self._notify_execution_result(result, risk_decision=risk_decision)
            return result

        # 2. PAPER enforcement
        if self.config.paper_only and not self._is_paper_environment():
            result = self._rejected(
                "LIVE_BLOCKED",
                "Executor is configured as PAPER ONLY.",
                token,
                intent,
            )
            self._notify_execution_result(result, risk_decision=risk_decision)
            return result

        # 3. Intent / risk consistency
        consistency_ok, consistency_message = self._validate_risk_consistency(
            intent, risk_decision
        )
        if not consistency_ok:
            result = self._rejected(
                "RISK_INTENT_MISMATCH", consistency_message, token, intent
            )
            self._notify_execution_result(result, risk_decision=risk_decision)
            return result

        # 4. Reconciliation immediately before execution
        try:
            report = self.reconciliation.reconcile_current_state()
        except Exception as exc:
            result = self._rejected(
                "RECONCILIATION_ERROR",
                f"Pre-execution reconciliation failed: {exc}",
                token,
                intent,
            )
            self._notify_execution_result(result, risk_decision=risk_decision)
            return result

        if not report.reconciled:
            result = self._rejected(
                "RECONCILIATION_FAILED",
                "Pre-execution reconciliation is not reconciled.",
                token,
                intent,
            )
            self._notify_execution_result(result, risk_decision=risk_decision)
            return result

        # 5. Position-state protection for existing portfolios
        # BUY orders may increase an existing long position.
        # SELL orders must be backed by an actual position with sufficient qty.
        if intent.side.lower() == "sell":
            position_ok, position_message = self._validate_sell_position(
                intent=intent,
                report=report,
            )
            if not position_ok:
                result = self._rejected(
                    "POSITION_STATE_INVALID",
                    position_message,
                    token,
                    intent,
                )
                self._notify_execution_result(result, risk_decision=risk_decision)
                return result

        # 6. Duplicate/open order protection
        if self.config.reject_duplicate_open_order:
            try:
                duplicate = self.orders.has_open_order(
                    symbol=intent.symbol, side=intent.side
                )
            except Exception as exc:
                result = self._rejected(
                    "OPEN_ORDER_CHECK_FAILED",
                    f"Unable to verify open orders: {exc}",
                    token,
                    intent,
                )
                self._notify_execution_result(result, risk_decision=risk_decision)
                return result

            if duplicate:
                result = self._rejected(
                    "DUPLICATE_ORDER",
                    f"An open {intent.side.upper()} order already exists for {intent.symbol}.",
                    token,
                    intent,
                )
                self._notify_execution_result(result, risk_decision=risk_decision)
                return result

        # 7. Final submit -- no Telegram authorization is consulted here.
        try:
            order = self._submit_to_alpaca(intent)
        except Exception as exc:
            result = self._rejected(
                "ALPACA_SUBMIT_FAILED",
                f"Alpaca rejected/failed the order submission: {exc}",
                token,
                intent,
            )
            self._notify_execution_result(result, risk_decision=risk_decision)
            return result

        order_id = self._extract_order_id(order)
        submitted_at = datetime.now(timezone.utc).isoformat()

        result = ExecutionResult(
            submitted=True,
            status="SUBMITTED",
            message=(
                "Order submitted successfully to Alpaca PAPER after Risk Gate approval, "
                "dynamic reconciliation, position-state validation, and duplicate-order checks."
            ),
            proposal_token=token,
            order_id=order_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            order_type=intent.order_type,
            submitted_at=submitted_at,
        )

        # 7. Telegram is informational only. A notification failure must not
        # turn a successfully submitted order into a failed execution result.
        self._notify_execution_result(result, risk_decision=risk_decision)
        return result

    # ------------------------------------------------------------------
    # Existing-position safety
    # ------------------------------------------------------------------

    def _validate_sell_position(
        self,
        intent: TradeIntent,
        report: Any,
    ) -> tuple[bool, str]:
        """Ensure a SELL is backed by a sufficient live Alpaca position."""
        symbol = intent.symbol.upper()

        try:
            positions = self.client.get_positions()
        except Exception as exc:
            return False, f"Unable to verify existing position for {symbol}: {exc}"

        for position in positions:
            if str(position.get("symbol", "")).upper() != symbol:
                continue

            side = str(position.get("side", "long")).lower()
            if "short" in side:
                return (
                    False,
                    f"Cannot SELL {symbol}: existing position is SHORT; "
                    "short-position handling is not enabled in this Executor.",
                )

            available_qty = float(position.get("qty") or 0.0)
            if available_qty + 1e-8 < float(intent.quantity):
                return (
                    False,
                    f"Cannot SELL {symbol}: requested {intent.quantity:g} shares "
                    f"but only {available_qty:g} are held.",
                )

            return True, "Existing position and quantity are valid for SELL."

        return (
            False,
            f"Cannot SELL {symbol}: no existing position was found in Alpaca.",
        )

    # ------------------------------------------------------------------
    # Alpaca submission
    # ------------------------------------------------------------------

    def _submit_to_alpaca(self, intent: TradeIntent) -> Any:
        """Build the appropriate alpaca-py request and submit it."""

        side = (
            OrderSide.BUY
            if intent.side.lower() == "buy"
            else OrderSide.SELL
        )

        tif = TimeInForce(intent.time_in_force.lower())

        if intent.order_type == "market":
            request = MarketOrderRequest(
                symbol=intent.symbol,
                qty=intent.quantity,
                side=side,
                time_in_force=tif,
            )

        elif intent.order_type == "limit":
            request = LimitOrderRequest(
                symbol=intent.symbol,
                qty=intent.quantity,
                side=side,
                time_in_force=tif,
                limit_price=intent.limit_price,
            )

        elif intent.order_type == "stop":
            request = StopOrderRequest(
                symbol=intent.symbol,
                qty=intent.quantity,
                side=side,
                time_in_force=tif,
                stop_price=intent.stop_price,
            )

        elif intent.order_type == "stop_limit":
            request = StopLimitOrderRequest(
                symbol=intent.symbol,
                qty=intent.quantity,
                side=side,
                time_in_force=tif,
                limit_price=intent.limit_price,
                stop_price=intent.stop_price,
            )

        else:
            raise ValueError(
                f"Unsupported order type: {intent.order_type}"
            )

        # This is the ONLY actual order-submission call in this module.
        return self.client._client.submit_order(order_data=request)

    # ------------------------------------------------------------------
    # Validation / safety
    # ------------------------------------------------------------------

    def _is_paper_environment(self) -> bool:
        health = self.client.health_check()
        environment = str(
            health.get("environment", "")
        ).upper()

        return environment == "PAPER"

    @staticmethod
    def _validate_risk_consistency(
        intent: TradeIntent,
        risk_decision: RiskDecision,
    ) -> tuple[bool, str]:
        if risk_decision.symbol != intent.symbol:
            return False, "Risk decision symbol does not match TradeIntent."

        if risk_decision.side != intent.side:
            return False, "Risk decision side does not match TradeIntent."

        if risk_decision.quantity != intent.quantity:
            return False, "Risk decision quantity does not match TradeIntent."

        if intent.order_type == "market":
            if risk_decision.estimated_notional <= 0:
                return False, "Market order has no valid Risk Gate estimated notional."
            return True, "Risk decision matches TradeIntent; market notional is risk-estimated from current price."

        intent_notional = intent.notional_value
        if intent_notional is None:
            return False, "TradeIntent notional is missing for non-market order."

        if abs(
            risk_decision.estimated_notional - intent_notional
        ) > 0.01:
            return False, "Risk decision notional does not match TradeIntent."

        return True, "Risk decision matches TradeIntent."

    @staticmethod
    def _create_proposal_token(intent: TradeIntent) -> str:
        """Short unique token bound to this proposal."""
        prefix = intent.symbol.upper()
        return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"

    @staticmethod
    def _extract_order_id(order: Any) -> Optional[str]:
        if order is None:
            return None

        order_id = getattr(order, "id", None)
        if order_id is None:
            return None

        return str(order_id)

    def _notify_execution_result(
        self,
        result: ExecutionResult,
        *,
        risk_decision: Optional[RiskDecision] = None,
    ) -> None:
        """Send an informational Telegram notification only."""
        if not self.config.telegram_notifications:
            return

        try:
            if not self.telegram.configured:
                return

            projected = (
                f"{risk_decision.projected_exposure_pct:.2f}%"
                if risk_decision is not None
                and hasattr(risk_decision, "projected_exposure_pct")
                else "-"
            )
            capital = (
                f"${risk_decision.capital_base:,.2f}"
                if risk_decision is not None
                and hasattr(risk_decision, "capital_base")
                else "-"
            )

            status_title = (
                "🚨 AI TRADER - ORDER EXECUTED"
                if result.submitted
                else "⚠️ AI TRADER - ORDER NOT EXECUTED"
            )

            message = (
                f"{status_title}\n\n"
                f"Symbol: {result.symbol or '-'}\n"
                f"Action: {(result.side or '-').upper()}\n"
                f"Quantity: {result.quantity:g}\n"
                f"Order: {(result.order_type or '-').upper()}\n"
                f"Status: {result.status}\n"
                f"Order ID: {result.order_id or '-'}\n\n"
                f"Projected Exposure: {projected}\n"
                f"Capital Base: {capital}\n\n"
                "Environment: ALPACA PAPER\n"
                f"Proposal Token: {result.proposal_token or '-'}\n\n"
                f"{result.message}"
            )
            self.telegram.send_message(message)
        except Exception:
            # Notification is intentionally non-blocking. The broker result
            # remains authoritative and is returned unchanged.
            return

    @staticmethod
    def _rejected(
        status: str,
        message: str,
        token: str,
        intent: TradeIntent,
    ) -> ExecutionResult:
        return ExecutionResult(
            submitted=False,
            status=status,
            message=message,
            proposal_token=token,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            order_type=intent.order_type,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_executor(
    config: Optional[ExecutorConfig] = None,
) -> AlpacaExecutor:
    """Create the controlled Alpaca Paper executor."""
    return AlpacaExecutor(config=config)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    """Safe AUTONOMOUS PAPER configuration self-test."""
    print("AI TRADER - ALPACA EXECUTOR")
    print("=" * 60)
    print()
    print("MODE: AUTONOMOUS PAPER")
    print("SELF-TEST: NO ORDER SUBMISSION")
    print()

    executor = create_executor()
    config = executor.config

    print("Telegram Approval: DISABLED (notification only)")
    print("First Order Confirmation: DISABLED")
    print("Paper Only: ENABLED" if config.paper_only else "Paper Only: DISABLED")
    print("Duplicate Open-Order Protection: ENABLED" if config.reject_duplicate_open_order else "Duplicate Open-Order Protection: DISABLED")
    print("Telegram Notifications: ENABLED" if config.telegram_notifications else "Telegram Notifications: DISABLED")
    print()

    assert config.require_telegram_approval is False
    assert config.test_mode is False
    assert config.first_order_confirmation_required is False
    assert config.paper_only is True
    assert config.reject_duplicate_open_order is True
    assert config.telegram_notifications is True

    print("AUTONOMOUS PAPER SAFETY CONFIG: PASS")
    print("No Alpaca order was submitted.")


if __name__ == "__main__":
    _run_self_test()
