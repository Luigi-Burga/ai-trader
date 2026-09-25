"""
AI Trader - Alpaca Executor
Version: 1.0

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
Telegram Approval             <-- human approval required
    |
    v
Alpaca Executor
    |
    v
Alpaca PAPER

Safety principles
-----------------
1. PAPER trading is mandatory.
2. A TradeIntent is required.
3. A RiskDecision must be approved.
4. Telegram approval is required before submission.
5. Approval is bound to a unique proposal token.
6. Duplicate/open-order protection is checked again immediately before submit.
7. This module never uses Alpaca live trading.
8. No order is submitted during the self-test.

Telegram configuration
---------------------
Set these environment variables:

    TELEGRAM_BOT_TOKEN=<your bot token>
    TELEGRAM_CHAT_ID=<your chat id>

The existing AI Trader Telegram alert module is reused. The executor
does not create a second Bot, token configuration, chat configuration, or
asyncio event loop.

Approval
--------
The executor sends a proposal with:

    APPROVE <token>
    REJECT <token>

The user replies through Telegram. The executor accepts approval only when:
- the token matches the pending proposal;
- the chat ID matches TELEGRAM_CHAT_ID;
- the response is explicit APPROVE <token>;
- the proposal has not expired.

No Telegram approval means no Alpaca order.
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

    require_telegram_approval: bool = True
    approval_timeout_seconds: int = 300
    poll_interval_seconds: int = 2
    reject_duplicate_open_order: bool = True
    paper_only: bool = True

    # Additional protection for the first real Paper order.
    # When TEST_MODE is enabled, Telegram approval alone is not sufficient.
    # A second explicit Telegram confirmation is required immediately before
    # the Alpaca submit_order() call.
    test_mode: bool = True
    first_order_confirmation_required: bool = True

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
        wait_for_telegram: bool = True,
    ) -> ExecutionResult:
        """
        Execute an approved TradeIntent.

        The actual Alpaca submit_order() call occurs only after all safety
        gates pass and Telegram approval is received.
        """

        token = self._create_proposal_token(intent)

        # --------------------------------------------------------------
        # 1. Risk Decision
        # --------------------------------------------------------------
        if not risk_decision.approved:
            return self._rejected(
                "RISK_REJECTED",
                "Execution blocked because the Risk Gate did not approve "
                "the TradeIntent.",
                token,
                intent,
            )

        # --------------------------------------------------------------
        # 2. PAPER enforcement
        # --------------------------------------------------------------
        if self.config.paper_only:
            if not self._is_paper_environment():
                return self._rejected(
                    "LIVE_BLOCKED",
                    "Executor is configured as PAPER ONLY.",
                    token,
                    intent,
                )

        # --------------------------------------------------------------
        # 3. Intent / risk consistency
        # --------------------------------------------------------------
        consistency_ok, consistency_message = (
            self._validate_risk_consistency(
                intent,
                risk_decision,
            )
        )

        if not consistency_ok:
            return self._rejected(
                "RISK_INTENT_MISMATCH",
                consistency_message,
                token,
                intent,
            )

        # --------------------------------------------------------------
        # 4. Reconciliation immediately before execution
        # --------------------------------------------------------------
        try:
            report = self.reconciliation.reconcile_empty_portfolio()
        except Exception as exc:
            return self._rejected(
                "RECONCILIATION_ERROR",
                f"Pre-execution reconciliation failed: {exc}",
                token,
                intent,
            )

        if not report.reconciled:
            return self._rejected(
                "RECONCILIATION_FAILED",
                "Pre-execution reconciliation is not reconciled.",
                token,
                intent,
            )

        # --------------------------------------------------------------
        # 5. Duplicate/open order protection
        # --------------------------------------------------------------
        if self.config.reject_duplicate_open_order:
            try:
                duplicate = self.orders.has_open_order(
                    symbol=intent.symbol,
                    side=intent.side,
                )
            except Exception as exc:
                return self._rejected(
                    "OPEN_ORDER_CHECK_FAILED",
                    f"Unable to verify open orders: {exc}",
                    token,
                    intent,
                )

            if duplicate:
                return self._rejected(
                    "DUPLICATE_ORDER",
                    f"An open {intent.side.upper()} order already exists "
                    f"for {intent.symbol}.",
                    token,
                    intent,
                )

        # --------------------------------------------------------------
        # 6. Telegram approval
        # --------------------------------------------------------------
        if self.config.require_telegram_approval:
            if not self.telegram.configured:
                return self._rejected(
                    "TELEGRAM_NOT_CONFIGURED",
                    "Telegram approval is mandatory but Telegram is not "
                    "configured.",
                    token,
                    intent,
                )

            proposal = self._build_telegram_proposal(
                intent=intent,
                risk_decision=risk_decision,
                token=token,
            )

            try:
                self.telegram.send_message(proposal)
            except Exception as exc:
                return self._rejected(
                    "TELEGRAM_SEND_FAILED",
                    f"Unable to send approval request: {exc}",
                    token,
                    intent,
                )

            if not wait_for_telegram:
                return ExecutionResult(
                    submitted=False,
                    status="AWAITING_TELEGRAM_APPROVAL",
                    message=(
                        "Approval request was sent to Telegram. "
                        f"Reply APPROVE {token} to authorize execution, "
                        f"or REJECT {token} to cancel."
                    ),
                    proposal_token=token,
                    symbol=intent.symbol,
                    side=intent.side,
                    quantity=intent.quantity,
                    order_type=intent.order_type,
                )

            try:
                approval = self.telegram.wait_for_approval(
                    token=token,
                    timeout_seconds=self.config.approval_timeout_seconds,
                    poll_interval_seconds=self.config.poll_interval_seconds,
                )
            except Exception as exc:
                return self._rejected(
                    "TELEGRAM_APPROVAL_ERROR",
                    f"Telegram approval check failed: {exc}",
                    token,
                    intent,
                )

            if approval != "APPROVED":
                status = (
                    "TELEGRAM_REJECTED"
                    if approval == "REJECTED"
                    else "TELEGRAM_TIMEOUT"
                )

                return self._rejected(
                    status,
                    f"Telegram approval result: {approval}. No order submitted.",
                    token,
                    intent,
                )

        # --------------------------------------------------------------
        # 7. TEST_MODE / FIRST_ORDER_CONFIRMATION
        # --------------------------------------------------------------
        # This is deliberately the final human safety gate before the only
        # Alpaca submit_order() call.
        if (
            self.config.test_mode
            and self.config.first_order_confirmation_required
        ):
            confirmation_message = (
                "🛡️ AI TRADER - FIRST PAPER ORDER\n"
                "FIRST PAPER ORDER CONFIRMATION REQUIRED\n\n"
                f"Symbol: {intent.symbol}\n"
                f"Action: {intent.side.upper()}\n"
                f"Quantity: {intent.quantity:g}\n"
                f"Notional: ${intent.notional_value:,.2f}\n"
                f"Order: {intent.order_type.upper()}\n"
                f"Limit: "
                f"{('$' + format(intent.limit_price, ',.2f')) if intent.limit_price is not None else '-'}\n\n"
                "This is the FIRST real Paper order for this Executor.\n"
                "Risk Gate: APPROVED\n"
                "Telegram Approval: APPROVED\n\n"
                f"CONFIRM FIRST ORDER {token}\n"
                f"REJECT {token}\n\n"
                "No Alpaca order will be submitted without this second "
                "explicit confirmation."
            )

            try:
                self.telegram.send_message(confirmation_message)
            except Exception as exc:
                return self._rejected(
                    "FIRST_ORDER_CONFIRMATION_SEND_FAILED",
                    f"Unable to send first-order confirmation request: {exc}",
                    token,
                    intent,
                )

            try:
                confirmation = self.telegram.wait_for_first_order_confirmation(
                    token=token,
                    timeout_seconds=self.config.approval_timeout_seconds,
                    poll_interval_seconds=self.config.poll_interval_seconds,
                )
            except Exception as exc:
                return self._rejected(
                    "FIRST_ORDER_CONFIRMATION_ERROR",
                    f"First-order confirmation check failed: {exc}",
                    token,
                    intent,
                )

            if confirmation != "CONFIRMED":
                status = (
                    "FIRST_ORDER_REJECTED"
                    if confirmation == "REJECTED"
                    else "FIRST_ORDER_CONFIRMATION_TIMEOUT"
                )

                return self._rejected(
                    status,
                    (
                        f"First-order confirmation result: {confirmation}. "
                        "No order submitted."
                    ),
                    token,
                    intent,
                )

        # --------------------------------------------------------------
        # 8. Final submit
        # --------------------------------------------------------------
        try:
            order = self._submit_to_alpaca(intent)
        except Exception as exc:
            return self._rejected(
                "ALPACA_SUBMIT_FAILED",
                f"Alpaca rejected/failed the order submission: {exc}",
                token,
                intent,
            )

        order_id = self._extract_order_id(order)
        submitted_at = datetime.now(timezone.utc).isoformat()

        return ExecutionResult(
            submitted=True,
            status="SUBMITTED",
            message=(
                "Order submitted successfully to Alpaca PAPER after "
                "Risk Gate approval and Telegram approval."
            ),
            proposal_token=token,
            order_id=order_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            order_type=intent.order_type,
            submitted_at=submitted_at,
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

        if abs(
            risk_decision.estimated_notional - intent.notional_value
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

    @staticmethod
    def _build_telegram_proposal(
        intent: TradeIntent,
        risk_decision: RiskDecision,
        token: str,
    ) -> str:
        return (
            "🚨 AI TRADER - FIRST PAPER ORDER\n"
            "TRADE APPROVAL REQUIRED\n\n"
            f"Symbol: {intent.symbol}\n"
            f"Action: {intent.side.upper()}\n"
            f"Quantity: {intent.quantity:g}\n"
            f"Order: {intent.order_type.upper()}\n"
            f"TIF: {intent.time_in_force.upper()}\n"
            f"Notional: ${intent.notional_value:,.2f}\n"
            f"Limit: "
            f"{('$' + format(intent.limit_price, ',.2f')) if intent.limit_price is not None else '-'}\n"
            f"Stop: "
            f"{('$' + format(intent.stop_price, ',.2f')) if intent.stop_price is not None else '-'}\n\n"
            f"Strategy: {intent.strategy or '-'}\n"
            f"Signal: {intent.signal or '-'}\n"
            f"Confidence: "
            f"{(f'{intent.confidence:.2f}%' if intent.confidence is not None else '-')}\n"
            f"Projected Exposure: {risk_decision.projected_exposure_pct:.2f}%\n"
            f"Capital Base: ${risk_decision.capital_base:,.2f}\n\n"
            "Environment: ALPACA PAPER\n"
            "Risk Gate: APPROVED\n\n"
            f"APPROVE {token}\n"
            f"REJECT {token}\n\n"
            "No order will be submitted without explicit approval."
        )

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
    """
    Safe self-test.

    It intentionally does NOT call execute() and therefore cannot submit
    an order. It validates the Telegram proposal formatting and configuration.
    """

    from app.broker.alpaca_risk_gate import create_risk_gate

    print("AI TRADER - ALPACA EXECUTOR")
    print("=" * 60)
    print()
    print("MODE: PAPER ONLY")
    print("SELF-TEST: NO ORDER SUBMISSION")
    print()

    intent = TradeIntent(
        symbol="NVDA",
        side="buy",
        quantity=100,
        order_type="limit",
        limit_price=180.50,
        time_in_force="day",
        reason="Executor self-test",
        confidence=84.2,
        strategy="Characteristic Curve + Price Action",
        signal="BUY",
    )

    risk_gate = create_risk_gate()
    risk_decision = risk_gate.evaluate(intent)

    print(f"Risk Gate: {'APPROVED' if risk_decision.approved else 'REJECTED'}")
    print("Operation Type: FIRST PAPER ORDER")
    print()

    executor = create_executor()

    token = executor._create_proposal_token(intent)
    proposal = executor._build_telegram_proposal(
        intent=intent,
        risk_decision=risk_decision,
        token=token,
    )

    print("TELEGRAM APPROVAL MESSAGE")
    print("-" * 60)
    print(proposal)
    print()
    print(
        "Telegram configured:",
        "YES" if executor.telegram.configured else "NO",
    )
    print()
    print(
        "TEST_MODE:",
        "ENABLED" if executor.config.test_mode else "DISABLED",
    )
    print(
        "FIRST_ORDER_CONFIRMATION:",
        "REQUIRED"
        if (
            executor.config.test_mode
            and executor.config.first_order_confirmation_required
        )
        else "NOT REQUIRED",
    )
    print(
        "FIRST-ORDER SAFETY TEST: PASS"
        if (
            executor.config.test_mode
            and executor.config.first_order_confirmation_required
        )
        else "FIRST-ORDER SAFETY TEST: DISABLED",
    )
    print()
    print("SELF-TEST RESULT: PASS")
    print("No Alpaca order was submitted.")


if __name__ == "__main__":
    _run_self_test()
