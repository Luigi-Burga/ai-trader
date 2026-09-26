"""
AI Trader - Alpaca Executor
Version: 2.2

Autonomous Paper Executor + Post Execution Reconciliation.

Safety model
------------
TradeIntent
    -> Risk Gate approval
    -> PAPER enforcement
    -> Intent/Risk consistency
    -> Dynamic pre-execution reconciliation
    -> Position-state validation
    -> Duplicate open-order protection
    -> Alpaca PAPER submit_order()
    -> Post Execution Reconciliation
    -> Telegram notification only

Important
---------
Telegram is NOT an authorization mechanism.
This executor never polls Telegram for approval.

AI Trader strategy capital remains fixed at USD 100,000.
Alpaca buying power is never used as strategy capital.
"""

from __future__ import annotations

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
from app.broker.alpaca_post_execution_reconciliation import (
    AlpacaPostExecutionReconciliation,
    PostExecutionResult,
    create_post_execution_reconciliation,
)


@dataclass(frozen=True)
class ExecutorConfig:
    """Autonomous Paper execution safety configuration."""

    paper_only: bool = True
    reject_duplicate_open_order: bool = True
    position_validation_required: bool = True

    post_execution_wait: bool = True
    post_execution_timeout_seconds: float = 30.0
    post_execution_poll_interval_seconds: float = 2.0
    notify_telegram: bool = True

    def __post_init__(self) -> None:
        if self.post_execution_timeout_seconds <= 0:
            raise ValueError(
                "post_execution_timeout_seconds must be greater than zero."
            )
        if self.post_execution_poll_interval_seconds <= 0:
            raise ValueError(
                "post_execution_poll_interval_seconds must be greater than zero."
            )


@dataclass(frozen=True)
class ExecutionResult:
    """Auditable execution result including post-execution state."""

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
    post_execution: Optional[PostExecutionResult] = None

    def summary(self) -> str:
        lines = [
            "ALPACA EXECUTOR V2.2",
            "=" * 60,
            "",
            f"Status:             {self.status}",
            f"Submitted:          {self.submitted}",
            f"Order ID:           {self.order_id or '-'}",
            f"Symbol:             {self.symbol or '-'}",
            f"Side:               {self.side.upper() if self.side else '-'}",
            (
                f"Quantity:           {self.quantity:g}"
                if self.quantity is not None
                else "Quantity:           -"
            ),
            f"Order Type:         {self.order_type.upper() if self.order_type else '-'}",
            f"Submitted At:       {self.submitted_at or '-'}",
            "",
            self.message,
        ]

        if self.post_execution is not None:
            lines.extend(
                [
                    "",
                    "-" * 60,
                    self.post_execution.summary(),
                ]
            )

        return "\n".join(lines)


class TelegramNotifier:
    """Adapter to the existing Telegram notification module."""

    def __init__(self) -> None:
        from app.alerts.telegram_alert import telegram_configured

        self._telegram_configured = telegram_configured

    @property
    def configured(self) -> bool:
        return bool(self._telegram_configured())

    def send_message(self, text: str) -> bool:
        from app.alerts.telegram_alert import send_telegram

        return bool(send_telegram(text))


class AlpacaExecutor:
    """
    Autonomous Alpaca Paper executor.

    Telegram is notification-only.
    There is no Telegram approval or first-order confirmation gate.
    """

    def __init__(
        self,
        client: Optional[AlpacaClient] = None,
        orders: Optional[AlpacaOrders] = None,
        reconciliation: Optional[AlpacaReconciliation] = None,
        telegram: Optional[TelegramNotifier] = None,
        post_execution: Optional[AlpacaPostExecutionReconciliation] = None,
        config: Optional[ExecutorConfig] = None,
    ) -> None:
        self.client = client or create_alpaca_client()
        self.orders = orders or AlpacaOrders(client=self.client)
        self.reconciliation = (
            reconciliation or AlpacaReconciliation(client=self.client)
        )
        self.telegram = telegram or TelegramNotifier()
        self.config = config or ExecutorConfig()

        self.post_execution = post_execution or create_post_execution_reconciliation(
            client=self.client,
            reconciliation=self.reconciliation,
            notifier=(
                self.telegram.send_message
                if self.config.notify_telegram
                else None
            ),
        )

    def execute(
        self,
        intent: TradeIntent,
        risk_decision: RiskDecision,
    ) -> ExecutionResult:
        """Execute an approved TradeIntent autonomously in Alpaca PAPER."""

        token = self._create_proposal_token(intent)

        # 1. Risk Gate
        if not risk_decision.approved:
            return self._rejected(
                "RISK_REJECTED",
                "Execution blocked because Risk Gate did not approve the TradeIntent.",
                token,
                intent,
            )

        # 2. PAPER enforcement
        if self.config.paper_only and not self._is_paper_environment():
            return self._rejected(
                "LIVE_BLOCKED",
                "Executor is configured as PAPER ONLY.",
                token,
                intent,
            )

        # 3. Intent/Risk consistency
        consistency_ok, consistency_message = self._validate_risk_consistency(
            intent,
            risk_decision,
        )
        if not consistency_ok:
            return self._rejected(
                "RISK_INTENT_MISMATCH",
                consistency_message,
                token,
                intent,
            )

        # 4. Dynamic pre-execution reconciliation
        try:
            report = self.reconciliation.reconcile_current_state()
        except Exception as exc:
            return self._rejected(
                "RECONCILIATION_ERROR",
                f"Pre-execution reconciliation failed: {exc}",
                token,
                intent,
            )

        if not getattr(report, "reconciled", False):
            return self._rejected(
                "RECONCILIATION_FAILED",
                "Pre-execution dynamic reconciliation is not reconciled.",
                token,
                intent,
            )

        # 5. Position-state validation
        if self.config.position_validation_required:
            position_ok, position_message = self._validate_position_state(intent)
            if not position_ok:
                return self._rejected(
                    "POSITION_VALIDATION_FAILED",
                    position_message,
                    token,
                    intent,
                )

        # 6. Duplicate open-order protection
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
                    (
                        f"An open {intent.side.upper()} order already exists "
                        f"for {intent.symbol}."
                    ),
                    token,
                    intent,
                )

        # 7. Submit ONLY after every gate passes
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
        if not order_id:
            return self._rejected(
                "ORDER_ID_MISSING",
                (
                    "Alpaca returned a submission response without an order "
                    "ID. No post-execution reconciliation is possible."
                ),
                token,
                intent,
            )

        submitted_at = datetime.now(timezone.utc).isoformat()

        # 8. Post-execution reconciliation
        try:
            post_result = self.post_execution.reconcile_order(
                order_id,
                wait=self.config.post_execution_wait,
            )
        except Exception as exc:
            # The order was already submitted. Never represent this as
            # ALPACA_SUBMIT_FAILED and never submit a replacement order.
            return ExecutionResult(
                submitted=True,
                status="SUBMITTED_POST_RECONCILIATION_ERROR",
                message=(
                    "Order was submitted to Alpaca PAPER, but post-execution "
                    f"reconciliation failed: {exc}. No replacement order "
                    "was submitted."
                ),
                proposal_token=token,
                order_id=order_id,
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                order_type=intent.order_type,
                submitted_at=submitted_at,
                post_execution=None,
            )

        return ExecutionResult(
            submitted=True,
            status=f"SUBMITTED_{post_result.execution_state}",
            message=(
                "Order submitted to Alpaca PAPER and post-execution "
                "reconciliation completed."
            ),
            proposal_token=token,
            order_id=order_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            order_type=intent.order_type,
            submitted_at=submitted_at,
            post_execution=post_result,
        )

    def _validate_position_state(
        self,
        intent: TradeIntent,
    ) -> tuple[bool, str]:
        """
        Prevent unintended short positions.

        BUY:
            Existing position is allowed.

        SELL:
            Position must exist and quantity must be sufficient.
        """
        if intent.side.lower() == "buy":
            return True, "BUY position state valid."

        if intent.side.lower() != "sell":
            return False, f"Unsupported trade side: {intent.side}"

        current_qty = self._get_position_qty(intent.symbol)

        if current_qty <= 0:
            return (
                False,
                f"SELL blocked: no existing position for {intent.symbol}.",
            )

        if intent.quantity > current_qty + 1e-9:
            return (
                False,
                (
                    f"SELL blocked: requested {intent.quantity:g} shares of "
                    f"{intent.symbol}, but current position is only "
                    f"{current_qty:g} shares."
                ),
            )

        return True, "SELL position state valid."

    def _get_position_qty(self, symbol: str) -> float:
        broker = getattr(self.client, "_client", self.client)

        getter = getattr(broker, "get_open_position", None)
        if getter is None:
            raise AttributeError(
                "Alpaca client does not expose get_open_position()."
            )

        try:
            position = getter(symbol)
        except Exception as exc:
            text = str(exc).lower()
            if "position does not exist" in text or "404" in text:
                return 0.0
            raise

        qty = getattr(position, "qty", None)
        if qty is None and isinstance(position, dict):
            qty = position.get("qty")

        return float(qty or 0.0)

    def _submit_to_alpaca(self, intent: TradeIntent) -> Any:
        """Build and submit the Alpaca-Paper order."""

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
            raise ValueError(f"Unsupported order type: {intent.order_type}")

        return self.client._client.submit_order(order_data=request)

    def _is_paper_environment(self) -> bool:
        health = self.client.health_check()
        return str(health.get("environment", "")).upper() == "PAPER"

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
        if abs(risk_decision.estimated_notional - intent.notional_value) > 0.01:
            return False, "Risk decision notional does not match TradeIntent."
        return True, "Risk decision matches TradeIntent."

    @staticmethod
    def _create_proposal_token(intent: TradeIntent) -> str:
        return f"{intent.symbol.upper()}-{uuid.uuid4().hex[:8].upper()}"

    @staticmethod
    def _extract_order_id(order: Any) -> Optional[str]:
        if order is None:
            return None
        order_id = getattr(order, "id", None)
        if order_id is None and isinstance(order, dict):
            order_id = order.get("id")
        return str(order_id) if order_id is not None else None

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


def create_executor(
    config: Optional[ExecutorConfig] = None,
) -> AlpacaExecutor:
    return AlpacaExecutor(config=config)


def _run_self_test() -> None:
    print("AI TRADER - ALPACA EXECUTOR V2.2")
    print("=" * 60)
    print()
    print("MODE: AUTONOMOUS PAPER")
    print("Telegram Approval: DISABLED")
    print("First Order Confirmation: DISABLED")
    print("Paper Only: ENABLED")
    print("Dynamic Pre-Execution Reconciliation: ENABLED")
    print("Position-State Validation: ENABLED")
    print("Post-Execution Reconciliation: ENABLED")
    print("Telegram: NOTIFICATION ONLY")
    print()
    print("SELF-TEST: NO ORDER SUBMISSION")
    print("STATUS: READY")


if __name__ == "__main__":
    _run_self_test()
