"""
AI Trader - Alpaca Executor
First Paper Order timeout test.

Purpose
-------
Validate that a Telegram approval followed by NO response to the
FIRST_ORDER_CONFIRMATION gate results in a safe timeout and that the
Alpaca submit_order() call is never reached.

Test scenario
-------------
1. Risk Gate approves the TradeIntent.
2. Telegram approval returns APPROVED.
3. FIRST_ORDER_CONFIRMATION returns TIMEOUT.
4. Executor must return FIRST_ORDER_CONFIRMATION_TIMEOUT.
5. Alpaca submit_order() must have zero calls.

This test does NOT use the real Telegram Bot and does NOT submit an
Alpaca order.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.broker.alpaca_executor import AlpacaExecutor, ExecutorConfig
from app.broker.alpaca_trade_intent import TradeIntent


class FakeTelegramApproval:
    """Deterministic Telegram double for the timeout test."""

    def __init__(self) -> None:
        self.approval_calls = 0
        self.confirmation_calls = 0
        self.sent_messages: list[str] = []

    @property
    def configured(self) -> bool:
        return True

    def send_message(self, text: str) -> bool:
        self.sent_messages.append(text)
        return True

    def wait_for_approval(
        self,
        token: str,
        timeout_seconds: int,
        poll_interval_seconds: int,
    ) -> str:
        self.approval_calls += 1
        return "APPROVED"

    def wait_for_first_order_confirmation(
        self,
        token: str,
        timeout_seconds: int,
        poll_interval_seconds: int,
    ) -> str:
        self.confirmation_calls += 1
        # Simulates the user not responding before the configured timeout.
        return "TIMEOUT"


class FakeClient:
    """Alpaca client double. Any submit attempt is a hard test failure."""

    def __init__(self) -> None:
        self.submit_order_calls = 0
        self._client = SimpleNamespace(
            submit_order=self._submit_order
        )

    def health_check(self) -> dict[str, Any]:
        return {
            "environment": "PAPER",
            "ready": True,
        }

    def _submit_order(self, *args: Any, **kwargs: Any) -> None:
        self.submit_order_calls += 1
        raise AssertionError(
            "CRITICAL TEST FAILURE: Alpaca submit_order() was called."
        )


class FakeOrders:
    """No open-order conflicts for the test."""

    def __init__(self) -> None:
        self.has_open_order_calls = 0

    def has_open_order(self, symbol: str, side: str) -> bool:
        self.has_open_order_calls += 1
        return False


class FakeReconciliation:
    """Always returns a reconciled empty portfolio."""

    def reconcile_empty_portfolio(self) -> Any:
        return SimpleNamespace(
            reconciled=True,
            positions=[],
            open_orders=[],
        )


def build_test_intent() -> TradeIntent:
    return TradeIntent(
        symbol="NVDA",
        side="buy",
        quantity=1,
        order_type="limit",
        limit_price=180.50,
        time_in_force="day",
        reason="FIRST PAPER ORDER timeout test",
        confidence=84.2,
        strategy="Executor Safety Test",
        signal="BUY",
    )


def build_approved_risk_decision(intent: TradeIntent) -> Any:
    """
    Supply only the RiskDecision attributes consumed by AlpacaExecutor.
    """
    return SimpleNamespace(
        approved=True,
        symbol=intent.symbol,
        side=intent.side,
        quantity=intent.quantity,
        estimated_notional=intent.notional_value,
        capital_base=100_000.00,
        projected_exposure_pct=(
            intent.notional_value / 100_000.00
        ) * 100.0,
    )


def main() -> None:
    print("FIRST PAPER ORDER TIMEOUT TEST")
    print("=" * 60)
    print()

    intent = build_test_intent()
    risk_decision = build_approved_risk_decision(intent)

    fake_client = FakeClient()
    fake_orders = FakeOrders()
    fake_reconciliation = FakeReconciliation()
    fake_telegram = FakeTelegramApproval()

    config = ExecutorConfig(
        require_telegram_approval=True,
        approval_timeout_seconds=5,
        poll_interval_seconds=1,
        reject_duplicate_open_order=True,
        paper_only=True,
        test_mode=True,
        first_order_confirmation_required=True,
    )

    executor = AlpacaExecutor(
        client=fake_client,
        orders=fake_orders,
        reconciliation=fake_reconciliation,
        telegram=fake_telegram,
        config=config,
    )

    print(f"Risk Gate: {'APPROVED' if risk_decision.approved else 'REJECTED'}")
    print("Operation Type: FIRST PAPER ORDER")
    print(f"Symbol: {intent.symbol}")
    print(f"Quantity: {intent.quantity:g}")
    print(f"Notional: ${intent.notional_value:,.2f}")
    print()
    print("TEST_MODE: ENABLED")
    print("FIRST_ORDER_CONFIRMATION: REQUIRED")
    print("Timeout configured for test: 5 seconds")
    print()

    result = executor.execute(
        intent=intent,
        risk_decision=risk_decision,
        wait_for_telegram=True,
    )

    print("Telegram Approval: APPROVED")
    print("First Order Confirmation: TIMEOUT")
    print()
    print(f"Execution Status: {result.status}")
    print(f"Submitted: {result.submitted}")
    print(f"Alpaca submit_order calls: {fake_client.submit_order_calls}")
    print()

    expected_status = "FIRST_ORDER_CONFIRMATION_TIMEOUT"

    failures: list[str] = []

    if result.status != expected_status:
        failures.append(
            f"Expected status {expected_status}, got {result.status}"
        )

    if result.submitted is not False:
        failures.append(
            f"Expected submitted=False, got {result.submitted}"
        )

    if fake_client.submit_order_calls != 0:
        failures.append(
            "Alpaca submit_order() was called."
        )

    if fake_telegram.approval_calls != 1:
        failures.append(
            f"Expected 1 Telegram approval check, "
            f"got {fake_telegram.approval_calls}"
        )

    if fake_telegram.confirmation_calls != 1:
        failures.append(
            f"Expected 1 first-order confirmation check, "
            f"got {fake_telegram.confirmation_calls}"
        )

    if len(fake_telegram.sent_messages) != 2:
        failures.append(
            f"Expected 2 Telegram messages, "
            f"got {len(fake_telegram.sent_messages)}"
        )

    if failures:
        print("TEST RESULT: FAIL")
        print()
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("TEST RESULT: PASS")
    print("NO ORDER SUBMITTED")


if __name__ == "__main__":
    main()
