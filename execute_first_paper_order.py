"""
AI Trader - FIRST PAPER ORDER
Controlled execution: 1 share of NVDA.

IMPORTANT
---------
This script is intentionally NOT connected to main.py or the scheduler.

It uses the production:
- TradeIntent
- Risk Gate
- Alpaca Executor
- existing Telegram integration

The Executor remains responsible for:
1. PAPER-only enforcement
2. Risk Gate approval
3. reconciliation
4. duplicate/open-order protection
5. Telegram APPROVE <token>
6. FIRST PAPER ORDER confirmation
7. final Alpaca Paper submission

Order:
    NVDA
    BUY
    1 share
    LIMIT
    $180.50
    DAY

No order is submitted until the two Telegram confirmations are received.
"""

from __future__ import annotations

from app.broker.alpaca_executor import create_executor
from app.broker.alpaca_risk_gate import create_risk_gate
from app.broker.alpaca_trade_intent import TradeIntent


def main() -> None:
    print("AI TRADER - CONTROLLED FIRST PAPER ORDER")
    print("=" * 60)
    print()
    print("WARNING: THIS TEST CAN SUBMIT ONE REAL ALPACA PAPER ORDER.")
    print("Production main.py and scheduler are NOT used.")
    print()

    intent = TradeIntent(
        symbol="NVDA",
        side="buy",
        quantity=1,
        order_type="limit",
        limit_price=180.50,
        time_in_force="day",
        reason="Controlled FIRST PAPER ORDER validation",
        confidence=84.2,
        strategy="Characteristic Curve + Price Action",
        signal="BUY",
    )

    print("TRADE INTENT")
    print("-" * 60)
    print(f"Symbol:        {intent.symbol}")
    print(f"Action:        {intent.side.upper()}")
    print(f"Quantity:      {intent.quantity:g}")
    print(f"Order:         {intent.order_type.upper()}")
    print(f"TIF:           {intent.time_in_force.upper()}")
    print(f"Limit Price:   ${intent.limit_price:,.2f}")
    print(f"Notional:      ${intent.notional_value:,.2f}")
    print()

    risk_gate = create_risk_gate()
    risk_decision = risk_gate.evaluate(intent)

    print(
        "Risk Gate:",
        "APPROVED" if risk_decision.approved else "REJECTED",
    )

    if not risk_decision.approved:
        print()
        print("ORDER NOT SUBMITTED.")
        print("Risk Gate rejected the TradeIntent.")
        if getattr(risk_decision, "rejection_reasons", None):
            for reason in risk_decision.rejection_reasons:
                print(f"- {reason}")
        return

    print()
    print("FIRST PAPER ORDER SAFETY")
    print("-" * 60)
    print("TEST_MODE: ENABLED")
    print("FIRST_ORDER_CONFIRMATION: REQUIRED")
    print("Telegram approval: REQUIRED")
    print("Second first-order confirmation: REQUIRED")
    print()
    print("The Executor will now send the proposal to Telegram.")
    print("It will NOT submit an order without:")
    print("  1. APPROVE <TOKEN>")
    print("  2. CONFIRM FIRST ORDER <TOKEN>")
    print()

    executor = create_executor()

    result = executor.execute(
        intent=intent,
        risk_decision=risk_decision,
        wait_for_telegram=True,
    )

    print()
    print("EXECUTION RESULT")
    print("-" * 60)
    print(result.summary())

    if result.submitted:
        print()
        print("FIRST PAPER ORDER SUBMITTED SUCCESSFULLY.")
        print(f"Order ID: {result.order_id or '-'}")
    else:
        print()
        print("NO ORDER WAS SUBMITTED.")


if __name__ == "__main__":
    main()
