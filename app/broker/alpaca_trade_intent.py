"""
AI Trader - Alpaca Trade Intent
V1.0

Purpose:
- Represent a proposed trade before it reaches the execution layer.
- Validate symbol, side, quantity, order type, prices and time-in-force.
- Calculate notional value for risk checks.
- Keep the intent independent from Alpaca order submission.

IMPORTANT:
- This module DOES NOT connect to Alpaca.
- This module DOES NOT submit, cancel or modify orders.
- A TradeIntent is only a proposed action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


SUPPORTED_SIDES = {"buy", "sell"}
SUPPORTED_ORDER_TYPES = {"market", "limit", "stop", "stop_limit"}
SUPPORTED_TIME_IN_FORCE = {
    "day",
    "gtc",
    "opg",
    "cls",
    "ioc",
    "fok",
}


@dataclass(frozen=True)
class TradeIntent:
    """
    Immutable representation of a proposed trade.

    The intent is deliberately broker-independent.
    """

    symbol: str
    side: str
    quantity: float

    order_type: str = "market"
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "day"

    reason: str = ""
    confidence: float | None = None

    strategy: str = ""
    signal: str = ""

    created_at: str = ""

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        side = self.side.strip().lower()
        order_type = self.order_type.strip().lower()
        time_in_force = self.time_in_force.strip().lower()

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "order_type", order_type)
        object.__setattr__(self, "time_in_force", time_in_force)

        if not self.symbol:
            raise ValueError("symbol must not be empty.")

        if self.side not in SUPPORTED_SIDES:
            raise ValueError(
                f"side must be one of: {sorted(SUPPORTED_SIDES)}"
            )

        if self.quantity <= 0:
            raise ValueError("quantity must be greater than zero.")

        if self.order_type not in SUPPORTED_ORDER_TYPES:
            raise ValueError(
                "order_type must be one of: "
                f"{sorted(SUPPORTED_ORDER_TYPES)}"
            )

        if self.time_in_force not in SUPPORTED_TIME_IN_FORCE:
            raise ValueError(
                "time_in_force must be one of: "
                f"{sorted(SUPPORTED_TIME_IN_FORCE)}"
            )

        if self.confidence is not None and not 0 <= self.confidence <= 100:
            raise ValueError("confidence must be between 0 and 100.")

        if self.order_type in {"limit", "stop_limit"}:
            if self.limit_price is None or self.limit_price <= 0:
                raise ValueError(
                    f"{self.order_type} order requires a positive limit_price."
                )

        if self.order_type in {"stop", "stop_limit"}:
            if self.stop_price is None or self.stop_price <= 0:
                raise ValueError(
                    f"{self.order_type} order requires a positive stop_price."
                )

        if self.order_type == "market" and (
            self.limit_price is not None or self.stop_price is not None
        ):
            raise ValueError(
                "market order cannot contain limit_price or stop_price."
            )

        if self.order_type == "limit" and self.stop_price is not None:
            raise ValueError("limit order cannot contain stop_price.")

        if self.order_type == "stop" and self.limit_price is not None:
            raise ValueError("stop order cannot contain limit_price.")

    @property
    def notional_value(self) -> float | None:
        """
        Return estimated order notional.

        Market orders return None because the execution price is unknown.
        """
        if self.order_type == "market":
            return None

        if self.limit_price is not None:
            return self.quantity * self.limit_price

        if self.stop_price is not None:
            return self.quantity * self.stop_price

        return None

    @property
    def is_buy(self) -> bool:
        return self.side == "buy"

    @property
    def is_sell(self) -> bool:
        return self.side == "sell"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    def summary(self) -> str:
        """Return a concise human-readable description."""
        price = ""

        if self.order_type == "limit":
            price = f" @ LIMIT ${self.limit_price:,.2f}"
        elif self.order_type == "stop":
            price = f" @ STOP ${self.stop_price:,.2f}"
        elif self.order_type == "stop_limit":
            price = (
                f" @ STOP ${self.stop_price:,.2f}"
                f" / LIMIT ${self.limit_price:,.2f}"
            )

        return (
            f"{self.side.upper()} {self.quantity:g} {self.symbol}"
            f" {self.order_type.upper()}{price}"
            f" TIF={self.time_in_force.upper()}"
        )


class TradeIntentFactory:
    """Factory for creating validated TradeIntent objects."""

    @staticmethod
    def create(
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str = "day",
        reason: str = "",
        confidence: float | None = None,
        strategy: str = "",
        signal: str = "",
    ) -> TradeIntent:
        return TradeIntent(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            reason=reason,
            confidence=confidence,
            strategy=strategy,
            signal=signal,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


def create_trade_intent(
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "market",
    limit_price: float | None = None,
    stop_price: float | None = None,
    time_in_force: str = "day",
    reason: str = "",
    confidence: float | None = None,
    strategy: str = "",
    signal: str = "",
) -> TradeIntent:
    """Convenience factory for AI Trader components."""
    return TradeIntentFactory.create(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        time_in_force=time_in_force,
        reason=reason,
        confidence=confidence,
        strategy=strategy,
        signal=signal,
    )


def print_intent(intent: TradeIntent) -> None:
    print("AI TRADER - TRADE INTENT")
    print("=" * 60)
    print()
    print(f"Symbol:             {intent.symbol}")
    print(f"Side:               {intent.side.upper()}")
    print(f"Quantity:           {intent.quantity:g}")
    print(f"Order Type:         {intent.order_type.upper()}")
    print(f"Time in Force:      {intent.time_in_force.upper()}")

    if intent.limit_price is not None:
        print(f"Limit Price:        ${intent.limit_price:,.2f}")

    if intent.stop_price is not None:
        print(f"Stop Price:         ${intent.stop_price:,.2f}")

    if intent.notional_value is not None:
        print(f"Estimated Notional: ${intent.notional_value:,.2f}")
    else:
        print("Estimated Notional: UNKNOWN (market order)")

    print(f"Strategy:           {intent.strategy or 'N/A'}")
    print(f"Signal:             {intent.signal or 'N/A'}")
    print(f"Confidence:         {intent.confidence if intent.confidence is not None else 'N/A'}")
    print(f"Reason:             {intent.reason or 'N/A'}")
    print(f"Created At:         {intent.created_at}")
    print()
    print(f"Intent:             {intent.summary()}")
    print()
    print("STATUS: PROPOSED ONLY")
    print("No Alpaca order was submitted.")


if __name__ == "__main__":
    # Demonstration only. This creates an intent in memory.
    # It does NOT connect to Alpaca and does NOT submit an order.
    demo_intent = create_trade_intent(
        symbol="NVDA",
        side="buy",
        quantity=100,
        order_type="limit",
        limit_price=180.50,
        time_in_force="day",
        reason="Demonstration of Trade Intent layer",
        confidence=84.2,
        strategy="Characteristic Curve + Price Action",
        signal="BUY",
    )

    print_intent(demo_intent)
