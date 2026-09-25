"""
AI Trader - Alpaca Orders Manager
V1.0

Purpose:
- Read and normalize Alpaca Paper Trading orders.
- Inspect open, closed, and all orders.
- Retrieve an individual order by ID.
- Calculate basic order statistics.
- Provide duplicate-order lookup helpers.

IMPORTANT:
- Paper Trading only.
- READ ONLY.
- No submit_order().
- No cancel_order().
- No replace_order().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from app.broker.alpaca_client import (
    AlpacaClient,
    create_alpaca_client,
)


@dataclass(frozen=True)
class OrderSnapshot:
    id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    status: str

    qty: float
    filled_qty: float
    remaining_qty: float

    filled_avg_price: float | None
    limit_price: float | None
    stop_price: float | None

    time_in_force: str

    submitted_at: str | None
    filled_at: str | None
    canceled_at: str | None
    expired_at: str | None
    failed_at: str | None


@dataclass(frozen=True)
class OrdersSummary:
    environment: str
    total_orders: int
    open_orders: int
    filled_orders: int
    canceled_orders: int
    rejected_orders: int
    partially_filled_orders: int
    other_orders: int


class AlpacaOrders:
    """Read-only order management and inspection layer."""

    OPEN_STATUSES = {
        "new",
        "accepted",
        "pending_new",
        "partially_filled",
        "pending_replace",
        "pending_cancel",
        "held",
        "calculated",
    }

    FILLED_STATUSES = {"filled"}
    CANCELED_STATUSES = {"canceled", "expired", "done_for_day"}
    REJECTED_STATUSES = {"rejected", "suspended", "stopped"}

    def __init__(self, client: AlpacaClient | None = None) -> None:
        self.client = client or create_alpaca_client()

        if self.client.environment != "PAPER":
            raise RuntimeError(
                "AI Trader AlpacaOrders requires PAPER environment."
            )

    def get_open_orders(self, limit: int = 500) -> list[OrderSnapshot]:
        """Return currently open orders."""
        return self._get_orders(
            status=QueryOrderStatus.OPEN,
            limit=limit,
        )

    def get_closed_orders(self, limit: int = 500) -> list[OrderSnapshot]:
        """Return closed orders."""
        return self._get_orders(
            status=QueryOrderStatus.CLOSED,
            limit=limit,
        )

    def get_all_orders(self, limit: int = 500) -> list[OrderSnapshot]:
        """Return all orders available through the requested limit."""
        return self._get_orders(
            status=QueryOrderStatus.ALL,
            limit=limit,
        )

    def get_order(self, order_id: str) -> OrderSnapshot | None:
        """Return a single order by Alpaca order ID."""
        normalized_id = order_id.strip()

        if not normalized_id:
            raise ValueError("order_id must not be empty.")

        order = self.client._client.get_order_by_id(normalized_id)

        if order is None:
            return None

        return self._normalize_order(order)

    def find_open_orders_for_symbol(
        self,
        symbol: str,
    ) -> list[OrderSnapshot]:
        """Return open orders for a specific symbol."""
        normalized_symbol = symbol.strip().upper()

        if not normalized_symbol:
            raise ValueError("symbol must not be empty.")

        return [
            order
            for order in self.get_open_orders()
            if order.symbol.upper() == normalized_symbol
        ]

    def has_open_order_for_symbol(self, symbol: str) -> bool:
        """Return True if there is at least one open order for the symbol."""
        return bool(self.find_open_orders_for_symbol(symbol))

    def has_open_order(
        self,
        symbol: str,
        side: str | None = None,
    ) -> bool:
        """
        Return True when a matching open order exists.

        If side is omitted, any open order for the symbol matches.
        """
        normalized_symbol = symbol.strip().upper()

        if not normalized_symbol:
            raise ValueError("symbol must not be empty.")

        normalized_side = side.strip().lower() if side else None

        if normalized_side and normalized_side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'.")

        for order in self.get_open_orders():
            if order.symbol.upper() != normalized_symbol:
                continue

            if normalized_side is None:
                return True

            if order.side.lower() == normalized_side:
                return True

        return False

    def get_summary(self) -> OrdersSummary:
        """Return aggregate statistics for currently available orders."""
        orders = self.get_all_orders()

        counts = {
            "open": 0,
            "filled": 0,
            "canceled": 0,
            "rejected": 0,
            "partially_filled": 0,
            "other": 0,
        }

        for order in orders:
            status = order.status.lower()

            if status == "partially_filled":
                counts["partially_filled"] += 1
            elif status in self.FILLED_STATUSES:
                counts["filled"] += 1
            elif status in self.CANCELED_STATUSES:
                counts["canceled"] += 1
            elif status in self.REJECTED_STATUSES:
                counts["rejected"] += 1
            elif status in self.OPEN_STATUSES:
                counts["open"] += 1
            else:
                counts["other"] += 1

        return OrdersSummary(
            environment=self.client.environment,
            total_orders=len(orders),
            open_orders=counts["open"],
            filled_orders=counts["filled"],
            canceled_orders=counts["canceled"],
            rejected_orders=counts["rejected"],
            partially_filled_orders=counts["partially_filled"],
            other_orders=counts["other"],
        )

    def report(self) -> dict[str, Any]:
        """Return a JSON-serializable order report."""
        summary = self.get_summary()
        orders = self.get_all_orders()

        return {
            "summary": asdict(summary),
            "orders": [asdict(order) for order in orders],
        }

    def _get_orders(
        self,
        status: QueryOrderStatus,
        limit: int,
    ) -> list[OrderSnapshot]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500.")

        request = GetOrdersRequest(
            status=status,
            limit=limit,
            nested=True,
        )

        orders = self.client._client.get_orders(filter=request)

        return [self._normalize_order(order) for order in orders]

    @staticmethod
    def _normalize_order(order: Any) -> OrderSnapshot:
        qty = AlpacaOrders._number(getattr(order, "qty", None))
        filled_qty = AlpacaOrders._number(
            getattr(order, "filled_qty", None)
        )

        return OrderSnapshot(
            id=str(getattr(order, "id", "")),
            client_order_id=str(
                getattr(order, "client_order_id", "")
            ),
            symbol=str(getattr(order, "symbol", "")),
            side=str(getattr(order, "side", "")),
            order_type=str(getattr(order, "type", "")),
            status=str(getattr(order, "status", "")),
            qty=qty,
            filled_qty=filled_qty,
            remaining_qty=max(qty - filled_qty, 0.0),
            filled_avg_price=AlpacaOrders._optional_number(
                getattr(order, "filled_avg_price", None)
            ),
            limit_price=AlpacaOrders._optional_number(
                getattr(order, "limit_price", None)
            ),
            stop_price=AlpacaOrders._optional_number(
                getattr(order, "stop_price", None)
            ),
            time_in_force=str(
                getattr(order, "time_in_force", "")
            ),
            submitted_at=AlpacaOrders._iso(
                getattr(order, "submitted_at", None)
            ),
            filled_at=AlpacaOrders._iso(
                getattr(order, "filled_at", None)
            ),
            canceled_at=AlpacaOrders._iso(
                getattr(order, "canceled_at", None)
            ),
            expired_at=AlpacaOrders._iso(
                getattr(order, "expired_at", None)
            ),
            failed_at=AlpacaOrders._iso(
                getattr(order, "failed_at", None)
            ),
        )

    @staticmethod
    def _number(value: Any) -> float:
        if value is None:
            return 0.0

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _optional_number(value: Any) -> float | None:
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _iso(value: Any) -> str | None:
        if value is None:
            return None

        return value.isoformat() if hasattr(value, "isoformat") else str(value)


def create_alpaca_orders() -> AlpacaOrders:
    """Factory for future AI Trader components."""
    return AlpacaOrders()


def print_report(orders_manager: AlpacaOrders) -> None:
    summary = orders_manager.get_summary()

    print("AI TRADER - ALPACA ORDERS")
    print("=" * 60)
    print()
    print(f"Environment:             {summary.environment}")
    print(f"Total Orders:            {summary.total_orders}")
    print(f"Open Orders:             {summary.open_orders}")
    print(f"Filled Orders:           {summary.filled_orders}")
    print(f"Canceled/Expired:        {summary.canceled_orders}")
    print(f"Rejected Orders:         {summary.rejected_orders}")
    print(f"Partially Filled:        {summary.partially_filled_orders}")
    print(f"Other Orders:            {summary.other_orders}")
    print()

    open_orders = orders_manager.get_open_orders()

    print("[OPEN ORDERS]")

    if not open_orders:
        print("No open orders.")
    else:
        for order in open_orders:
            print()
            print(
                f"{order.symbol} | "
                f"{order.side} | "
                f"{order.order_type} | "
                f"status={order.status}"
            )
            print(
                f"  ID={order.id} | "
                f"Qty={order.qty:.4f} | "
                f"Filled={order.filled_qty:.4f} | "
                f"Remaining={order.remaining_qty:.4f}"
            )
            print(
                f"  Limit={order.limit_price} | "
                f"Stop={order.stop_price} | "
                f"TIF={order.time_in_force}"
            )

    print()
    print("READ-ONLY MODE: no order was submitted, canceled, or modified.")


if __name__ == "__main__":
    orders_manager = create_alpaca_orders()
    print_report(orders_manager)
