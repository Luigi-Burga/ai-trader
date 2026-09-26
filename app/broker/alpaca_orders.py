"""
AI Trader - Alpaca Orders
Version: 1.1

Read-only order manager for Alpaca Paper Trading.

V1.1 CHANGE:
Only the normalization of Alpaca enum values for `side` and `status`
was changed. Alpaca-py returns enum objects such as OrderSide.BUY and
OrderStatus.ACCEPTED; V1.1 stores their `.value` (`buy`, `accepted`).

No order submission, cancellation, replacement, or portfolio modification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from app.broker.alpaca_client import AlpacaClient, create_alpaca_client


@dataclass(frozen=True)
class OrderSnapshot:
    id: str
    symbol: str
    side: str
    status: str
    qty: float
    filled_qty: float
    order_type: str
    time_in_force: str
    limit_price: float | None
    stop_price: float | None
    submitted_at: str | None
    filled_at: str | None
    canceled_at: str | None


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


class AlpacaOrders:
    """Read-only order manager for Alpaca Paper Trading."""

    def __init__(self, client: AlpacaClient | None = None) -> None:
        self.client = client or create_alpaca_client()
        if self.client.environment != "PAPER":
            raise RuntimeError(
                "AI Trader AlpacaOrders requires PAPER environment."
            )

    def get_open_orders(self, limit: int = 500) -> list[OrderSnapshot]:
        return self._get_orders(QueryOrderStatus.OPEN, limit)

    def get_closed_orders(self, limit: int = 500) -> list[OrderSnapshot]:
        return self._get_orders(QueryOrderStatus.CLOSED, limit)

    def get_all_orders(self, limit: int = 500) -> list[OrderSnapshot]:
        return self._get_orders(QueryOrderStatus.ALL, limit)

    def get_order(self, order_id: str) -> OrderSnapshot | None:
        order = self.client._client.get_order_by_id(order_id)
        return None if order is None else self._normalize_order(order)

    def find_open_orders_for_symbol(self, symbol: str) -> list[OrderSnapshot]:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol cannot be empty.")
        return [
            order for order in self.get_open_orders()
            if order.symbol.upper() == normalized_symbol
        ]

    def has_open_order_for_symbol(self, symbol: str) -> bool:
        return bool(self.find_open_orders_for_symbol(symbol))

    def has_open_order(self, symbol: str, side: str | None = None) -> bool:
        normalized_symbol = symbol.strip().upper()
        normalized_side = side.strip().lower() if side else None

        if not normalized_symbol:
            raise ValueError("symbol cannot be empty.")
        if normalized_side and normalized_side not in {"buy", "sell"}:
            raise ValueError(
                f"Unsupported order side: {side}. Expected BUY or SELL."
            )

        for order in self.get_open_orders():
            if order.symbol.upper() != normalized_symbol:
                continue
            if normalized_side is None or order.side.lower() == normalized_side:
                return True
        return False

    def get_summary(self) -> dict[str, Any]:
        open_orders = self.get_open_orders()
        closed_orders = self.get_closed_orders()
        return {
            "environment": self.client.environment,
            "open_orders": len(open_orders),
            "closed_orders": len(closed_orders),
            "open_symbols": sorted({order.symbol for order in open_orders}),
        }

    def report(self) -> dict[str, Any]:
        return {
            "summary": self.get_summary(),
            "open_orders": [asdict(order) for order in self.get_open_orders()],
        }

    def _get_orders(
        self,
        status: QueryOrderStatus,
        limit: int,
    ) -> list[OrderSnapshot]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        request = GetOrdersRequest(status=status, limit=limit, nested=True)
        orders = self.client._client.get_orders(filter=request)
        return [self._normalize_order(order) for order in orders]

    @staticmethod
    def _normalize_enum(value: Any, default: str = "") -> str:
        """Use Alpaca enum `.value`; fall back to normalized string."""
        if value is None:
            return default
        enum_value = getattr(value, "value", None)
        if enum_value is not None:
            return str(enum_value).strip().lower()
        return str(value).strip().lower()

    @classmethod
    def _normalize_order(cls, order: Any) -> OrderSnapshot:
        """Normalize an Alpaca order; V1.1 changes only side/status enum handling."""
        return OrderSnapshot(
            id=str(getattr(order, "id", "")),
            symbol=str(getattr(order, "symbol", "")).strip().upper(),
            side=cls._normalize_enum(getattr(order, "side", None)),
            status=cls._normalize_enum(getattr(order, "status", None)),
            qty=float(getattr(order, "qty", 0) or 0),
            filled_qty=float(getattr(order, "filled_qty", 0) or 0),
            order_type=str(getattr(order, "order_type", "")),
            time_in_force=str(getattr(order, "time_in_force", "")),
            limit_price=(
                float(getattr(order, "limit_price"))
                if getattr(order, "limit_price", None) is not None else None
            ),
            stop_price=(
                float(getattr(order, "stop_price"))
                if getattr(order, "stop_price", None) is not None else None
            ),
            submitted_at=(
                str(getattr(order, "submitted_at"))
                if getattr(order, "submitted_at", None) is not None else None
            ),
            filled_at=(
                str(getattr(order, "filled_at"))
                if getattr(order, "filled_at", None) is not None else None
            ),
            canceled_at=(
                str(getattr(order, "canceled_at"))
                if getattr(order, "canceled_at", None) is not None else None
            ),
        )


def create_alpaca_orders() -> AlpacaOrders:
    return AlpacaOrders()
