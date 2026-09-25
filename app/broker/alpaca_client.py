"""
AI Trader - Alpaca Trading Client
Phase 1: Read-only connectivity layer.

This module connects AI Trader to Alpaca Paper Trading and exposes:
- account information
- open positions
- orders

IMPORTANT:
- No order submission is implemented in this phase.
- Paper Trading is forced by default.
- Credentials are read from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest


@dataclass(frozen=True)
class AlpacaConfig:
    api_key: str
    secret_key: str
    paper: bool = True


class AlpacaClient:
    """Read-only Alpaca Paper Trading client for AI Trader."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ) -> None:
        if not paper:
            raise ValueError(
                "Live Alpaca trading is intentionally disabled in Phase 1. "
                "Use paper=True."
            )

        self.config = AlpacaConfig(
            api_key=api_key or os.getenv("ALPACA_API_KEY", ""),
            secret_key=secret_key or os.getenv("ALPACA_SECRET_KEY", ""),
            paper=True,
        )

        if not self.config.api_key:
            raise ValueError("Missing ALPACA_API_KEY environment variable.")

        if not self.config.secret_key:
            raise ValueError("Missing ALPACA_SECRET_KEY environment variable.")

        self._client = TradingClient(
            self.config.api_key,
            self.config.secret_key,
            paper=True,
        )

    @property
    def environment(self) -> str:
        return "PAPER"

    def get_account(self) -> dict[str, Any]:
        """Return the current Alpaca account state."""
        account = self._client.get_account()

        return {
            "id": str(account.id),
            "status": str(account.status),
            "currency": str(account.currency),
            "cash": self._to_float(account.cash),
            "portfolio_value": self._to_float(account.portfolio_value),
            "equity": self._to_float(account.equity),
            "last_equity": self._to_float(account.last_equity),
            "buying_power": self._to_float(account.buying_power),
            "long_market_value": self._to_float(account.long_market_value),
            "short_market_value": self._to_float(account.short_market_value),
            "trading_blocked": bool(account.trading_blocked),
            "transfers_blocked": bool(account.transfers_blocked),
            "account_blocked": bool(account.account_blocked),
            "pattern_day_trader": bool(account.pattern_day_trader),
        }

    def get_positions(self) -> list[dict[str, Any]]:
        """Return all currently open positions."""
        positions = self._client.get_all_positions()

        return [
            {
                "symbol": position.symbol,
                "qty": self._to_float(position.qty),
                "side": str(position.side),
                "avg_entry_price": self._to_float(position.avg_entry_price),
                "current_price": self._to_float(position.current_price),
                "market_value": self._to_float(position.market_value),
                "cost_basis": self._to_float(position.cost_basis),
                "unrealized_pl": self._to_float(position.unrealized_pl),
                "unrealized_plpc": self._to_float(position.unrealized_plpc),
                "change_today": self._to_float(position.change_today),
            }
            for position in positions
        ]

    def get_orders(
        self,
        status: str = "open",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return Alpaca orders without submitting or modifying any order."""
        status_map = {
            "open": QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
            "all": QueryOrderStatus.ALL,
        }

        normalized_status = status.lower()
        if normalized_status not in status_map:
            raise ValueError("status must be one of: open, closed, all")

        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500.")

        request = GetOrdersRequest(
            status=status_map[normalized_status],
            limit=limit,
            nested=True,
        )

        orders = self._client.get_orders(filter=request)

        return [
            {
                "id": str(order.id),
                "client_order_id": str(order.client_order_id),
                "symbol": order.symbol,
                "side": str(order.side),
                "type": str(order.type),
                "status": str(order.status),
                "qty": self._to_float(order.qty),
                "filled_qty": self._to_float(order.filled_qty),
                "filled_avg_price": self._to_float(order.filled_avg_price),
                "time_in_force": str(order.time_in_force),
                "submitted_at": self._iso(order.submitted_at),
                "filled_at": self._iso(order.filled_at),
                "canceled_at": self._iso(order.canceled_at),
            }
            for order in orders
        ]

    def health_check(self) -> dict[str, Any]:
        """Validate credentials and basic account connectivity."""
        account = self.get_account()

        return {
            "connected": True,
            "environment": self.environment,
            "account_id": account["id"],
            "status": account["status"],
            "currency": account["currency"],
            "equity": account["equity"],
            "buying_power": account["buying_power"],
            "trading_blocked": account["trading_blocked"],
        }

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _iso(value: Any) -> str | None:
        if value is None:
            return None
        return value.isoformat() if hasattr(value, "isoformat") else str(value)


def create_alpaca_client() -> AlpacaClient:
    """Factory used by future AI Trader modules."""
    return AlpacaClient()


if __name__ == "__main__":
    import json

    client = create_alpaca_client()

    print("AI TRADER - ALPACA CONNECTIVITY TEST")
    print("=" * 60)

    health = client.health_check()
    print("\n[ACCOUNT]")
    print(json.dumps(health, indent=2))

    positions = client.get_positions()
    print("\n[POSITIONS]")
    print(json.dumps(positions, indent=2))

    orders = client.get_orders(status="open")
    print("\n[OPEN ORDERS]")
    print(json.dumps(orders, indent=2))

    print("\nNo trading operation was submitted.")
