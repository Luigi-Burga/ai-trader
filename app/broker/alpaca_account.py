"""
AI Trader - Alpaca Account Manager
V1.0

Purpose:
- Load Alpaca credentials from .env through alpaca_client.py.
- Read the Paper Trading account.
- Establish AI Trader's fixed capital base at USD 100,000.
- Calculate available capital and current exposure.
- Explicitly ignore Alpaca buying power as AI Trader investment capital.

IMPORTANT:
- Paper Trading only.
- No order submission.
- No modification to production trading engines.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.broker.alpaca_client import AlpacaClient, create_alpaca_client


AI_TRADER_CAPITAL_BASE = 100_000.00


@dataclass(frozen=True)
class AccountSnapshot:
    environment: str
    account_id: str
    status: str
    currency: str

    broker_equity: float
    broker_cash: float
    broker_buying_power: float

    ai_trader_capital_base: float
    invested_capital: float
    available_capital: float
    exposure_pct: float

    position_count: int
    open_order_count: int

    trading_blocked: bool
    ready: bool


class AlpacaAccount:
    """Account-control layer for AI Trader."""

    def __init__(
        self,
        client: AlpacaClient | None = None,
        capital_base: float = AI_TRADER_CAPITAL_BASE,
    ) -> None:
        if capital_base <= 0:
            raise ValueError("capital_base must be greater than zero.")

        self.client = client or create_alpaca_client()
        self.capital_base = float(capital_base)

        if self.client.environment != "PAPER":
            raise RuntimeError(
                "AI Trader AlpacaAccount requires PAPER environment."
            )

    def snapshot(self) -> AccountSnapshot:
        """Build a complete account snapshot."""
        account = self.client.get_account()
        positions = self.client.get_positions()
        orders = self.client.get_orders(status="open")

        broker_equity = self._number(account.get("equity"))
        broker_cash = self._number(account.get("cash"))
        broker_buying_power = self._number(account.get("buying_power"))

        invested_capital = sum(
            max(self._number(position.get("market_value")), 0.0)
            for position in positions
        )

        # AI Trader capital is deliberately independent of Alpaca buying power.
        # We use the configured $100K capital base, not the broker's leverage.
        available_capital = max(
            self.capital_base - invested_capital,
            0.0,
        )

        exposure_pct = (
            (invested_capital / self.capital_base) * 100.0
            if self.capital_base > 0
            else 0.0
        )

        ready = (
            account.get("status") == "AccountStatus.ACTIVE"
            and not bool(account.get("trading_blocked"))
            and account.get("currency") == "USD"
        )

        return AccountSnapshot(
            environment=self.client.environment,
            account_id=str(account.get("id", "")),
            status=str(account.get("status", "")),
            currency=str(account.get("currency", "")),
            broker_equity=broker_equity,
            broker_cash=broker_cash,
            broker_buying_power=broker_buying_power,
            ai_trader_capital_base=self.capital_base,
            invested_capital=invested_capital,
            available_capital=available_capital,
            exposure_pct=exposure_pct,
            position_count=len(positions),
            open_order_count=len(orders),
            trading_blocked=bool(account.get("trading_blocked")),
            ready=ready,
        )

    def is_ready(self) -> bool:
        """Return whether the account is usable by AI Trader."""
        return self.snapshot().ready

    def get_available_capital(self) -> float:
        """Return AI Trader available capital, never broker buying power."""
        return self.snapshot().available_capital

    def get_exposure(self) -> float:
        """Return current AI Trader invested capital."""
        return self.snapshot().invested_capital

    def get_exposure_pct(self) -> float:
        """Return current portfolio exposure as percentage of $100K."""
        return self.snapshot().exposure_pct

    def report(self) -> dict[str, Any]:
        """Return a JSON-serializable account report."""
        return asdict(self.snapshot())

    @staticmethod
    def _number(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


def create_alpaca_account() -> AlpacaAccount:
    """Factory for future AI Trader components."""
    return AlpacaAccount()


def print_report(account: AlpacaAccount) -> None:
    snapshot = account.snapshot()

    print("AI TRADER - ALPACA ACCOUNT")
    print("=" * 60)
    print()
    print(f"Environment:        {snapshot.environment}")
    print(f"Account ID:         {snapshot.account_id}")
    print(f"Status:             {snapshot.status}")
    print(f"Currency:           {snapshot.currency}")
    print()
    print(f"Broker Equity:      ${snapshot.broker_equity:,.2f}")
    print(f"Broker Cash:        ${snapshot.broker_cash:,.2f}")
    print(f"Broker Buying Power:${snapshot.broker_buying_power:,.2f}")
    print()
    print(f"AI Trader Capital:  ${snapshot.ai_trader_capital_base:,.2f}")
    print(f"Invested Capital:   ${snapshot.invested_capital:,.2f}")
    print(f"Available Capital:  ${snapshot.available_capital:,.2f}")
    print(f"Portfolio Exposure: {snapshot.exposure_pct:.2f}%")
    print()
    print(f"Positions:          {snapshot.position_count}")
    print(f"Open Orders:        {snapshot.open_order_count}")
    print()
    print(f"Trading Blocked:    {snapshot.trading_blocked}")
    print(f"ACCOUNT READY:      {snapshot.ready}")
    print()
    print("Capital policy: AI Trader uses $100,000 as its capital base.")
    print("Broker buying power is NOT used as AI Trader capital.")
    print()
    print("No trading operation was submitted.")


if __name__ == "__main__":
    account = create_alpaca_account()
    print_report(account)
