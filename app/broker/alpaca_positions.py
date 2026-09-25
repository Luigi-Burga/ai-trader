"""
AI Trader - Alpaca Positions Manager
V1.0

Purpose:
- Read current Alpaca Paper Trading positions.
- Normalize position data for AI Trader.
- Calculate market value, P/L and portfolio exposure.
- Calculate exposure against AI Trader's $100,000 capital base.
- Provide read-only position summaries.

IMPORTANT:
- Paper Trading only.
- No order submission.
- No position modification.
- Broker buying power is NOT used for exposure calculations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.broker.alpaca_account import AI_TRADER_CAPITAL_BASE, AlpacaAccount
from app.broker.alpaca_client import AlpacaClient, create_alpaca_client


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    qty: float
    side: str

    avg_entry_price: float
    current_price: float

    market_value: float
    cost_basis: float

    unrealized_pl: float
    unrealized_pl_pct: float
    change_today_pct: float

    capital_exposure_pct: float


@dataclass(frozen=True)
class PositionsSummary:
    environment: str
    capital_base: float

    position_count: int
    gross_market_value: float
    long_market_value: float
    short_market_value: float

    total_unrealized_pl: float
    total_exposure_pct: float
    available_capital: float

    positions: list[dict[str, Any]]


class AlpacaPositions:
    """Read-only position layer for AI Trader."""

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
                "AI Trader AlpacaPositions requires PAPER environment."
            )

    def get_positions(self) -> list[PositionSnapshot]:
        """Return normalized open positions."""
        raw_positions = self.client.get_positions()
        result: list[PositionSnapshot] = []

        for position in raw_positions:
            market_value = self._number(position.get("market_value"))
            cost_basis = self._number(position.get("cost_basis"))
            unrealized_pl = self._number(position.get("unrealized_pl"))

            # Prefer broker-provided percentage when available.
            raw_pl_pct = position.get("unrealized_plpc")
            unrealized_pl_pct = self._number(raw_pl_pct)

            # AI Trader exposure is always based on the $100K capital base.
            capital_exposure_pct = (
                abs(market_value) / self.capital_base * 100.0
            )

            result.append(
                PositionSnapshot(
                    symbol=str(position.get("symbol", "")),
                    qty=self._number(position.get("qty")),
                    side=str(position.get("side", "")),
                    avg_entry_price=self._number(
                        position.get("avg_entry_price")
                    ),
                    current_price=self._number(
                        position.get("current_price")
                    ),
                    market_value=market_value,
                    cost_basis=cost_basis,
                    unrealized_pl=unrealized_pl,
                    unrealized_pl_pct=unrealized_pl_pct,
                    change_today_pct=self._number(
                        position.get("change_today")
                    ),
                    capital_exposure_pct=capital_exposure_pct,
                )
            )

        return result

    def get_position(self, symbol: str) -> PositionSnapshot | None:
        """Return a single position by ticker symbol."""
        normalized_symbol = symbol.strip().upper()

        if not normalized_symbol:
            raise ValueError("symbol must not be empty.")

        for position in self.get_positions():
            if position.symbol.upper() == normalized_symbol:
                return position

        return None

    def has_position(self, symbol: str) -> bool:
        """Return True when AI Trader currently holds the symbol."""
        return self.get_position(symbol) is not None

    def get_summary(self) -> PositionsSummary:
        """Return aggregate portfolio position statistics."""
        positions = self.get_positions()

        gross_market_value = sum(
            abs(position.market_value) for position in positions
        )

        long_market_value = sum(
            position.market_value
            for position in positions
            if position.market_value > 0
            and not position.side.lower().endswith("short")
        )

        short_market_value = sum(
            abs(position.market_value)
            for position in positions
            if position.market_value < 0
            or position.side.lower().endswith("short")
        )

        total_unrealized_pl = sum(
            position.unrealized_pl for position in positions
        )

        total_exposure_pct = (
            gross_market_value / self.capital_base * 100.0
            if self.capital_base > 0
            else 0.0
        )

        available_capital = max(
            self.capital_base - gross_market_value,
            0.0,
        )

        return PositionsSummary(
            environment=self.client.environment,
            capital_base=self.capital_base,
            position_count=len(positions),
            gross_market_value=gross_market_value,
            long_market_value=long_market_value,
            short_market_value=short_market_value,
            total_unrealized_pl=total_unrealized_pl,
            total_exposure_pct=total_exposure_pct,
            available_capital=available_capital,
            positions=[asdict(position) for position in positions],
        )

    def get_total_exposure(self) -> float:
        """Return gross market-value exposure in USD."""
        return self.get_summary().gross_market_value

    def get_total_exposure_pct(self) -> float:
        """Return gross exposure as percentage of $100K."""
        return self.get_summary().total_exposure_pct

    def get_available_capital(self) -> float:
        """Return capital not currently allocated to positions."""
        return self.get_summary().available_capital

    def get_total_unrealized_pl(self) -> float:
        """Return total unrealized P/L across positions."""
        return self.get_summary().total_unrealized_pl

    def report(self) -> dict[str, Any]:
        """Return a JSON-serializable position report."""
        return asdict(self.get_summary())

    @staticmethod
    def _number(value: Any) -> float:
        if value is None:
            return 0.0

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


def create_alpaca_positions() -> AlpacaPositions:
    """Factory for future AI Trader components."""
    return AlpacaPositions()


def print_report(positions_manager: AlpacaPositions) -> None:
    summary = positions_manager.get_summary()

    print("AI TRADER - ALPACA POSITIONS")
    print("=" * 60)
    print()
    print(f"Environment:        {summary.environment}")
    print(f"Capital Base:       ${summary.capital_base:,.2f}")
    print()
    print(f"Positions:          {summary.position_count}")
    print(f"Long Market Value:  ${summary.long_market_value:,.2f}")
    print(f"Short Market Value: ${summary.short_market_value:,.2f}")
    print(f"Gross Exposure:     ${summary.gross_market_value:,.2f}")
    print(f"Total Exposure:     {summary.total_exposure_pct:.2f}%")
    print(f"Available Capital:  ${summary.available_capital:,.2f}")
    print(f"Unrealized P/L:     ${summary.total_unrealized_pl:,.2f}")
    print()

    if not summary.positions:
        print("[POSITIONS]")
        print("No open positions.")
    else:
        print("[POSITIONS]")

        for position in summary.positions:
            print()
            print(
                f"{position['symbol']} | "
                f"Qty={position['qty']:.4f} | "
                f"Side={position['side']} | "
                f"Entry=${position['avg_entry_price']:.2f} | "
                f"Current=${position['current_price']:.2f}"
            )
            print(
                f"  Market Value=${position['market_value']:,.2f} | "
                f"Cost Basis=${position['cost_basis']:,.2f}"
            )
            print(
                f"  Unrealized P/L=${position['unrealized_pl']:,.2f} | "
                f"P/L={position['unrealized_pl_pct'] * 100:.2f}%"
            )
            print(
                f"  Capital Exposure="
                f"{position['capital_exposure_pct']:.2f}%"
            )

    print()
    print("No trading operation was submitted.")


if __name__ == "__main__":
    positions = create_alpaca_positions()
    print_report(positions)
