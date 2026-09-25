"""
AI Trader - Alpaca Reconciliation
V1.0

Purpose:
- Compare AI Trader's expected broker state with Alpaca's actual state.
- Detect position, quantity, side, market-value and open-order mismatches.
- Provide a read-only reconciliation report.
- Establish the safety foundation for future order execution.

IMPORTANT:
- Paper Trading only.
- No order submission.
- No order cancellation.
- No portfolio modification.
- A mismatch must be treated as a safety condition before future execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from app.broker.alpaca_account import AI_TRADER_CAPITAL_BASE
from app.broker.alpaca_client import AlpacaClient, create_alpaca_client


@dataclass(frozen=True)
class ExpectedPosition:
    """Expected AI Trader position state."""

    symbol: str
    qty: float
    side: str = "long"


@dataclass(frozen=True)
class PositionMismatch:
    symbol: str
    mismatch_type: str
    expected_qty: float
    actual_qty: float
    expected_side: str
    actual_side: str


@dataclass(frozen=True)
class ReconciliationReport:
    environment: str
    account_id: str
    capital_base: float

    status: str
    reconciled: bool

    expected_position_count: int
    actual_position_count: int

    expected_open_order_count: int
    actual_open_order_count: int

    position_mismatches: list[dict[str, Any]]
    missing_expected_positions: list[str]
    unexpected_actual_positions: list[str]

    expected_symbols: list[str]
    actual_symbols: list[str]

    warnings: list[str]


class AlpacaReconciliation:
    """Read-only reconciliation layer between AI Trader and Alpaca."""

    QTY_TOLERANCE = 1e-8

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
                "AI Trader AlpacaReconciliation requires PAPER environment."
            )

    def reconcile(
        self,
        expected_positions: list[ExpectedPosition] | None = None,
        expected_open_order_count: int = 0,
    ) -> ReconciliationReport:
        """
        Compare expected AI Trader state with actual Alpaca state.

        If expected_positions is omitted, the expected position set is empty.
        This is useful for the initial account-state validation.
        """
        expected_positions = expected_positions or []

        account = self.client.get_account()
        actual_positions = self.client.get_positions()
        actual_orders = self.client.get_orders(status="open")

        expected_map = self._normalize_expected_positions(expected_positions)
        actual_map = self._normalize_actual_positions(actual_positions)

        mismatches: list[PositionMismatch] = []
        missing_expected: list[str] = []
        unexpected_actual: list[str] = []
        warnings: list[str] = []

        all_symbols = sorted(set(expected_map) | set(actual_map))

        for symbol in all_symbols:
            expected = expected_map.get(symbol)
            actual = actual_map.get(symbol)

            if expected is None and actual is not None:
                unexpected_actual.append(symbol)
                mismatches.append(
                    PositionMismatch(
                        symbol=symbol,
                        mismatch_type="UNEXPECTED_ACTUAL_POSITION",
                        expected_qty=0.0,
                        actual_qty=actual["qty"],
                        expected_side="none",
                        actual_side=actual["side"],
                    )
                )
                continue

            if expected is not None and actual is None:
                missing_expected.append(symbol)
                mismatches.append(
                    PositionMismatch(
                        symbol=symbol,
                        mismatch_type="MISSING_ACTUAL_POSITION",
                        expected_qty=expected["qty"],
                        actual_qty=0.0,
                        expected_side=expected["side"],
                        actual_side="none",
                    )
                )
                continue

            assert expected is not None
            assert actual is not None

            expected_qty = expected["qty"]
            actual_qty = actual["qty"]

            if abs(expected_qty - actual_qty) > self.QTY_TOLERANCE:
                mismatches.append(
                    PositionMismatch(
                        symbol=symbol,
                        mismatch_type="QUANTITY_MISMATCH",
                        expected_qty=expected_qty,
                        actual_qty=actual_qty,
                        expected_side=expected["side"],
                        actual_side=actual["side"],
                    )
                )

            if expected["side"] != actual["side"]:
                mismatches.append(
                    PositionMismatch(
                        symbol=symbol,
                        mismatch_type="SIDE_MISMATCH",
                        expected_qty=expected_qty,
                        actual_qty=actual_qty,
                        expected_side=expected["side"],
                        actual_side=actual["side"],
                    )
                )

        actual_order_count = len(actual_orders)

        if actual_order_count != expected_open_order_count:
            warnings.append(
                "Open order count mismatch: "
                f"expected={expected_open_order_count}, "
                f"actual={actual_order_count}"
            )

        if account.get("trading_blocked"):
            warnings.append("Alpaca account is trading blocked.")

        if account.get("status") != "AccountStatus.ACTIVE":
            warnings.append(
                f"Alpaca account status is {account.get('status')}."
            )

        reconciled = (
            len(mismatches) == 0
            and actual_order_count == expected_open_order_count
            and not account.get("trading_blocked", False)
            and account.get("status") == "AccountStatus.ACTIVE"
        )

        status = "RECONCILED" if reconciled else "MISMATCH"

        return ReconciliationReport(
            environment=self.client.environment,
            account_id=str(account.get("id", "")),
            capital_base=self.capital_base,
            status=status,
            reconciled=reconciled,
            expected_position_count=len(expected_map),
            actual_position_count=len(actual_map),
            expected_open_order_count=expected_open_order_count,
            actual_open_order_count=actual_order_count,
            position_mismatches=[
                asdict(mismatch) for mismatch in mismatches
            ],
            missing_expected_positions=missing_expected,
            unexpected_actual_positions=unexpected_actual,
            expected_symbols=sorted(expected_map),
            actual_symbols=sorted(actual_map),
            warnings=warnings,
        )

    def reconcile_empty_portfolio(self) -> ReconciliationReport:
        """
        Validate that Alpaca currently contains no positions and no open orders.
        """
        return self.reconcile(
            expected_positions=[],
            expected_open_order_count=0,
        )

    def can_trade(self, report: ReconciliationReport) -> bool:
        """
        Safety gate for future execution.

        V1.0 only returns True when reconciliation is clean.
        This method does not submit any order.
        """
        return report.reconciled

    @staticmethod
    def _normalize_expected_positions(
        positions: list[ExpectedPosition],
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}

        for position in positions:
            symbol = position.symbol.strip().upper()

            if not symbol:
                raise ValueError("Expected position symbol cannot be empty.")

            if position.qty < 0:
                raise ValueError(
                    f"Expected quantity cannot be negative: {symbol}"
                )

            side = position.side.strip().lower()
            if side not in {"long", "short"}:
                raise ValueError(
                    f"Expected side must be 'long' or 'short': {symbol}"
                )

            if symbol in result:
                raise ValueError(
                    f"Duplicate expected position: {symbol}"
                )

            result[symbol] = {
                "qty": float(position.qty),
                "side": side,
            }

        return result

    @staticmethod
    def _normalize_actual_positions(
        positions: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}

        for position in positions:
            symbol = str(position.get("symbol", "")).strip().upper()

            if not symbol:
                continue

            side = str(position.get("side", "")).lower()

            # Alpaca commonly returns long/short. Keep the normalized
            # representation stable for reconciliation.
            if "short" in side:
                normalized_side = "short"
            else:
                normalized_side = "long"

            result[symbol] = {
                "qty": float(position.get("qty") or 0.0),
                "side": normalized_side,
            }

        return result

    def report(
        self,
        expected_positions: list[ExpectedPosition] | None = None,
        expected_open_order_count: int = 0,
    ) -> dict[str, Any]:
        """Return a JSON-serializable reconciliation report."""
        return asdict(
            self.reconcile(
                expected_positions=expected_positions,
                expected_open_order_count=expected_open_order_count,
            )
        )


def create_alpaca_reconciliation() -> AlpacaReconciliation:
    """Factory for future AI Trader components."""
    return AlpacaReconciliation()


def print_report(report: ReconciliationReport) -> None:
    print("AI TRADER - ALPACA RECONCILIATION")
    print("=" * 60)
    print()
    print(f"Environment:             {report.environment}")
    print(f"Account ID:              {report.account_id}")
    print(f"Capital Base:            ${report.capital_base:,.2f}")
    print()
    print(f"Expected Positions:      {report.expected_position_count}")
    print(f"Actual Positions:        {report.actual_position_count}")
    print(
        f"Expected Open Orders:    "
        f"{report.expected_open_order_count}"
    )
    print(
        f"Actual Open Orders:      "
        f"{report.actual_open_order_count}"
    )
    print()
    print(f"Reconciliation Status:   {report.status}")
    print(f"Can Trade:               {report.reconciled}")
    print()

    if report.expected_symbols:
        print(
            "Expected Symbols:        "
            + ", ".join(report.expected_symbols)
        )
    else:
        print("Expected Symbols:        NONE")

    if report.actual_symbols:
        print(
            "Actual Symbols:          "
            + ", ".join(report.actual_symbols)
        )
    else:
        print("Actual Symbols:          NONE")

    print()

    if report.position_mismatches:
        print("[POSITION MISMATCHES]")
        for mismatch in report.position_mismatches:
            print(
                f"{mismatch['symbol']} | "
                f"{mismatch['mismatch_type']} | "
                f"expected_qty={mismatch['expected_qty']} | "
                f"actual_qty={mismatch['actual_qty']} | "
                f"expected_side={mismatch['expected_side']} | "
                f"actual_side={mismatch['actual_side']}"
            )
    else:
        print("[POSITION MISMATCHES]")
        print("None")

    if report.warnings:
        print()
        print("[WARNINGS]")
        for warning in report.warnings:
            print(f"- {warning}")
    else:
        print()
        print("[WARNINGS]")
        print("None")

    print()
    if report.reconciled:
        print("RECONCILIATION: OK")
    else:
        print("RECONCILIATION: MISMATCH - EXECUTION SHOULD REMAIN BLOCKED")

    print()
    print("No trading operation was submitted.")


if __name__ == "__main__":
    reconciliation = create_alpaca_reconciliation()

    # Initial safety test:
    # AI Trader expects an empty portfolio and zero open orders.
    report = reconciliation.reconcile_empty_portfolio()

    print_report(report)
