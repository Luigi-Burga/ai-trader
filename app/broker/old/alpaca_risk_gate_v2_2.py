"""
AI Trader - Alpaca Risk Gate
Version: 2.2

Read-only safety barrier between Trade Intent and the future Executor.
This module NEVER submits, cancels, or modifies Alpaca orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.broker.alpaca_account import AI_TRADER_CAPITAL_BASE, AlpacaAccount
from app.broker.alpaca_orders import AlpacaOrders
from app.broker.alpaca_positions import AlpacaPositions
from app.broker.alpaca_reconciliation import AlpacaReconciliation
from app.broker.alpaca_trade_intent import TradeIntent


@dataclass(frozen=True)
class RiskConfig:
    """Configurable risk limits for AI Trader."""

    max_portfolio_exposure_pct: float = 80.0
    max_position_exposure_pct: float = 20.0
    market_order_slippage_buffer_pct: float = 2.0
    min_confidence: float = 60.0
    allowed_buy_signals: frozenset[str] = frozenset({"BUY", "STRONG_BUY"})
    allowed_sell_signals: frozenset[str] = frozenset({"SELL", "REDUCE"})
    require_reconciliation: bool = True
    reject_if_open_same_side: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.max_portfolio_exposure_pct <= 100:
            raise ValueError(
                "max_portfolio_exposure_pct must be > 0 and <= 100."
            )
        if not 0 < self.max_position_exposure_pct <= 100:
            raise ValueError(
                "max_position_exposure_pct must be > 0 and <= 100."
            )
        if self.market_order_slippage_buffer_pct < 0:
            raise ValueError("market_order_slippage_buffer_pct must be >= 0.")
        if not 0 <= self.min_confidence <= 100:
            raise ValueError(
                "min_confidence must be between 0 and 100."
            )


@dataclass(frozen=True)
class RiskCheck:
    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class RiskDecision:
    """Final Risk Gate decision. Approval never submits an order."""

    approved: bool
    symbol: str
    side: str
    quantity: float
    estimated_notional: float
    capital_base: float
    current_exposure: float
    current_exposure_pct: float
    requested_exposure_pct: float
    projected_exposure_pct: float
    checks: tuple[RiskCheck, ...] = field(default_factory=tuple)
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        status = "APPROVED" if self.approved else "REJECTED"

        lines = [
            "RISK GATE",
            "=" * 60,
            "",
            f"Risk Decision:       {status}",
            f"Symbol:              {self.symbol}",
            f"Side:                {self.side.upper()}",
            f"Quantity:            {self.quantity:g}",
            f"Estimated Notional:  ${self.estimated_notional:,.2f}",
            "",
            f"Capital Base:        ${self.capital_base:,.2f}",
            f"Current Exposure:     ${self.current_exposure:,.2f}",
            f"Current Exposure %:    {self.current_exposure_pct:.2f}%",
            f"Requested Exposure %:  {self.requested_exposure_pct:.2f}%",
            f"Projected Exposure %:  {self.projected_exposure_pct:.2f}%",
            "",
            "CHECKS",
            "-" * 60,
        ]

        for check in self.checks:
            status_text = "PASS" if check.passed else "FAIL"
            lines.append(
                f"{status_text:<6} {check.name}: {check.message}"
            )

        if self.rejection_reasons:
            lines.extend(["", "REJECTION REASONS", "-" * 60])
            lines.extend(
                f"- {reason}" for reason in self.rejection_reasons
            )

        lines.extend(
            [
                "",
                f"FINAL DECISION: {status}",
                "No Alpaca order was submitted.",
            ]
        )
        return "\n".join(lines)


class AlpacaRiskGate:
    """Read-only evaluator for TradeIntent objects."""

    def __init__(
        self,
        account: Optional[AlpacaAccount] = None,
        positions: Optional[AlpacaPositions] = None,
        orders: Optional[AlpacaOrders] = None,
        reconciliation: Optional[AlpacaReconciliation] = None,
        config: Optional[RiskConfig] = None,
    ) -> None:
        self.account = account or AlpacaAccount()
        self.positions = positions or AlpacaPositions()
        self.orders = orders or AlpacaOrders()
        self.reconciliation = reconciliation or AlpacaReconciliation()
        self.config = config or RiskConfig()

    def evaluate(self, intent: TradeIntent, reconciliation_report=None, market_price: Optional[float] = None) -> RiskDecision:
        checks: list[RiskCheck] = []
        rejection_reasons: list[str] = []

        account_snapshot = self.account.snapshot()
        positions_summary = self.positions.get_summary()
        capital_base = AI_TRADER_CAPITAL_BASE

        current_exposure = positions_summary.gross_market_value
        current_exposure_pct = current_exposure / capital_base * 100
        if intent.order_type == "market":
            if market_price is None or float(market_price) <= 0:
                estimated_notional = 0.0
                market_price_message = "Market order requires a positive current market price for risk estimation."
            else:
                market_price = float(market_price)
                buffer = 1.0 + (self.config.market_order_slippage_buffer_pct / 100.0)
                estimated_notional = intent.quantity * market_price * buffer
                market_price_message = (
                    f"Market price=${market_price:,.2f}; "
                    f"risk buffer={self.config.market_order_slippage_buffer_pct:.2f}%; "
                    f"risk notional=${estimated_notional:,.2f}."
                )
        else:
            estimated_notional = intent.notional_value or 0.0
            market_price_message = "Not applicable for non-market order."

        requested_exposure_pct = estimated_notional / capital_base * 100

        if intent.is_buy:
            projected_exposure = current_exposure + estimated_notional
        else:
            projected_exposure = current_exposure

        projected_exposure_pct = projected_exposure / capital_base * 100

        def add_check(name: str, passed: bool, message: str) -> None:
            checks.append(RiskCheck(name=name, passed=passed, message=message))
            if not passed:
                rejection_reasons.append(f"{name}: {message}")

        # 1. Trade Intent
        try:
            intent.to_dict()
            intent_valid = True
            intent_message = "TradeIntent structure is valid."
        except Exception as exc:
            intent_valid = False
            intent_message = f"TradeIntent validation failed: {exc}"

        add_check("TRADE_INTENT", intent_valid, intent_message)

        # 2. Paper environment
        environment = self._get_environment(account_snapshot)
        environment_ok = environment.upper() == "PAPER"
        add_check(
            "ENVIRONMENT",
            environment_ok,
            f"Environment={environment}; PAPER required.",
        )

        # 3. Account readiness
        account_ready = bool(account_snapshot.ready)
        add_check(
            "ACCOUNT",
            account_ready,
            (
                "Alpaca account is ready."
                if account_ready
                else "Alpaca account is not ready for trading."
            ),
        )

        # 4. Fixed strategy capital
        capital_ok = capital_base == 100_000.00
        add_check(
            "CAPITAL_BASE",
            capital_ok,
            f"AI Trader capital base=${capital_base:,.2f}.",
        )

        # 5. Notional
        notional_ok = estimated_notional > 0
        add_check(
            "NOTIONAL",
            notional_ok,
            f"Requested notional=${estimated_notional:,.2f}.",
        )

        if intent.order_type == "market":
            add_check("MARKET_PRICE", market_price is not None and float(market_price) > 0, market_price_message)

        # 6. Confidence
        confidence = intent.confidence
        confidence_ok = (
            confidence is not None
            and confidence >= self.config.min_confidence
        )
        confidence_message = (
            f"Confidence={confidence:.2f}%; "
            f"minimum={self.config.min_confidence:.2f}%."
            if confidence is not None
            else
            f"Confidence is missing; minimum="
            f"{self.config.min_confidence:.2f}%."
        )
        add_check("CONFIDENCE", confidence_ok, confidence_message)

        # 7. Signal
        signal = (intent.signal or "").strip().upper()
        if intent.is_buy:
            allowed_signals = self.config.allowed_buy_signals
        else:
            allowed_signals = self.config.allowed_sell_signals

        signal_ok = signal in allowed_signals
        add_check(
            "SIGNAL",
            signal_ok,
            (
                f"Signal={signal}; allowed={sorted(allowed_signals)}."
                if signal
                else f"Signal is missing; allowed={sorted(allowed_signals)}."
            ),
        )

        # 8. Position exposure
        position_ok = (
            requested_exposure_pct
            <= self.config.max_position_exposure_pct
        )
        add_check(
            "POSITION_EXPOSURE",
            position_ok,
            (
                f"Requested exposure={requested_exposure_pct:.2f}%; "
                f"maximum per position="
                f"{self.config.max_position_exposure_pct:.2f}%."
            ),
        )

        # 9. Portfolio exposure
        portfolio_ok = (
            projected_exposure_pct
            <= self.config.max_portfolio_exposure_pct
        )
        add_check(
            "PORTFOLIO_EXPOSURE",
            portfolio_ok,
            (
                f"Projected exposure={projected_exposure_pct:.2f}%; "
                f"maximum={self.config.max_portfolio_exposure_pct:.2f}%."
            ),
        )

        # 10. Available strategy capital
        available_capital = max(0.0, capital_base - current_exposure)
        capital_available_ok = (
            not intent.is_buy
            or estimated_notional <= available_capital
        )
        add_check(
            "AVAILABLE_CAPITAL",
            capital_available_ok,
            (
                f"Available strategy capital=${available_capital:,.2f}; "
                f"requested=${estimated_notional:,.2f}."
                if intent.is_buy
                else "SELL does not consume additional strategy capital."
            ),
        )

        # 11. Reconciliation
        reconciliation_error = None
        if reconciliation_report is None:
            try:
                reconciliation_report = (
                    self.reconciliation.reconcile_current_state()
                )
            except Exception as exc:
                reconciliation_error = str(exc)

        if self.config.require_reconciliation:
            reconciliation_ok = (
                reconciliation_report is not None
                and bool(reconciliation_report.reconciled)
            )
            if reconciliation_error:
                reconciliation_message = (
                    f"Unable to obtain reconciliation: {reconciliation_error}"
                )
            elif reconciliation_report is None:
                reconciliation_message = "No reconciliation report available."
            elif reconciliation_ok:
                reconciliation_message = "Account/portfolio state is reconciled."
            else:
                reconciliation_message = (
                    "Account/portfolio state is NOT reconciled."
                )
        else:
            reconciliation_ok = True
            reconciliation_message = (
                "Reconciliation requirement disabled by configuration."
            )

        add_check(
            "RECONCILIATION",
            reconciliation_ok,
            reconciliation_message,
        )

        # 12. Duplicate open order
        duplicate_ok = True
        duplicate_message = ""

        if self.config.reject_if_open_same_side:
            try:
                duplicate_order = self.orders.has_open_order(
                    symbol=intent.symbol,
                    side=intent.side,
                )
                duplicate_ok = not duplicate_order
                duplicate_message = (
                    f"No open {intent.side.upper()} order for {intent.symbol}."
                    if duplicate_ok
                    else
                    f"An open {intent.side.upper()} order already exists "
                    f"for {intent.symbol}."
                )
            except Exception as exc:
                duplicate_ok = False
                duplicate_message = (
                    f"Unable to verify open orders: {exc}"
                )
        else:
            duplicate_message = (
                "Duplicate-order check disabled by configuration."
            )

        add_check(
            "OPEN_ORDER_CONFLICT",
            duplicate_ok,
            duplicate_message,
        )

        approved = len(rejection_reasons) == 0

        return RiskDecision(
            approved=approved,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            estimated_notional=estimated_notional,
            capital_base=capital_base,
            current_exposure=current_exposure,
            current_exposure_pct=current_exposure_pct,
            requested_exposure_pct=requested_exposure_pct,
            projected_exposure_pct=projected_exposure_pct,
            checks=tuple(checks),
            rejection_reasons=tuple(rejection_reasons),
        )

    def approve(self, intent: TradeIntent, reconciliation_report=None) -> bool:
        """Return True only when the Risk Gate approves the intent."""
        return self.evaluate(
            intent,
            reconciliation_report=reconciliation_report,
        ).approved

    @staticmethod
    def _get_environment(account_snapshot) -> str:
        environment = getattr(account_snapshot, "environment", None)
        if environment is None:
            return "UNKNOWN"
        return str(getattr(environment, "value", environment))


def create_risk_gate(config: Optional[RiskConfig] = None) -> AlpacaRiskGate:
    """Create the default read-only Risk Gate."""
    return AlpacaRiskGate(config=config)


def _run_self_test() -> None:
    """Read-only self-test against the current Paper Trading account."""

    print("AI TRADER - ALPACA RISK GATE")
    print("=" * 60)
    print()
    print("Mode: PAPER / READ-ONLY")
    print(f"Strategy Capital Base: ${AI_TRADER_CAPITAL_BASE:,.2f}")
    print()

    intent = TradeIntent(
        symbol="NVDA",
        side="buy",
        quantity=100,
        order_type="market",
        time_in_force="day",
        reason="Risk Gate self-test",
        confidence=84.2,
        strategy="Characteristic Curve + Price Action",
        signal="BUY",
    )

    print("TRADE INTENT")
    print("-" * 60)
    print(intent.summary())
    print()

    gate = create_risk_gate()
    decision = gate.evaluate(intent, market_price=180.50)

    print(decision.summary())
    print()
    print(
        "SELF-TEST RESULT:",
        "PASS" if decision.approved else "REJECTED",
    )
    print("No Alpaca order was submitted.")


if __name__ == "__main__":
    _run_self_test()
