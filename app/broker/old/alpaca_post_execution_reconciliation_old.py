"""
AI Trader - Alpaca Post Execution Reconciliation
Version: 1.0

Purpose
-------
Validate the real broker state AFTER an Alpaca Paper order has been
submitted.

Important distinction
---------------------
SUBMITTED != FILLED

The executor is responsible for submitting an order.
This module is responsible for determining what actually happened at Alpaca
after submission.

Safety principles
-----------------
1. PAPER execution only is assumed by the caller.
2. No order is submitted by this module.
3. An order that remains OPEN is never treated as FILLED.
4. PARTIALLY_FILLED is reported explicitly.
5. FILLED triggers a dynamic broker-state reconciliation.
6. CANCELED / EXPIRED / REJECTED are terminal non-fill outcomes.
7. A timeout never causes a second order submission.
8. Telegram is notification-only through the optional notifier callback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Order status groups
# ---------------------------------------------------------------------------

OPEN_STATUSES = {
    "PENDING_NEW",
    "ACCEPTED",
    "NEW",
    "CALCULATED",
    "PENDING_REPLACE",
    "PENDING_CANCEL",
    "HELD",
    "SUSPENDED",
}

PARTIAL_STATUSES = {
    "PARTIALLY_FILLED",
}

FILLED_STATUSES = {
    "FILLED",
}

CANCELED_STATUSES = {
    "CANCELED",
    "CANCELLED",
}

EXPIRED_STATUSES = {
    "EXPIRED",
    "DONE_FOR_DAY",
}

REJECTED_STATUSES = {
    "REJECTED",
}


@dataclass(frozen=True)
class PostExecutionConfig:
    """Safety and polling configuration."""

    poll_interval_seconds: float = 2.0
    timeout_seconds: float = 30.0
    reconcile_on_partial: bool = True
    reconcile_on_filled: bool = True
    notify_telegram: bool = True

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")


@dataclass(frozen=True)
class PostExecutionResult:
    """Auditable post-submission result."""

    order_id: str
    symbol: Optional[str]
    side: Optional[str]
    requested_qty: Optional[float]
    filled_qty: float
    filled_avg_price: Optional[float]
    broker_status: str
    execution_state: str
    terminal: bool
    timed_out: bool
    reconciled: bool
    reconciliation_message: str
    checked_at: str
    message: str

    @property
    def is_filled(self) -> bool:
        return self.execution_state == "FILLED"

    @property
    def is_partial(self) -> bool:
        return self.execution_state == "PARTIALLY_FILLED"

    @property
    def is_open(self) -> bool:
        return self.execution_state == "OPEN"

    @property
    def is_terminal_non_fill(self) -> bool:
        return self.execution_state in {
            "CANCELED",
            "EXPIRED",
            "REJECTED",
        }

    def summary(self) -> str:
        lines = [
            "ALPACA POST-EXECUTION RECONCILIATION",
            "=" * 60,
            "",
            f"Order ID:            {self.order_id}",
            f"Symbol:              {self.symbol or '-'}",
            f"Side:                {(self.side or '-').upper()}",
            (
                f"Requested Qty:      {self.requested_qty:g}"
                if self.requested_qty is not None
                else "Requested Qty:      -"
            ),
            f"Filled Qty:          {self.filled_qty:g}",
            (
                f"Filled Avg Price:    ${self.filled_avg_price:,.4f}"
                if self.filled_avg_price is not None
                else "Filled Avg Price:    -"
            ),
            f"Broker Status:       {self.broker_status}",
            f"Execution State:     {self.execution_state}",
            f"Terminal:            {self.terminal}",
            f"Timed Out:           {self.timed_out}",
            f"Reconciled:          {self.reconciled}",
            f"Checked At:          {self.checked_at}",
            "",
            self.reconciliation_message,
            "",
            self.message,
        ]
        return "\n".join(lines)


class AlpacaPostExecutionReconciliation:
    """
    Post-submission reconciliation against the live broker state.

    Expected client contract:
        client._client.get_order_by_id(order_id)

    Expected reconciliation contract:
        reconciliation.reconcile_current_state()

    No submit/cancel/replace operation exists in this class.
    """

    def __init__(
        self,
        client: Any,
        reconciliation: Any,
        notifier: Optional[Callable[[str], bool]] = None,
        config: Optional[PostExecutionConfig] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.reconciliation = reconciliation
        self.notifier = notifier
        self.config = config or PostExecutionConfig()
        self._sleep = sleep_fn

    def reconcile_order(
        self,
        order_id: str,
        *,
        wait: bool = True,
    ) -> PostExecutionResult:
        """
        Check an already-submitted Alpaca order.

        If wait=True, poll until:
        - FILLED
        - CANCELED/CANCELLED
        - EXPIRED/DONE_FOR_DAY
        - REJECTED
        - timeout

        A timeout returns the latest broker state and NEVER submits another
        order.
        """
        if not order_id or not str(order_id).strip():
            raise ValueError("order_id is required.")

        deadline = time.monotonic() + self.config.timeout_seconds

        while True:
            order = self._get_order(order_id)
            result = self._build_result(order, timed_out=False)

            if self._requires_reconciliation(result):
                return self._finalize(result)

            if result.terminal:
                return self._finalize(result)

            if not wait:
                return self._finalize(result)

            if time.monotonic() >= deadline:
                timed_out_result = self._build_result(
                    self._get_order(order_id),
                    timed_out=True,
                )
                return self._finalize(timed_out_result)

            self._sleep(self.config.poll_interval_seconds)

    def _get_order(self, order_id: str) -> Any:
        broker = getattr(self.client, "_client", self.client)

        getter = getattr(broker, "get_order_by_id", None)
        if getter is None:
            raise AttributeError(
                "Alpaca client does not expose get_order_by_id()."
            )

        return getter(order_id)

    def _build_result(
        self,
        order: Any,
        *,
        timed_out: bool,
    ) -> PostExecutionResult:
        if order is None:
            raise RuntimeError("Alpaca returned no order for the supplied order_id.")

        order_id = str(self._value(order, "id", ""))
        if not order_id:
            raise RuntimeError("Alpaca order response has no order id.")

        raw_status = self._normalize_status(
            self._value(order, "status", "UNKNOWN")
        )
        state = self._map_execution_state(raw_status)

        filled_qty = self._to_float(
            self._value(order, "filled_qty", 0)
        )
        requested_qty = self._to_float_or_none(
            self._value(order, "qty", None)
        )
        filled_avg_price = self._to_float_or_none(
            self._value(order, "filled_avg_price", None)
        )

        symbol = self._string_or_none(self._value(order, "symbol", None))
        side = self._string_or_none(self._value(order, "side", None))

        terminal = state in {
            "FILLED",
            "CANCELED",
            "EXPIRED",
            "REJECTED",
        }

        checked_at = datetime.now(timezone.utc).isoformat()

        if state == "FILLED":
            message = "Order is FILLED at Alpaca."
        elif state == "PARTIALLY_FILLED":
            message = (
                "Order is PARTIALLY_FILLED. Remaining quantity is still "
                "subject to broker execution."
            )
        elif state == "OPEN":
            message = (
                "Order remains OPEN at Alpaca. It has not been treated as "
                "filled."
            )
        elif state == "CANCELED":
            message = "Order was CANCELED by Alpaca."
        elif state == "EXPIRED":
            message = "Order EXPIRED/DONE_FOR_DAY without a final fill."
        elif state == "REJECTED":
            message = "Order was REJECTED by Alpaca."
        else:
            message = f"Broker returned unclassified status: {raw_status}."

        return PostExecutionResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            requested_qty=requested_qty,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            broker_status=raw_status,
            execution_state=state,
            terminal=terminal,
            timed_out=timed_out,
            reconciled=False,
            reconciliation_message="Reconciliation not required yet.",
            checked_at=checked_at,
            message=message,
        )

    def _requires_reconciliation(
        self,
        result: PostExecutionResult,
    ) -> bool:
        if result.is_filled:
            return self.config.reconcile_on_filled

        if result.is_partial:
            return self.config.reconcile_on_partial

        return False

    def _finalize(self, result: PostExecutionResult) -> PostExecutionResult:
        reconciliation_ok = False
        reconciliation_message = "Broker state not reconciled."

        should_reconcile = (
            result.is_filled
            or (result.is_partial and self.config.reconcile_on_partial)
        )

        if should_reconcile:
            try:
                report = self.reconciliation.reconcile_current_state()
                reconciliation_ok = bool(
                    getattr(report, "reconciled", False)
                )
                reconciliation_message = (
                    "Dynamic broker-state reconciliation PASS."
                    if reconciliation_ok
                    else "Dynamic broker-state reconciliation FAILED."
                )
            except Exception as exc:
                reconciliation_message = (
                    f"Dynamic broker-state reconciliation ERROR: {exc}"
                )

        final_result = PostExecutionResult(
            order_id=result.order_id,
            symbol=result.symbol,
            side=result.side,
            requested_qty=result.requested_qty,
            filled_qty=result.filled_qty,
            filled_avg_price=result.filled_avg_price,
            broker_status=result.broker_status,
            execution_state=result.execution_state,
            terminal=result.terminal,
            timed_out=result.timed_out,
            reconciled=reconciliation_ok,
            reconciliation_message=reconciliation_message,
            checked_at=result.checked_at,
            message=result.message,
        )

        self._notify(final_result)
        return final_result

    def _notify(self, result: PostExecutionResult) -> None:
        if not self.config.notify_telegram or self.notifier is None:
            return

        try:
            self.notifier(result.summary())
        except Exception:
            # Notification failure must never change the broker execution
            # state or cause a second order.
            pass

    @staticmethod
    def _map_execution_state(status: str) -> str:
        if status in FILLED_STATUSES:
            return "FILLED"
        if status in PARTIAL_STATUSES:
            return "PARTIALLY_FILLED"
        if status in CANCELED_STATUSES:
            return "CANCELED"
        if status in EXPIRED_STATUSES:
            return "EXPIRED"
        if status in REJECTED_STATUSES:
            return "REJECTED"
        if status in OPEN_STATUSES:
            return "OPEN"
        return "UNKNOWN"

    @staticmethod
    def _normalize_status(value: Any) -> str:
        if hasattr(value, "value"):
            value = value.value
        return str(value).strip().upper()

    @staticmethod
    def _value(obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @staticmethod
    def _to_float(value: Any) -> float:
        if value is None:
            return 0.0
        return float(value)

    @staticmethod
    def _to_float_or_none(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        return float(value)

    @staticmethod
    def _string_or_none(value: Any) -> Optional[str]:
        if value is None:
            return None
        return str(value)


def create_post_execution_reconciliation(
    client: Any,
    reconciliation: Any,
    notifier: Optional[Callable[[str], bool]] = None,
    config: Optional[PostExecutionConfig] = None,
) -> AlpacaPostExecutionReconciliation:
    """Factory for the post-execution reconciliation layer."""
    return AlpacaPostExecutionReconciliation(
        client=client,
        reconciliation=reconciliation,
        notifier=notifier,
        config=config,
    )


if __name__ == "__main__":
    print("AI TRADER - POST EXECUTION RECONCILIATION")
    print("=" * 60)
    print("Module: alpaca_post_execution_reconciliation.py")
    print("Version: 1.0")
    print()
    print("Safety: NO ORDER SUBMISSION")
    print("Purpose: validate an already-submitted Alpaca order.")
    print("Telegram: notification only")
    print("Status: READY")
