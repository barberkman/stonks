"""In-memory FakeContext + helpers for pure-Python unit tests of strategies.

The engine-bound `Context` cannot be constructed from Python directly (it
originates from the C++ engine). `FakeContext` mirrors its surface so
strategies can be exercised without spinning up the C++ binary.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from stonks import OrderSide, TimeInForce


@dataclass
class FakeKLine:
    """Stand-in for the C++-bound KLine. `timestamp` accepts int (ms) or a
    `stonks.Timestamp` — the engine-bound KLine uses Timestamp; tests
    typically use ints for brevity."""

    timestamp: Any
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class FakeOrder:
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC


class FakeContext:
    """Drop-in stand-in for `Context` in unit tests.

    Construct with a pre-built list of bars; call `advance()` to step the
    cursor before each `on_tick`. Placed orders accumulate in `.orders` for
    test assertions.
    """

    def __init__(self, bars: List[FakeKLine], cash: float = 100_000.0):
        self._bars = bars
        self._cursor = 0
        self._cash = cash
        self.orders: List[FakeOrder] = []

    def advance(self) -> None:
        self._cursor += 1

    def now(self) -> Any:
        if self._cursor == 0:
            return 0
        return self._bars[self._cursor - 1].timestamp

    def cash(self) -> float:
        return self._cash

    def equity(self) -> float:
        return self._cash

    def klines(self, *args, **kwargs) -> List[FakeKLine]:
        if len(args) == 1 and isinstance(args[0], int) and not kwargs:
            count = args[0]
            return list(self._bars[max(0, self._cursor - count) : self._cursor])
        if len(args) >= 1:
            start = args[0]
            end = args[1] if len(args) > 1 else kwargs.get("end")
            return [
                b
                for b in self._bars[: self._cursor]
                if b.timestamp >= start and (end is None or b.timestamp <= end)
            ]
        raise TypeError(
            "klines() expects (count: int) or (start, end=None)"
        )

    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> bool:
        self.orders.append(FakeOrder(symbol, side, quantity, None, time_in_force))
        return True

    def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> bool:
        self.orders.append(FakeOrder(symbol, side, quantity, price, time_in_force))
        return True
