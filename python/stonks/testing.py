"""In-memory FakeContext + helpers for pure-Python unit tests of strategies.

The engine-bound `Context` cannot be constructed from Python directly (it
originates from the C++ engine). `FakeContext` mirrors its surface so
strategies can be exercised without spinning up the C++ binary.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from stonks import OrderSide, OrderType


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


def _to_ms(ts: Any) -> int:
    """Normalize a FakeKLine timestamp (int ms or stonks.Timestamp) to int ms."""
    if hasattr(ts, "to_millis"):
        return int(ts.to_millis())
    return int(ts)


@dataclass
class FakeMarketWindow:
    """Stand-in for the C++-bound MarketWindow returned by Context.history():
    all printing symbols gathered into one long frame, `symbol` per row. Mirrors
    the real columns so strategies build a pandas DataFrame unchanged."""

    symbol: List[str]
    timestamp: "np.ndarray"
    open: "np.ndarray"
    high: "np.ndarray"
    low: "np.ndarray"
    close: "np.ndarray"
    volume: "np.ndarray"

    def __len__(self) -> int:
        return len(self.symbol)


@dataclass
class FakeOrder:
    symbol: str
    side: OrderSide
    quantity: float
    type: OrderType = OrderType.Market
    price: Optional[float] = None
    stop_loss: Optional[float] = None      # exit-on-loss level carried by the entry
    take_profit: Optional[float] = None    # exit-on-profit level
    ttl_ms: Optional[int] = None           # unfilled-order lifetime in ms; None = good-till-cancelled
    id: Optional[int] = None               # the OrderID place_order handed back


class FakeContext:
    """Drop-in stand-in for `Context` in unit tests.

    Construct with a pre-built list of bars (multi-symbol, in time order). Ticks
    are per timestamp: call `advance()` once per distinct timestamp before each
    `on_tick`. Placed orders accumulate in `.orders` for test assertions.

    Fills are not simulated, so `position()` always reports flat — strategies are
    exercised as if they hold nothing.
    """

    def __init__(self, bars: List[FakeKLine], cash: float = 100_000.0):
        self._bars = bars
        self._cash = cash
        self.orders: List[FakeOrder] = []
        self.closes: List[str] = []                        # symbols passed to close()
        self.exit_updates: List[Any] = []                  # (symbol, stop_loss, take_profit) tuples
        self._timestamps = sorted({_to_ms(b.timestamp) for b in bars})
        self._group = -1   # advance() steps to the first timestamp
        self._next_order_id = 1   # broker hands back monotonic OrderIDs

    def advance(self) -> None:
        self._group += 1

    def now(self) -> int:
        if self._group < 0:
            return 0
        return self._timestamps[self._group]

    def cash(self) -> float:
        return self._cash

    def equity(self) -> float:
        return self._cash

    def position(self, symbol: str):
        return None

    def history(self, count: int) -> FakeMarketWindow:
        now_ms = self.now()
        printers = [b for b in self._bars if _to_ms(b.timestamp) == now_ms]

        symbol: List[str] = []
        ts, op, hi, lo, cl, vo = [], [], [], [], [], []
        for pb in printers:
            hist = [
                b for b in self._bars
                if b.symbol == pb.symbol and _to_ms(b.timestamp) <= now_ms
            ]
            window = hist[-count:] if count > 0 else []
            for b in window:
                symbol.append(b.symbol)
                ts.append(_to_ms(b.timestamp))
                op.append(b.open); hi.append(b.high); lo.append(b.low)
                cl.append(b.close); vo.append(b.volume)

        return FakeMarketWindow(
            symbol=symbol,
            timestamp=np.array(ts, dtype=np.int64),
            open=np.array(op, dtype=np.float64),
            high=np.array(hi, dtype=np.float64),
            low=np.array(lo, dtype=np.float64),
            close=np.array(cl, dtype=np.float64),
            volume=np.array(vo, dtype=np.float64),
        )

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        type: OrderType = OrderType.Market,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        ttl_ms: Optional[int] = None,
    ) -> int:
        oid = self._next_order_id
        self._next_order_id += 1
        self.orders.append(
            FakeOrder(symbol, side, quantity, type, price, stop_loss, take_profit, ttl_ms, oid)
        )
        return oid

    def close(self, symbol: str) -> bool:
        self.closes.append(symbol)
        return True

    def update_exits(
        self,
        symbol: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        self.exit_updates.append((symbol, stop_loss, take_profit))
        return True
