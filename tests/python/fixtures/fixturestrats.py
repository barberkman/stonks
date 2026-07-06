"""Fixture strategies used by tests/python/pythonstrategy_test.cpp.

Each class is small and observable through the orders it places on the
StubBroker — letting the C++ side verify dispatch behavior without needing
to peek inside the Python instance.
"""

import stonks
from stonks import OrderSide


class NoOnTick:
    """Bare class without on_tick. PythonStrategy must reject this at
    construction with a clear runtime_error."""
    pass


class BareTickOnly:
    """No inheritance, on_tick only. Verifies that on_start / on_stop are
    silently skipped when not defined."""

    def on_tick(self, ctx):
        ctx.place_market_order(symbol="X", side=OrderSide.Buy, quantity=2.0)


class CallRecording(stonks.Strategy):
    """Places a buy in each lifecycle hook with a quantity that encodes which
    hook fired. Tests assert the order/quantities sequence to verify dispatch."""

    def on_start(self, ctx):
        ctx.place_market_order(symbol="X", side=OrderSide.Buy, quantity=1.0)

    def on_tick(self, ctx):
        ctx.place_market_order(symbol="X", side=OrderSide.Buy, quantity=2.0)

    def on_stop(self, ctx):
        ctx.place_market_order(symbol="X", side=OrderSide.Buy, quantity=3.0)


class RaisingStrategy:
    """on_tick raises a Python exception. PythonStrategy must wrap it as
    std::runtime_error containing the traceback."""

    def on_tick(self, ctx):
        raise RuntimeError("boom from python")


class BracketPlacing(stonks.Strategy):
    """Places one order of each kind through the real bindings — a stop entry,
    a stop-loss child, and a limit take-profit child — so the C++ side can
    verify the pybind11 lambdas build the right params (type, price, leverage,
    parent linkage)."""

    def on_tick(self, ctx):
        entry = ctx.place_stop_order(
            symbol="X", side=OrderSide.Buy, quantity=2.0, price=110.0, leverage=5.0
        )
        ctx.place_stop_order(
            symbol="X", side=OrderSide.Sell, quantity=2.0, price=95.0, parent=entry
        )
        ctx.place_limit_order(
            symbol="X", side=OrderSide.Sell, quantity=2.0, price=130.0, parent=entry
        )


class ApiProbe(stonks.Strategy):
    """Probes the observability API through the real bindings: position() on a
    flat book, order() status round-trip, cancel_order() semantics, and the
    reduce_only flag. Raises (surfacing as runtime_error) on any mismatch, then
    places one reduce-only stop marker the C++ side inspects."""

    def on_tick(self, ctx):
        from stonks import OrderStatus

        assert ctx.position("X") is None
        oid = ctx.place_limit_order(symbol="X", side=OrderSide.Buy, quantity=1.0, price=50.0)
        o = ctx.order(oid)
        assert o is not None and o.status == OrderStatus.Open
        assert o.reduce_only is False
        assert ctx.cancel_order(oid) is True
        assert ctx.order(oid).status == OrderStatus.Cancelled
        assert ctx.cancel_order(oid) is False
        assert ctx.order(987654321) is None
        ctx.place_stop_order(symbol="X", side=OrderSide.Sell, quantity=7.0, price=90.0,
                             reduce_only=True)


class CashAwareStrategy(stonks.Strategy):
    """Reads ctx.cash() and places a sell with that quantity. Used to verify
    the Python -> C++ -> Python query path for context state."""

    def on_tick(self, ctx):
        ctx.place_market_order(
            symbol="X", side=OrderSide.Sell, quantity=ctx.cash()
        )


class HistoryInspector(stonks.Strategy):
    """Inspects ctx.history(): builds a pandas DataFrame from the combined
    multi-symbol window, asserts dtypes and per-symbol groupby contents, and
    places one buy per symbol encoding its latest close so the C++ side can
    confirm the round-trip. Any failed assertion surfaces as a
    std::runtime_error on the C++ side."""

    def on_tick(self, ctx):
        import numpy as np
        import pandas as pd

        w = ctx.history(3)
        assert len(w) == 6, len(w)                       # 2 symbols x 3 bars
        assert w.timestamp.dtype == np.int64, w.timestamp.dtype
        assert w.close.dtype == np.float64, w.close.dtype

        df = pd.DataFrame({
            "symbol": w.symbol,
            "timestamp": w.timestamp,
            "open": w.open,
            "high": w.high,
            "low": w.low,
            "close": w.close,
            "volume": w.volume,
        })
        assert df.shape == (6, 7), df.shape

        groups = {sym: sub for sym, sub in df.groupby("symbol")}
        assert list(groups["A"]["close"]) == [100.0, 101.0, 102.0], list(groups["A"]["close"])
        assert list(groups["B"]["close"]) == [200.0, 201.0, 202.0], list(groups["B"]["close"])

        for symbol, sub in groups.items():
            ctx.place_market_order(
                symbol=symbol,
                side=OrderSide.Buy,
                quantity=float(sub["close"].iloc[-1]),
            )
