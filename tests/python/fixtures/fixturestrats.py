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


class CashAwareStrategy(stonks.Strategy):
    """Reads ctx.cash() and places a sell with that quantity. Used to verify
    the Python -> C++ -> Python query path for context state."""

    def on_tick(self, ctx):
        ctx.place_market_order(
            symbol="X", side=OrderSide.Sell, quantity=ctx.cash()
        )
