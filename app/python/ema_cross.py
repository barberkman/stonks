"""Sample Python strategy: EMA crossover.

Long-only, per-symbol. Buys when the short EMA crosses above the long EMA;
sells when it crosses back. Mirrors the spirit of the C++ EMA50Strategy at a
different periodicity.
"""

import stonks
from stonks import OrderSide


def _ema(values, period):
    k = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema


class EMACross(stonks.Strategy):
    SHORT = 12
    LONG = 26

    def on_start(self, ctx):
        self.held = {}

    def on_tick(self, ctx):
        bars = ctx.klines(self.LONG + 1)
        if len(bars) < self.LONG + 1:
            return
        symbol = bars[-1].symbol
        closes = [b.close for b in bars]
        short = _ema(closes[-(self.SHORT + 1):], self.SHORT)
        long_ = _ema(closes, self.LONG)
        held = self.held.get(symbol, 0.0)
        last = bars[-1].close
        if short > long_ and held == 0.0:
            qty = ctx.cash() / last
            if qty > 0 and ctx.place_market_order(
                symbol=symbol, side=OrderSide.Buy, quantity=qty
            ):
                self.held[symbol] = qty
        elif short < long_ and held > 0.0:
            if ctx.place_market_order(
                symbol=symbol, side=OrderSide.Sell, quantity=held
            ):
                self.held[symbol] = 0.0
