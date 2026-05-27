"""Python mirror of app/strategies/ema50strategy.h.

Long-only trend-following strategy, applied independently per symbol: for
each symbol the feed surfaces, hold while its close is above its own 50-bar
EMA and stay flat otherwise. No shorting and no cross-symbol coupling.
"""

import stonks
from stonks import OrderSide


class EMA50Strategy(stonks.Strategy):
    PERIOD = 50
    # Standard EMA smoothing factor: 2/(N+1) gives the latest bar a weight that
    # makes the EMA's responsiveness comparable to an N-period SMA.
    ALPHA = 2.0 / (PERIOD + 1)

    def on_start(self, ctx):
        self.states = {}

    def on_tick(self, ctx):
        # klines() interleaves bars across every symbol the feed surfaces, so a
        # per-symbol EMA must consume only its own symbol's stream. Treat the
        # latest bar as the one that triggered this tick and route it to its
        # symbol's state.
        bars = ctx.klines(1)
        if not bars:
            return
        bar = bars[-1]

        state = self.states.setdefault(
            bar.symbol,
            { "ema": None, "seed_sum": 0.0, "seed_count": 0, "held_quantity": 0.0 },
        )

        if state["ema"] is None:
            # Accumulate closes until we have PERIOD samples for this symbol,
            # then seed the EMA with their SMA — standard bootstrap that avoids
            # anchoring the recursion to a single bar.
            state["seed_sum"] += bar.close
            state["seed_count"] += 1
            if state["seed_count"] < self.PERIOD:
                return
            state["ema"] = state["seed_sum"] / self.PERIOD
        else:
            state["ema"] = self.ALPHA * bar.close + (1.0 - self.ALPHA) * state["ema"]

        # Enter long on an upside crossover; flat-only means we never short on the downside.
        if bar.close > state["ema"] and state["held_quantity"] == 0.0:
            qty = ctx.cash() / bar.close
            if qty <= 0.0:
                return
            if ctx.place_market_order(symbol=bar.symbol, side=OrderSide.Buy, quantity=qty):
                state["held_quantity"] = qty
        elif bar.close < state["ema"] and state["held_quantity"] > 0.0:
            if ctx.place_market_order(
                symbol=bar.symbol, side=OrderSide.Sell, quantity=state["held_quantity"]
            ):
                state["held_quantity"] = 0.0
