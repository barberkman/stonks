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
        # One tick per timestamp: loop the symbols that printed today and run the
        # per-symbol EMA on each. EMA is incremental, so one bar per symbol
        # (history(1)) suffices; each symbol keeps its own independent state.
        window = ctx.history(1)
        for symbol, close in zip(window.symbol, window.close):
            close = float(close)
            state = self.states.setdefault(
                symbol,
                { "ema": None, "seed_sum": 0.0, "seed_count": 0, "held_quantity": 0.0 },
            )

            if state["ema"] is None:
                # Accumulate closes until we have PERIOD samples for this symbol,
                # then seed the EMA with their SMA — standard bootstrap that avoids
                # anchoring the recursion to a single bar.
                state["seed_sum"] += close
                state["seed_count"] += 1
                if state["seed_count"] < self.PERIOD:
                    continue
                state["ema"] = state["seed_sum"] / self.PERIOD
            else:
                state["ema"] = self.ALPHA * close + (1.0 - self.ALPHA) * state["ema"]

            # Enter long on an upside crossover; flat-only means we never short on the downside.
            if close > state["ema"] and state["held_quantity"] == 0.0:
                qty = ctx.cash() / close
                if qty <= 0.0:
                    continue
                if ctx.place_order(symbol=symbol, side=OrderSide.Buy, quantity=qty):
                    state["held_quantity"] = qty
            elif close < state["ema"] and state["held_quantity"] > 0.0:
                if ctx.close(symbol):   # market-close the position at the next bar's open
                    state["held_quantity"] = 0.0
