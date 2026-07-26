"""Short-term cross-sectional momentum — port of trade_algo/backtest.py.

Each month, hold the `top` strongest names of the past `window` trading days,
equally weighted; everything else goes to cash. Ranking is on cumulative
cleaned return, and only names with a full window are rankable — a stock listed
last week has no 20-day return, and letting it in on a partial window is a
subtle bias.

Caveat carried over from the original: there is no liquidity filter, so the top
`window`-bar gainers are mostly micro-caps that spiked on almost no volume.
Lookahead-free but largely untradable; a cross-sectional percentile of median
traded value is the usual fix.

Deviations from the source, which expresses a target weight vector the engine
nets into share deltas:

 1. The source ranks on `sanitize`'s `ret`, where a bar-to-bar move outside the
    BIST daily band is flagged as a corporate action or a bad tick and booked as
    0%. The engine's feed carries no such field, so returns are computed from
    close and the same rule is applied here via `price_limit_pct` — the raw
    prices are not split-adjusted and a bonus issue looks like a 2800x gain.
 2. `market.listed()` becomes "printed a bar this tick": `ctx.history()` only
    returns symbols that printed at the current timestamp, so halted and
    delisted names drop out of the ranking on their own. The source carries a
    halted name's last good price forward, so its `window` lookback always spans
    `window` calendar bars; here a halt shortens the span the lookback covers.
 3. The engine never adds to a position, so a name that stays in the top `top`
    keeps the position it already has instead of being re-sized back to
    1/`top`. Only the delta trades, as in the source, but the weights drift
    between rebalances rather than being reset at each one.
 4. Exits and entries are split across two bars: exits go out on the rebalance
    bar (filling at the next open, like the source), entries on the following
    tick, once the exits' proceeds have actually settled. The broker settles
    symbol by symbol and rejects — never queues — an order it cannot fund at
    fill time, so funding entries out of the same bar's sale proceeds would drop
    positions depending on the order symbols happen to settle in. Entries
    therefore fill one bar later than the source's.
 5. Entry size is capped by free cash less `cash_buffer`. A weight vector cannot
    overdraw by construction; a market order sized on today's close and filled
    at tomorrow's open can, and the excess would be rejected outright.
 6. Long-only and unlevered, like the source (weights >= 0 summing to <= 1).
"""

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide


class ShortTermMomentumStrategy(stonks.Strategy):
    window = 20
    top = 10
    # BIST equities trade inside a +/-20% daily band; the exchange rounds the
    # band edges to the tick grid, which lets a genuine limit close print a
    # shade past it, hence the 0.5% headroom.
    price_limit_pct = 20.5
    cash_buffer = 0.02

    params = {
        "window": stonks.Param("ranking lookback", unit="bars"),
        "top": stonks.Param("names held, equally weighted"),
        "price_limit_pct": stonks.Param(
            "bar-to-bar move treated as a corporate action and booked as 0%", unit="%"),
        "cash_buffer": stonks.Param("fraction of cash held back from entries for fees and gaps"),
    }

    def on_start(self, ctx):
        self.bars = 0
        self.month = None
        self.book = set()    # symbols we believe we hold, refreshed each rebalance
        self.entries = []    # symbols to buy on the next tick (deviation 4)

    def on_tick(self, ctx):
        w = ctx.history(self.window + 1)
        if len(w) == 0:
            return
        self.bars += 1
        ts = int(np.max(w.timestamp))
        day = pd.Timestamp(ts, unit="ms", tz="UTC")
        month = (day.year, day.month)

        # Warmup mirrors the source: the first rebalance fires as soon as a full
        # window exists, not on the month boundary after it.
        rebalance = self.bars > self.window and month != self.month
        if not rebalance and not self.entries:
            return

        df = pd.DataFrame({
            "symbol": w.symbol, "timestamp": w.timestamp, "close": w.close,
        })
        closes = {}
        for symbol, sub in df.groupby("symbol", sort=False):
            closes[symbol] = sub.sort_values("timestamp")["close"].to_numpy()
        latest = {s: float(c[-1]) for s, c in closes.items()}

        if rebalance:
            self.month = month
            # A rebalance supersedes entries still waiting from the previous one
            # (only reachable when the warmup rebalance lands on a month's last
            # bar): the new target replaces a decision that never traded.
            self._rebalance(ctx, ts, closes, latest)
        else:
            self._enter(ctx, ts, latest)

    def _rebalance(self, ctx, ts, closes, latest):
        """Rank, close what dropped out, and queue what came in."""
        scores = {}
        for symbol, c in closes.items():
            if len(c) < self.window + 1:
                continue                        # partial window: the source's `complete` mask
            r = c[1:] / c[:-1] - 1.0
            if not np.isfinite(r).all():
                continue
            r = np.where(np.abs(r) > self.price_limit_pct / 100.0, 0.0, r)
            scores[symbol] = float(np.prod(1.0 + r))

        # Positions the broker still reports: anything closed behind our back
        # (a liquidation) drops out of the book here.
        held = {s for s in self.book if ctx.position(s) is not None}
        # Fewer rankable names than slots -> go to cash, like the source.
        target = set()
        if len(scores) >= self.top:
            target = set(sorted(scores, key=lambda s: (-scores[s], s))[:self.top])

        exits = sorted(held - target)
        for symbol in exits:
            position = ctx.position(symbol)
            ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                   quantity=abs(position.quantity), reduce_only=True)
        self.book = held - set(exits)

        self.entries = sorted(target - held)
        self._print(ts, f"rebalance: {len(scores)} ranked, {len(held)} held, "
                        f"{len(exits)} exits, {len(self.entries)} entries")
        if not exits:
            # Nothing to fund from, so the entries need no settled proceeds.
            self._enter(ctx, ts, latest)

    def _enter(self, ctx, ts, latest):
        """Buy the queued names, equally weighted within what cash can fund."""
        pending = [s for s in self.entries if s in latest]
        self.entries = []
        if not pending:
            return
        budget = min(ctx.equity() / self.top,
                     ctx.cash() * (1.0 - self.cash_buffer) / len(pending))
        if budget <= 0.0 or not np.isfinite(budget):
            return
        for symbol in pending:
            qty = budget / latest[symbol]
            if qty <= 0.0 or not np.isfinite(qty):
                continue
            ctx.place_market_order(symbol=symbol, side=OrderSide.Buy, quantity=qty)
            self.book.add(symbol)
        self._print(ts, f"entering {len(pending)} names at {budget:.2f} each")

    @staticmethod
    def _print(ts, msg):
        '''
            when = pd.Timestamp(ts, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")
            print(f"[{when} UTC] {msg}", flush=True)
        '''
        pass
