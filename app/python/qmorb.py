"""Qullamaggie opening-range breakout as a day trade — Setup 1b standalone.

The pine defines the opening range with TradingView's session calendar; this
engine has no session API and its intraday data is continuous crypto, so a
"session" here is a UTC day (timestamp // 86_400_000). On daily bars every
bar is its own session and the setup is inert by construction — this file
is only meaningful on intraday data.

  range   the high of the first orb_bars bars of the UTC day
  entry   once the range is complete: a resting buy-stop just above it,
          gated by the full QM universe (price, ADR, momentum, MA alignment);
          one attempt per day, and the order lives only until the day ends
  stop    ADR-based, tightened to the arm bar's low if closer
  exit    whatever survives the stop is flattened at the day rollover with a
          reduce-only market order. Decisions happen on closed bars, so the
          flatten is detected on the new day's first bar and fills at the
          SECOND bar's open — the engine-honest reading of "close by end of
          session". No partials, no trail: in by the range, out by the bell.

Intraday horizon: the stop stays at its planned level (no gap re-anchoring).
"""

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide, OrderStatus

MS_PER_DAY = 86_400_000


def sma(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.sum(a[-n:]) / n)


def adr_pct(high, low, n):
    if n <= 0 or len(high) < n:
        return None
    h = high[-n:]
    l = low[-n:]
    return float(np.sum(100.0 * (h / l - 1.0)) / n)


def gain_pct(close, n):
    if len(close) < n + 1:
        return None
    base = close[len(close) - 1 - n]
    if base == 0.0:
        return None
    return float(100.0 * (close[-1] / base - 1.0))


class QMORBStrategy(stonks.Strategy):
    # Universe & trend
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    min_adr = 0.1
    mom_len = 24
    min_gain = 0.5
    require_mas = True
    # Opening range
    orb_bars = 1
    entry_buffer_bps = 5.0
    # Stop
    adr_stop_mult = 1.0
    use_lod_stop = True
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "orb_bars": stonks.Param("opening range length, bars from the day open", unit="bars"),
        "adr_stop_mult": stonks.Param("stop distance from entry, in ADRs", unit="ADR"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "or_high": stonks.Indicator("opening-range high of the current UTC day"),
    }

    def lookback(self):
        return max(51, self.mom_len, self.adr_len, 21) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"pending": None, "entry_id": None, "sl": None,
                "entry_qty": 0.0, "day": None, "day_bars": 0, "or_high": None,
                "armed_day": None, "taken_day": None,
                "exiting": False, "was_in": False, "cooldown": 0, "bar_count": 0}

    def on_tick(self, ctx):
        w = ctx.history(self.lookback())
        if len(w) == 0:
            return
        df = pd.DataFrame({
            "symbol": w.symbol, "timestamp": w.timestamp,
            "open": w.open, "high": w.high, "low": w.low,
            "close": w.close, "volume": w.volume,
        })
        for symbol, sub in df.groupby("symbol", sort=False):
            sub = sub.sort_values("timestamp")
            hi = sub["high"].to_numpy()
            lo = sub["low"].to_numpy()
            cl = sub["close"].to_numpy()
            vo = sub["volume"].to_numpy()
            ts = sub["timestamp"].to_numpy()

            st = self.state.setdefault(symbol, self._fresh())
            st["bar_count"] += 1

            # ── Session bookkeeping runs from the very first bar ──────────────
            day = int(ts[-1]) // MS_PER_DAY
            rollover = st["day"] is not None and day != st["day"]
            if st["day"] is None or rollover:
                st["day"] = day
                st["day_bars"] = 1
                st["or_high"] = float(hi[-1])
            else:
                st["day_bars"] += 1
                if st["day_bars"] <= self.orb_bars:
                    st["or_high"] = max(st["or_high"], float(hi[-1]))
            if st["or_high"] is not None:
                ctx.plot("or_high", symbol, st["or_high"])

            pos = ctx.position(symbol)
            in_pos = pos is not None

            if rollover:
                # The day is over: cancel the unfilled attempt, flatten the rest.
                if st["pending"] is not None:
                    entry = ctx.order(st["pending"])
                    if entry is not None and entry.status == OrderStatus.Open:
                        ctx.cancel_order(st["pending"])
                if in_pos and not st["exiting"]:
                    ctx.place_market_order(symbol=symbol,
                                           side=OrderSide.Sell if pos.quantity > 0.0 else OrderSide.Buy,
                                           quantity=abs(pos.quantity),
                                           parent=st["entry_id"], reduce_only=True)
                    st["exiting"] = True

            closed = st["was_in"] and not in_pos
            if st["pending"] is not None:
                entry = ctx.order(st["pending"])
                status = entry.status if entry is not None else OrderStatus.Cancelled
                if status == OrderStatus.Filled:
                    if in_pos:
                        st["entry_id"] = st["pending"]
                        st["taken_day"] = day
                    else:
                        closed = True
                    st["pending"] = None
                elif status != OrderStatus.Open:
                    st["pending"] = None
            if closed:
                st["cooldown"] = self.cooldown_bars
                st["exiting"] = False
                st["sl"] = st["entry_id"] = None
            elif not in_pos and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_pos

            if in_pos or st["pending"] is not None or st["cooldown"] > 0:
                continue
            if len(cl) < self.lookback():
                continue
            if st["day_bars"] <= self.orb_bars:      # the range is still forming
                continue
            if st["armed_day"] == day or st["taken_day"] == day:
                continue                             # one attempt per session
            if not self._universe_up(hi, lo, cl, vo):
                continue

            adr = adr_pct(hi, lo, self.adr_len)
            entry_px = st["or_high"] * (1.0 + self.entry_buffer_bps / 10_000.0)
            adr_stop = entry_px * (1.0 - self.adr_stop_mult * adr / 100.0)
            stop_px = max(adr_stop, float(lo[-1])) if self.use_lod_stop else adr_stop
            stop_px = min(stop_px, entry_px * 0.999)
            risk = entry_px - stop_px
            if risk <= 0.0 or not np.isfinite(risk):
                continue
            qty = ctx.equity() * self.risk_fraction / risk
            if qty <= 0.0 or not np.isfinite(qty):
                continue
            entry_id = ctx.place_stop_order(symbol=symbol, side=OrderSide.Buy,
                                            quantity=qty, price=entry_px)
            st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                            quantity=qty, price=stop_px,
                                            parent=entry_id, reduce_only=True)
            st["pending"] = entry_id
            st["entry_qty"] = qty
            st["armed_day"] = day

    def _universe_up(self, hi, lo, cl, vo):
        adr = adr_pct(hi, lo, self.adr_len)
        g = gain_pct(cl, self.mom_len)
        if adr is None or g is None:
            return False
        av20 = sma(vo, 20)
        if cl[-1] < self.min_price or av20 is None or av20 < self.min_avg_vol:
            return False
        if adr < self.min_adr or g < self.min_gain:
            return False
        if self.require_mas:
            s10 = sma(cl, 10)
            s20 = sma(cl, 20)
            if s10 is None or s20 is None or not (cl[-1] > s20 and s10 > s20):
                return False
        return True
