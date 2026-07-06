"""Python mirror of app/strategies/qmsignals.h — the Qullamaggie momentum scanner.

Five setups run against each symbol's latest closed bar; whenever one fires the
strategy places a 3-leg bracket (entry + stop + take-profit limit orders), exactly
like the C++ reference:

  breakout        — long: break above a tight base's pivot high
  short_breakout  — short: breakdown below a base's pivot low (downtrend mirror)
  episodic_pivot  — long: gap up on volume with a strong close
  parabolic_short — short: first red bar after a parabolic run-up
  orb             — long: opening-range breakout (intraday sessions only)

The C++ side hands each setup a per-symbol columnar `SeriesView`; the Python
`ctx.history(n)` instead returns one flattened long frame over every printing
symbol, so on_tick regroups it by symbol before scanning. The setup math is a
faithful, index-for-index port — base-pivot scans keep the *last* extreme (`>=`
/`<=`), so the loops are explicit rather than vectorized argmax/argmin.

Fire-and-forget, like the C++: no holdings are tracked. The broker's
one-position-per-symbol semantics reject duplicate same-side entries.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide


@dataclass
class Signal:
    """A fired setup on a closed bar, with the trade levels to place."""

    setup: str
    timestamp: int   # ms since epoch (the closed bar)
    entry: float
    stop: float
    sell: float


# ─── Stateless TA helpers (1:1 with the C++ statics) ─────────────────────────
# Each returns None when there is not enough history, mirroring std::optional.
def sma(a, n: int) -> Optional[float]:
    if n <= 0 or len(a) < n:
        return None
    return float(np.sum(a[-n:]) / n)


def adr_pct(high, low, n: int) -> Optional[float]:
    if n <= 0 or len(high) < n:
        return None
    h = high[-n:]
    l = low[-n:]
    return float(np.sum(100.0 * (h / l - 1.0)) / n)


def gain_pct(close, n: int) -> Optional[float]:
    if len(close) < n + 1:
        return None
    base = close[len(close) - 1 - n]
    if base == 0.0:
        return None
    return float(100.0 * (close[-1] / base - 1.0))


def highest(a, n: int) -> Optional[float]:
    if n <= 0 or len(a) < n:
        return None
    return float(np.max(a[-n:]))


def lowest(a, n: int) -> Optional[float]:
    if n <= 0 or len(a) < n:
        return None
    return float(np.min(a[-n:]))


class QMSignalsStrategy(stonks.Strategy):
    # ─── Configurable parameters (C++ defaults verbatim) ─────────────────────
    # Universe & trend filters
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    min_adr = 0.1
    mom_len = 24
    min_gain = 0.5
    require_mas = True
    # Breakout / short-breakout base
    base_max_len = 40
    min_base_days = 3
    max_depth = 40.0
    use_vol_dry = False
    vol_dry_ratio = 1.0
    buf_ticks = 0
    use_bo_vol = True
    bo_vol_mult = 1.3
    wait_close = True       # close-confirm the break (vs intrabar high/low)
    # Opening range breakout
    orb_bars = 1
    # Episodic pivot
    ep_min_gap = 0.5
    ep_vol_mult = 1.3
    ep_strong_close = True
    # Parabolic short
    ps_lookback = 10
    ps_min_gain = 8.0
    ps_streak = 3
    ps_stop_lb = 3
    # Stop & take-profit
    adr_stop_mult = 1.0     # stop distance from entry, in ADRs
    use_lod_stop = True     # tighten the stop to the signal bar's extreme if closer
    target_rr = 2.0         # take-profit ("sell") target, in R multiples
    # Sizing & leverage
    risk_fraction = 0.02    # fraction of equity risked per trade (stop-out loss target)
    maint_margin = 0.0      # maintenance-margin rate for the leverage calc (engine has none)
    max_leverage = 125.0    # cap on computed isolated leverage

    # The dataset carries no tick size, so the "buffer beyond the pivot" is zero.
    MINTICK = 0.0
    MS_PER_DAY = 86_400_000

    def on_start(self, ctx):
        # The setups that fired on each symbol's last processed bar (for tests).
        self._last = {}

    def lookback(self) -> int:
        return max(self.base_max_len, 51, self.mom_len, self.adr_len,
                   self.ps_lookback, self.ps_streak + 1, self.ps_stop_lb) + 5

    def on_tick(self, ctx):
        w = ctx.history(self.lookback())
        if len(w) == 0:
            return

        df = pd.DataFrame({
            "symbol": w.symbol,
            "timestamp": w.timestamp,
            "open": w.open, "high": w.high, "low": w.low,
            "close": w.close, "volume": w.volume,
        })
        for symbol, sub in df.groupby("symbol", sort=False):
            sub = sub.sort_values("timestamp")
            op = sub["open"].to_numpy()
            hi = sub["high"].to_numpy()
            lo = sub["low"].to_numpy()
            cl = sub["close"].to_numpy()
            vo = sub["volume"].to_numpy()
            ts = sub["timestamp"].to_numpy()

            sigs = self.scan(op, hi, lo, cl, vo, ts)
            if sigs:
                s = sigs[0]
                is_long = s.stop < s.entry              # long setups stop below entry
                # Risk-mode sizing: a stop-out loses risk_fraction of equity (no fees in this engine).
                risk_per_unit = abs(s.entry - s.stop)
                qty = ctx.equity() * self.risk_fraction / risk_per_unit if risk_per_unit > 0.0 else 0.0
                if qty > 0.0:
                    entry_side = OrderSide.Buy if is_long else OrderSide.Sell
                    exit_side = OrderSide.Sell if is_long else OrderSide.Buy
                    lev = self.entry_leverage(s.entry, s.stop, is_long)

                    # Stop-entry at the signal price — its id parents the protective
                    # legs, so the broker keeps them dormant until the entry fills and
                    # then OCO-cancels the loser once one side closes the position.
                    # A stop (not limit) entry fills at s.entry when reached, keeping
                    # the risk math calibrated; a reversal leaves it unfilled.
                    entry_id = ctx.place_stop_order(symbol=symbol, side=entry_side,
                                                    quantity=qty, price=s.entry,
                                                    leverage=lev)
                    # Stop loss order (child of the entry)
                    ctx.place_stop_order(symbol=symbol, side=exit_side,
                                         quantity=qty, price=s.stop, parent=entry_id)
                    # Take profit order (child of the entry)
                    ctx.place_limit_order(symbol=symbol, side=exit_side,
                                          quantity=qty, price=s.sell, parent=entry_id)

            self._last[symbol] = sigs

    # The setups that fired on this symbol's last processed bar (for tests).
    def last_signals(self, symbol: str) -> List[Signal]:
        return self._last.get(symbol, [])

    # Run every setup against the closed bar; collect the ones that fired.
    def scan(self, op, hi, lo, cl, vo, ts) -> List[Signal]:
        out: List[Signal] = []
        for setup in (self.breakout, self.short_breakout, self.episodic_pivot,
                      self.parabolic_short, self.orb):
            s = setup(op, hi, lo, cl, vo, ts)
            if s is not None:
                out.append(s)
        return out

    # Largest isolated leverage keeping liquidation just beyond the stop (formulas §9),
    # floored with a one-step buffer, clamped to [1, max_leverage].
    def entry_leverage(self, entry: float, stop: float, is_long: bool) -> float:
        denom = (entry - stop * (1.0 - self.maint_margin)) if is_long \
            else (stop * (1.0 + self.maint_margin) - entry)
        if denom <= 0.0:
            return 1.0
        lmax = entry / denom
        l = math.floor(lmax)
        if l == lmax:
            l -= 1                          # exact integer -> step below so liquidation stays under the stop
        l = min(float(l), self.max_leverage)
        return max(l, 1.0)                  # broker requires leverage >= 1

    # ─── Setup 1 — momentum breakout (long) ──────────────────────────────────
    def breakout(self, op, hi, lo, cl, vo, ts) -> Optional[Signal]:
        N = len(cl)
        if N < self.base_max_len + 1:
            return None

        adr = adr_pct(hi, lo, self.adr_len)
        g = gain_pct(cl, self.mom_len)
        universe = (self.liq_ok(cl, vo) and adr is not None and adr >= self.min_adr
                    and g is not None and g >= self.min_gain and self.ma_ok_up(cl))
        if not universe:
            return None

        base_end = N - 1
        w_start = base_end - self.base_max_len
        mx = hi[w_start]
        last_pos = 0
        for i in range(self.base_max_len):
            if hi[w_start + i] >= mx:
                mx = hi[w_start + i]
                last_pos = i
        since_pk = (self.base_max_len - 1) - last_pos
        pull_n = max(since_pk, 1)
        pull_low = lo[base_end - pull_n]
        for i in range(base_end - pull_n, base_end):
            pull_low = min(pull_low, lo[i])
        retrace = 100.0 * (mx - pull_low) / mx if mx > 0.0 else 1e9

        vd = (not self.use_vol_dry) or self.vol_dry_ok(vo)
        if not (since_pk >= self.min_base_days and retrace <= self.max_depth and vd):
            return None

        entry = mx + self.buf_ticks * self.MINTICK
        broke = cl[-1] >= entry if self.wait_close else hi[-1] >= entry
        if not broke or not self.break_vol_ok(vo) or adr is None:
            return None

        stop, sell = self.long_levels(entry, adr, lo[-1])
        return Signal("breakout", int(ts[-1]), float(entry), stop, sell)

    # ─── Setup 1c — short breakout / breakdown (short, mirror) ────────────────
    def short_breakout(self, op, hi, lo, cl, vo, ts) -> Optional[Signal]:
        N = len(cl)
        if N < self.base_max_len + 1:
            return None

        adr = adr_pct(hi, lo, self.adr_len)
        g = gain_pct(cl, self.mom_len)
        universe = (self.liq_ok(cl, vo) and adr is not None and adr >= self.min_adr
                    and g is not None and g <= -self.min_gain and self.ma_ok_dn(cl))
        if not universe:
            return None

        base_end = N - 1
        w_start = base_end - self.base_max_len
        mn = lo[w_start]
        last_pos = 0
        for i in range(self.base_max_len):
            if lo[w_start + i] <= mn:
                mn = lo[w_start + i]
                last_pos = i
        since_tr = (self.base_max_len - 1) - last_pos
        bounce_n = max(since_tr, 1)
        bounce_high = hi[base_end - bounce_n]
        for i in range(base_end - bounce_n, base_end):
            bounce_high = max(bounce_high, hi[i])
        retrace_up = 100.0 * (bounce_high - mn) / mn if mn > 0.0 else 1e9

        vd = (not self.use_vol_dry) or self.vol_dry_ok(vo)
        if not (since_tr >= self.min_base_days and retrace_up <= self.max_depth and vd):
            return None

        entry = mn - self.buf_ticks * self.MINTICK
        broke = cl[-1] <= entry if self.wait_close else lo[-1] <= entry
        if not broke or not self.break_vol_ok(vo) or adr is None:
            return None

        stop, sell = self.short_levels(entry, adr, hi[-1])
        return Signal("short_breakout", int(ts[-1]), float(entry), stop, sell)

    # ─── Setup 2 — episodic pivot (gap bar, long) ─────────────────────────────
    def episodic_pivot(self, op, hi, lo, cl, vo, ts) -> Optional[Signal]:
        if len(cl) < 2:
            return None
        if not self.liq_ok(cl, vo):
            return None

        o = op[-1]
        h = hi[-1]
        l = lo[-1]
        c = cl[-1]
        prev_close = cl[len(cl) - 2]
        if prev_close <= 0.0:
            return None

        if 100.0 * (o / prev_close - 1.0) < self.ep_min_gap:
            return None

        prev_avg50 = sma(vo[:-1], 50)
        if prev_avg50 is None or vo[-1] < self.ep_vol_mult * prev_avg50:
            return None

        if self.ep_strong_close and not (c > o and c >= (h + l) / 2.0):
            return None

        adr = adr_pct(hi, lo, self.adr_len)
        risk_ps = c - l
        if adr is None or not (risk_ps > 0.0 and risk_ps <= self.adr_stop_mult * adr / 100.0 * c):
            return None

        stop, sell = self.long_levels(c, adr, l)
        return Signal("episodic_pivot", int(ts[-1]), float(c), stop, sell)

    # ─── Setup 3 — parabolic short (first red bar) ────────────────────────────
    def parabolic_short(self, op, hi, lo, cl, vo, ts) -> Optional[Signal]:
        need = max(self.ps_lookback, self.ps_streak + 1, self.ps_stop_lb) + 1
        if len(cl) < need:
            return None
        if not self.liq_ok(cl, vo):
            return None

        hh = highest(hi, self.ps_lookback)
        ll = lowest(lo, self.ps_lookback)
        if hh is None or ll is None or ll <= 0.0:
            return None
        if 100.0 * (hh / ll - 1.0) < self.ps_min_gain:
            return None

        up_streak = 0
        i = len(cl) - 2
        while i >= 1 and cl[i] > cl[i - 1]:
            up_streak += 1
            i -= 1
        if not (cl[-1] < cl[len(cl) - 2] and up_streak >= self.ps_streak):
            return None

        ps_stop = highest(hi, self.ps_stop_lb)
        if ps_stop is None:
            return None

        entry = float(cl[-1])
        stop = ps_stop + self.buf_ticks * self.MINTICK
        sell = entry - self.target_rr * (stop - entry)
        return Signal("parabolic_short", int(ts[-1]), entry, stop, sell)

    # ─── Setup 1b — opening range breakout (intraday, long) ───────────────────
    def orb(self, op, hi, lo, cl, vo, ts) -> Optional[Signal]:
        if len(cl) < 1:
            return None

        adr = adr_pct(hi, lo, self.adr_len)
        g = gain_pct(cl, self.mom_len)
        universe = (self.liq_ok(cl, vo) and adr is not None and adr >= self.min_adr
                    and g is not None and g >= self.min_gain and self.ma_ok_up(cl))
        if not universe:
            return None

        n = len(ts)
        cur_day = int(ts[-1]) // self.MS_PER_DAY
        start = n - 1
        while start > 0 and int(ts[start - 1]) // self.MS_PER_DAY == cur_day:
            start -= 1
        bars_into_session = (n - 1) - start
        if bars_into_session < self.orb_bars:
            return None

        or_high = hi[start]
        for k in range(self.orb_bars):
            or_high = max(or_high, hi[start + k])

        entry = or_high + self.buf_ticks * self.MINTICK
        broke = cl[-1] >= entry if self.wait_close else hi[-1] >= entry
        if not broke or adr is None:
            return None

        stop, sell = self.long_levels(entry, adr, lo[-1])
        return Signal("orb", int(ts[-1]), float(entry), stop, sell)

    # ─── Trade levels ─────────────────────────────────────────────────────────
    def long_levels(self, entry, adr, bar_low):
        adr_stop = entry * (1.0 - self.adr_stop_mult * adr / 100.0)
        stop = max(adr_stop, bar_low) if self.use_lod_stop else adr_stop
        stop = min(stop, entry * 0.999)
        return float(stop), float(entry + self.target_rr * (entry - stop))

    def short_levels(self, entry, adr, bar_high):
        adr_stop = entry * (1.0 + self.adr_stop_mult * adr / 100.0)
        stop = min(adr_stop, bar_high) if self.use_lod_stop else adr_stop
        stop = max(stop, entry * 1.001)
        return float(stop), float(entry - self.target_rr * (stop - entry))

    # ─── Gate predicates ──────────────────────────────────────────────────────
    def liq_ok(self, cl, vo) -> bool:
        av20 = sma(vo, 20)
        return cl[-1] >= self.min_price and av20 is not None and av20 >= self.min_avg_vol

    def ma_ok_up(self, cl) -> bool:
        if not self.require_mas:
            return True
        s10 = sma(cl, 10)
        s20 = sma(cl, 20)
        return s10 is not None and s20 is not None and cl[-1] > s20 and s10 > s20

    def ma_ok_dn(self, cl) -> bool:
        if not self.require_mas:
            return True
        s10 = sma(cl, 10)
        s20 = sma(cl, 20)
        return s10 is not None and s20 is not None and cl[-1] < s20 and s10 < s20

    def vol_dry_ok(self, vo) -> bool:
        v5 = sma(vo, 5)
        v50 = sma(vo, 50)
        return v5 is not None and v50 is not None and v5 < self.vol_dry_ratio * v50

    def break_vol_ok(self, vo) -> bool:
        if not self.use_bo_vol:
            return True
        if len(vo) < 51:
            return False
        prev_avg50 = sma(vo[:-1], 50)
        return prev_avg50 is not None and vo[-1] >= self.bo_vol_mult * prev_avg50
