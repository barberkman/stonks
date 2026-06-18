"""Setup 3 — parabolic short (first red bar).

After a steep parabolic run-up of consecutive up-closes, short the first red bar
(the reversal). Pine enters at that bar's close; here we enter at the next open.
Distinct management: cover on a stop above the recent highs, on any green close,
or after a max holding period — no partial. See qm_common for the model.
"""

from qm_common import QMBase, Params, EntryPlan, liq_ok, highest, lowest


class QMParabolicShortStrategy(QMBase):
    DIRECTION = "SHORT"
    MGMT = "PARA"
    SAME_BAR_BAIL = False
    PARAMS = Params()

    def signal(self, bars):
        p = self.p
        if not liq_ok(bars, p):
            return None
        close = bars.close
        if len(close) < max(p.ps_lookback, p.ps_streak + 1, p.ps_stop_lb) + 1:
            return None

        hi = highest(bars.high, p.ps_lookback)
        lo = lowest(bars.low, p.ps_lookback)
        if hi is None or lo is None or lo <= 0.0:
            return None
        ps_runup = 100.0 * (hi / lo - 1.0)
        if ps_runup < p.ps_min_gain:
            return None

        # First red bar after >= ps_streak consecutive up-closes ending at bar[-2].
        up_streak = 0
        i = -2
        while i - 1 >= -len(close) and float(close[i]) > float(close[i - 1]):
            up_streak += 1
            i -= 1
        if not (float(close[-1]) < float(close[-2]) and up_streak >= p.ps_streak):
            return None

        ps_stop = highest(bars.high, p.ps_stop_lb) + p.buf_ticks * 0.0
        return EntryPlan(entry_ref=float(close[-1]), explicit_stop=float(ps_stop), use_target=False)
