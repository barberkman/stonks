"""Setup 2 — episodic pivot (gap bar, long).

A liquid stock gaps up on volume with a strong close — a news/earnings "episodic
pivot". Pine enters at the close of the gap bar; here we enter at the next open.
R-based management, with the stop tightened to the gap bar's low (epWithin keeps
that risk within the ADR stop distance). See qm_common for the execution model.
"""

from qm_common import QMBase, Params, EntryPlan, liq_ok, adr_pct, sma


class QMEpisodicPivotStrategy(QMBase):
    DIRECTION = "LONG"
    MGMT = "R"
    SAME_BAR_BAIL = False  # Pine's EP entry has no same-bar bail
    PARAMS = Params()

    def signal(self, bars):
        p = self.p
        if not liq_ok(bars, p):
            return None
        o, h, l, c = float(bars.open[-1]), float(bars.high[-1]), float(bars.low[-1]), float(bars.close[-1])
        prev_close = float(bars.close[-2])
        if prev_close <= 0.0:
            return None

        ep_gap = 100.0 * (o / prev_close - 1.0)
        if ep_gap < p.ep_min_gap:
            return None

        prev_avg50 = sma(bars.volume[:-1], 50)
        if prev_avg50 is None or float(bars.volume[-1]) < p.ep_vol_mult * prev_avg50:
            return None

        if p.ep_strong_close and not (c > o and c >= (h + l) / 2.0):
            return None

        adr = adr_pct(bars.high, bars.low, p.adr_len)
        risk_ps = c - l
        if adr is None or not (risk_ps > 0.0 and risk_ps <= p.adr_stop_mult * adr / 100.0 * c):
            return None

        return EntryPlan(entry_ref=c)
