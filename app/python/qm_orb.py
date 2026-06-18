"""Setup 1b — opening range breakout (intraday, long).

Buy the break above the high of the first N bars of a session. A session is one
calendar day (UTC); the opening range is the high of its first ``orb_bars`` bars.

NOTE: this is intraday-only. On the daily dataset every bar is its own calendar
day, so a session never has more than one bar and ``bars_into_session`` never
reaches ``orb_bars`` — the strategy produces no signals by construction. It fires
correctly once intraday bars are fed. See qm_common for the execution model.
"""

import numpy as np

from qm_common import QMBase, Params, EntryPlan, universe_long

_MS_PER_DAY = 86_400_000


class QMORBStrategy(QMBase):
    DIRECTION = "LONG"
    MGMT = "R"
    SAME_BAR_BAIL = True
    PARAMS = Params()

    def signal(self, bars):
        p = self.p
        if not universe_long(bars, p):
            return None

        days = np.asarray(bars.timestamp) // _MS_PER_DAY
        session = np.where(days == days[-1])[0]      # bar indices in the current session
        bars_into_session = len(session) - 1         # 0 on the session's first bar
        if bars_into_session < p.orb_bars:
            return None

        or_high = float(bars.high[session[:p.orb_bars]].max())
        pivot = or_high + p.buf_ticks * 0.0
        broke = bars.close[-1] >= pivot if p.wait_close else bars.high[-1] >= pivot
        if not broke:
            return None
        return EntryPlan(entry_ref=pivot)
