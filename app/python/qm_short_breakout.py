"""Setup 1c — short breakout (breakdown).

Mirror of the long breakout: in a downtrend a stock bounces into a base, and we
short the break below the base's pivot low. Mirrors the Pine "Short BO" resting
sell-stop, adapted to a daily close-confirmed break that fills at the next open.
Same R-based management as longs. See qm_common for the execution model.
"""

from qm_common import QMBase, Params, universe_short, flag_base_short


class QMShortBreakoutStrategy(QMBase):
    DIRECTION = "SHORT"
    MGMT = "R"
    SAME_BAR_BAIL = True
    PARAMS = Params()

    def signal(self, bars):
        p = self.p
        if not universe_short(bars, p):
            return None
        return flag_base_short(bars, p)
