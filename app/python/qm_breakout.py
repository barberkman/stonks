"""Setup 1 — momentum breakout (long).

After a strong run a stock digests in a tight base; we buy the break above the
base's pivot high. Mirrors the Pine "Buy BO" resting buy-stop, adapted to a
daily close-confirmed break that fills at the next open. See qm_common for the
execution model and trade management.
"""

from qm_common import QMBase, Params, universe_long, flag_base_long


class QMBreakoutStrategy(QMBase):
    DIRECTION = "LONG"
    MGMT = "R"
    SAME_BAR_BAIL = True
    PARAMS = Params()

    def signal(self, bars):
        p = self.p
        if not universe_long(bars, p):
            return None
        return flag_base_long(bars, p)
