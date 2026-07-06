"""Discovery fixture: a single strategy class with declared parameters."""

import stonks
from stonks import Param


class ParamFixture(stonks.Strategy):
    risk = 0.05
    lookback = 30
    use_trend = False

    params = {
        "risk": Param("risk per trade", unit="%"),
        "lookback": Param("bars of history", unit="bars"),
        "use_trend": Param("require the trend filter"),
    }

    def on_tick(self, ctx):
        pass
