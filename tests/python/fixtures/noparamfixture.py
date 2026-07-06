"""Discovery fixture: a single strategy class with no declared parameters."""

import stonks


class NoParamFixture(stonks.Strategy):
    def on_tick(self, ctx):
        pass
