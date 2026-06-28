"""stonks — Python strategy authoring for the stonks C++ engine.

Strategies subclass `Strategy` and implement `on_tick(ctx)`. The `Context`,
`KLine`, `Timestamp`, and enum types are imported from the compiled `_core`
extension; see python/README.md for the full API and runtime setup.
"""

from stonks._core import (
    Context,
    KLine,
    MarketWindow,
    OrderSide,
    OrderType,
    Position,
    Timestamp,
)


class Strategy:
    """Base class for stonks Python strategies.

    Subclass and override `on_tick(ctx)` — required. `on_start(ctx)` and
    `on_stop(ctx)` are optional and skipped if not defined.
    """

    def on_start(self, ctx: "Context") -> None:
        pass

    def on_tick(self, ctx: "Context") -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must override on_tick(ctx)"
        )

    def on_stop(self, ctx: "Context") -> None:
        pass


__all__ = [
    "Context",
    "KLine",
    "MarketWindow",
    "OrderSide",
    "OrderType",
    "Position",
    "Strategy",
    "Timestamp",
]
