"""stonks — Python strategy authoring for the stonks C++ engine.

Strategies subclass `Strategy` and implement `on_tick(ctx)`. The `Context`,
`KLine`, `Timestamp`, and enum types are imported from the compiled `_core`
extension; see python/README.md for the full API and runtime setup.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from stonks._core import (
    Context,
    KLine,
    MarketWindow,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
    Timestamp,
)


@dataclass
class Param:
    """Metadata for a class attribute exposed to the GUI as an editable
    per-run parameter. The plain class attribute (e.g. `risk_fraction = 0.02`)
    stays the single source of truth for the default value and its type;
    Param only carries documentation.

    `choices` turns an int param into a named selection: the GUI renders a
    dropdown of these labels and the parameter still travels as the *index*
    into the list, so the override transport stays numeric. A strategy with
    212 named alternatives (see `patterns.PatternsStrategy.pattern`) is
    unusable as a free-text integer box; this is what makes it pickable."""

    doc: str = ""
    unit: str = ""
    choices: Sequence[str] = ()


@dataclass
class Indicator:
    """Metadata for a named indicator series a strategy publishes via
    `ctx.plot(name, symbol, value)`. The `indicators` dict's key IS the series
    name passed to `ctx.plot` — unlike `Param`, there is no backing class
    attribute. Read-only/display-only: an indicator never influences trading
    logic and, unlike params, is never GUI-editable."""

    doc: str = ""
    color: str = ""   # optional "#rrggbb"; "" lets the GUI assign a palette color


class Strategy:
    """Base class for stonks Python strategies.

    Subclass and override `on_tick(ctx)` — required. `on_start(ctx)` and
    `on_stop(ctx)` are optional and skipped if not defined.

    Declare GUI-editable parameters via `params`: a dict mapping existing
    class-attribute names to `Param` metadata. The engine overrides declared
    attributes on the instance (before `on_start`) with values chosen in the
    setup screen, and stamps the effective values into the run's report.

    Declare read-only chart overlays via `indicators`: a dict mapping series
    names to `Indicator` metadata. Publish values from `on_tick` with
    `ctx.plot(name, symbol, value)`; the GUI draws each series over the
    symbol's candles.
    """

    params: Dict[str, Param] = {}
    indicators: Dict[str, Indicator] = {}

    def on_start(self, ctx: "Context") -> None:
        pass

    def on_tick(self, ctx: "Context") -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must override on_tick(ctx)"
        )

    def on_stop(self, ctx: "Context") -> None:
        pass


def param_specs(cls: type) -> List[Dict[str, Any]]:
    """The GUI-facing spec for a strategy class's declared parameters, in
    declaration order: [{name, default, type, doc, unit, choices}]. Defaults
    (and their types) are read from the class attributes the strategy actually
    uses. `choices` is empty unless the param declared named alternatives, in
    which case the value is the index into that list."""
    out: List[Dict[str, Any]] = []
    for name, meta in cls.params.items():
        default = getattr(cls, name)   # AttributeError = author typo; let it propagate
        if isinstance(default, bool):          # must precede int: bool subclasses int
            type_name = "bool"
        elif isinstance(default, int):
            type_name = "int"
        elif isinstance(default, float):
            type_name = "float"
        else:
            raise TypeError(f"{cls.__name__}.{name}: param default must be "
                            f"float/int/bool, got {type(default).__name__}")
        choices = [str(c) for c in meta.choices]
        if choices:
            if type_name != "int":
                raise TypeError(f"{cls.__name__}.{name}: choices need an int "
                                f"default (the index), got {type_name}")
            if not 0 <= int(default) < len(choices):
                raise ValueError(f"{cls.__name__}.{name}: default {default} is "
                                 f"outside its {len(choices)} choices")
        out.append({"name": name, "default": default, "type": type_name,
                    "doc": meta.doc, "unit": meta.unit, "choices": choices})
    return out


def indicator_specs(cls: type) -> List[Dict[str, Any]]:
    """The GUI-facing spec for a strategy class's declared indicator series,
    in declaration order: [{name, doc, color}]. Unlike param_specs there is no
    class attribute to read a value from — the dict key IS the series name
    `ctx.plot(name, ...)` must use."""
    return [{"name": name, "doc": meta.doc, "color": meta.color}
            for name, meta in cls.indicators.items()]


__all__ = [
    "Context",
    "Indicator",
    "KLine",
    "MarketWindow",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Param",
    "Position",
    "Strategy",
    "TimeInForce",
    "Timestamp",
    "indicator_specs",
    "param_specs",
]
