"""stonks — Python strategy authoring for the stonks C++ engine.

Strategies subclass `Strategy` and implement `on_tick(ctx)`. The `Context`,
`KLine`, `Timestamp`, and enum types are imported from the compiled `_core`
extension; see python/README.md for the full API and runtime setup.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

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
    Param only carries documentation."""

    doc: str = ""
    unit: str = ""


class Strategy:
    """Base class for stonks Python strategies.

    Subclass and override `on_tick(ctx)` — required. `on_start(ctx)` and
    `on_stop(ctx)` are optional and skipped if not defined.

    Declare GUI-editable parameters via `params`: a dict mapping existing
    class-attribute names to `Param` metadata. The engine overrides declared
    attributes on the instance (before `on_start`) with values chosen in the
    setup screen, and stamps the effective values into the run's report.
    """

    params: Dict[str, Param] = {}

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
    declaration order: [{name, default, type, doc, unit}]. Defaults (and their
    types) are read from the class attributes the strategy actually uses."""
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
        out.append({"name": name, "default": default, "type": type_name,
                    "doc": meta.doc, "unit": meta.unit})
    return out


__all__ = [
    "Context",
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
    "param_specs",
]
