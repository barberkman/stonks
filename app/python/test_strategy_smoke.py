"""Parametrized smoke sweep over ALL strategy files in app/python/.

Four layers, cheap to strict:
  1. discovery compliance — every file resolves to exactly one Strategy
     subclass (the resolver mirrors app/strategies/strategydiscovery.cpp) and
     its param/indicator specs are valid;
  2. meta — the set of strategy files on disk IS the parametrize list, so a
     new/renamed file can't dodge the sweep;
  3. no exceptions — each strategy runs 300 bars x 2 symbols through three
     synthetic regimes (trending / choppy / gappy) under conftest's settle()
     mini-broker, so fills happen and the management paths actually execute;
  4. order well-formedness — every order any run ever placed obeys the house
     bracket grammar (finite positive quantities, protective children are
     reduce-only with opposite side and no more than the parent's size,
     entries are stop/market only, bracket geometry is on the correct side).
"""

import importlib
import inspect
import pathlib

import numpy as np
import pytest

import stonks
from stonks import OrderSide
from conftest import make_bars, run_with_broker

STRATEGIES = [
    ("qmliteral", "QMLiteralStrategy"),
    ("qmcloseconfirm", "QMCloseConfirmStrategy"),
    ("qmbreakoutpure", "QMBreakoutPureStrategy"),
    ("qmepisodic", "QMEpisodicStrategy"),
    ("qmorb", "QMORBStrategy"),
    ("qmshortbo", "QMShortBOStrategy"),
    ("qmparabolic", "QMParabolicStrategy"),
    ("qmfullsuite", "QMFullSuiteStrategy"),
    ("qmatr", "QMATRStrategy"),
    ("qmrunner", "QMRunnerStrategy"),
    ("qmthirds", "QMThirdsStrategy"),
    ("qmpullback", "QMPullbackStrategy"),
    ("qmswing", "QMSwingStrategy"),
    ("darvasclassic", "DarvasClassicStrategy"),
    ("darvasstrict", "DarvasStrictStrategy"),
    ("darvasshort", "DarvasShortStrategy"),
    ("darvasboth", "DarvasBothStrategy"),
    ("darvasvolume", "DarvasVolumeStrategy"),
    ("darvasboxrisk", "DarvasBoxRiskStrategy"),
    ("darvastight", "DarvasTightStrategy"),
    ("darvastrend", "DarvasTrendStrategy"),
    ("darvasrebreak", "DarvasRebreakStrategy"),
    ("qmdarvasbase", "QMDarvasBaseStrategy"),
    ("qmdarvasconsensus", "QMDarvasConsensusStrategy"),
    ("qmdarvasuniverse", "QMDarvasUniverseStrategy"),
    ("qmdarvasepbox", "QMDarvasEPBoxStrategy"),
    ("qmdarvasexit", "QMDarvasExitStrategy"),
    ("qmdarvasshort", "QMDarvasShortStrategy"),
    ("qmdarvasregime", "QMDarvasRegimeStrategy"),
    ("qmdarvasfirstbox", "QMDarvasFirstBoxStrategy"),
    ("qmdarvasboxtrail", "QMDarvasBoxTrailStrategy"),
]

# Short-only files get a falling trend so their setups (and management) run.
SHORT_BIASED = {"qmshortbo", "qmparabolic", "darvasshort", "qmdarvasshort"}


def _resolve(module, cls_name):
    m = importlib.import_module(module)
    found = [c for _, c in inspect.getmembers(m, inspect.isclass)
             if issubclass(c, stonks.Strategy) and c is not stonks.Strategy
             and c.__module__ == module]
    assert [c.__name__ for c in found] == [cls_name], (
        f"{module}.py must hold exactly one Strategy subclass named {cls_name}, "
        f"found {[c.__name__ for c in found]}")
    return found[0]


@pytest.mark.parametrize("module,cls_name", STRATEGIES)
def test_discovery_compliance_and_specs(module, cls_name):
    cls = _resolve(module, cls_name)
    param_specs = stonks.param_specs(cls)
    assert param_specs, f"{module} declares no GUI params"
    for spec in param_specs:
        assert spec["type"] in ("bool", "int", "float")
        assert getattr(cls, spec["name"]) == spec["default"]
        assert isinstance(spec["doc"], str)
        assert isinstance(spec["unit"], str)
    indicator_specs = stonks.indicator_specs(cls)
    assert indicator_specs, f"{module} publishes no chart overlays"
    for spec in indicator_specs:
        assert isinstance(spec["name"], str) and spec["name"]
        assert isinstance(spec["doc"], str)
        assert isinstance(spec["color"], str)


def test_exactly_the_thirty_one_strategy_files_exist():
    here = pathlib.Path(__file__).parent
    on_disk = {p.stem for p in here.glob("*.py")
               if not (p.stem.startswith("test_") or p.stem.startswith("__")
                       or p.stem.endswith("_test") or p.stem == "conftest")}
    listed = {m for m, _ in STRATEGIES}
    assert on_disk == listed, (f"strategy files and the smoke list disagree: "
                               f"only on disk {sorted(on_disk - listed)}, "
                               f"only listed {sorted(listed - on_disk)}")
    assert len(STRATEGIES) == 31


@pytest.mark.parametrize("regime", ["trending", "choppy", "gappy"])
@pytest.mark.parametrize("module,cls_name", STRATEGIES)
def test_smoke_run_places_only_well_formed_orders(module, cls_name, regime):
    cls = _resolve(module, cls_name)
    strategy = cls()
    direction = -1 if module in SHORT_BIASED else 1
    bars = make_bars(["AAA", "BBB"], 300, regime, seed=7, direction=direction)

    ctx = run_with_broker(strategy, bars)

    by_id = {o.id: o for o in ctx.orders}
    for o in ctx.orders:
        assert np.isfinite(o.quantity) and o.quantity > 0.0, (module, regime, o)
        if o.price is not None:
            assert np.isfinite(o.price) and o.price > 0.0, (module, regime, o)
        if o.parent is not None:
            parent = by_id.get(o.parent)
            assert parent is not None, (module, regime, o)
            assert o.reduce_only, ("bracket child must be reduce-only", module, o)
            assert o.side != parent.side, ("child must oppose its parent", module, o)
            assert o.quantity <= parent.quantity * (1.0 + 1e-9), (
                "child bigger than its parent", module, o)
        elif not o.reduce_only:
            # A fresh entry: never a resting limit in any of the 31 designs.
            assert o.order_type in ("stop", "market"), (module, regime, o)
            # Take-profit geometry versus a priced (stop) entry. Stop-loss
            # children are deliberately NOT side-checked here: breakeven
            # moves and box/chandelier ratchets re-place them ABOVE a long
            # entry by design — the per-strategy behavior tests pin the
            # initial bracket geometry exactly instead.
            kids = [c for c in ctx.orders if c.parent == o.id]
            if o.price is not None:
                for c in kids:
                    if c.order_type == "limit":
                        assert (c.price > o.price) == (o.side == OrderSide.Buy), (
                            "take-profit on the wrong side of its entry", module, o, c)
