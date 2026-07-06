"""Unit tests for the Indicator / indicator_specs framework and the
FakeContext.plot recording used to test strategies' published series.

Run from the project root with the app-local venv:

    app/python/.venv/bin/pytest app/python/ -q
"""

import stonks
from stonks import Indicator, indicator_specs
from stonks.testing import FakeContext, FakeKLine, FakePlot


class _Overlays(stonks.Strategy):
    indicators = {
        "ema50": Indicator("50-bar EMA of close"),
        "adr": Indicator("average daily range", color="#e0a64e"),
    }

    def on_tick(self, ctx):
        pass


def test_indicator_specs_returns_name_doc_color():
    assert indicator_specs(_Overlays)[0] == {
        "name": "ema50", "doc": "50-bar EMA of close", "color": ""}
    assert indicator_specs(_Overlays)[1] == {
        "name": "adr", "doc": "average daily range", "color": "#e0a64e"}


def test_indicator_specs_preserves_declaration_order():
    assert [s["name"] for s in indicator_specs(_Overlays)] == ["ema50", "adr"]


def test_indicator_specs_empty_when_none_declared():
    class Bare(stonks.Strategy):
        def on_tick(self, ctx):
            pass

    assert indicator_specs(Bare) == []


def test_indicator_doc_and_color_default_to_empty_string():
    i = Indicator("just a doc")
    assert i.doc == "just a doc"
    assert i.color == ""
    assert Indicator().doc == ""


def test_fake_context_plot_records_name_symbol_value_and_timestamp():
    ctx = FakeContext([
        FakeKLine(1000, "X", 1.0, 1.0, 1.0, 1.0, 1.0),
        FakeKLine(2000, "X", 2.0, 2.0, 2.0, 2.0, 1.0),
    ])

    ctx.advance()   # -> ts 1000
    ctx.plot("ema", "X", 10.0)
    ctx.advance()   # -> ts 2000
    ctx.plot("ema", "X", 11.0)
    ctx.plot("adr", "Y", 3.5)

    assert ctx.plots == [
        FakePlot("ema", "X", 10.0, 1000),
        FakePlot("ema", "X", 11.0, 2000),
        FakePlot("adr", "Y", 3.5, 2000),
    ]
