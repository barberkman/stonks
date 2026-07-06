"""Unit tests for the Param / param_specs framework in the stonks package.

Run from the project root with the app-local venv:

    app/python/.venv/bin/pytest app/python/ -q
"""

import pytest

import stonks
from stonks import Param, param_specs


class _Full(stonks.Strategy):
    risk = 0.02
    lookback = 20
    use_filter = True

    params = {
        "risk": Param("fraction of equity risked", unit="%"),
        "lookback": Param("bars of history", unit="bars"),
        "use_filter": Param("toggle the filter"),
    }

    def on_tick(self, ctx):
        pass


def test_param_specs_extracts_default_via_getattr():
    spec = param_specs(_Full)[0]
    assert spec == {"name": "risk", "default": 0.02, "type": "float",
                    "doc": "fraction of equity risked", "unit": "%"}


def test_param_specs_infers_int_and_bool_and_bool_precedes_int():
    by_name = {s["name"]: s for s in param_specs(_Full)}
    assert by_name["lookback"]["type"] == "int"
    # bool subclasses int in Python — the bool check must win.
    assert by_name["use_filter"]["type"] == "bool"
    assert by_name["use_filter"]["default"] is True


def test_param_specs_preserves_declaration_order():
    assert [s["name"] for s in param_specs(_Full)] == ["risk", "lookback", "use_filter"]


def test_param_specs_empty_when_no_params_declared():
    class Bare(stonks.Strategy):
        def on_tick(self, ctx):
            pass

    assert param_specs(Bare) == []


def test_param_specs_raises_on_undeclared_attribute():
    class Typo(stonks.Strategy):
        risk = 0.02
        params = {"rsik": Param("oops")}

        def on_tick(self, ctx):
            pass

    with pytest.raises(AttributeError):
        param_specs(Typo)


def test_param_specs_raises_on_unsupported_type():
    class Stringy(stonks.Strategy):
        mode = "fast"
        params = {"mode": Param("unsupported")}

        def on_tick(self, ctx):
            pass

    with pytest.raises(TypeError):
        param_specs(Stringy)


def test_param_doc_and_unit_default_to_empty_string():
    p = Param("just a doc")
    assert p.doc == "just a doc"
    assert p.unit == ""
    assert Param().doc == ""
