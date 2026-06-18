"""Unit tests for the shared indicators, sizing, and window-splitting helpers."""

import numpy as np
import pytest

import qm_common as qc
from qm_common import Params
from qm_helpers import DAY, bar, to_klines
from stonks.testing import FakeContext


def _arr(xs):
    return np.array(xs, dtype=np.float64)


def test_sma_and_insufficient_history():
    assert qc.sma(_arr([1, 2, 3, 4]), 2) == pytest.approx(3.5)
    assert qc.sma(_arr([1, 2]), 5) is None


def test_ema_of_constant_series_is_the_constant():
    assert qc.ema(_arr([5.0] * 30), 10) == pytest.approx(5.0)


def test_highest_lowest():
    a = _arr([3, 1, 4, 1, 5, 9, 2])
    assert qc.highest(a, 3) == pytest.approx(9.0)
    assert qc.lowest(a, 3) == pytest.approx(2.0)


def test_adr_pct_is_mean_bar_range():
    high = _arr([2.0, 2.0, 2.0])
    low = _arr([1.0, 1.0, 1.0])
    assert qc.adr_pct(high, low, 3) == pytest.approx(100.0)


def test_gain_pct_over_lookback():
    close = _arr([100.0, 105.0, 110.0])
    assert qc.gain_pct(close, 2) == pytest.approx(10.0)
    assert qc.gain_pct(close, 5) is None


def test_size_by_risk_uses_risk_fraction():
    p = Params()  # risk_per_trade_pct = 0.5
    qty = qc.size_by_risk(equity=100_000.0, cash=100_000.0, entry=100.0, stop=90.0, p=p)
    assert qty == pytest.approx(50.0)  # 500$ risk / 10$ per share


def test_size_by_risk_caps_at_available_cash():
    p = Params()
    qty = qc.size_by_risk(equity=1_000.0, cash=1_000.0, entry=100.0, stop=99.9, p=p)
    assert qty == pytest.approx(1_000.0 * 0.99 / 100.0)  # cap binds, not 5$/0.1


def test_size_by_risk_zero_when_no_risk():
    assert qc.size_by_risk(100_000.0, 100_000.0, 100.0, 100.0, Params()) == 0.0


def test_symbol_slices_splits_per_symbol():
    bars = to_klines("AAA", [bar(10), bar(11), bar(12)])
    bars += to_klines("BBB", [bar(20), bar(21), bar(22)])
    ctx = FakeContext(bars)
    for _ in range(3):
        ctx.advance()
    slices = qc.symbol_slices(ctx.history(3))
    assert set(slices) == {"AAA", "BBB"}
    assert len(slices["AAA"]) == 3 and len(slices["BBB"]) == 3
    assert slices["AAA"].close[-1] == pytest.approx(12.0)
    assert slices["BBB"].close[-1] == pytest.approx(22.0)


def test_universe_long_requires_uptrend():
    from qm_helpers import uptrend
    bars = to_klines("AAA", uptrend(60))
    ctx = FakeContext(bars)
    for _ in range(60):
        ctx.advance()
    up = qc.symbol_slices(ctx.history(60))["AAA"]
    assert qc.universe_long(up, Params()) is True
    assert qc.universe_short(up, Params()) is False
