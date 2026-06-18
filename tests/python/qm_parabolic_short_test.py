"""Tests for Setup 3 — parabolic short (first red bar)."""

from qm_parabolic_short import QMParabolicShortStrategy
from qm_helpers import bar, buys, run, sells, to_klines

WARMUP = [bar(20.0) for _ in range(15)]      # flat (no run-up)
RUNUP = [bar(c) for c in [21, 23, 26, 30, 35]]  # 5 consecutive up-closes
RED = bar(33.0, hi=35.5, lo=32.5)            # first red bar after the run-up


def test_parabolic_short_fires_on_first_red_bar():
    ctx = run(QMParabolicShortStrategy(), to_klines("AAA", WARMUP + RUNUP + [RED]))
    assert len(sells(ctx)) == 1 and len(buys(ctx)) == 0  # sell-to-open


def test_parabolic_short_rejects_without_runup():
    rows = [bar(20.0) for _ in range(20)] + [bar(19.5)]  # red, but no parabolic run
    ctx = run(QMParabolicShortStrategy(), to_klines("AAA", rows))
    assert ctx.orders == []


def test_parabolic_short_rejects_when_bar_is_green():
    rows = WARMUP + RUNUP + [bar(36.0)]  # up-close, not a reversal
    ctx = run(QMParabolicShortStrategy(), to_klines("AAA", rows))
    assert ctx.orders == []


def test_parabolic_short_covers_on_green_close():
    rows = WARMUP + RUNUP + [RED, bar(32.0), bar(34.0)]  # green close -> cover
    s = QMParabolicShortStrategy()
    ctx = run(s, to_klines("AAA", rows))
    assert len(sells(ctx)) == 1 and len(buys(ctx)) == 1
    assert s.pos == {}


def test_parabolic_short_covers_on_stop():
    rows = WARMUP + RUNUP + [RED, bar(33.0), bar(33.0, hi=36.0, lo=32.0)]  # high tags stop
    s = QMParabolicShortStrategy()
    ctx = run(s, to_klines("AAA", rows))
    assert len(buys(ctx)) == 1
    assert s.pos == {}


def test_parabolic_short_covers_on_time():
    rows = WARMUP + RUNUP + [RED] + [bar(c) for c in [31, 30, 29, 28, 27, 26]]  # no green/stop
    s = QMParabolicShortStrategy()
    ctx = run(s, to_klines("AAA", rows))
    assert len(buys(ctx)) == 1  # covered after ps_max_hold bars
    assert s.pos == {}
