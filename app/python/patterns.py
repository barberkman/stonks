"""Bulkowski chart-pattern library — one strategy, one selectable pattern.

Every detector in this file is transcribed from the corresponding page of
Thomas Bulkowski's *Encyclopedia of Chart Patterns* site,
https://thepatternsite.com/chartpatterns.html — the "Identification
Guidelines" table (shape, symmetry, volume, confirmation) and the "Trading
Tactics" table (entry, stop placement, measure rule) of each individual
pattern page. Each detector's docstring names its source page and quotes the
guidelines it mechanizes.

The strategy trades exactly ONE pattern per run, chosen with the `pattern`
parameter (an index into `PATTERNS`; see `PATTERN_NAMES`). That keeps a
backtest attributable to a single pattern's edge instead of a blend.

──────────────────────────────────────────────────────────────────────────────
How a pattern becomes a trade

Bulkowski's tactics are uniform across the book, so the engine is too:

  detect        On each bar close, the selected pattern's detector runs over
                every symbol's recent history and returns a `Setup` — the
                confirmation price, the protective stop, and the measure-rule
                target — or None.
  confirm       Nothing is bought at detection. The `Setup.trigger` is the
                page's *confirmation* price ("the pattern confirms as a valid
                one when price closes above ..."), and the engine arms a
                resting stop order there, good for `order_bars` bars. A
                pattern that never confirms never trades, which is the whole
                point of the confirmation column.
  bracket       The moment the entry rests, a protective stop (the page's
                stop-placement rule) and a limit at the measure-rule target
                are bracketed under it, reduce-only.
  size          Position size comes from the stop distance: a stop-out costs
                `risk_fraction` of equity, fees included, clamped by a
                notional cap and by free cash.
  time stop     A position still open after `max_hold_bars` bars is closed at
                market — the measure rule is a target, not a promise.

──────────────────────────────────────────────────────────────────────────────
Deviations from the book (engine reality, applied to every pattern)

 1. Decisions are made on completed bar closes and fills happen from the next
    bar on. Intraday "buy as it pierces the trendline" is not expressible.
 2. Bulkowski identifies patterns by eye. Every visual phrase is given a
    numeric definition — "near the same price" becomes a percentage, "wide and
    rounded" becomes a bar count, "several weeks apart" becomes a bar range.
    Those thresholds are module constants or class parameters, and each
    detector's docstring says which phrase each one stands for.
 3. Minor highs and lows follow minorhl.html: a peak with "no higher price
    within 5 days surrounding the peak. That's 2 days before to 2 days after"
    (`PIVOT_SPAN` = 2 either side). Every multi-turn pattern is built from
    those pivots, so a pivot is only usable `PIVOT_SPAN` bars after it prints.
 4. A pattern is armed in the single breakout direction its page names under
    "Breakout" — the engine does not arm both sides of a two-sided pattern
    (triangles, rectangles), because opposing resting orders on one symbol
    would net rather than cancel.
 5. Volume guidelines that are statistical rather than definitional ("volume
    trends downward 78% of the time") are enforced only where the page states
    them as identification requirements, and always through
    `require_volume_rules` so they can be switched off.
 6. Measure-rule targets follow the page's own Trading Tactics wording, in
    this order of preference: a page that names a price LEVEL as the target
    ("the lowest valley in the pattern is the price target") uses that level;
    a page that gives height x "percentage meeting price target" uses that
    percentage, recorded in `Spec.target_pct`; a page that gives the height
    with no published percentage uses the full height (`target_pct` 100).
    A few pages measure the target from a point other than the breakout
    (flags and pennants add the flagpole to the flag's BOTTOM, not to its
    top), which can put the target below the confirmation price on an
    unusually tall flag. That bracket is not tradable, so `_arm` skips it
    rather than quietly moving the target the book specifies.
 7. No pyramiding and one position per symbol: the broker rejects same-side
    adds, and a symbol already holding a position is skipped by the scan.

Execution timeline: decisions on bar close, fills from the next bar on.
"""

import bisect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

import numpy as np

import stonks
from stonks import OrderSide, OrderStatus

# minorhl.html: "find peaks that have no higher price within 5 days
# surrounding the peak. That's 2 days before to 2 days after the tallest peak."
PIVOT_SPAN = 2

CASH_USE = 0.99          # fraction of free cash one entry may consume
EMA_WINDOW = 100         # bars feeding ema_last


# ─── numeric helpers ─────────────────────────────────────────────────────────


def sma(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.mean(a[-n:]))


def ema_last(a, n):
    """Last value of an EMA(n) — an adjust-style EWM over the most recent
    EMA_WINDOW bars. Stateless, so a detector can be called on any slice."""
    if n <= 0 or len(a) < n:
        return None
    k = min(len(a), EMA_WINDOW)
    w = np.power(1.0 - 2.0 / (n + 1.0), np.arange(k, dtype=np.float64))
    return float(np.dot(a[-k:][::-1], w) / np.sum(w))


def linfit(x, y):
    """Least-squares (slope, intercept) of y on x; (0, mean) for a single
    point. Used wherever Bulkowski draws a trendline through pivots."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def pct(a, b):
    """(a - b) / b as a percentage; +inf guards a non-positive base."""
    if b == 0.0:
        return float("inf")
    return 100.0 * (a - b) / abs(b)


def near(a, b, tol_pct):
    """"...near the same price": within tol_pct of the larger magnitude."""
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return True
    return 100.0 * abs(a - b) / scale <= tol_pct


def segments(ts):
    """Per-symbol slice bounds of the combined history frame. Rows are
    contiguous per symbol and every printing symbol's slice ends at the
    tick's timestamp."""
    ends = np.flatnonzero(ts == ts[-1])
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    return starts, ends


# ─── the bar container every detector receives ───────────────────────────────


class Bars:
    """One symbol's recent history, plus the minor highs/lows of minorhl.html
    computed once and shared by every detector that needs them.

    Index 0 is the oldest bar, index `n - 1` the bar that just closed.
    """

    __slots__ = ("o", "h", "l", "c", "v", "n", "_peaks", "_valleys")

    def __init__(self, o, h, l, c, v, peaks=None, valleys=None):
        self.o, self.h, self.l, self.c, self.v = o, h, l, c, v
        self.n = len(c)
        # `peaks`/`valleys` let a caller that already knows this window's
        # pivots hand them over instead of paying for them again. A pivot at
        # index i depends only on bars i-PIVOT_SPAN..i+PIVOT_SPAN, so a prefix
        # of a longer window has exactly that window's pivots truncated — see
        # `_busted`, the one caller that exploits it.
        self._peaks = peaks
        self._valleys = valleys

    # minorhl.html — "peaks that have no higher price within 5 days
    # surrounding the peak", i.e. the extreme of a 2*PIVOT_SPAN+1 window.
    # A pivot at index i is only knowable at bar i + PIVOT_SPAN, so the last
    # PIVOT_SPAN bars can never host one: no lookahead.
    @property
    def peaks(self):
        if self._peaks is None:
            self._peaks = _pivots(self.h, True)
        return self._peaks

    @property
    def valleys(self):
        if self._valleys is None:
            self._valleys = _pivots(self.l, False)
        return self._valleys

    def peak_price(self, i):
        return float(self.h[i])

    def valley_price(self, i):
        return float(self.l[i])

    def highest(self, a, b):
        """Highest high over the inclusive bar range [a, b]."""
        a, b = max(0, a), min(self.n - 1, b)
        return float(np.max(self.h[a:b + 1])) if b >= a else float("nan")

    def lowest(self, a, b):
        a, b = max(0, a), min(self.n - 1, b)
        return float(np.min(self.l[a:b + 1])) if b >= a else float("nan")

    def arghighest(self, a, b):
        a, b = max(0, a), min(self.n - 1, b)
        return a + int(np.argmax(self.h[a:b + 1]))

    def arglowest(self, a, b):
        a, b = max(0, a), min(self.n - 1, b)
        return a + int(np.argmin(self.l[a:b + 1]))


def _pivots(series, want_high):
    """Indices of minor highs (or lows) under the 2-days-either-side rule.
    Ties resolve to the earliest bar of the plateau so a flat top yields one
    pivot, not several.

    Vectorised over a sliding 2*PIVOT_SPAN+1 window: bar i qualifies when it
    equals the window's extreme AND is the first bar to do so. `np.argmax` /
    `np.argmin` return the first extreme index, which is what makes the
    plateau tie-break come out the same as a bar-by-bar scan."""
    n = len(series)
    if n < 2 * PIVOT_SPAN + 1:
        return []
    win = np.lib.stride_tricks.sliding_window_view(series, 2 * PIVOT_SPAN + 1)
    centre = series[PIVOT_SPAN:n - PIVOT_SPAN]
    if want_high:
        ok = (centre >= win.max(axis=1)) & (win.argmax(axis=1) == PIVOT_SPAN)
    else:
        ok = (centre <= win.min(axis=1)) & (win.argmin(axis=1) == PIVOT_SPAN)
    return (np.flatnonzero(ok) + PIVOT_SPAN).tolist()


def valley_width(b, i, tol_pct):
    """Bars in the neighbourhood of valley `i` whose low sits within
    `tol_pct` of it — the numeric stand-in for Bulkowski's Adam/Eve shape
    test. "Adam bottoms are narrow, V-shaped"; "Eve bottoms are wide and more
    rounded appearing"."""
    base = float(b.l[i])
    lo, hi = max(0, i - 6), min(b.n - 1, i + 6)
    band = base * (1.0 + tol_pct / 100.0)
    return int(np.count_nonzero(b.l[lo:hi + 1] <= band))


def peak_width(b, i, tol_pct):
    """Mirror of `valley_width` for tops: "Adam tops are narrow, inverted
    V's"; "an Eve top is rounded and wide looking"."""
    base = float(b.h[i])
    lo, hi = max(0, i - 6), min(b.n - 1, i + 6)
    band = base * (1.0 - tol_pct / 100.0)
    return int(np.count_nonzero(b.h[lo:hi + 1] >= band))


def trend_down_into(b, i, window, min_drop_pct):
    """"Price trend: downward leading to the pattern." Somewhere in the
    `window` bars before index `i` price closed at least `min_drop_pct` above
    the low at `i`.

    The test looks for the highest close in the window rather than the close
    at its far edge: Bulkowski's guideline is about the trend into the
    pattern, not about a move of a fixed length, and a leg shorter than
    `window` should still qualify."""
    j = max(0, i - window)
    if i <= j:
        return False
    return pct(float(np.max(b.c[j:i + 1])), float(b.l[i])) >= min_drop_pct


def trend_up_into(b, i, window, min_rise_pct):
    """"Price trend: upward leading to the pattern" — the mirror: the high at
    `i` sits at least `min_rise_pct` above the lowest close in the window."""
    j = max(0, i - window)
    if i <= j:
        return False
    return pct(float(b.h[i]), float(np.min(b.c[j:i + 1]))) >= min_rise_pct


def volume_recedes(b, a, z):
    """"Volume trends downward" — least-squares slope of volume over the
    pattern's span is negative."""
    a, z = max(0, a), min(b.n - 1, z)
    if z - a < 3:
        return True
    slope, _ = linfit(np.arange(z - a + 1), b.v[a:z + 1])
    return slope < 0.0


# ─── what a detector returns ─────────────────────────────────────────────────


@dataclass
class Setup:
    """One armed trade: the page's confirmation price, its stop-placement
    rule, and its measure-rule target."""

    side: str          # "long" | "short"
    trigger: float     # confirmation price — the resting entry level
    stop: float        # protective stop, from the page's stop-placement rule
    target: Optional[float]   # measure-rule target; None where the page
                              # publishes no target, in which case no limit
                              # is bracketed and the trade leaves on its stop
                              # or on `max_hold_bars`
    note: str = ""     # what was matched, for the run log
    hold_bars: Optional[int] = None   # per-pattern time stop where the page
                                      # names one (e.g. "exit at the close 3
                                      # trading days after entry"); otherwise
                                      # `max_hold_bars` applies


@dataclass
class Spec:
    """Registry entry: a named pattern, its source page, and its detector."""

    name: str
    url: str
    kind: str          # "reversal" | "continuation" | "event" | "other"
    side: str          # the breakout direction the page names
    detect: Callable[["Bars", "PatternsStrategy"], Optional[Setup]]
    target_pct: float = 100.0   # page's "percentage meeting price target"
    tradeable: bool = True      # False where the page states the pattern is
                                # not tradeable and gives no entry rule; the
                                # detector still identifies it, but nothing
                                # is ever armed


PATTERNS: List[Spec] = []


def pattern(name, url, kind, side, target_pct=100.0, tradeable=True):
    """Register a detector under its book name."""

    def wrap(fn):
        PATTERNS.append(Spec(name, url, kind, side, fn, target_pct, tradeable))
        return fn

    return wrap


def measure_long(trigger, height, target_pct):
    """"Compute the height ... multiply it by the percentage meeting price
    target ... add the result to the breakout price."""
    return trigger + height * target_pct / 100.0


def measure_short(trigger, height, target_pct):
    return trigger - height * target_pct / 100.0


# ═══════════════════════════════════════════════════════════════════════════
# Head-and-shoulders family
# ═══════════════════════════════════════════════════════════════════════════


def _hs_neckline(b, left_armpit, right_armpit):
    """"Neckline: joins the two armpits." Returns its value projected to the
    last bar, plus its slope."""
    x = np.array([left_armpit, right_armpit], dtype=np.float64)
    y = np.array([b.h[left_armpit], b.h[right_armpit]], dtype=np.float64)
    slope, intercept = linfit(x, y)
    return slope * (b.n - 1) + intercept, slope


def _hs_neckline_low(b, left_armpit, right_armpit):
    """Top-side mirror: the neckline of a head-and-shoulders TOP joins the
    two armpits, which are valleys."""
    x = np.array([left_armpit, right_armpit], dtype=np.float64)
    y = np.array([b.l[left_armpit], b.l[right_armpit]], dtype=np.float64)
    slope, intercept = linfit(x, y)
    return slope * (b.n - 1) + intercept, slope


@pattern("head_and_shoulders_bottom", "hsb.html", "reversal", "long", 71.0)
def _hs_bottom(b, p):
    """hsb.html — "A 3-valley pattern with the middle valley below the
    others ... like an inverted person's head and shoulders, proportional,
    and not lopsided."

      price trend    "downward leading to the pattern"
      symmetry       "the two shoulders should bottom near the same price,
                     be nearly the same distance from the head"
      volume         "highest on the left shoulder or head, diminished on
                     the right shoulder"
      neckline       "joins the two armpits"
      confirmation   "price closes above a down-sloping neckline or above
                     the right armpit when the neckline slopes upward"
      measure rule   head's low to the neckline above it, x 71%, added to
                     the breakout price
    """
    vs = b.valleys
    if len(vs) < 3:
        return None
    ls, head, rs = vs[-3], vs[-2], vs[-1]
    if not (b.l[head] < b.l[ls] and b.l[head] < b.l[rs]):
        return None
    if not near(float(b.l[ls]), float(b.l[rs]), p.shoulder_tol_pct):
        return None
    dl, dr = head - ls, rs - head
    if min(dl, dr) <= 0 or max(dl, dr) > p.symmetry_ratio * min(dl, dr):
        return None
    if not trend_down_into(b, ls, p.trend_window, p.min_trend_pct):
        return None
    if p.require_volume_rules and not volume_recedes(b, ls, rs):
        return None
    la = b.arghighest(ls, head)          # left armpit
    ra = b.arghighest(head, rs)          # right armpit
    level, slope = _hs_neckline(b, la, ra)
    # "above the right armpit when the neckline slopes upward"
    trigger = float(b.h[ra]) if slope > 0.0 else max(level, float(b.c[-1]) * 1.0001)
    height = trigger - float(b.l[head])
    if height <= 0.0:
        return None
    return Setup("long", trigger, float(b.l[head]) * 0.999,
                 measure_long(trigger, height, 71.0),
                 f"HSB head {b.l[head]:.4f} neckline {trigger:.4f}")


@pattern("head_and_shoulders_top", "hst.html", "reversal", "short", 62.0)
def _hs_top(b, p):
    """hst.html — "Looks like a head perched atop two shoulders. A
    three-peak pattern with the middle peak above the others."

      price trend    "upward leading to the pattern"
      symmetry       "the two shoulders should peak near the same price, be
                     nearly the same distance from the head"
      volume         "highest on the left shoulder followed by the head"
      neckline       "joins the two armpits"
      confirmation   "price closes below an up-sloping neckline or below the
                     right armpit when the neckline slopes downward"
      stop           above the pattern's high (the head)
      measure rule   head to the neckline below it, x the target percentage,
                     subtracted from the breakout price
    """
    ps = b.peaks
    if len(ps) < 3:
        return None
    ls, head, rs = ps[-3], ps[-2], ps[-1]
    if not (b.h[head] > b.h[ls] and b.h[head] > b.h[rs]):
        return None
    if not near(float(b.h[ls]), float(b.h[rs]), p.shoulder_tol_pct):
        return None
    dl, dr = head - ls, rs - head
    if min(dl, dr) <= 0 or max(dl, dr) > p.symmetry_ratio * min(dl, dr):
        return None
    if not trend_up_into(b, ls, p.trend_window, p.min_trend_pct):
        return None
    if p.require_volume_rules and b.v[ls] < b.v[rs]:
        return None
    la = b.arglowest(ls, head)
    ra = b.arglowest(head, rs)
    level, slope = _hs_neckline_low(b, la, ra)
    trigger = float(b.l[ra]) if slope < 0.0 else min(level, float(b.c[-1]) * 0.9999)
    height = float(b.h[head]) - trigger
    if height <= 0.0:
        return None
    return Setup("short", trigger, float(b.h[head]) * 1.001,
                 measure_short(trigger, height, 62.0),
                 f"HST head {b.h[head]:.4f} neckline {trigger:.4f}")


@pattern("complex_head_and_shoulders_bottom", "chsb.html", "reversal", "long", 71.0)
def _complex_hs_bottom(b, p):
    """chsb.html — "a head-and-shoulders bottom with multiple shoulders or
    multiple heads, but rarely both".

      shape          five valleys, the centre one lowest, the outer pairs
                     mirroring each other ("look similar to their mirror
                     opposite")
      neckline       "joins the highest armpits"
      confirmation   close above the neckline (or the right armpit when the
                     neckline slopes up)
      measure rule   head to neckline, x 71%
    """
    vs = b.valleys
    if len(vs) < 5:
        return None
    s = vs[-5:]
    head = s[2]
    if any(b.l[head] >= b.l[i] for i in (s[0], s[1], s[3], s[4])):
        return None
    if not near(float(b.l[s[0]]), float(b.l[s[4]]), p.shoulder_tol_pct):
        return None
    if not near(float(b.l[s[1]]), float(b.l[s[3]]), p.shoulder_tol_pct):
        return None
    if not trend_down_into(b, s[0], p.trend_window, p.min_trend_pct):
        return None
    if p.require_volume_rules and not volume_recedes(b, s[0], s[4]):
        return None
    la = b.arghighest(s[0], head)        # "joins the highest armpits"
    ra = b.arghighest(head, s[4])
    level, slope = _hs_neckline(b, la, ra)
    trigger = float(b.h[ra]) if slope > 0.0 else max(level, float(b.c[-1]) * 1.0001)
    height = trigger - float(b.l[head])
    if height <= 0.0:
        return None
    return Setup("long", trigger, float(b.l[head]) * 0.999,
                 measure_long(trigger, height, 71.0),
                 f"complex HSB head {b.l[head]:.4f}")


@pattern("complex_head_and_shoulders_top", "chst.html", "reversal", "short", 62.0)
def _complex_hs_top(b, p):
    """chst.html — "a head-and-shoulders top with multiple shoulders or
    multiple heads, but rarely both".

      neckline       "joins the lowest armpits and is often nearly
                     horizontal. Rarely does it slope steeply."
      confirmation   close below the neckline (or the right armpit when the
                     neckline slopes down)
      stop           above the highest head
    """
    ps = b.peaks
    if len(ps) < 5:
        return None
    s = ps[-5:]
    head = s[2]
    if any(b.h[head] <= b.h[i] for i in (s[0], s[1], s[3], s[4])):
        return None
    if not near(float(b.h[s[0]]), float(b.h[s[4]]), p.shoulder_tol_pct):
        return None
    if not near(float(b.h[s[1]]), float(b.h[s[3]]), p.shoulder_tol_pct):
        return None
    if not trend_up_into(b, s[0], p.trend_window, p.min_trend_pct):
        return None
    la = b.arglowest(s[0], head)         # "joins the lowest armpits"
    ra = b.arglowest(head, s[4])
    level, slope = _hs_neckline_low(b, la, ra)
    trigger = float(b.l[ra]) if slope < 0.0 else min(level, float(b.c[-1]) * 0.9999)
    height = float(b.h[head]) - trigger
    if height <= 0.0:
        return None
    return Setup("short", trigger, float(b.h[head]) * 1.001,
                 measure_short(trigger, height, 62.0),
                 f"complex HST head {b.h[head]:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Double bottoms and tops (Adam/Eve family)
# ═══════════════════════════════════════════════════════════════════════════


def _double_bottom(b, p, left_shape, right_shape, target_pct, sep_lo, sep_hi,
                   bottom_tol, label):
    """Shared skeleton of every double bottom. The pages differ only in the
    shape of each bottom, the allowed price variation between them, and the
    measure-rule percentage; everything else is identical:

      price trend    "downward leading to the pattern"
      peak           "the rise between bottoms should measure at least 10%"
      confirmation   "confirms as a true double bottom once price closes
                     above the peak between the two valleys"
      stop           "slightly below the lower of the two bottoms"
      measure rule   highest peak between the bottoms to the lowest valley,
                     x the target percentage, added to the breakout price

    `left_shape`/`right_shape` are "adam" (narrow, V-shaped) or "eve" (wide,
    rounded), tested with `valley_width`.
    """
    vs = b.valleys
    if len(vs) < 2:
        return None
    v1, v2 = vs[-2], vs[-1]
    sep = v2 - v1
    if not sep_lo <= sep <= sep_hi:
        return None
    p1, p2 = float(b.l[v1]), float(b.l[v2])
    if not near(p1, p2, bottom_tol):
        return None
    if not trend_down_into(b, v1, p.trend_window, p.min_trend_pct):
        return None
    if not _shape_ok(b, v1, left_shape, p, low=True):
        return None
    if not _shape_ok(b, v2, right_shape, p, low=True):
        return None
    pk = b.arghighest(v1, v2)
    peak = float(b.h[pk])
    if pct(peak, min(p1, p2)) < p.min_double_rise_pct:
        return None                       # "should measure at least 10%"
    if p.require_volume_rules and float(np.mean(b.v[v1 - 2:v1 + 3])) < \
            float(np.mean(b.v[v2 - 2:v2 + 3])):
        return None                       # "usually higher on the first bottom"
    trigger = peak
    height = peak - min(p1, p2)
    return Setup("long", trigger, min(p1, p2) * 0.999,
                 measure_long(trigger, height, target_pct),
                 f"{label} bottoms {p1:.4f}/{p2:.4f} peak {peak:.4f}")


def _double_top(b, p, left_shape, right_shape, target_pct, sep_lo, sep_hi,
                top_tol, label):
    """Shared skeleton of every double top — the mirror of `_double_bottom`.

      price trend    "upward leading to the pattern"
      valley         "the valley drop between the tops should measure at
                     least 10%"
      confirmation   "confirms as a true double top once price closes below
                     the valley between the two peaks"
      stop           "a few pennies above the highest peak"
    """
    ps = b.peaks
    if len(ps) < 2:
        return None
    t1, t2 = ps[-2], ps[-1]
    sep = t2 - t1
    if not sep_lo <= sep <= sep_hi:
        return None
    p1, p2 = float(b.h[t1]), float(b.h[t2])
    if not near(p1, p2, top_tol):
        return None
    if not trend_up_into(b, t1, p.trend_window, p.min_trend_pct):
        return None
    if not _shape_ok(b, t1, left_shape, p, low=False):
        return None
    if not _shape_ok(b, t2, right_shape, p, low=False):
        return None
    vl = b.arglowest(t1, t2)
    valley = float(b.l[vl])
    if pct(max(p1, p2), valley) < p.min_double_rise_pct:
        return None
    if p.require_volume_rules and float(np.mean(b.v[t1 - 2:t1 + 3])) < \
            float(np.mean(b.v[t2 - 2:t2 + 3])):
        return None                       # "usually higher on the left peak"
    trigger = valley
    height = max(p1, p2) - valley
    return Setup("short", trigger, max(p1, p2) * 1.001,
                 measure_short(trigger, height, target_pct),
                 f"{label} tops {p1:.4f}/{p2:.4f} valley {valley:.4f}")


def _shape_ok(b, i, shape, p, low):
    """"Adam bottoms are narrow, V-shaped, sometimes with one long price
    spike"; "Eve bottoms are wide and more rounded appearing". Width is the
    count of nearby bars sharing the extreme (see `valley_width`)."""
    if shape == "any":
        return True
    w = (valley_width(b, i, p.shape_tol_pct) if low
         else peak_width(b, i, p.shape_tol_pct))
    if shape == "adam":
        return w <= p.adam_max_width
    return w >= p.eve_min_width


@pattern("adam_adam_double_bottom", "aadb.html", "reversal", "long", 73.0)
def _aadb(b, p):
    """aadb.html — "two distinct valleys that look similar. Adam bottoms are
    narrow, V-shaped, sometimes with one long price spike." Price variation
    between bottoms averages 1%; "the twin valleys are usually several weeks
    apart (16 days is the median)". Measure rule x 73%."""
    return _double_bottom(b, p, "adam", "adam", 73.0, p.dbl_sep_min,
                          p.dbl_sep_max, p.adam_bottom_tol_pct, "Adam&Adam")


@pattern("adam_adam_double_top", "aadt.html", "reversal", "short", 64.0)
def _aadt(b, p):
    """aadt.html — "usually twin spikes poking above the surrounding price
    landscape. Adam tops are narrow, inverted V's." "The variation between
    price peaks is small, usually less than 3%." Measure rule x 64%."""
    return _double_top(b, p, "adam", "adam", 64.0, p.dbl_sep_min,
                       p.dbl_sep_max, p.top_tol_pct, "Adam&Adam")


@pattern("adam_eve_double_bottom", "aedb.html", "reversal", "long", 69.0)
def _aedb(b, p):
    """aedb.html — "two distinct valleys appearing different: Adam is narrow
    and V-shaped; Eve is wide and rounded." "Average separation is nearly two
    months." Measure rule x 69%."""
    return _double_bottom(b, p, "adam", "eve", 69.0, p.dbl_sep_min,
                          p.dbl_sep_max, p.bottom_tol_pct, "Adam&Eve")


@pattern("adam_eve_double_top", "aedt.html", "reversal", "short", 54.0)
def _aedt(b, p):
    """aedt.html — "Adam tops appear first and are narrow, inverted Vs but
    Eve follows Adam and is more rounded looking and wider." Variation
    between peaks "less than 3%"; separation 2-7 weeks. Stop goes "above the
    Eve peak. The wide and rounded top makes for a good resistance area."
    Measure rule x 54%."""
    return _double_top(b, p, "adam", "eve", 54.0, p.dbl_sep_min,
                       p.dbl_sep_max, p.top_tol_pct, "Adam&Eve")


@pattern("eve_adam_double_bottom", "eadb.html", "reversal", "long", 72.0)
def _eadb(b, p):
    """eadb.html — "Eve appears first (wider, rounded), Adam follows (narrow,
    V-shaped with potential price spike)." Bottom variation "usually between
    0% and 4%"; separation 2-7 weeks, median 23 days. Measure rule x 72%."""
    return _double_bottom(b, p, "eve", "adam", 72.0, p.dbl_sep_min,
                          p.dbl_sep_max, 4.0, "Eve&Adam")


@pattern("eve_adam_double_top", "eadt.html", "reversal", "short", 55.0)
def _eadt(b, p):
    """eadt.html — "Eve appears first and is rounded looking and wider than
    Adam. Adam comes second and is narrow, an inverted V, often appears as a
    1-day price spike." Separation 2-6 weeks. Measure rule x 55%."""
    return _double_top(b, p, "eve", "adam", 55.0, p.dbl_sep_min,
                       p.dbl_sep_max, p.top_tol_pct, "Eve&Adam")


@pattern("eve_eve_double_bottom", "eedb.html", "reversal", "long", 65.0)
def _eedb(b, p):
    """eedb.html — "two distinct valleys that look similar. Eve bottoms are
    wide and more rounded appearing." Bottom variation "usually between 0%
    and 6%"; "the twin valleys are several weeks apart with most falling in
    the 2 to 7 week range". Measure rule x 65%."""
    return _double_bottom(b, p, "eve", "eve", 65.0, p.dbl_sep_min,
                          p.dbl_sep_max, 6.0, "Eve&Eve")


@pattern("eve_eve_double_top", "eedt.html", "reversal", "short", 43.0)
def _eedt(b, p):
    """eedt.html — "two distinct tops that look similar. An Eve top is
    rounded and wide looking." Peak variation "often less than 3%";
    separation typically 2-6 weeks. Measure rule x 43%."""
    return _double_top(b, p, "eve", "eve", 43.0, p.dbl_sep_min,
                       p.dbl_sep_max, p.top_tol_pct, "Eve&Eve")


@pattern("ugly_double_bottom", "udb.html", "reversal", "long", 63.0)
def _ugly_double_bottom(b, p):
    """udb.html — "looks like a double bottom with unequal bottoms. The
    second bottom should be between 5% and 15% higher than the first, and a
    consecutive minor low (no intervening low)."

      volume         "recedes 80% of the time"
      breakout       "upward when price closes above the highest high
                     between the two bottoms"
      measure rule   peak to the LEFT bottom, x 63%, added to the breakout
    """
    vs = b.valleys
    if len(vs) < 2:
        return None
    v1, v2 = vs[-2], vs[-1]              # consecutive minor lows by construction
    p1, p2 = float(b.l[v1]), float(b.l[v2])
    rise = pct(p2, p1)
    if not p.ugly_rise_min <= rise <= p.ugly_rise_max:
        return None
    if not trend_down_into(b, v1, p.trend_window, p.min_trend_pct):
        return None
    if p.require_volume_rules and not volume_recedes(b, v1, v2):
        return None
    pk = b.arghighest(v1, v2)
    trigger = float(b.h[pk])
    height = trigger - p1               # "height from peak (C) to left bottom (A)"
    if height <= 0.0:
        return None
    return Setup("long", trigger, p1 * 0.999,
                 measure_long(trigger, height, 63.0),
                 f"ugly DB {p1:.4f}->{p2:.4f} (+{rise:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# Triple bottoms and tops
# ═══════════════════════════════════════════════════════════════════════════


@pattern("triple_bottom", "tb.html", "reversal", "long", 64.0)
def _triple_bottom(b, p):
    """tb.html — "three distinct valleys that look similar", with "the price
    variation between bottoms ... small such that it appears the three
    valleys bottom near the same price".

      price trend    "downward leading to the pattern but should not drop
                     below the first bottom"
      volume         "usually higher on the first bottom than on the last"
      confirmation   "once price closes above the highest peak between the
                     valleys"
      stop           below the pattern's lowest valley
      measure rule   highest peak (A) to the lowest valley (B), x the target
                     percentage, added to the breakout price
    """
    vs = b.valleys
    if len(vs) < 3:
        return None
    v1, v2, v3 = vs[-3], vs[-2], vs[-1]
    lows = [float(b.l[i]) for i in (v1, v2, v3)]
    if not (near(lows[0], lows[1], p.triple_tol_pct)
            and near(lows[1], lows[2], p.triple_tol_pct)
            and near(lows[0], lows[2], p.triple_tol_pct)):
        return None
    if not trend_down_into(b, v1, p.trend_window, p.min_trend_pct):
        return None
    # "should not drop below the first bottom"
    if b.lowest(v1 + 1, b.n - 1) < lows[0] * (1.0 - p.triple_tol_pct / 100.0):
        return None
    if p.require_volume_rules and float(b.v[v1]) < float(b.v[v3]):
        return None
    pk = b.arghighest(v1, v3)
    trigger = float(b.h[pk])
    height = trigger - min(lows)
    if height <= 0.0:
        return None
    return Setup("long", trigger, min(lows) * 0.999,
                 measure_long(trigger, height, 64.0),
                 f"triple bottom {lows[0]:.4f}/{lows[1]:.4f}/{lows[2]:.4f}")


@pattern("triple_top", "tt.html", "reversal", "short", 49.0)
def _triple_top(b, p):
    """tt.html — "three peaks near the same price with a downward breakout";
    "sometimes the middle peak is priced marginally below the other two".

      price trend    "upward leading to the pattern"
      volume         "trends downward 62% of the time"
      confirmation   "the pattern becomes valid when price closes below the
                     lowest valley in the pattern"
      stop           above the highest peak
      measure rule   highest peak to the lowest valley, x 49%, subtracted
                     from the lowest valley
    """
    ps = b.peaks
    if len(ps) < 3:
        return None
    t1, t2, t3 = ps[-3], ps[-2], ps[-1]
    highs = [float(b.h[i]) for i in (t1, t2, t3)]
    if not (near(highs[0], highs[1], p.triple_tol_pct)
            and near(highs[1], highs[2], p.triple_tol_pct)
            and near(highs[0], highs[2], p.triple_tol_pct)):
        return None
    if not trend_up_into(b, t1, p.trend_window, p.min_trend_pct):
        return None
    if p.require_volume_rules and not volume_recedes(b, t1, t3):
        return None
    vl = b.arglowest(t1, t3)
    trigger = float(b.l[vl])
    height = max(highs) - trigger
    if height <= 0.0:
        return None
    return Setup("short", trigger, max(highs) * 1.001,
                 measure_short(trigger, height, 49.0),
                 f"triple top {highs[0]:.4f}/{highs[1]:.4f}/{highs[2]:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Triangles
# ═══════════════════════════════════════════════════════════════════════════


def _recent_pivots(idxs, lo, hi):
    return [i for i in idxs if lo <= i <= hi]


def _triangle(b, p, top_kind, bot_kind):
    """Shared triangle geometry (at.html, dt.html, st.html):

      shape      "prices move between two converging trendlines"
      touches    "price must touch one trendline at least three times, the
                 other at least twice, forming distinct valleys and peaks"
      crossing   "price must cross the pattern from side to side, filling
                 the triangle with price movement, not white space"
      volume     "trends downward"

    `top_kind`/`bot_kind` are "flat", "down" or "up", matching the page's
    trendline column. Returns (top_slope, top_at_end, bot_slope, bot_at_end,
    first_pivot_index) or None.
    """
    lo = max(0, b.n - 1 - p.triangle_lookback)
    hi = b.n - 1
    pk = _recent_pivots(b.peaks, lo, hi)
    vl = _recent_pivots(b.valleys, lo, hi)
    if len(pk) < 2 or len(vl) < 2 or len(pk) + len(vl) < 5:
        return None                       # 3 touches + 2 touches
    ts, ti = linfit(pk, [b.h[i] for i in pk])
    bs, bi = linfit(vl, [b.l[i] for i in vl])
    span = hi - min(pk[0], vl[0])
    if span < p.triangle_min_bars:
        return None
    scale = float(np.mean(b.c[lo:hi + 1]))
    flat = p.trendline_flat_pct / 100.0 * scale / max(span, 1)
    if top_kind == "flat" and abs(ts) > flat:
        return None
    if top_kind == "down" and ts >= -flat:
        return None
    if bot_kind == "flat" and abs(bs) > flat:
        return None
    if bot_kind == "up" and bs <= flat:
        return None
    top_end = ts * hi + ti
    bot_end = bs * hi + bi
    if top_end <= bot_end:
        return None                       # already past the apex
    # "converging": the opening must be wider at the start than at the end
    top_start = ts * min(pk[0], vl[0]) + ti
    bot_start = bs * min(pk[0], vl[0]) + bi
    if (top_start - bot_start) <= (top_end - bot_end):
        return None
    if p.require_volume_rules and not volume_recedes(b, min(pk[0], vl[0]), hi):
        return None
    return ts, top_end, bs, bot_end, min(pk[0], vl[0])


@pattern("ascending_triangle", "at.html", "continuation", "long", 70.0)
def _ascending_triangle(b, p):
    """at.html — "two trendlines bound prices; the top trendline is
    horizontal and the bottom one slopes upward". Volume "trends downward at
    least 78% of the time". Breakout is "upward 63% of the time".

      stop           "on the side opposite the breakout" — the minor lows
                     near point A, i.e. the lower trendline
      measure rule   the horizontal trendline (B) to the lowest valley (A),
                     x 70%, added to the breakout price
    """
    t = _triangle(b, p, "flat", "up")
    if t is None:
        return None
    _, top_end, _, bot_end, first = t
    trigger = top_end
    height = top_end - b.lowest(first, b.n - 1)
    if height <= 0.0:
        return None
    return Setup("long", trigger, bot_end * 0.999,
                 measure_long(trigger, height, 70.0),
                 f"ascending triangle top {trigger:.4f}")


@pattern("descending_triangle", "dt.html", "continuation", "short", 50.0)
def _descending_triangle(b, p):
    """dt.html — "bounded by two trendlines, the bottom one horizontal and
    the top sloping downward". Volume "recedes 78% of the time and gets quite
    low just before the breakout". The engine arms the classic downward
    break (deviation 4).

      confirmation   "price closes outside one of the trendlines"
      stop           "a penny beyond the opposite trendline"
      measure rule   highest peak (A) to the horizontal trendline (B), x 50%
                     for a down breakout, subtracted from the breakout price
    """
    t = _triangle(b, p, "down", "flat")
    if t is None:
        return None
    _, top_end, _, bot_end, first = t
    trigger = bot_end
    height = b.highest(first, b.n - 1) - bot_end
    if height <= 0.0:
        return None
    return Setup("short", trigger, top_end * 1.001,
                 measure_short(trigger, height, 50.0),
                 f"descending triangle base {trigger:.4f}")


@pattern("symmetrical_triangle", "st.html", "continuation", "long", 58.0)
def _symmetrical_triangle(b, p):
    """st.html — "the bottom trendline slopes up and the top one slopes
    down". Volume "trends downward 84% to 86% of the time". Breakout is
    "upward 60% of the time", which is the side the engine arms.

      stop           beyond the opposite trendline
      measure rule   pattern height (peak to valley), x 58% for an up
                     breakout, added to the breakout price
    """
    t = _triangle(b, p, "down", "up")
    if t is None:
        return None
    _, top_end, _, bot_end, first = t
    trigger = top_end
    height = b.highest(first, b.n - 1) - b.lowest(first, b.n - 1)
    if height <= 0.0:
        return None
    return Setup("long", trigger, bot_end * 0.999,
                 measure_long(trigger, height, 58.0),
                 f"symmetrical triangle apex-side {trigger:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Rectangles, broadening formations and wedges
#
# All of these are "price bounces between two trendlines" patterns; they
# differ only in the sign of each slope and in whether the lines converge or
# diverge. `_lines` fits both trendlines and counts touches; each detector
# then applies its page's Trendlines / Touches / Breakout rows.
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Lines:
    ts: float          # top-trendline slope
    ti: float
    bs: float          # bottom-trendline slope
    bi: float
    first: int         # first pivot index of the pattern
    last: int
    top_start: float
    top_end: float
    bot_start: float
    bot_end: float
    n_top: int         # touches of the top line
    n_bot: int


def _lines(b, p, lookback, min_span):
    """Fit the two bounding trendlines through the minor highs and minor lows
    of the last `lookback` bars and count touches.

    "Price should touch one trendline at least three times and the other at
    least twice" is the touch rule shared by rectangles, broadening
    formations, wedges and triangles; a touch is a pivot sitting within
    `touch_tol_pct` of its line.
    """
    lo, hi = max(0, b.n - 1 - lookback), b.n - 1
    pk = [i for i in b.peaks if lo <= i <= hi]
    vl = [i for i in b.valleys if lo <= i <= hi]
    if len(pk) < 2 or len(vl) < 2:
        return None
    first = min(pk[0], vl[0])
    if hi - first < min_span:
        return None
    ts, ti = linfit(pk, [b.h[i] for i in pk])
    bs, bi = linfit(vl, [b.l[i] for i in vl])
    scale = float(np.mean(b.c[lo:hi + 1]))
    tol = p.touch_tol_pct / 100.0 * scale
    n_top = sum(1 for i in pk if abs(b.h[i] - (ts * i + ti)) <= tol)
    n_bot = sum(1 for i in vl if abs(b.l[i] - (bs * i + bi)) <= tol)
    if max(n_top, n_bot) < 3 or min(n_top, n_bot) < 2:
        return None
    return Lines(ts, ti, bs, bi, first, hi,
                 ts * first + ti, ts * hi + ti,
                 bs * first + bi, bs * hi + bi, n_top, n_bot)


def _flat(b, p, slope, lo, hi):
    """"Two near horizontal trendlines": the line must not travel more than
    `trendline_flat_pct` of price across the pattern."""
    scale = float(np.mean(b.c[max(0, lo):hi + 1]))
    span = max(hi - lo, 1)
    return abs(slope) * span <= p.trendline_flat_pct / 100.0 * scale


def _rectangle(b, p, want_up, target_pct, label):
    """Shared body of rectbots.html / recttops.html — "prices have flat tops
    and flat bottoms, crossing the pattern from side to side following two
    parallel trendlines", "two near horizontal trendlines bound price
    action", 5-touch minimum, volume "trends downward".

      entry          "wait for price to close outside the trendline"
      stop           beyond the opposing trendline
      measure rule   the height between the two trendlines x the target
                     percentage, added to the top trendline
    """
    ln = _lines(b, p, p.channel_lookback, p.channel_min_bars)
    if ln is None:
        return None
    if not (_flat(b, p, ln.ts, ln.first, ln.last)
            and _flat(b, p, ln.bs, ln.first, ln.last)):
        return None
    top = (ln.top_start + ln.top_end) / 2.0
    bot = (ln.bot_start + ln.bot_end) / 2.0
    height = top - bot
    if height <= 0.0:
        return None
    if want_up:
        if not trend_down_into(b, ln.first, p.trend_window, p.min_trend_pct):
            return None
    elif not trend_up_into(b, ln.first, p.trend_window, p.min_trend_pct):
        return None
    if p.require_volume_rules and not volume_recedes(b, ln.first, ln.last):
        return None
    # Both rectangles break upward more often than not (59% / 63%).
    return Setup("long", top, bot * 0.999,
                 measure_long(top, height, target_pct),
                 f"{label} {bot:.4f}-{top:.4f}")


@pattern("rectangle_bottom", "rectbots.html", "continuation", "long", 79.0)
def _rectangle_bottom(b, p):
    """rectbots.html — a rectangle with "price trend: downward leading to the
    chart pattern". Volume "trends downward at least 71% of the time";
    breakout is "upward 59% of the time" and 79% of up breakouts meet the
    target. "Compute the height between the two trendlines ... multiply it by
    the percentage meeting price target. Add it to the price of the top
    trendline."""
    return _rectangle(b, p, True, 79.0, "rectangle bottom")


@pattern("rectangle_top", "recttops.html", "continuation", "long", 78.0)
def _rectangle_top(b, p):
    """recttops.html — a rectangle with "price trend: upward leading to the
    chart pattern". Volume trends downward 70% of the time; breakout is
    "upward 63% of the time" and 78% of up breakouts meet the target."""
    return _rectangle(b, p, False, 78.0, "rectangle top")


@pattern("broadening_bottom", "broadb.html", "reversal", "long", 100.0)
def _broadening_bottom(b, p):
    """broadb.html — "higher peaks and lower valleys, a megaphone shape",
    "the top trend line slopes upward, the bottom one slopes downward",
    "price trend: downward leading to the pattern". Breakout is "upward 60%
    of the time ... when price pierces a trendline or moves above/below the
    top/bottom of the pattern".

      stop           beyond the pattern boundary
      measure rule   the difference between the highest peak (A) and lowest
                     valley (B), added to the pattern's top
    """
    ln = _lines(b, p, p.channel_lookback, p.channel_min_bars)
    if ln is None or ln.ts <= 0.0 or ln.bs >= 0.0:
        return None                        # not diverging
    if not trend_down_into(b, ln.first, p.trend_window, p.min_trend_pct):
        return None
    top = b.highest(ln.first, ln.last)
    bot = b.lowest(ln.first, ln.last)
    height = top - bot
    if height <= 0.0:
        return None
    return Setup("long", top, bot * 0.999, measure_long(top, height, 100.0),
                 f"broadening bottom {bot:.4f}-{top:.4f}")


@pattern("broadening_top", "bt.html", "reversal", "long", 100.0)
def _broadening_top(b, p):
    """bt.html — the same megaphone with "price trend: upward leading to the
    pattern. That is, the trend start is below the pattern's start."
    Breakout "can occur in any direction (upward 60%)", which is the side
    armed here (deviation 4).

      stop           "a penny below the pattern's low"
      measure rule   highest peak (A) minus lowest valley (B), added to the
                     pattern's top
    """
    ln = _lines(b, p, p.channel_lookback, p.channel_min_bars)
    if ln is None or ln.ts <= 0.0 or ln.bs >= 0.0:
        return None
    if not trend_up_into(b, ln.first, p.trend_window, p.min_trend_pct):
        return None
    top = b.highest(ln.first, ln.last)
    bot = b.lowest(ln.first, ln.last)
    height = top - bot
    if height <= 0.0:
        return None
    return Setup("long", top, bot * 0.999, measure_long(top, height, 100.0),
                 f"broadening top {bot:.4f}-{top:.4f}")


@pattern("right_angled_broadening_ascending", "rabfa.html", "reversal", "long", 67.0)
def _rabfa(b, p):
    """rabfa.html — "a megaphone tilted up with the bottom horizontal": "the
    bottom trendline is horizontal, the top one slopes upward". Volume
    "trends upward 62% to 63% of the time"; breakout "upward 55% of the
    time"; 67% of up breakouts meet the target.

      entry          "buy at the horizontal trendline when price starts
                     rising" — mechanized as the break of the pattern top
      stop           below the lower trendline
      measure rule   the height from the highest peak (A), x 67%, added to
                     the highest peak
    """
    ln = _lines(b, p, p.channel_lookback, p.channel_min_bars)
    if ln is None or ln.ts <= 0.0:
        return None
    if not _flat(b, p, ln.bs, ln.first, ln.last):
        return None
    top = b.highest(ln.first, ln.last)
    bot = (ln.bot_start + ln.bot_end) / 2.0
    height = top - bot
    if height <= 0.0:
        return None
    return Setup("long", top, bot * 0.999, measure_long(top, height, 67.0),
                 f"RABF ascending base {bot:.4f} top {top:.4f}")


@pattern("right_angled_broadening_descending", "rabfd.html", "reversal", "long", 65.0)
def _rabfd(b, p):
    """rabfd.html — "a megaphone tilted down with the top horizontal": "the
    top trendline is horizontal, the bottom one slopes downward". Breakout
    "upward 64% of the time"; 65% of up breakouts meet the target.

      stop           below the bottom trendline
      measure rule   the distance from the horizontal top trendline to the
                     lowest valley, x 65%, added to the horizontal trendline
    """
    ln = _lines(b, p, p.channel_lookback, p.channel_min_bars)
    if ln is None or ln.bs >= 0.0:
        return None
    if not _flat(b, p, ln.ts, ln.first, ln.last):
        return None
    top = (ln.top_start + ln.top_end) / 2.0
    bot = b.lowest(ln.first, ln.last)
    height = top - bot
    if height <= 0.0:
        return None
    return Setup("long", top, ln.bot_end * 0.999,
                 measure_long(top, height, 65.0),
                 f"RABF descending top {top:.4f} low {bot:.4f}")


@pattern("ascending_broadening_wedge", "abw.html", "reversal", "short", 100.0)
def _ascending_broadening_wedge(b, p):
    """abw.html — "a megaphone tilted up": "both trendlines slope upward. The
    top one slopes more steeply than the bottom one." Volume "trends upward
    66% to 67% of the time"; breakout is "downward 52% of the time", the side
    armed here.

      measure rule   for downward breaks, "use the lowest valley as target"
      stop           above the pattern's high (the mirror of the page's
                     "a penny below the lowest price bar" long-side stop)
    """
    ln = _lines(b, p, p.channel_lookback, p.channel_min_bars)
    if ln is None or ln.ts <= 0.0 or ln.bs <= 0.0 or ln.ts <= ln.bs:
        return None
    bot = b.lowest(ln.first, ln.last)
    top = b.highest(ln.first, ln.last)
    trigger = ln.bot_end
    if not bot < trigger < top:
        return None
    return Setup("short", trigger, top * 1.001, bot,
                 f"ascending broadening wedge low {bot:.4f}")


@pattern("descending_broadening_wedge", "dbw.html", "continuation", "long", 100.0)
def _descending_broadening_wedge(b, p):
    """dbw.html — "a megaphone tilted down": both trendlines slope downward
    and diverge, with "at least five trendline touches". Volume trends
    upward. Roughly 72% of breakouts are upward.

      entry          "buy when price breaks above the top trendline"
      stop           "a penny below the bottom of the chart pattern"
      measure rule   for upward breakouts, "use the highest peak as target"
    """
    ln = _lines(b, p, p.channel_lookback, p.channel_min_bars)
    if ln is None or ln.ts >= 0.0 or ln.bs >= 0.0 or ln.bs >= ln.ts:
        return None                        # both down, bottom steeper: diverging
    top = b.highest(ln.first, ln.last)
    bot = b.lowest(ln.first, ln.last)
    trigger = ln.top_end
    if not bot < trigger < top:
        return None
    return Setup("long", trigger, bot * 0.999, top,
                 f"descending broadening wedge high {top:.4f}")


@pattern("rising_wedge", "risewedge.html", "reversal", "short", 32.0)
def _rising_wedge(b, p):
    """risewedge.html — "a narrowing and rising triangle shape": "price
    bounces between two up-sloping and converging trendlines", minimum
    duration 3 weeks "otherwise it's a pennant". Volume "trends downward 79%
    of the time until the breakout". Breakout is "downward 60% of the time";
    32% of down breakouts meet the target.

      confirmation   "price closes outside one of the trendlines"
      stop           above the opposite trendline
      measure rule   "lowest valley in the pattern (A) is the price target"
    """
    ln = _lines(b, p, p.channel_lookback, p.wedge_min_bars)
    if ln is None or ln.ts <= 0.0 or ln.bs <= 0.0:
        return None
    if (ln.top_start - ln.bot_start) <= (ln.top_end - ln.bot_end):
        return None                        # must converge
    if p.require_volume_rules and not volume_recedes(b, ln.first, ln.last):
        return None
    bot = b.lowest(ln.first, ln.last)
    trigger = ln.bot_end
    if not bot < trigger:
        return None
    return Setup("short", trigger, ln.top_end * 1.001, bot,
                 f"rising wedge, target lowest valley {bot:.4f}")


@pattern("falling_wedge", "fallwedge.html", "reversal", "long", 100.0)
def _falling_wedge(b, p):
    """fallwedge.html — "price follows two down-sloping and converging
    trendlines", "3 weeks is the minimum duration, otherwise it's a pennant".
    Volume trends downward until the breakout; breakout is "upward 68% of the
    time".

      confirmation   "price closes outside one of the trendlines"
      stop           below the opposite trendline
      measure rule   the height between the highest peak (A) and lowest
                     valley (B), added to the breakout price
    """
    ln = _lines(b, p, p.channel_lookback, p.wedge_min_bars)
    if ln is None or ln.ts >= 0.0 or ln.bs >= 0.0:
        return None
    if (ln.top_start - ln.bot_start) <= (ln.top_end - ln.bot_end):
        return None
    if p.require_volume_rules and not volume_recedes(b, ln.first, ln.last):
        return None
    top = b.highest(ln.first, ln.last)
    bot = b.lowest(ln.first, ln.last)
    trigger = ln.top_end
    height = top - bot
    if height <= 0.0 or trigger <= ln.bot_end:
        return None
    return Setup("long", trigger, ln.bot_end * 0.999,
                 measure_long(trigger, height, 100.0),
                 f"falling wedge break {trigger:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Cups
# ═══════════════════════════════════════════════════════════════════════════


@pattern("cup_with_handle", "cup.html", "continuation", "long", 61.0)
def _cup_with_handle(b, p):
    """cup.html — "the cup should be U-shaped, not V-shaped", "the cup must
    have a handle on the right", "cup rims should be near the same price
    level", cup duration "from 7 to 65 weeks", handle "1 week minimum with no
    maximum, forming in the upper half of the cup".

      entry          "a close above the right cup rim"
      stop           "the handle low (point C) is a good place to put a stop"
      measure rule   the height from the right cup lip to the lowest valley,
                     x 61%, added to the breakout price
    """
    ps = b.peaks
    if len(ps) < 2:
        return None
    for ri in range(len(ps) - 1, -1, -1):
        right = ps[ri]
        if b.n - 1 - right < p.handle_min_bars:
            continue
        for li in range(ri - 1, -1, -1):
            left = ps[li]
            span = right - left
            if span < p.cup_min_bars:
                continue
            if span > p.cup_max_bars:
                break
            if not near(float(b.h[left]), float(b.h[right]), p.cup_rim_tol_pct):
                continue
            cup_low_i = b.arglowest(left, right)
            cup_low = float(b.l[cup_low_i])
            rim = float(b.h[right])
            depth = rim - cup_low
            if depth <= 0.0:
                continue
            # "U-shaped, not V-shaped": the base must be broad, not a spike
            if valley_width(b, cup_low_i, p.shape_tol_pct) < p.cup_base_min_width:
                continue
            handle_low = b.lowest(right + 1, b.n - 1)
            # "forming in the upper half of the cup"
            if handle_low < cup_low + depth / 2.0:
                continue
            if handle_low >= rim:
                continue
            return Setup("long", rim, handle_low * 0.999,
                         measure_long(rim, rim - cup_low, 61.0),
                         f"cup {cup_low:.4f} rim {rim:.4f} handle {handle_low:.4f}")
    return None


@pattern("inverted_cup_with_handle", "icup.html", "reversal", "short", 62.0)
def _inverted_cup_with_handle(b, p):
    """icup.html — "a smooth, rounded looking turn (an inverted cup)"; "the
    two cup rims should bottom near the same price"; "to the right of the cup
    should be a handle"; "handle must not rise above the cup top but often
    retrace 30% to 60% up the height of the cup".

      confirmation   "the pattern confirms as valid when price closes below
                     the right cup lip"
      entry          short below the right rim low
      measure rule   the handle height subtracted from the right rim low
    """
    vs = b.valleys
    if len(vs) < 2:
        return None
    for ri in range(len(vs) - 1, -1, -1):
        right = vs[ri]
        if b.n - 1 - right < p.handle_min_bars:
            continue
        for li in range(ri - 1, -1, -1):
            left = vs[li]
            span = right - left
            if span < p.cup_min_bars:
                continue
            if span > p.cup_max_bars:
                break
            if not near(float(b.l[left]), float(b.l[right]), p.cup_rim_tol_pct):
                continue
            cup_hi_i = b.arghighest(left, right)
            cup_hi = float(b.h[cup_hi_i])
            rim = float(b.l[right])
            depth = cup_hi - rim
            if depth <= 0.0:
                continue
            if peak_width(b, cup_hi_i, p.shape_tol_pct) < p.cup_base_min_width:
                continue
            handle_high = b.highest(right + 1, b.n - 1)
            # "must not rise above the cup top", "retrace 30% to 60%"
            if handle_high >= cup_hi or handle_high > rim + depth / 2.0:
                continue
            if handle_high <= rim:
                continue
            return Setup("short", rim, handle_high * 1.001,
                         measure_short(rim, handle_high - rim, 62.0),
                         f"inverted cup {cup_hi:.4f} rim {rim:.4f}")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Flags and pennants (flagpole patterns)
# ═══════════════════════════════════════════════════════════════════════════


def _flagpole(b, p, end):
    """"The flagpole which leads to the flag should be unusually steep and
    last several days." Returns (start_index, height) of the steepest
    qualifying run ending at `end`, or None."""
    best = None
    for length in range(p.pole_min_bars, p.pole_max_bars + 1):
        s = end - length
        if s < 0:
            continue
        low = b.lowest(s, end)
        high = b.highest(s, end)
        if low <= 0.0:
            continue
        if pct(high, low) >= p.pole_min_pct:
            if best is None or high - low > best[1]:
                best = (s, high - low)
    return best


@pattern("flag", "flags.html", "continuation", "long", 46.0)
def _flag(b, p):
    """flags.html — "looks like a small rectangle often tilted against the
    prevailing price trend", "price moves between two parallel, or near
    parallel, trendlines", "flags are short, less than 3 weeks long", volume
    trends downward. Breakout is "upward 60% of the time"; 46% meet target.

      entry          at the breakout beyond the trendline
      stop           beyond the opposite trendline of the flag
      measure rule   the flagpole height from trend start (A) to end (B),
                     x 46%, added to the flag bottom (C)
    """
    fl = p.flag_max_bars
    if b.n < fl + p.pole_max_bars + 2:
        return None
    pole = _flagpole(b, p, b.n - 1 - fl)
    if pole is None:
        return None
    top = b.highest(b.n - fl, b.n - 1)
    bot = b.lowest(b.n - fl, b.n - 1)
    if top <= bot:
        return None
    # "a small rectangle": the flag must be a fraction of the pole, and must
    # not have given back the pole's gain
    if (top - bot) > p.flag_max_height * pole[1]:
        return None
    if bot < b.highest(pole[0], b.n - 1 - fl) - pole[1]:
        return None
    if p.require_volume_rules and not volume_recedes(b, b.n - fl, b.n - 1):
        return None
    return Setup("long", top, bot * 0.999, measure_long(bot, pole[1], 46.0),
                 f"flag {bot:.4f}-{top:.4f} on a {pole[1]:.4f} pole")


@pattern("pennant", "pennants.html", "continuation", "long", 35.0)
def _pennant(b, p):
    """pennants.html — "looks like a short symmetrical triangle": "prices
    move between two converging trendlines", "pennants are short, 3 weeks
    long or less", volume "downward trend 86% of the time". Breakout is
    "upward 57% of the time"; 35% of up breakouts meet the target.

      stop           beyond the opposite trendline
      measure rule   the flagpole height, x 35%, added to the pennant bottom
    """
    fl = p.flag_max_bars
    if b.n < fl + p.pole_max_bars + 2:
        return None
    pole = _flagpole(b, p, b.n - 1 - fl)
    if pole is None:
        return None
    s = b.n - fl
    top = b.highest(s, b.n - 1)
    bot = b.lowest(s, b.n - 1)
    if top <= bot:
        return None
    half = fl // 2
    # "converging": the second half of the pennant is narrower than the first
    if (b.highest(s + half, b.n - 1) - b.lowest(s + half, b.n - 1)) >= \
            (b.highest(s, s + half) - b.lowest(s, s + half)):
        return None
    if bot < b.highest(pole[0], b.n - 1 - fl) - pole[1]:
        return None
    if p.require_volume_rules and not volume_recedes(b, s, b.n - 1):
        return None
    return Setup("long", top, bot * 0.999, measure_long(bot, pole[1], 35.0),
                 f"pennant {bot:.4f}-{top:.4f} on a {pole[1]:.4f} pole")


@pattern("high_and_tight_flag", "htf.html", "continuation", "long", 82.0)
def _high_and_tight_flag(b, p):
    """htf.html — "price must rise at least 90% (shoot for a double) in 2
    months or less"; "a consolidation pattern forms after price doubles. It
    usually doesn't look like a flag or pennant, just a pause in the price
    rise." Volume "recedes for best performance".

      confirmation   "price closes above the highest peak in the pattern,
                     which is usually the flagpole top"
      entry          "place buy stop above the highest peak"
      measure rule   half the flagpole height added to the flag's bottom
                     (82% of patterns meet that target)
    """
    win = p.htf_pole_bars
    if b.n < win + 5:
        return None
    low = b.lowest(b.n - 1 - win, b.n - 1)
    high = b.highest(b.n - 1 - win, b.n - 1)
    if low <= 0.0 or pct(high, low) < p.htf_min_rise_pct:
        return None
    hi_i = b.arghighest(b.n - 1 - win, b.n - 1)
    if b.n - 1 - hi_i < p.flag_min_bars:
        return None                        # no pause yet
    bot = b.lowest(hi_i, b.n - 1)
    if bot <= low:
        return None
    if p.require_volume_rules and not volume_recedes(b, hi_i, b.n - 1):
        return None
    trigger = high
    return Setup("long", trigger, bot * 0.999,
                 bot + 0.5 * (high - low),
                 f"high and tight flag +{pct(high, low):.0f}% pole")


# ═══════════════════════════════════════════════════════════════════════════
# Diamonds
# ═══════════════════════════════════════════════════════════════════════════


def _diamond(b, p):
    """"Looks like a diamond, but often tilted to the side": price makes
    "higher peaks and lower valleys initially, then lower peaks and higher
    valleys". Returns (first, mid, top, bot) or None."""
    lo = max(0, b.n - 1 - p.diamond_lookback)
    hi = b.n - 1
    pk = [i for i in b.peaks if lo <= i <= hi]
    vl = [i for i in b.valleys if lo <= i <= hi]
    if len(pk) < 3 or len(vl) < 3:
        return None
    first = min(pk[0], vl[0])
    if hi - first < p.diamond_min_bars:
        return None
    mid = (first + hi) // 2
    w1 = b.highest(first, mid) - b.lowest(first, mid)
    w2 = b.highest(mid, hi) - b.lowest(mid, hi)
    if w1 <= 0.0 or w2 >= w1:
        return None                        # broadening then narrowing
    return first, mid, b.highest(first, hi), b.lowest(first, hi)


@pattern("diamond_bottom", "diamondb.html", "reversal", "long", 73.0)
def _diamond_bottom(b, p):
    """diamondb.html — a diamond with "price trend: downward leading to the
    pattern"; volume "downward trend 67% of the time"; breakout "upward 74%
    of the time, when price closes outside one of the trendline boundaries";
    73% of up breakouts meet the target.

      stop           below the pattern's lowest valley
      measure rule   the height from the highest peak (A) to the lowest
                     valley (B), x 73%, added to the breakout price
    """
    d = _diamond(b, p)
    if d is None:
        return None
    first, mid, top, bot = d
    if not trend_down_into(b, first, p.trend_window, p.min_trend_pct):
        return None
    if p.require_volume_rules and not volume_recedes(b, first, b.n - 1):
        return None
    trigger = b.highest(mid, b.n - 1)
    height = top - bot
    if height <= 0.0 or trigger <= bot:
        return None
    return Setup("long", trigger, bot * 0.999,
                 measure_long(trigger, height, 73.0),
                 f"diamond bottom {bot:.4f}-{top:.4f}")


@pattern("diamond_top", "diamondt.html", "reversal", "short", 63.0)
def _diamond_top(b, p):
    """diamondt.html — a diamond with "price trend: upward leading to the
    pattern"; breakout "downward 54% of the time"; 63% of down breakouts meet
    the target.

      measure rule   highest peak (A) to lowest valley (B), x 63%,
                     subtracted from the breakout price
    """
    d = _diamond(b, p)
    if d is None:
        return None
    first, mid, top, bot = d
    if not trend_up_into(b, first, p.trend_window, p.min_trend_pct):
        return None
    trigger = b.lowest(mid, b.n - 1)
    height = top - bot
    if height <= 0.0 or trigger >= top:
        return None
    return Setup("short", trigger, top * 1.001,
                 measure_short(trigger, height, 63.0),
                 f"diamond top {bot:.4f}-{top:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Bump-and-run reversals
# ═══════════════════════════════════════════════════════════════════════════


@pattern("bump_and_run_reversal_bottom", "barrb.html", "reversal", "long", 76.0)
def _barr_bottom(b, p):
    """barrb.html — "resembles a tilted frying pan with the handle on the
    left".

      trendline      a down-sloping line along the price peaks
      lead-in        "at least one month duration"; the lead-in height is the
                     "widest vertical distance between trendline and low in
                     the first quarter"
      bump           the trendline steepens; the bump height "should be at
                     least twice the lead-in height"
      confirmation   "the pattern confirms when price closes above the
                     down-sloping trendline"
      measure rule   "the highest high in the pattern" is the target
    """
    win = p.barr_lookback
    if b.n < win + 2:
        return None
    lo, hi = b.n - 1 - win, b.n - 1
    lead_end = lo + max(p.barr_leadin_bars, win // 4)
    if lead_end >= hi:
        return None
    pk = [i for i in b.peaks if lo <= i <= lead_end]
    if len(pk) < 2:
        return None
    slope, inter = linfit(pk, [b.h[i] for i in pk])
    if slope >= 0.0:
        return None                        # "down-sloping trendline"
    line = lambda i: slope * i + inter     # noqa: E731 - the trendline itself
    lead_h = max(line(i) - float(b.l[i]) for i in range(lo, lead_end + 1))
    bump_h = max(line(i) - float(b.l[i]) for i in range(lead_end + 1, hi + 1))
    if lead_h <= 0.0 or bump_h < p.barr_bump_mult * lead_h:
        return None
    trigger = line(hi)
    top = b.highest(lo, hi)
    bot = b.lowest(lead_end, hi)
    if not bot < trigger < top:
        return None
    return Setup("long", trigger, bot * 0.999, top,
                 f"BARR bottom bump {bump_h:.4f} vs lead-in {lead_h:.4f}")


@pattern("bump_and_run_reversal_top", "barrt.html", "reversal", "short", 44.0)
def _barr_top(b, p):
    """barrt.html — the mirror: "a trendline connecting the price valleys
    rises upward at 30 to 45 degrees" (the lead-in), then "price rises
    following a steeper trendline (45 to 60 degrees) on high volume" (the
    bump), whose height "should be at least twice the lead-in height".

      confirmation   "pattern confirms as valid when price closes below the
                     30-degree trendline"
      entry          "sell short when price closes below the 30-degree
                     trendline"
      measure rule   "use the bottom of the chart pattern as the target"
    """
    win = p.barr_lookback
    if b.n < win + 2:
        return None
    lo, hi = b.n - 1 - win, b.n - 1
    lead_end = lo + max(p.barr_leadin_bars, win // 4)
    if lead_end >= hi:
        return None
    vl = [i for i in b.valleys if lo <= i <= lead_end]
    if len(vl) < 2:
        return None
    slope, inter = linfit(vl, [b.l[i] for i in vl])
    if slope <= 0.0:
        return None                        # "rises upward at 30 to 45 degrees"
    line = lambda i: slope * i + inter     # noqa: E731
    lead_h = max(float(b.h[i]) - line(i) for i in range(lo, lead_end + 1))
    bump_h = max(float(b.h[i]) - line(i) for i in range(lead_end + 1, hi + 1))
    if lead_h <= 0.0 or bump_h < p.barr_bump_mult * lead_h:
        return None
    trigger = line(hi)
    top = b.highest(lead_end, hi)
    bot = b.lowest(lo, hi)
    if not bot < trigger < top:
        return None
    return Setup("short", trigger, top * 1.001, bot,
                 f"BARR top bump {bump_h:.4f} vs lead-in {lead_h:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Rounding turns
# ═══════════════════════════════════════════════════════════════════════════


def _rounded(b, first, last, want_bowl):
    """"Look for a rounded bowl shape" / "prices form a gentle curve, a half
    moon shape": the extreme sits in the middle third of the span, and each
    outer third is further from the extreme than the middle third."""
    span = last - first
    if span < 9:
        return False
    a, c = first + span // 3, last - span // 3
    if want_bowl:
        mid = b.lowest(a, c)
        return mid < b.lowest(first, a) and mid < b.lowest(c, last)
    mid = b.highest(a, c)
    return mid > b.highest(first, a) and mid > b.highest(c, last)


@pattern("rounding_bottom", "roundb.html", "continuation", "long", 65.0)
def _rounding_bottom(b, p):
    """roundb.html — "look for a rounded bowl shape, usually over many
    months"; "price trends upward to the pattern 67% of the time", so the
    page files it as a continuation.

      confirmation   "use a close above the left peak as confirmation"
      stop           below the pattern's low point
      measure rule   "compute the height from the left saucer lip to the
                     lowest valley", x 65%, added to the right rim price
    """
    ps = b.peaks
    if not ps:
        return None
    for li in range(len(ps) - 1, -1, -1):
        left = ps[li]
        span = b.n - 1 - left
        if span < p.rounding_min_bars:
            continue
        if span > p.rounding_lookback:
            break
        if not _rounded(b, left, b.n - 1, True):
            continue
        lip = float(b.h[left])
        low = b.lowest(left, b.n - 1)
        if low <= 0.0 or lip <= low:
            continue
        return Setup("long", lip, low * 0.999,
                     measure_long(lip, lip - low, 65.0),
                     f"rounding bottom lip {lip:.4f} low {low:.4f}")
    return None


@pattern("rounding_top", "roundingtop.html", "reversal", "long", 58.0)
def _rounding_top(b, p):
    """roundingtop.html — "prices form a gentle curve, a half moon shape";
    "the rims of the inverted bowl bottom near the same price". The engine
    arms the upward break ("close above the highest high"), which the page
    ranks 2 of 39; 58% of those meet the target.

      measure rule   the height from the highest peak (A) to the right rim
                     low (B), x 58%, added to the highest peak
    """
    vs = b.valleys
    if not vs:
        return None
    for li in range(len(vs) - 1, -1, -1):
        left = vs[li]
        span = b.n - 1 - left
        if span < p.rounding_min_bars:
            continue
        if span > p.rounding_lookback:
            break
        if not _rounded(b, left, b.n - 1, False):
            continue
        rim_l = float(b.l[left])
        rim_r = b.lowest(b.n - 1 - p.rounding_min_bars // 2, b.n - 1)
        if not near(rim_l, rim_r, p.cup_rim_tol_pct):
            continue
        top = b.highest(left, b.n - 1)
        if top <= rim_r:
            continue
        if not trend_up_into(b, left, p.trend_window, p.min_trend_pct):
            continue
        return Setup("long", top, rim_r * 0.999,
                     measure_long(top, top - rim_r, 58.0),
                     f"rounding top {top:.4f} right rim {rim_r:.4f}")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Pipes and horns — the two- and three-bar spike patterns
# ═══════════════════════════════════════════════════════════════════════════


def _overlap(lo1, hi1, lo2, hi2):
    """Fraction of the combined range the two bars share — "the 2 weeks often
    have a large price overlap"."""
    inner = min(hi1, hi2) - max(lo1, lo2)
    outer = max(hi1, hi2) - min(lo1, lo2)
    if outer <= 0.0:
        return 0.0
    return max(0.0, inner) / outer


@pattern("pipe_bottom", "pipeb.html", "reversal", "long", 77.0)
def _pipe_bottom(b, p):
    """pipeb.html — "twin and adjacent downward spikes" forming two price
    bars, which "often have a large price overlap but need not bottom at the
    same price"; "most pipes display above-average volume on one or both
    spikes"; "the pipe should stand-alone and be obvious on the chart", and
    "price trend: usually downward leading to the pattern".

      confirmation   "the pattern confirms when price closes above the
                     highest high in the pattern"
      entry          "buy when price closes above the higher of the two
                     spikes"
      stop           "if price closes below the lower of the two spikes,
                     then close out your position"
      measure rule   the pattern height x 77%, added to the higher spike
    """
    if b.n < p.pipe_lookback + 3:
        return None
    i, j = b.n - 2, b.n - 1
    lo1, hi1 = float(b.l[i]), float(b.h[i])
    lo2, hi2 = float(b.l[j]), float(b.h[j])
    if _overlap(lo1, hi1, lo2, hi2) < p.pipe_overlap:
        return None
    # "the spikes should be longer than most in the past year": both bars must
    # undercut the surrounding landscape
    prior_low = b.lowest(i - p.pipe_lookback, i - 1)
    if min(lo1, lo2) > prior_low:
        return None
    if p.require_volume_rules:
        avg = sma(b.v[:i], 20)
        if avg is None or max(float(b.v[i]), float(b.v[j])) < avg:
            return None
    top = max(hi1, hi2)
    bot = min(lo1, lo2)
    if top <= bot:
        return None
    return Setup("long", top, bot * 0.999,
                 measure_long(top, top - bot, 77.0),
                 f"pipe bottom {bot:.4f}-{top:.4f}")


@pattern("pipe_top", "pipet.html", "reversal", "short", 54.0)
def _pipe_top(b, p):
    """pipet.html — "twin and adjacent upward spikes" that "should be longer
    than most past year spikes, towering over surrounding landscape", with a
    "large price overlap" and "small variation between tops"; "the right
    spike shows lower volume than the left spike"; "price trend: usually
    upward leading to the pattern".

      confirmation   "pattern confirms when price closes below the lowest
                     price in the pattern"
      measure rule   the pattern height x 54%, subtracted from the lowest
                     price
    """
    if b.n < p.pipe_lookback + 3:
        return None
    i, j = b.n - 2, b.n - 1
    lo1, hi1 = float(b.l[i]), float(b.h[i])
    lo2, hi2 = float(b.l[j]), float(b.h[j])
    if _overlap(lo1, hi1, lo2, hi2) < p.pipe_overlap:
        return None
    prior_high = b.highest(i - p.pipe_lookback, i - 1)
    if max(hi1, hi2) < prior_high:
        return None
    if not near(hi1, hi2, p.top_tol_pct):
        return None
    if p.require_volume_rules and float(b.v[j]) >= float(b.v[i]):
        return None                        # "right spike shows lower volume"
    top = max(hi1, hi2)
    bot = min(lo1, lo2)
    if top <= bot:
        return None
    return Setup("short", bot, top * 1.001,
                 measure_short(bot, top - bot, 54.0),
                 f"pipe top {bot:.4f}-{top:.4f}")


@pattern("horn_bottom", "hornb.html", "reversal", "long", 74.0)
def _horn_bottom(b, p):
    """hornb.html — "looks like an inverted steer's horn, two parallel price
    spikes separated by a week", the spikes "plummeting below the surrounding
    price landscape, including the middle week"; "price trend: downward
    leading to the pattern".

      confirmation   "the pattern confirms as valid when price closes above
                     the highest price in the 3-week pattern"
      entry          "place a buy stop a penny above the top of the 3-week
                     pattern"
      stop           "a penny below the lower of the 3-bar horn pattern"
      measure rule   the 3-bar height x 74%, added to the highest high
    """
    if b.n < p.pipe_lookback + 4:
        return None
    a, m, z = b.n - 3, b.n - 2, b.n - 1
    if not (b.l[a] < b.l[m] and b.l[z] < b.l[m]):
        return None                        # spikes straddle the middle bar
    prior_low = b.lowest(a - p.pipe_lookback, a - 1)
    if min(float(b.l[a]), float(b.l[z])) > prior_low:
        return None
    if not trend_down_into(b, a, p.trend_window, p.min_trend_pct):
        return None
    top = b.highest(a, z)
    bot = b.lowest(a, z)
    if top <= bot:
        return None
    return Setup("long", top, bot * 0.999,
                 measure_long(top, top - bot, 74.0),
                 f"horn bottom {bot:.4f}-{top:.4f}")


@pattern("horn_top", "hornt.html", "reversal", "short", 54.0)
def _horn_top(b, p):
    """hornt.html — "looks like a steer's horn, two parallel price spikes
    separated by a week"; "the spikes should be longer than most in the past
    year"; "price trend: upward leading to the pattern".

      confirmation   "the pattern confirms as valid when price closes below
                     the lowest price in the 3-week pattern"
      stop           above the highest high in the 3-week pattern
      measure rule   the 3-bar height x 54%, subtracted from the lowest low
    """
    if b.n < p.pipe_lookback + 4:
        return None
    a, m, z = b.n - 3, b.n - 2, b.n - 1
    if not (b.h[a] > b.h[m] and b.h[z] > b.h[m]):
        return None
    prior_high = b.highest(a - p.pipe_lookback, a - 1)
    if max(float(b.h[a]), float(b.h[z])) < prior_high:
        return None
    if not trend_up_into(b, a, p.trend_window, p.min_trend_pct):
        return None
    top = b.highest(a, z)
    bot = b.lowest(a, z)
    if top <= bot:
        return None
    return Setup("short", bot, top * 1.001,
                 measure_short(bot, top - bot, 54.0),
                 f"horn top {bot:.4f}-{top:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Islands
# ═══════════════════════════════════════════════════════════════════════════


def _island(b, p, up_exit, max_bars, aligned):
    """islandrev.html / longisland.html — "gaps separate a price island from
    the mainland". The last bar must be the second gap (the breakout gap);
    the island is everything between the two gaps.

    `aligned` True demands the island reversal's "two gaps must share some or
    all of the same price"; False demands the long island's "two gaps ... do
    not share the same price". Returns (island_first, island_last) or None.
    """
    z = b.n - 1
    if up_exit:
        if float(b.l[z]) <= float(b.h[z - 1]):
            return None                    # no gap up out of the island
        g2_lo, g2_hi = float(b.h[z - 1]), float(b.l[z])
    else:
        if float(b.h[z]) >= float(b.l[z - 1]):
            return None
        g2_lo, g2_hi = float(b.h[z]), float(b.l[z - 1])
    if pct(g2_hi, g2_lo) < p.island_min_gap_pct:
        return None
    for k in range(z - 1, max(0, z - max_bars) - 1, -1):
        if k == 0:
            break
        if up_exit:                        # entered by gapping DOWN
            if float(b.h[k]) >= float(b.l[k - 1]):
                continue
            g1_lo, g1_hi = float(b.h[k]), float(b.l[k - 1])
        else:                              # entered by gapping UP
            if float(b.l[k]) <= float(b.h[k - 1]):
                continue
            g1_lo, g1_hi = float(b.h[k - 1]), float(b.l[k])
        if pct(g1_hi, g1_lo) < p.island_min_gap_pct:
            continue
        shares = min(g1_hi, g2_hi) > max(g1_lo, g2_lo)
        if shares != aligned:
            continue
        return k, z - 1
    return None


@pattern("island_bottom", "islandrev.html", "reversal", "long", 82.0)
def _island_bottom(b, p):
    """islandrev.html — "gaps separate a price island from the mainland" and
    "two gaps must share some or all of the same price"; "bottoms have price
    trending downward" into the island; volume is "high on the day price
    makes the second gap"; "the island can be one day to several months
    long".

      stop           beyond the island boundary opposite the breakout
      measure rule   the island height (peak A to valley B) x 82%, added to
                     the peak
    """
    isl = _island(b, p, True, p.island_max_bars, True)
    if isl is None:
        return None
    first, last = isl
    if not trend_down_into(b, first, p.trend_window, p.min_trend_pct):
        return None
    if p.require_volume_rules:
        avg = sma(b.v[:b.n - 1], 20)
        if avg is None or float(b.v[-1]) < avg:
            return None
    top = b.highest(first, last)
    bot = b.lowest(first, last)
    if top <= bot:
        return None
    trigger = max(top, float(b.h[-1]))
    return Setup("long", trigger, bot * 0.999,
                 measure_long(top, top - bot, 82.0),
                 f"island bottom {bot:.4f}-{top:.4f}")


@pattern("island_top", "islandrev.html", "reversal", "short", 62.0)
def _island_top(b, p):
    """islandrev.html — the mirror: "tops have price trending upward to the
    island", the island is entered by a gap up and left by a gap down, and
    the two gaps share price.

      measure rule   the island height x 62%, subtracted from the valley
    """
    isl = _island(b, p, False, p.island_max_bars, True)
    if isl is None:
        return None
    first, last = isl
    if not trend_up_into(b, first, p.trend_window, p.min_trend_pct):
        return None
    if p.require_volume_rules:
        avg = sma(b.v[:b.n - 1], 20)
        if avg is None or float(b.v[-1]) < avg:
            return None
    top = b.highest(first, last)
    bot = b.lowest(first, last)
    if top <= bot:
        return None
    trigger = min(bot, float(b.l[-1]))
    return Setup("short", trigger, top * 1.001,
                 measure_short(bot, top - bot, 62.0),
                 f"island top {bot:.4f}-{top:.4f}")


@pattern("long_island_bullish", "longisland.html", "continuation", "long", 71.0)
def _long_island_bullish(b, p):
    """longisland.html — "non-aligned gaps separate a price island from the
    mainland": "the two gaps that setoff the long island do not share the
    same price". "Look for gaps at least $1 wide" and "islands shorter than 4
    months". "The day after the second gap is the breakout day"; 71% of up
    breakouts meet the target.

      measure rule   the island height (highest peak to lowest valley)
                     divided by two, added to the prior day's close
    """
    isl = _island(b, p, True, p.long_island_max_bars, False)
    if isl is None:
        return None
    first, last = isl
    top = b.highest(first, last)
    bot = b.lowest(first, last)
    if top <= bot:
        return None
    trigger = float(b.h[-1])
    return Setup("long", trigger, bot * 0.999,
                 float(b.c[-1]) + (top - bot) / 2.0,
                 f"long island {bot:.4f}-{top:.4f}")


@pattern("long_island_bearish", "longisland.html", "continuation", "short", 55.0)
def _long_island_bearish(b, p):
    """longisland.html — the same non-aligned island left by a gap DOWN;
    55% of down breakouts meet the target. Measure rule: half the island
    height subtracted from the prior day's close."""
    isl = _island(b, p, False, p.long_island_max_bars, False)
    if isl is None:
        return None
    first, last = isl
    top = b.highest(first, last)
    bot = b.lowest(first, last)
    if top <= bot:
        return None
    trigger = float(b.l[-1])
    return Setup("short", trigger, top * 1.001,
                 float(b.c[-1]) - (top - bot) / 2.0,
                 f"long island {bot:.4f}-{top:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Small patterns (SmallPatterns.html)
#
# Bulkowski's short bar patterns all share one set of trading rules, which he
# states on every page: "a buy stop placed a penny above the top of the chart
# pattern (above the highest price bar in the pattern)", "a stop loss order
# placed a penny below the lowest price bar in the pattern", and "a limit
# order to sell at [N] times the height of the pattern ... added to the price
# of the highest bar". `_small` is exactly that, so each detector below only
# has to recognise its bars.
# ═══════════════════════════════════════════════════════════════════════════


def _small(b, first, last, side, mult, note, top=None, bot=None):
    """The shared small-pattern bracket. `mult` is the page's height
    multiple: 2.0 for "twice the height of the pattern added to the top of
    it", 1.0 where the page adds the height once."""
    top = b.highest(first, last) if top is None else top
    bot = b.lowest(first, last) if bot is None else bot
    height = top - bot
    if height <= 0.0:
        return None
    if side == "long":
        return Setup("long", top, bot * 0.999, top + mult * height, note)
    return Setup("short", bot, top * 1.001, bot - mult * height, note)


def avg_bar_height(b, end, n):
    """"The average price bar [height] measured one month (22 price bars)
    before the start of the pattern"."""
    s = end - n
    if s < 0:
        return None
    return float(np.mean(b.h[s:end] - b.l[s:end]))


def _body(b, i):
    return abs(float(b.c[i]) - float(b.o[i]))


def _upper_shadow(b, i):
    return float(b.h[i]) - max(float(b.c[i]), float(b.o[i]))


def _lower_shadow(b, i):
    return min(float(b.c[i]), float(b.o[i])) - float(b.l[i])


@pattern("bullish_2_close_reversal", "2Closebull.html", "reversal", "long", 38.0)
def _two_close_bull(b, p):
    """2Closebull.html — "this pattern is three price bars long".

      bar 1          any price bar
      bar 2          "price makes a lower low (below bar 1), and closes
                     below bar 1's close"
      bar 3          "price posts a lower low (below bar 2's low), but
                     closes above bar 1 and 2's close"
      breakout       "breaks out upward 71% of the time"
      target         "twice the height of the pattern ... added to the price
                     of the highest bar" (hit 38% of the time)
    """
    if b.n < 4:
        return None
    a, m, z = b.n - 3, b.n - 2, b.n - 1
    if not (b.l[m] < b.l[a] and b.c[m] < b.c[a]):
        return None
    if not (b.l[z] < b.l[m] and b.c[z] > b.c[a] and b.c[z] > b.c[m]):
        return None
    return _small(b, a, z, "long", 2.0, "bullish 2-close reversal")


@pattern("bearish_2_close_reversal", "2Closebear.html", "reversal", "long", 38.0)
def _two_close_bear(b, p):
    """2Closebear.html — the mirror image of the bullish 2-close:

      bar 2          "price makes a higher high (above bar 1), and closes
                     above bar 1's close"
      bar 3          "price posts a higher high (above bar 2's high), but
                     closes below bar 1's close"

    The page reports the pattern "breaks out downward 65% of the time in
    stocks", but the tactics it publishes and tested are the long-side ones
    ("a buy stop ... one penny above the pattern's highest price bar", stop
    below the lowest bar, target twice the height above the top) and it
    explicitly advises against the short, "as downward breakouts resulted in
    consistent losses". The engine therefore arms the page's tested long.
    """
    if b.n < 4:
        return None
    a, m, z = b.n - 3, b.n - 2, b.n - 1
    if not (b.h[m] > b.h[a] and b.c[m] > b.c[a]):
        return None
    if not (b.h[z] > b.h[m] and b.c[z] < b.c[a]):
        return None
    return _small(b, a, z, "long", 2.0, "bearish 2-close reversal")


@pattern("two_dance", "2Dance.html", "reversal", "long", 100.0)
def _two_dance(b, p):
    """2Dance.html — "two adjacent price bars where each must have a shadow
    at least 3x the body height" and "the longer shadow must be at least 2x
    the length of the shorter shadow"; four-price dojis are excluded. The
    best form has "a tall downward shadow on the first price bar, and a tall
    upward shadow for the second price bar".

      entry          "buy stop a penny above the top of the tallest price
                     bar in the dance pattern"
      stop           "a penny below the lowest price bar in the pattern"
      target         "twice the height of the pattern added to the top of it"
    """
    if b.n < 3:
        return None
    a, z = b.n - 2, b.n - 1
    shadows = []
    for i in (a, z):
        if b.h[i] == b.l[i]:
            return None                    # four-price doji
        sh = max(_upper_shadow(b, i), _lower_shadow(b, i))
        if sh < p.dance_shadow_body_mult * _body(b, i):
            return None
        shadows.append(sh)
    lo, hi = min(shadows), max(shadows)
    if lo <= 0.0 or hi < p.dance_shadow_ratio * lo:
        return None
    return _small(b, a, z, "long", 2.0, "2-dance")


@pattern("two_did", "2Did.html", "reversal", "long", 100.0)
def _two_did(b, p):
    """2Did.html — two consecutive price bars "more than 1.5 times the height
    of the average price bar measured one month (22 price bars) before the
    start of the 2-did pattern", where "the height of the second bar in the
    pair must be inside the prior day's high-low range. No ties allowed."

      entry          "a buy stop a penny above the top of the FIRST price
                     bar in the pattern"
      stop           "a penny below the bottom of the first price bar"
      target         "the height of the 2-did pair added to the top of it"
    """
    if b.n < p.tall_avg_bars + 3:
        return None
    a, z = b.n - 2, b.n - 1
    avg = avg_bar_height(b, a, p.tall_avg_bars)
    if avg is None or avg <= 0.0:
        return None
    if float(b.h[a] - b.l[a]) <= p.tall_bar_mult * avg:
        return None
    if float(b.h[z] - b.l[z]) <= p.tall_bar_mult * avg:
        return None
    if not (b.h[z] < b.h[a] and b.l[z] > b.l[a]):
        return None                        # "inside ... no ties allowed"
    pair_height = b.highest(a, z) - b.lowest(a, z)
    return Setup("long", float(b.h[a]), float(b.l[a]) * 0.999,
                 b.highest(a, z) + pair_height, "2-did")


@pattern("two_tall", "TallDance.html", "reversal", "long", 100.0)
def _two_tall(b, p):
    """TallDance.html — "two tall (at least 1.5 times the 1-month average
    price bar height, but for the best performance, look for twice the
    average) and adjacent price bars".

      entry          "a buy stop a penny above the top of the higher of the
                     two price bars"
      stop           "a stop loss order a penny below the bottom of the
                     lowest of the two"
      target         "twice the height of the 2-tall pattern added to the
                     top of it"; "discard trades where the target exceeds
                     20% away, as these are considered unrealistic gains"
    """
    if b.n < p.tall_avg_bars + 3:
        return None
    a, z = b.n - 2, b.n - 1
    avg = avg_bar_height(b, a, p.tall_avg_bars)
    if avg is None or avg <= 0.0:
        return None
    if min(float(b.h[a] - b.l[a]), float(b.h[z] - b.l[z])) < p.tall_bar_mult * avg:
        return None
    st = _small(b, a, z, "long", 2.0, "2-tall")
    if st is None or pct(st.target, st.trigger) > p.tall_max_target_pct:
        return None
    return st


@pattern("three_bar", "3Bar.html", "reversal", "long", 100.0)
def _three_bar(b, p):
    """3Bar.html — a three-bar setup that "breaks above 85% of the time":

      bar 1          "price closes lower than the prior day's close"
      bar 2          "has a low below the two adjacent price bars"
      bar 3          "closes above the highs of the other price bars"

    Buy stop "a penny above the top of the pattern", stop "a penny below the
    bottom of the pattern", target "twice the height of the pattern added to
    the top of it".
    """
    if b.n < 5:
        return None
    a, m, z = b.n - 3, b.n - 2, b.n - 1
    if b.c[a] >= b.c[a - 1]:
        return None
    if not (b.l[m] < b.l[a] and b.l[m] < b.l[z]):
        return None
    if not (b.c[z] > b.h[a] and b.c[z] > b.h[m]):
        return None
    return _small(b, a, z, "long", 2.0, "3-bar")


@pattern("three_day_compression", "3DC.html", "continuation", "long", 100.0)
def _three_day_compression(b, p):
    """3DC.html — "add the height of the first two bars of the pattern. The
    third bar must be less than one-third of the total." Where the three bars
    sit relative to each other "is irrelevant to identification".

    Buy stop a penny above the highest point of the three bars, stop a penny
    below the lowest, target twice the pattern height added to the top.
    """
    if b.n < 4:
        return None
    a, m, z = b.n - 3, b.n - 2, b.n - 1
    total = float(b.h[a] - b.l[a]) + float(b.h[m] - b.l[m])
    if total <= 0.0 or float(b.h[z] - b.l[z]) >= total / 3.0:
        return None
    return _small(b, a, z, "long", 2.0, "3-day compression")


@pattern("inside_day", "InsideDays.html", "continuation", "long", 100.0)
def _inside_day(b, p):
    """InsideDays.html — "look for a lower high and higher low on the second
    day. The price bar fits inside the prior day's range"; "the last bar
    cannot have the high price equal to the low price". There is "no
    requirement of a price trend leading to the inside day", but "since
    inside days act as a continuation pattern, expect the breakout to be in
    the same direction as the inbound price trend" — the engine arms the
    upward break.

      stop           "a penny below the bottom of the pattern"
      target         twice the pattern's height added to the top
    """
    if b.n < 3:
        return None
    a, z = b.n - 2, b.n - 1
    if not (b.h[z] < b.h[a] and b.l[z] > b.l[a]):
        return None
    if b.h[z] == b.l[z]:
        return None
    return _small(b, a, z, "long", 2.0, "inside day")


@pattern("outside_day", "OutsideDays.html", "continuation", "long", 100.0)
def _outside_day(b, p):
    """OutsideDays.html — "look for a higher high and lower low on the second
    day. The price bar fits outside the prior day's range"; "the first bar
    cannot have the high price equal to the low price".

      breakout       "when the stock closes either above the top of the
                     pattern or below the bottom of it"
      stop           "a penny beyond the opposite extreme of the outside day"
      target         "the height of the chart pattern added to the top of
                     the pattern"
    """
    if b.n < 3:
        return None
    a, z = b.n - 2, b.n - 1
    if not (b.h[z] > b.h[a] and b.l[z] < b.l[a]):
        return None
    if b.h[a] == b.l[a]:
        return None
    return _small(b, a, z, "long", 1.0, "outside day")


@pattern("narrow_range_7", "nr7.html", "continuation", "long", 43.0)
def _nr7(b, p):
    """nr7.html — "the most recent bar must have a smaller high-low price
    range than the prior six bars (seven bars, total)".

      breakout       a close beyond the pattern boundary; "buy at the open
                     the next day" — mechanized as a resting buy stop at the
                     pattern top
      stop           "a penny below the bottom of the pattern"
      target         the pattern height added to the highest price (met 43%
                     of the time)
    """
    if b.n < 8:
        return None
    rng = b.h[b.n - 7:b.n] - b.l[b.n - 7:b.n]
    if float(rng[-1]) >= float(np.min(rng[:-1])):
        return None
    return _small(b, b.n - 7, b.n - 1, "long", 1.0, "NR7")


# ═══════════════════════════════════════════════════════════════════════════
# Reversal bars — the one- and two-bar reversals
# ═══════════════════════════════════════════════════════════════════════════


def short_trend(b, end, n):
    """KRB2.html: "I check the slope of a line found using 5-day linear
    regression on the high-low price range to determine the trend." Returns
    the slope of that regression over the `n` bars ending at `end`."""
    s = end - n + 1
    if s < 0:
        return 0.0
    mid = (b.h[s:end + 1] + b.l[s:end + 1]) / 2.0
    slope, _ = linfit(np.arange(len(mid)), mid)
    return slope


def _in_range(b, i, price, frac, from_high):
    """"The open must be within 25% of the intraday high" — `price` sits in
    the top (or bottom) `frac` of bar `i`'s high-low range."""
    hi, lo = float(b.h[i]), float(b.l[i])
    span = hi - lo
    if span <= 0.0:
        return False
    return ((hi - price) <= frac * span) if from_high else ((price - lo) <= frac * span)


@pattern("narrow_range_4", "NR4.html", "continuation", "long", 50.0)
def _nr4(b, p):
    """NR4.html — "the pattern is composed of four bars" and "the most recent
    bar must have a smaller high-low price range than the prior three bars
    (four bars, total)". "A breakout occurs when price closes above the top
    or below the bottom of the NR4."

      stop           "a penny below the bottom of the pattern"
      target         "measure the height of pattern and add it to the highest
                     price in the pattern to get an upward target"
    """
    if b.n < 5:
        return None
    rng = b.h[b.n - 4:b.n] - b.l[b.n - 4:b.n]
    if float(rng[-1]) >= float(np.min(rng[:-1])):
        return None
    return _small(b, b.n - 4, b.n - 1, "long", 1.0, "NR4")


@pattern("key_reversal_uptrend", "KRU.html", "reversal", "long", 71.0)
def _key_reversal_up(b, p):
    """KRU.html — "the pattern is composed of two bars"; "look for the
    pattern in a short-term uptrend"; "the pattern forms an outside day.
    However, in this case, look for an open above the prior close, a close
    below the prior low, and a high above the prior day's high."

    The page notes that "51% of these formations actually continue upward
    rather than reverse the trend", and the tactics it publishes are the
    long-side ones, so that is the side armed.

      stop           "a penny below the pattern's lowest point"
      target         "multiply this height by two and add to the top of the
                     pattern"
    """
    if b.n < p.trend_bars + 3:
        return None
    a, z = b.n - 2, b.n - 1
    if short_trend(b, a, p.trend_bars) <= 0.0:
        return None
    if not (b.o[z] > b.c[a] and b.c[z] < b.l[a] and b.h[z] > b.h[a]):
        return None
    return _small(b, a, z, "long", 2.0, "key reversal (uptrend)")


@pattern("key_reversal_downtrend", "KRD.html", "reversal", "long", 69.0)
def _key_reversal_down(b, p):
    """KRD.html — "2 bars", "pattern appears in short-term downtrends", and
    an outside day where the "current day closes above prior high, opens
    below prior close, and reaches below prior low".

      entry          "trade in the breakout direction ... a breakout occurs
                     when price closes either above the top or below the
                     bottom of the pattern"
      stop           "a penny below the bottom of the pattern"
      target         "twice as high as the height of the key reversal
                     pattern added to the price of the top of the key"
    """
    if b.n < p.trend_bars + 3:
        return None
    a, z = b.n - 2, b.n - 1
    if short_trend(b, a, p.trend_bars) >= 0.0:
        return None
    if not (b.c[z] > b.h[a] and b.o[z] < b.c[a] and b.l[z] < b.l[a]):
        return None
    return _small(b, a, z, "long", 2.0, "key reversal (downtrend)")


@pattern("key_reversal_bar_bullish_v2", "KRB2.html", "reversal", "long", 100.0)
def _key_reversal_bar_bull(b, p):
    """KRB2.html — a one-bar pattern that "is supposed to act as a reversal
    of the short-term downtrend": "price opens much lower than the prior
    close (more than half the average 1-month bar height lower) but closes
    near or above the prior close", and "the bar in the pattern is at least
    50% taller than the one-month average price bar height".

      entry          "a buy stop a penny above the price bar in the pattern"
      stop           "a penny below the bottom of the pattern"
      target         "twice the height of the reversal bar added to the
                     price of the top of it"
    """
    if b.n < p.tall_avg_bars + p.trend_bars + 2:
        return None
    z = b.n - 1
    avg = avg_bar_height(b, z, p.tall_avg_bars)
    if avg is None or avg <= 0.0:
        return None
    if short_trend(b, z - 1, p.trend_bars) >= 0.0:
        return None
    if float(b.c[z - 1]) - float(b.o[z]) <= 0.5 * avg:
        return None
    if float(b.c[z]) < float(b.c[z - 1]) - p.krb_close_tol * avg:
        return None
    if float(b.h[z] - b.l[z]) < (1.0 + p.krb_tall_mult) * avg:
        return None
    return _small(b, z, z, "long", 2.0, "bullish key reversal bar v2")


@pattern("key_reversal_bar_bearish_v2", "KRB2Bear.html", "reversal", "long", 100.0)
def _key_reversal_bar_bear(b, p):
    """KRB2Bear.html — the mirror one-bar pattern, a "reversal of [the]
    short-term uptrend": it "opens significantly higher than [the] prior
    close (>50% of 1-month average bar height), closes near or below [the]
    prior close (within 15% of 1-month average bar height)", and is "at least
    50% taller than [the] one-month average bar height".

    As on the bullish page, the tested tactics are long-side: "buy stop
    placed one penny above the pattern's top price bar", stop "one penny
    below the pattern's bottom", target "twice the height of the reversal bar
    added to the pattern's high price".
    """
    if b.n < p.tall_avg_bars + p.trend_bars + 2:
        return None
    z = b.n - 1
    avg = avg_bar_height(b, z, p.tall_avg_bars)
    if avg is None or avg <= 0.0:
        return None
    if short_trend(b, z - 1, p.trend_bars) <= 0.0:
        return None
    if float(b.o[z]) - float(b.c[z - 1]) <= 0.5 * avg:
        return None
    if float(b.c[z]) > float(b.c[z - 1]) + p.krb_close_tol * avg:
        return None
    if float(b.h[z] - b.l[z]) < (1.0 + p.krb_tall_mult) * avg:
        return None
    return _small(b, z, z, "long", 2.0, "bearish key reversal bar v2")


@pattern("closing_price_reversal_uptrend", "CPRU.html", "reversal", "short", 64.0)
def _cpr_up(b, p):
    """CPRU.html — "the pattern is composed of one bar, but it uses the
    closing price of the prior bar"; "look for the pattern in a short-term
    uptrend"; "the open must be within 25% of the intraday high"; "the close
    must be within 25% of the intraday low and be below the prior day's
    close".

      entry          short "at the open the next day after the pattern"
      stop           "a penny above the top of the closing price reversal
                     pattern"
      target         "measure the height of the pattern and subtract it from
                     the low price" (met 64% of the time in bull markets)
    """
    if b.n < p.trend_bars + 2:
        return None
    z = b.n - 1
    if short_trend(b, z - 1, p.trend_bars) <= 0.0:
        return None
    if not _in_range(b, z, float(b.o[z]), p.quarter, True):
        return None
    if not _in_range(b, z, float(b.c[z]), p.quarter, False):
        return None
    if float(b.c[z]) >= float(b.c[z - 1]):
        return None
    return _small(b, z, z, "short", 1.0, "closing price reversal (uptrend)")


@pattern("closing_price_reversal_downtrend", "CPRD.html", "reversal", "long", 72.0)
def _cpr_down(b, p):
    """CPRD.html — the mirror: "look for the pattern in a short-term
    downtrend"; "the open must be within 25% of the intraday low"; "the close
    must be within 25% of the intraday high and be above the prior day's
    close".

      entry          "buy at the open the next day after the pattern"
      stop           "a penny below the bottom of the closing price reversal
                     pattern"
      target         "measure from the highest high to the lowest low in the
                     pattern to get the height. Add the height to the highest
                     high" (met 72% of the time)
    """
    if b.n < p.trend_bars + 2:
        return None
    z = b.n - 1
    if short_trend(b, z - 1, p.trend_bars) >= 0.0:
        return None
    if not _in_range(b, z, float(b.o[z]), p.quarter, False):
        return None
    if not _in_range(b, z, float(b.c[z]), p.quarter, True):
        return None
    if float(b.c[z]) <= float(b.c[z - 1]):
        return None
    return _small(b, z, z, "long", 1.0, "closing price reversal (downtrend)")


@pattern("hook_reversal_uptrend", "HRU.html", "reversal", "short", 63.0)
def _hook_reversal_up(b, p):
    """HRU.html — "the pattern is composed of two bars"; "look for the
    pattern in a short-term uptrend"; "the pattern forms an inside day"; "the
    last bar of the pattern has an open within 25% of the intraday high" and
    "the last bar's close must be within 25% of the intraday low"; "the last
    bar's high and low cannot be the same".

    The page's headline statistic is for downward breakouts (the measure rule
    "succeeds 63% of the time in bull markets with downward breakouts"), so
    the engine arms the short.

      stop           "above the top" for downward trades
      target         the pattern height subtracted from its bottom
    """
    if b.n < p.trend_bars + 3:
        return None
    a, z = b.n - 2, b.n - 1
    if short_trend(b, a, p.trend_bars) <= 0.0:
        return None
    if not (b.h[z] < b.h[a] and b.l[z] > b.l[a]):
        return None
    if b.h[z] == b.l[z]:
        return None
    if not _in_range(b, z, float(b.o[z]), p.quarter, True):
        return None
    if not _in_range(b, z, float(b.c[z]), p.quarter, False):
        return None
    return _small(b, a, z, "short", 1.0, "hook reversal (uptrend)")


@pattern("hook_reversal_downtrend", "HRD.html", "reversal", "long", 69.0)
def _hook_reversal_down(b, p):
    """HRD.html — "2 bars" in "a short-term downtrend" forming an inside day
    where "the first bar makes a higher high and lower low than the second
    bar"; the "last bar's open [is] within 25% of the intraday low" and its
    "close must be within 25% of the intraday high"; "the last bar's high and
    low cannot be the same".

      stop           "one penny below the pattern's bottom"
      target         the pattern height added twice to the top (met 69% of
                     the time in bull markets with upward breakouts)
    """
    if b.n < p.trend_bars + 3:
        return None
    a, z = b.n - 2, b.n - 1
    if short_trend(b, a, p.trend_bars) >= 0.0:
        return None
    if not (b.h[a] > b.h[z] and b.l[a] < b.l[z]):
        return None
    if b.h[z] == b.l[z]:
        return None
    if not _in_range(b, z, float(b.o[z]), p.quarter, False):
        return None
    if not _in_range(b, z, float(b.c[z]), p.quarter, True):
        return None
    return _small(b, a, z, "long", 2.0, "hook reversal (downtrend)")


@pattern("one_day_reversal_bottom", "ODRB.html", "reversal", "long", 73.0)
def _one_day_reversal_bottom(b, p):
    """ODRB.html — a three-bar pattern in "a short-term downtrend": "the open
    and close on the one-day reversal must be within 25% of the intraday
    high"; "the low price of the two adjacent bars must be above the mid
    point of the one-day reversal"; the bar is "at least as tall as the
    one-month average bar height".

      entry          "a penny above the top of the middle price bar"
      stop           "a penny below the bottom of the ODR day"
      target         twice the height of the middle bar added to its top
                     (met 73% of the time in bull markets)
    """
    if b.n < p.tall_avg_bars + p.trend_bars + 3:
        return None
    a, m, z = b.n - 3, b.n - 2, b.n - 1
    avg = avg_bar_height(b, a, p.tall_avg_bars)
    if avg is None or avg <= 0.0 or float(b.h[m] - b.l[m]) < avg:
        return None
    if short_trend(b, a, p.trend_bars) >= 0.0:
        return None
    if not _in_range(b, m, float(b.o[m]), p.quarter, True):
        return None
    if not _in_range(b, m, float(b.c[m]), p.quarter, True):
        return None
    midpoint = (float(b.h[m]) + float(b.l[m])) / 2.0
    if not (float(b.l[a]) > midpoint and float(b.l[z]) > midpoint):
        return None
    return _small(b, m, m, "long", 2.0, "one-day reversal bottom")


@pattern("one_day_reversal_top", "ODRT.html", "reversal", "short", 67.0)
def _one_day_reversal_top(b, p):
    """ODRT.html — "the pattern is composed of one bar, but for
    identification, I use three bars, one day before to one day after the
    one-day reversal." Look for it "in a short-term uptrend; trade only
    downward breakouts". "The open and close on the one-day reversal must be
    within 25% of the intraday low"; "adjacent bar highs must remain below
    the midpoint of the reversal bar"; the bar is "at least as tall as the
    one-month average bar height".

      entry          "once price closes below the bottom of the pattern,
                     sell short at the open the next day"
      stop           "a stop loss 7% above the short entry price" — the only
                     stop this page gives, so it is used instead of the
                     pattern-boundary stop the sibling pages use
      target         "measure the height of the pattern and subtract it from
                     the low price" (met 67% of the time in bull markets)
    """
    if b.n < p.tall_avg_bars + p.trend_bars + 3:
        return None
    a, m, z = b.n - 3, b.n - 2, b.n - 1
    avg = avg_bar_height(b, a, p.tall_avg_bars)
    if avg is None or avg <= 0.0 or float(b.h[m] - b.l[m]) < avg:
        return None
    if short_trend(b, a, p.trend_bars) <= 0.0:
        return None
    if not _in_range(b, m, float(b.o[m]), p.quarter, False):
        return None
    if not _in_range(b, m, float(b.c[m]), p.quarter, False):
        return None
    midpoint = (float(b.h[m]) + float(b.l[m])) / 2.0
    if not (float(b.h[a]) < midpoint and float(b.h[z]) < midpoint):
        return None
    trigger = float(b.l[m])
    height = float(b.h[m]) - trigger
    if height <= 0.0:
        return None
    return Setup("short", trigger, trigger * (1.0 + p.pct_stop / 100.0),
                 trigger - height, "one-day reversal top")


@pattern("open_close_reversal_uptrend", "OCRU.html", "reversal", "long", 84.0)
def _ocr_up(b, p):
    """OCRU.html — "the pattern is composed of two bars, including the
    reference to the close of the first bar"; "look for the pattern in a
    short-term up trend"; "the open must be within 25% of the intraday high";
    "the close must be within 25% of the intraday low, but also be above the
    prior day's close".

      entry          buy at the next open once price closes above the top
      stop           "a penny below the pattern's lowest point"
      target         the pattern height x 2 added to the pattern top (met
                     ~84% of the time in bull markets with up breakouts)
    """
    if b.n < p.trend_bars + 2:
        return None
    a, z = b.n - 2, b.n - 1
    if short_trend(b, a, p.trend_bars) <= 0.0:
        return None
    if not _in_range(b, z, float(b.o[z]), p.quarter, True):
        return None
    if not _in_range(b, z, float(b.c[z]), p.quarter, False):
        return None
    if float(b.c[z]) <= float(b.c[a]):
        return None
    return _small(b, a, z, "long", 2.0, "open-close reversal (uptrend)")


@pattern("open_close_reversal_downtrend", "OCRD.html", "reversal", "long", 82.0)
def _ocr_down(b, p):
    """OCRD.html — "1 or 2 bars", the pattern "composed of one bar
    referencing [the] prior bar's close", in "a short-term downtrend". "The
    open must be within 25% of the intraday low"; "the close must be within
    25% of the intraday high, but also be below the prior day's close".

      entry          "buy at the open the next day" after "price closes
                     above the top of the pattern"
      stop           "a penny below the bottom of the pattern"
      target         "measure the height of the pattern and add it to the
                     high price" (works 82% of the time)
    """
    if b.n < p.trend_bars + 2:
        return None
    a, z = b.n - 2, b.n - 1
    if short_trend(b, a, p.trend_bars) >= 0.0:
        return None
    if not _in_range(b, z, float(b.o[z]), p.quarter, False):
        return None
    if not _in_range(b, z, float(b.c[z]), p.quarter, True):
        return None
    if float(b.c[z]) >= float(b.c[a]):
        return None
    return _small(b, a, z, "long", 1.0, "open-close reversal (downtrend)")


@pattern("pivot_point_reversal_uptrend", "PPRU.html", "reversal", "long", 80.0)
def _ppr_up(b, p):
    """PPRU.html — "the pattern is composed of two price bars because it
    references the prior day's low". Look for it "during a short-term
    uptrend"; "the close must be below the prior day's low".

      entry          "once price closes above the top or below the bottom of
                     the pattern, buy/short at the open the next day"
      stop           "one penny below the pattern's lowest point"
      target         "measure the height of the pattern and add it to the
                     high price" (met roughly 80% of the time in bull markets
                     with upward breakouts)
    """
    if b.n < p.trend_bars + 2:
        return None
    a, z = b.n - 2, b.n - 1
    if short_trend(b, a, p.trend_bars) <= 0.0:
        return None
    if float(b.c[z]) >= float(b.l[a]):
        return None
    return _small(b, a, z, "long", 1.0, "pivot point reversal (uptrend)")


@pattern("wide_ranging_day_upside_reversal", "WRDUR.html", "reversal", "long", 40.0)
def _wrd_up(b, p):
    """WRDUR.html — "the pattern is composed of one bar" appearing "in a
    short-term downtrend"; "the close must be within 25% of the intraday
    high"; "look for an unusually tall price bar ... a high-low range on the
    reversal day that was at least three times the one-month average".

      entry          "a buy stop a penny above the top of the pattern"
      stop           "a penny below the bottom of the pattern"
      target         the pattern height added to the high (met 40% of the
                     time in bull markets with upward breakouts)
    """
    if b.n < p.tall_avg_bars + p.trend_bars + 2:
        return None
    z = b.n - 1
    avg = avg_bar_height(b, z, p.tall_avg_bars)
    if avg is None or avg <= 0.0:
        return None
    if float(b.h[z] - b.l[z]) < p.wide_range_mult * avg:
        return None
    if short_trend(b, z - 1, p.trend_bars) >= 0.0:
        return None
    if not _in_range(b, z, float(b.c[z]), p.quarter, True):
        return None
    return _small(b, z, z, "long", 1.0, "wide ranging day upside reversal")


@pattern("wide_ranging_day_downside_reversal", "WRDDR.html", "reversal", "short", 42.0)
def _wrd_down(b, p):
    """WRDDR.html — the mirror: "the pattern is composed of one bar", "look
    for the pattern in a short-term upward trend", "the close must be within
    25% of the intraday low", and the same "at least three times the
    one-month average" height rule.

      entry          "short at the open the next day" once price closes
                     below the bottom
      stop           the mirror of the page's "a penny below the bottom",
                     i.e. a penny above the top
      target         "subtract [the height] from the intraday low to get a
                     downward price target"
    """
    if b.n < p.tall_avg_bars + p.trend_bars + 2:
        return None
    z = b.n - 1
    avg = avg_bar_height(b, z, p.tall_avg_bars)
    if avg is None or avg <= 0.0:
        return None
    if float(b.h[z] - b.l[z]) < p.wide_range_mult * avg:
        return None
    if short_trend(b, z - 1, p.trend_bars) <= 0.0:
        return None
    if not _in_range(b, z, float(b.c[z]), p.quarter, False):
        return None
    return _small(b, z, z, "short", 1.0, "wide ranging day downside reversal")


@pattern("shark_32", "Shark32.html", "continuation", "long", 100.0)
def _shark_32(b, p):
    """Shark32.html — a three-bar pattern: "look for two consecutively lower
    highs and higher lows" (two consecutive inside days); "the last bar
    cannot have the high price equal to the low price".

      entry          at the next open after a close beyond the boundary
      stop           "a penny below the bottom of the Shark-32 pattern"
      target         "the height of the chart pattern added to the top of
                     the pattern"
    """
    if b.n < 4:
        return None
    a, m, z = b.n - 3, b.n - 2, b.n - 1
    if not (b.h[m] < b.h[a] and b.l[m] > b.l[a]):
        return None
    if not (b.h[z] < b.h[m] and b.l[z] > b.l[m]):
        return None
    if b.h[z] == b.l[z]:
        return None
    return _small(b, a, z, "long", 1.0, "Shark-32")


@pattern("gap_2h", "Gap2H.html", "continuation", "long", 100.0)
def _gap_2h(b, p):
    """Gap2H.html — three bars: "look for price to gap higher. Yesterday's
    low price is above the prior day's high, forming a gap"; "the third bar
    in the pattern makes a higher high"; "the third bar ... makes a higher
    low, but it remains below the 2nd bar's high".

      entry          "wait for breakout" — a close above the pattern's high
      stop           "a penny below the bottom of the Gap 2H pattern"
      target         the pattern height added to the pattern's top
    """
    if b.n < 4:
        return None
    a, m, z = b.n - 3, b.n - 2, b.n - 1
    if float(b.l[m]) <= float(b.h[a]):
        return None
    if not (b.h[z] > b.h[m] and b.l[z] > b.l[m] and b.l[z] < b.h[m]):
        return None
    return _small(b, a, z, "long", 1.0, "gap 2H")


@pattern("inverted_gap_2h", "Gap2Hi.html", "continuation", "short", 44.0)
def _inverted_gap_2h(b, p):
    """Gap2Hi.html — the bearish mirror: "price gaps lower; yesterday's high
    is below the prior day's low"; "the third bar makes a lower high and
    lower low"; "the third bar's high is above the second bar's low".

      entry          short at the next open once "price closes below the
                     pattern's bottom"
      stop           "a penny above the top of the inverted gap 2H pattern"
      target         the pattern height subtracted from the lowest low
                     (fulfilled ~44% of the time)
    """
    if b.n < 4:
        return None
    a, m, z = b.n - 3, b.n - 2, b.n - 1
    if float(b.h[m]) >= float(b.l[a]):
        return None
    if not (b.h[z] < b.h[m] and b.l[z] < b.l[m] and b.h[z] > b.l[m]):
        return None
    return _small(b, a, z, "short", 1.0, "inverted gap 2H")


# ═══════════════════════════════════════════════════════════════════════════
# Harmonic (Fibonacci) patterns — Gartley, bat, butterfly, crab, AB=CD
#
# Every one of these is five turning points X-A-B-C-D whose legs must sit on
# a published list of Fibonacci ratios, with the trade taken when "price
# turns at D". Bulkowski notes on each page that "the peaks and valleys in
# the pattern need not be consecutive" and that he allows "a 3 percentage
# point window" on the ratios, both of which `zigzag` and `FIB_TOL` provide.
# ═══════════════════════════════════════════════════════════════════════════


FIB_TOL = 3.0        # "plus or minus 3 percentage points ... to qualify"


def zigzag(b):
    """The alternating peak/valley skeleton of the bars: minor highs and lows
    with runs of the same kind collapsed to their extreme. This is what lets
    the harmonic points "not be consecutive" pivots.

    Returns a list of (index, kind, price) with kind "H" or "L", oldest
    first."""
    pts = [(i, "H", float(b.h[i])) for i in b.peaks]
    pts += [(i, "L", float(b.l[i])) for i in b.valleys]
    pts.sort(key=lambda t: t[0])
    out = []
    for pt in pts:
        if out and out[-1][1] == pt[1]:
            keep = pt if ((pt[2] > out[-1][2]) == (pt[1] == "H")) else out[-1]
            out[-1] = keep
        else:
            out.append(pt)
    return out


def _fib(value, targets, tol=FIB_TOL):
    """Is `value` (a percentage) within `tol` percentage points of any of the
    page's published Fibonacci ratios?"""
    return any(abs(value - t) <= tol for t in targets)


def _last_turn(z, want, b, size, max_age):
    """The most recent run of `size` alternating turns whose last point is of
    kind `want` — that last point is the pattern's final turn.

    The final turn is deliberately NOT required to be the newest zigzag
    point: a page's entry is "once price turns at D" (or "a close above the
    highest peak"), and that turn or breakout is itself a later pivot, so by
    the time the trade is armed the pattern's last point sits one turn back.
    `max_age` bounds how stale it may be — short for the harmonics, whose
    entry is the turn itself, longer for the structural patterns, whose
    entry waits for a breakout.
    """
    for j in range(len(z) - 1, size - 2, -1):
        if z[j][1] != want:
            continue
        if b.n - 1 - z[j][0] > max_age:
            return None                   # every earlier one is staler still
        # `zigzag` alternates strictly, so the kinds of the preceding turns
        # follow from D's kind and `size` — no further check is needed.
        return z[j - size + 1:j + 1]
    return None


def _harmonic_points(b, p, bullish):
    """The five alternating turns X-A-B-C-D, D being a valley for a bullish
    pattern and a peak for a bearish one. Returns the five indices followed
    by the five prices, or None."""
    z = zigzag(b)
    if len(z) < 5:
        return None
    pts = _last_turn(z, "L" if bullish else "H", b, 5, p.harmonic_max_age)
    if pts is None:
        return None
    return [q[0] for q in pts] + [q[2] for q in pts]


def _harmonic(b, p, bullish, ab_xa, bc_ab, cd_bc, ad_xa, stop_at_x,
              target_point, label):
    """Shared harmonic body. Each ratio argument is the list of percentages
    that page publishes for that leg, or None when the page states no
    constraint. `ad_xa` is the defining XAD retrace/extension.

      entry          "once price turns at D, buy/short" — armed as a stop
                     order beyond the turn that has formed since D
      stop           point X where the page says "a close below/above X",
                     otherwise point D ("if price closes below D, exit")
      target         the pattern point the page reaches most often
    """
    pts = _harmonic_points(b, p, bullish)
    if pts is None:
        return None
    xi, ai, bi, ci, di, x, a, bp, c, d = pts
    xa = abs(a - x)
    ab = abs(a - bp)
    bc = abs(c - bp)
    cd = abs(c - d)
    ad = abs(a - d)
    if min(xa, ab, bc, cd) <= 0.0:
        return None
    if ab_xa is not None and not _fib(100.0 * ab / xa, ab_xa):
        return None
    if bc_ab is not None and not _fib(100.0 * bc / ab, bc_ab):
        return None
    if cd_bc is not None and not _fib(100.0 * cd / bc, cd_bc):
        return None
    if ad_xa is not None and not _fib(100.0 * ad / xa, ad_xa):
        return None
    target = {"X": x, "A": a, "B": bp, "C": c}[target_point]
    stop_px = x if stop_at_x else d
    note = f"{label} X{x:.4f} A{a:.4f} B{bp:.4f} C{c:.4f} D{d:.4f}"
    if bullish:
        # "if price drops below X on the way to finding D, ignore the pattern"
        if stop_at_x and b.lowest(xi, di) < x:
            return None
        trigger = b.highest(di, b.n - 1)
        return Setup("long", trigger, min(stop_px, d) * 0.999, target, note)
    if stop_at_x and b.highest(xi, di) > x:
        return None
    trigger = b.lowest(di, b.n - 1)
    return Setup("short", trigger, max(stop_px, d) * 1.001, target, note)


@pattern("gartley_bullish", "GartleyBull.html", "reversal", "long", 98.0)
def _gartley_bull(b, p):
    """GartleyBull.html — "price rises from X to peak at A"; "price retraces
    from the peak A to valley B about 61.8% of the XA move"; "the BC move
    retraces 61.8% or 78.6% of the AB drop"; "the CD move is 127% or 161.8%
    of the BC move". "If price drops below X on the way to finding D, then
    the pattern should be ignored."

      entry          "once price turns at D, buy"
      stop           "a close below X"
      target         point B, reached 98% of the time (A 45%, C 59%)
    """
    return _harmonic(b, p, True, [61.8], [61.8, 78.6], [127.0, 161.8],
                     None, True, "B", "bullish Gartley")


@pattern("gartley_bearish", "GartleyBear.html", "reversal", "short", 99.0)
def _gartley_bear(b, p):
    """GartleyBear.html — the mirror: "price drops from X to valley at A",
    AB rebounds "approximately 61.8% of the XA decline", "the BC move
    retraces 61.8% or 78.6% of the AB rise", "the CD move is 127% or 161.8%
    of the BC move". "If price rises above X on the way to finding D, then
    the pattern should be ignored."

      entry          "once price turns at D, short the stock"
      stop           "a close above X"
      target         point B, reached 99% of the time (A 34%, C 51%)
    """
    return _harmonic(b, p, False, [61.8], [61.8, 78.6], [127.0, 161.8],
                     None, True, "B", "bearish Gartley")


@pattern("bat_bullish", "BatBull.html", "reversal", "long", 86.0)
def _bat_bull(b, p):
    """BatBull.html — "price drops to valley X, the first point in the
    pattern. It rises from there to A and retraces to B. The AB/AX retrace
    should be either 38.2% or 50%." The CB/AB retrace must be one of "38.2%,
    50%, 61.8%, 70.7%, 78.6% or 88.6%"; the CD/CB extension one of "161.8%,
    200%, 224% or 261.8%"; and "retrace AD/AX is 88.6%" with the page's
    3-percentage-point allowance.

      entry          price turns at D 91% of the time
      stop           point D — the page publishes no stop, so the sibling
                     butterfly page's "if price closes below D, exit" is used
      target         point B, reached 86% of the time (C 64%, A 58%)
    """
    return _harmonic(b, p, True, [38.2, 50.0],
                     [38.2, 50.0, 61.8, 70.7, 78.6, 88.6],
                     [161.8, 200.0, 224.0, 261.8], [88.6], False, "B",
                     "bullish bat")


@pattern("bat_bearish", "BatBear.html", "reversal", "short", 81.0)
def _bat_bear(b, p):
    """BatBear.html — "BA/XA retrace should be either 38.2% or 50%"; "BC/BA
    retrace of one of the following Fibonacci ratios: 38.2%, 50%, 61.8%,
    70.7%, 78.6% or 88.6%"; "DC/BC is one of the following Fibonacci ratios:
    161.8%, 200%, 224% or 261.8%"; "DA/XA is 88.6%" (85.6%-91.6%).

      entry          "once point D is located, price should turn lower
                     there. It does, too, 86% of the time!"
      target         point B, reached 81% of the time (C 48%, A 35%)
    """
    return _harmonic(b, p, False, [38.2, 50.0],
                     [38.2, 50.0, 61.8, 70.7, 78.6, 88.6],
                     [161.8, 200.0, 224.0, 261.8], [88.6], False, "B",
                     "bearish bat")


@pattern("butterfly_bullish", "ButterflyBull.html", "reversal", "long", 79.0)
def _butterfly_bull(b, p):
    """ButterflyBull.html — "price rises from X to peak at A and retraces
    78.6% to B" and "the AD move measures 127% of AX". The page shows the
    BC and CD ratios only in a figure, so the ratio lists published in text
    on the mirror page (ButterflyBear.html: "BC as a function of BA ...
    between and including 38.2% to 88.6%", "the DC/BC extension ... from
    161.8% to 224%") are used for both.

      entry          "once price turns at D ... buy the stock. Price may
                     continue dropping below D but that happens just 9% of
                     the time"
      stop           "if price closes below D, then exit the trade"
      target         point B, reached 79% of the time (C 49%, A 39%)
    """
    return _harmonic(b, p, True, [78.6],
                     [38.2, 50.0, 61.8, 70.7, 78.6, 88.6],
                     [161.8, 200.0, 224.0], [127.0], False, "B",
                     "bullish butterfly")


@pattern("butterfly_bearish", "ButterflyBear.html", "reversal", "short", 76.0)
def _butterfly_bear(b, p):
    """ButterflyBear.html — "price drops from X to valley A, then retraces up
    to B. The BA retrace of XA measures 78.6%"; "retrace BC as a function of
    BA should be a Fibonacci ratio between and including 38.2% to 88.6%";
    "the DC/BC extension measures one of the Fibonacci ratios from 161.8% to
    224%"; "the extension AD as a percentage of XA is 127 but I allow plus or
    minus 3 percentage points (124% to 130%)".

      entry          "once price turns at D, short the stock"
      stop           "use a close above D as the stop location"
      target         point B, reached 76% of the time (C 38%, A 24%)
    """
    return _harmonic(b, p, False, [78.6],
                     [38.2, 50.0, 61.8, 70.7, 78.6, 88.6],
                     [161.8, 200.0, 224.0], [127.0], False, "B",
                     "bearish butterfly")


@pattern("crab_bullish", "CrabBull.html", "reversal", "long", 65.0)
def _crab_bull(b, p):
    """CrabBull.html — "price drops to valley X, the first point in the
    pattern. It rises from there to A and retraces to B", and the defining
    leg is "retrace AD/AX is 161.8%". The page shows the other ratios only in
    a figure, so the mirror page's published table is used (CrabBear.html:
    BA/XA "38.2%, 50%, or 61.8%", BC/BA "38.2%, 50%, 61.8%, 78.6%, 88.6%, or
    100%", DC/BC "161%, 200%, or 314%").

      entry          "92% of trades reverse upward at point D"
      target         point X, reached 65% of the time (B 48%, C 36%, A 33%)
    """
    return _harmonic(b, p, True, [38.2, 50.0, 61.8],
                     [38.2, 50.0, 61.8, 78.6, 88.6, 100.0],
                     [161.0, 200.0, 314.0], [161.8], False, "X",
                     "bullish crab")


@pattern("crab_bearish", "CrabBear.html", "reversal", "short", 59.0)
def _crab_bear(b, p):
    """CrabBear.html — five turns X, A, B, C, D with "BA/XA: 38.2%, 50%, or
    61.8%", "BC/BA: 38.2%, 50%, 61.8%, 78.6%, 88.6%, or 100%", "DC/BC: 161%,
    200%, or 314%", and "DA/XA: 161.8% (±3 percentage points:
    158.8%-164.8%)".

      target         point X, reached 59% of the time (B 37%, C 23%, A 18%);
                     "87% of patterns show price dips below point D"
    """
    return _harmonic(b, p, False, [38.2, 50.0, 61.8],
                     [38.2, 50.0, 61.8, 78.6, 88.6, 100.0],
                     [161.0, 200.0, 314.0], [161.8], False, "X",
                     "bearish crab")


def _abcd(b, p, bullish, label):
    """ABCDBull.html / ABCDBear.html — a four-turn pattern (A, B, C, D):
    "retrace CB as a percentage of AB should be one of the following
    Fibonacci ratios: 38.2%, 50%, 61.8%, 70.7%, 78.6% or 88.6%" and "the
    CD/CB extension measures one of 113%, 127%, 141%, 161.8%, 200%, 224%,
    261.8%, or 314%". The page's rule is to "find four turns where the ratio
    of one leg to another is close to the Fibonacci numbers listed"."""
    z = zigzag(b)
    if len(z) < 4:
        return None
    pts = _last_turn(z, "L" if bullish else "H", b, 4, p.harmonic_max_age)
    if pts is None:
        return None
    ai, bi, ci, di = (q[0] for q in pts)
    a, bp, c, d = (q[2] for q in pts)
    ab = abs(a - bp)
    cb = abs(c - bp)
    cd = abs(c - d)
    if min(ab, cb, cd) <= 0.0:
        return None
    if not _fib(100.0 * cb / ab, [38.2, 50.0, 61.8, 70.7, 78.6, 88.6]):
        return None
    if not _fib(100.0 * cd / cb, [113.0, 127.0, 141.0, 161.8, 200.0, 224.0,
                                  261.8, 314.0]):
        return None
    note = f"{label} A{a:.4f} B{bp:.4f} C{c:.4f} D{d:.4f}"
    if bullish:
        return Setup("long", b.highest(di, b.n - 1), d * 0.999, bp, note)
    return Setup("short", b.lowest(di, b.n - 1), d * 1.001, bp, note)


@pattern("ab_cd_bullish", "ABCDBull.html", "reversal", "long", 83.0)
def _abcd_bull(b, p):
    """ABCDBull.html — "price drops from peak A to valley B then retraces to
    C", then "after reaching a low at B, price rises to C followed by an
    extension to D".

      entry          the turn upward at D (which happens 38% of the time —
                     the page's stated limitation)
      stop           point D; the page publishes no stop rule
      target         point B, reached 83% of the time (C 47%, A 40%)
    """
    return _abcd(b, p, True, "bullish AB=CD")


@pattern("ab_cd_bearish", "ABCDBear.html", "reversal", "short", 83.0)
def _abcd_bear(b, p):
    """ABCDBear.html — the mirror of the bullish AB=CD: price rises from
    valley A to peak B, retraces to C, then extends down to D, with the same
    published CB/AB and CD/CB Fibonacci ratio lists.

      entry          the turn downward at D
      stop           point D
      target         point B
    """
    return _abcd(b, p, False, "bearish AB=CD")


@pattern("turnkey_bullish", "TurnkeyBull.html", "reversal", "long", 100.0)
def _turnkey_bull(b, p):
    """TurnkeyBull.html — a four-bar pattern:

      bar 1          "any price bar"
      bar 2          "price makes a low below bar 1, but closes above bar
                     1's close"
      bar 3          "price posts a higher high (above bar 2's high), and
                     closes below bar 2's close but remains above bar 1's
                     close"
      bar 4          "makes a lower low (below bar 3) but a close above bar
                     3's close"

    Buy stop a penny above the pattern top, stop a penny below the pattern
    bottom, target twice the height added to the highest bar. The page
    reports the pattern "significantly underperforms benchmarks" and advises
    avoiding it — it is implemented because it is in the index, not because
    the page recommends it.
    """
    if b.n < 5:
        return None
    i1, i2, i3, i4 = b.n - 4, b.n - 3, b.n - 2, b.n - 1
    if not (b.l[i2] < b.l[i1] and b.c[i2] > b.c[i1]):
        return None
    if not (b.h[i3] > b.h[i2] and b.c[i1] < b.c[i3] < b.c[i2]):
        return None
    if not (b.l[i4] < b.l[i3] and b.c[i4] > b.c[i3]):
        return None
    return _small(b, i1, i4, "long", 2.0, "bullish turn-key")


@pattern("turnkey_bearish", "TurnkeyBear.html", "reversal", "long", 100.0)
def _turnkey_bear(b, p):
    """TurnkeyBear.html — the mirror four-bar pattern:

      bar 2          "price makes a higher high (above bar 1), but closes
                     below bar 1's close"
      bar 3          "price posts a lower low (below bar 2's low), and closes
                     above bar 2's close but remains below bar 1's close"
      bar 4          "makes a higher high (above bar 3) but a close below bar
                     3's close"

    The page publishes the same long-side tactics as the bullish version ("a
    buy stop placed a penny above the top of the chart pattern"), so that is
    the side armed.
    """
    if b.n < 5:
        return None
    i1, i2, i3, i4 = b.n - 4, b.n - 3, b.n - 2, b.n - 1
    if not (b.h[i2] > b.h[i1] and b.c[i2] < b.c[i1]):
        return None
    if not (b.l[i3] < b.l[i2] and b.c[i2] < b.c[i3] < b.c[i1]):
        return None
    if not (b.h[i4] > b.h[i3] and b.c[i4] < b.c[i3]):
        return None
    return _small(b, i1, i4, "long", 2.0, "bearish turn-key")


@pattern("two_step_bullish", "2StepBull.html", "reversal", "long", 100.0)
def _two_step_bull(b, p):
    """2StepBull.html — five bars, "breaks out upward 79% of the time in
    stocks":

      bar 2          a low below bar 1 with a lower close
      bar 3          low below bar 2, but close above bar 1 and bar 2's
                     closes ("bars 1 to 3 form a 2-close reversal pattern")
      bar 4          close below bar 3's close
      bar 5          low below bar 4 but closes above bars 3 and 4

    Buy stop above the pattern top, stop below the pattern bottom, target
    twice the height added to the highest bar.
    """
    if b.n < 6:
        return None
    i1, i2, i3, i4, i5 = (b.n - 5 + k for k in range(5))
    if not (b.l[i2] < b.l[i1] and b.c[i2] < b.c[i1]):
        return None
    if not (b.l[i3] < b.l[i2] and b.c[i3] > b.c[i1] and b.c[i3] > b.c[i2]):
        return None
    if b.c[i4] >= b.c[i3]:
        return None
    if not (b.l[i5] < b.l[i4] and b.c[i5] > b.c[i3] and b.c[i5] > b.c[i4]):
        return None
    return _small(b, i1, i5, "long", 2.0, "bullish 2-step")


@pattern("two_step_bearish", "2StepBear.html", "reversal", "long", 100.0)
def _two_step_bear(b, p):
    """2StepBear.html — the mirror five bars, which "breaks out downward 74%
    of the time in stocks":

      bar 2          "price makes a high above bar 1 with a higher close"
      bar 3          "price has a high above bar 2 but a close below bar 1"
      bar 4          "makes a close above bar 3's close"
      bar 5          "has a high above bar 4 but closes below bars 3 and 4"

    As on the bullish page the published tactics are the long-side ones, so
    that is the side armed.
    """
    if b.n < 6:
        return None
    i1, i2, i3, i4, i5 = (b.n - 5 + k for k in range(5))
    if not (b.h[i2] > b.h[i1] and b.c[i2] > b.c[i1]):
        return None
    if not (b.h[i3] > b.h[i2] and b.c[i3] < b.c[i1]):
        return None
    if b.c[i4] <= b.c[i3]:
        return None
    if not (b.h[i5] > b.h[i4] and b.c[i5] < b.c[i3] and b.c[i5] < b.c[i4]):
        return None
    return _small(b, i1, i5, "long", 2.0, "bearish 2-step")


@pattern("double_key_reversal_bullish", "DoubleKeyBull.html", "reversal", "long", 100.0)
def _double_key_bull(b, p):
    """DoubleKeyBull.html — three bars in a 5-day downtrend:

      bar 1          "price must close in [the] lower 25% of the price bar"
      bar 2          "price makes a lower low (below bar 1), but closes
                     above bar 1's close"
      bar 3          "price posts a lower low (below bar 2), but closes
                     above bar 2's close"

    Buy stop above the pattern top, stop below the pattern bottom, target
    twice the height added to the highest bar.
    """
    if b.n < p.trend_bars + 4:
        return None
    i1, i2, i3 = b.n - 3, b.n - 2, b.n - 1
    if short_trend(b, i1, p.trend_bars) >= 0.0:
        return None
    if not _in_range(b, i1, float(b.c[i1]), p.quarter, False):
        return None
    if not (b.l[i2] < b.l[i1] and b.c[i2] > b.c[i1]):
        return None
    if not (b.l[i3] < b.l[i2] and b.c[i3] > b.c[i2]):
        return None
    return _small(b, i1, i3, "long", 2.0, "bullish double-key reversal")


@pattern("double_key_reversal_bearish", "DoubleKeyBear.html", "reversal", "long", 100.0)
def _double_key_bear(b, p):
    """DoubleKeyBear.html — three bars in a 5-day uptrend:

      bar 1          "price must close in [the] upper 25% of the price bar"
      bar 2          a higher high than bar 1 but "closes below bar 1's close"
      bar 3          a higher high than bar 2 but "closes below bar 2's close"

    The page publishes the long-side tactics, so that is the side armed.
    """
    if b.n < p.trend_bars + 4:
        return None
    i1, i2, i3 = b.n - 3, b.n - 2, b.n - 1
    if short_trend(b, i1, p.trend_bars) <= 0.0:
        return None
    if not _in_range(b, i1, float(b.c[i1]), p.quarter, True):
        return None
    if not (b.h[i2] > b.h[i1] and b.c[i2] < b.c[i1]):
        return None
    if not (b.h[i3] > b.h[i2] and b.c[i3] < b.c[i2]):
        return None
    return _small(b, i1, i3, "long", 2.0, "bearish double-key reversal")


@pattern("fakey_bullish", "FakeyBull.html", "reversal", "long", 100.0)
def _fakey_bull(b, p):
    """FakeyBull.html — a four-day pattern built on an inside day:

      days 1-2       "the high of day 1 is above the high of day 2. Low of
                     day 1 is below the low of day 2."
      day 3          "the high of day 3 is below the high of day 1. The low
                     of day 3 is below the low of day 1."
      day 4          "the low of day 4 is below the low of day 3. The high
                     of day 4 is below the high of day 3."

      entry          "place a buy stop a penny above the top (high) of
                     candle 1" — which is the pattern top
      confirmation   "price must first rise above the high of day 1 to be a
                     valid fakey pattern"; "if price drops below the low of
                     day 4, cancel the buy stop"
      stop           "a penny below the bottom of fakey"
      target         the fakey's height x 2 added to the pattern's top
    """
    if b.n < 5:
        return None
    d1, d2, d3, d4 = b.n - 4, b.n - 3, b.n - 2, b.n - 1
    if not (b.h[d1] > b.h[d2] and b.l[d1] < b.l[d2]):
        return None
    if not (b.h[d3] < b.h[d1] and b.l[d3] < b.l[d1]):
        return None
    if not (b.l[d4] < b.l[d3] and b.h[d4] < b.h[d3]):
        return None
    return _small(b, d1, d4, "long", 2.0, "bullish fakey")


@pattern("fakey_bearish", "FakeyBear.html", "reversal", "short", 100.0)
def _fakey_bear(b, p):
    """FakeyBear.html — the mirror:

      days 1-2       the same inside day
      day 3          "the high of day 3 is above the high of day 1. The low
                     of day 3 is above the low of day 1."
      day 4          "the low of day 4 is above the low of day 3. The high
                     of day 4 is above the high of day 3."

      entry          short "a penny below the bottom (low) of candle 1",
                     which is the pattern bottom
      confirmation   "price must first drop below the low of day 1 to be a
                     valid bearish fakey pattern"; cancel if "price rises
                     above the high of day 4"
      stop           the high of day 4, which is the pattern top
      target         the page states no target, so the bullish page's rule
                     is mirrored: twice the height below the pattern bottom
    """
    if b.n < 5:
        return None
    d1, d2, d3, d4 = b.n - 4, b.n - 3, b.n - 2, b.n - 1
    if not (b.h[d1] > b.h[d2] and b.l[d1] < b.l[d2]):
        return None
    if not (b.h[d3] > b.h[d1] and b.l[d3] > b.l[d1]):
        return None
    if not (b.l[d4] > b.l[d3] and b.h[d4] > b.h[d3]):
        return None
    return _small(b, d1, d4, "short", 2.0, "bearish fakey")


@pattern("carl_v_bullish", "CarlVBull.html", "reversal", "long", 57.0)
def _carl_v_bull(b, p):
    """CarlVBull.html — "look for a minor low (X) which leads to a broadening
    pattern where a second peak is above the first (C is above A), and a
    second valley is below the first (D is below B)":

      X              the lowest point
      A              "minor high above X; highest peak between X and A"
      B              "valley below A but above X"
      C              "peak above A; highest point between A and C"
      D              "valley below B but above X"

      entry          Vanhaesendonck's method — "25% of the XC distance added
                     to the low at D"
      stop           "a penny below D"
      measure rule   "compute the height of the chart pattern from the
                     highest peak (C) to the lowest valley (X). Add the
                     result to the low price of turn D." (reached 57%)
    """
    pts = _harmonic_points(b, p, True)
    if pts is None:
        return None
    xi, ai, bi, ci, di, x, a, bp, c, d = pts
    if not (a > x and bp > x and c > a and d < bp and d > x):
        return None
    height = c - x
    if height <= 0.0:
        return None
    trigger = d + 0.25 * height
    if trigger <= d:
        return None
    return Setup("long", trigger, d * 0.999, d + height,
                 f"bullish Carl V X{x:.4f} A{a:.4f} B{bp:.4f} C{c:.4f} D{d:.4f}")


@pattern("carl_v_bearish", "CarlVBear.html", "reversal", "short", 100.0)
def _carl_v_bear(b, p):
    """CarlVBear.html — "look for a minor high (X) which leads to a
    broadening pattern where a second valley is below the first (C is below
    A), and a second peak is above the first (D is above B)":

      X              a minor high
      A              "minor low below X"
      B              "peak above A but below X"
      C              "valley below A"
      D              "above B but below X"

      entry          "he takes 25% of the pattern's height (X to C) and
                     subtracts that from the high price at D"
      stop           "a stop-loss order slightly above peak D"
      target         the 100% target — "full pattern height subtracted from
                     D" (the page also gives 50% and 200% scale-out levels)
    """
    pts = _harmonic_points(b, p, False)
    if pts is None:
        return None
    xi, ai, bi, ci, di, x, a, bp, c, d = pts
    if not (a < x and bp < x and c < a and d > bp and d < x):
        return None
    height = x - c
    if height <= 0.0:
        return None
    trigger = d - 0.25 * height
    if trigger >= d:
        return None
    return Setup("short", trigger, d * 1.001, d - height,
                 f"bearish Carl V X{x:.4f} A{a:.4f} B{bp:.4f} C{c:.4f} D{d:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Gaps (gaps.html)
#
# One page covers five kinds of gap. They are all "a gap on the latest bar";
# what separates them is WHERE in a trend the gap happens, which the page
# describes qualitatively. Those descriptions become the numeric tests below
# (deviation 2), each named in its detector's docstring.
# ═══════════════════════════════════════════════════════════════════════════


def _gap(b):
    """The gap on the last bar, as (direction, low_edge, high_edge, size_pct)
    or None. Direction is +1 for an up gap, -1 for a down gap."""
    z = b.n - 1
    if float(b.l[z]) > float(b.h[z - 1]):
        lo, hi = float(b.h[z - 1]), float(b.l[z])
        return 1, lo, hi, pct(hi, lo)
    if float(b.h[z]) < float(b.l[z - 1]):
        lo, hi = float(b.h[z]), float(b.l[z - 1])
        return -1, lo, hi, pct(hi, lo)
    return None


def _congested(b, end, bars, max_pct):
    """"Occurs in congestion (trendless markets)" — the `bars` before `end`
    span less than `max_pct` of their mean price."""
    s = max(0, end - bars)
    if end - s < 3:
        return False
    hi, lo = b.highest(s, end), b.lowest(s, end)
    base = float(np.mean(b.c[s:end + 1]))
    return base > 0.0 and 100.0 * (hi - lo) / base <= max_pct


def _heavy_volume(b, mult):
    """"On elevated volume" / "with high volume" / "usually heavy"."""
    avg = sma(b.v[:b.n - 1], 20)
    return avg is not None and avg > 0.0 and float(b.v[-1]) >= mult * avg


@pattern("area_gap", "gaps.html", "other", "none", 100.0, tradeable=False)
def _area_gap(b, p):
    """gaps.html, "area/common/pattern gaps" — "occurs in congestion
    (trendless markets) and closes quickly, usually in a few days", with "a
    distinctive price curl as the gap closes" and a median closure of 3-4
    days.

    The page states outright that these are not tradeable — "these close too
    quickly to be of trading significance ... use as weak support/resistance
    only" — so the detector identifies the gap and arms nothing rather than
    inventing an entry the page does not give.
    """
    g = _gap(b)
    if g is None:
        return None
    if not _congested(b, b.n - 2, p.gap_congestion_bars, p.gap_congestion_pct):
        return None
    return None


@pattern("ex_dividend_gap", "gaps.html", "other", "none", 100.0, tradeable=False)
def _ex_dividend_gap(b, p):
    """gaps.html, "ex-dividend gaps" — "caused by a dividend distribution.
    Price moves down by the amount of the dividend", a "mechanical price
    adjustment [that] distinguishes these from technical gaps".

    Like area gaps the page calls these untradeable ("close too quickly to be
    of trading significance", typically by day's end), so nothing is armed.
    The detector recognises the shape: a small down gap on ordinary volume,
    with no trend or congestion signature.
    """
    g = _gap(b)
    if g is None or g[0] != -1:
        return None
    if g[3] > p.ex_div_max_gap_pct or _heavy_volume(b, p.gap_volume_mult):
        return None
    return None


@pattern("breakaway_gap", "gaps.html", "continuation", "long", 100.0)
def _breakaway_gap(b, p):
    """gaps.html, "breakaway gaps" — the gap "starts a new trend" when "price
    exits consolidation on elevated volume that may persist several days.
    Price trends onward without filling the gap." Unlike a continuation gap
    it happens at a trend's start, not its midpoint; median closure is 84-89
    days.

      entry          "enter in the trend direction"
      stop           "a few cents beyond the gap"
      target         the page publishes no measure rule for this gap, so no
                     limit is bracketed
    """
    g = _gap(b)
    if g is None or g[0] != 1:
        return None
    if not _congested(b, b.n - 2, p.gap_congestion_bars, p.gap_congestion_pct):
        return None
    if not _heavy_volume(b, p.gap_volume_mult):
        return None
    return Setup("long", float(b.h[-1]), g[1] * 0.999, None,
                 f"breakaway gap {g[1]:.4f}-{g[2]:.4f} out of congestion")


@pattern("continuation_gap", "gaps.html", "continuation", "long", 100.0)
def _continuation_gap(b, p):
    """gaps.html, "continuation/measuring/runaway gaps" — these "occur during
    a straight-line advance or decline" on high volume and "usually mark the
    halfway point in an upward price move". Mid-trend occurrence is what
    separates them from breakaway gaps.

      entry          in the trend direction
      measure rule   "measure from [the] swing low to [the] gap center;
                     project to [the] predicted high" — so the target is the
                     gap centre plus the distance from the swing low to it
      stop           beyond the gap
    """
    g = _gap(b)
    if g is None or g[0] != 1:
        return None
    if _congested(b, b.n - 2, p.gap_congestion_bars, p.gap_congestion_pct):
        return None                        # that would be a breakaway gap
    swing_low = b.lowest(max(0, b.n - 1 - p.gap_trend_bars), b.n - 2)
    if swing_low <= 0.0 or pct(float(b.c[-2]), swing_low) < p.gap_trend_pct:
        return None                        # no straight-line advance
    if not _heavy_volume(b, p.gap_volume_mult):
        return None
    centre = (g[1] + g[2]) / 2.0
    return Setup("long", float(b.h[-1]), g[1] * 0.999,
                 centre + (centre - swing_low),
                 f"continuation gap, halfway target {centre + (centre - swing_low):.4f}")


@pattern("exhaustion_gap", "gaps.html", "reversal", "short", 100.0)
def _exhaustion_gap(b, p):
    """gaps.html, "exhaustion gaps" — the gap "happens at the end of a trend
    on high volume", is often "unusually tall", and "usually closes within a
    week"; "price consolidates or reverses after" and "violent reversals can
    follow".

      entry          "consider entering the new direction" — so an up
                     exhaustion gap is armed short, below the gap bar
      stop           "beyond the gap", i.e. above the gap bar's high
      target         the gap closing, which the page says happens within a
                     week: the far edge of the gap (the prior bar's high)
    """
    g = _gap(b)
    if g is None or g[0] != 1:
        return None
    swing_low = b.lowest(max(0, b.n - 1 - p.gap_trend_bars), b.n - 2)
    if swing_low <= 0.0 or pct(float(b.c[-2]), swing_low) < p.gap_trend_pct:
        return None                        # not at the end of a trend
    if not _heavy_volume(b, p.gap_volume_mult):
        return None
    avg = avg_bar_height(b, b.n - 1, p.tall_avg_bars)
    if avg is None or (g[2] - g[1]) < p.gap_tall_mult * avg:
        return None                        # "unusually tall"
    trigger = float(b.l[-1])
    if not g[1] < trigger:
        return None
    return Setup("short", trigger, float(b.h[-1]) * 1.001, g[1],
                 f"exhaustion gap {g[1]:.4f}-{g[2]:.4f}, target the gap close")


# ═══════════════════════════════════════════════════════════════════════════
# Scallops
# ═══════════════════════════════════════════════════════════════════════════


@pattern("ascending_scallop", "ascscallop.html", "continuation", "long", 62.0)
def _ascending_scallop(b, p):
    """ascscallop.html — "the chart pattern looks like the letter J. Find two
    peaks with a rounded valley in between and a higher right peak", with
    "price trend: upward leading to the chart pattern". Breakout is upward
    83% of the time in bull markets; 62% of those meet the target.

      confirmation   "a close above the highest peak in the chart pattern
                     signals an upward breakout"
      stop           "place a stop below the lowest valley (B) if it's not
                     too far away"
      measure rule   the height from the highest peak to the lowest valley,
                     x 62%, added to the peak price
    """
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 3, p.structure_max_age)
    if pts is None:
        return None
    (li, _, left), (vi, _, valley), (ri, _, right) = pts
    if right <= left:
        return None                        # "a higher right peak"
    if not trend_up_into(b, li, p.trend_window, p.min_trend_pct):
        return None
    if valley_width(b, vi, p.shape_tol_pct) < p.scallop_round_width:
        return None                        # "a rounded valley in between"
    height = right - valley
    if height <= 0.0:
        return None
    return Setup("long", right, valley * 0.999,
                 measure_long(right, height, 62.0),
                 f"ascending scallop {valley:.4f} -> {right:.4f}")


@pattern("descending_scallop", "descscallops.html", "reversal", "short", 34.0)
def _descending_scallop(b, p):
    """descscallops.html — "the descending scallop looks like the backward
    letter J. Find two peaks with a rounded valley in between and the left
    peak higher than the right one", with price "usually downward leading to
    the descending scallop". "The pattern breaks out downward 78% of the
    time"; 34% of those meet the target.

      confirmation   a close "below the lowest valley (downward breakout)"
      stop           "for downward breakouts, a stop above the right peak
                     (C) works well"
      measure rule   the height from the highest peak (A) to the lowest
                     valley (B), x 34%, subtracted from the lowest valley
    """
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 3, p.structure_max_age)
    if pts is None:
        return None
    (li, _, left), (vi, _, valley), (ri, _, right) = pts
    if left <= right:
        return None                        # "the left peak higher than the right"
    if not trend_down_into(b, li, p.trend_window, p.min_trend_pct):
        return None
    if valley_width(b, vi, p.shape_tol_pct) < p.scallop_round_width:
        return None
    height = left - valley
    if height <= 0.0:
        return None
    return Setup("short", valley, right * 1.001,
                 measure_short(valley, height, 34.0),
                 f"descending scallop {left:.4f} -> {valley:.4f}")


@pattern("ascending_inverted_scallop", "aiscallop.html", "continuation", "long", 64.0)
def _ascending_inverted_scallop(b, p):
    """aiscallop.html — "inverted and backward J shape. It looks like the
    right half of an umbrella": a valley (A), a rounded top (B), then a right
    edge (C) where "the end of the pattern on the right usually retraces 54%
    of the prior up move". Volume "trends downward 70% of the time"; the
    breakout is upward 95% of the time.

      confirmation   "the pattern confirms when price closes above the
                     highest high in the pattern"
      entry          "buy when price closes above the highest peak (point B)"
      stop           "a few pennies below the right scallop edge (point C)"
      measure rule   peak B minus valley A, x 64%, added to peak B
    """
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 3, p.structure_max_age)
    if pts is None:
        return None
    (ai, _, a), (bi, _, peak), (ci, _, c) = pts
    if not a < c < peak:
        return None
    up = peak - a
    if up <= 0.0:
        return None
    retrace = 100.0 * (peak - c) / up
    if abs(retrace - p.aiscallop_retrace_pct) > p.scallop_retrace_tol:
        return None                        # "usually retraces 54%"
    if peak_width(b, bi, p.shape_tol_pct) < p.scallop_round_width:
        return None                        # "the peaks should form a rounded turn"
    if p.require_volume_rules and not volume_recedes(b, ai, b.n - 1):
        return None
    return Setup("long", peak, c * 0.999, measure_long(peak, up, 64.0),
                 f"ascending inverted scallop A{a:.4f} B{peak:.4f} C{c:.4f}")


@pattern("inverted_descending_scallop", "idscallops.html", "reversal", "short", 29.0)
def _inverted_descending_scallop(b, p):
    """idscallops.html — "looks like an inverted J" with "a rounded top, not
    V-shaped": price runs from the start (A) to its high (B), which "averages
    56% of the following down move", then falls to the lowest valley (C).
    Price trends "usually downward leading to the scallop or at bearish
    turning points", and "both the scallop start and end should form at price
    turning points".

      confirmation   "price closes below [the] lowest valley without first
                     closing above [the] pattern peak"
      entry          "short when price closes below the pattern's lowest
                     valley"
      stop           "cover [the] short if price rises 67% of the decline
                     from peak to valley"
      measure rule   the height (peak to valley) x 29%, subtracted from the
                     valley
    """
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 3, p.structure_max_age)
    if pts is None:
        return None
    (ai, _, a), (bi, _, peak), (ci, _, c) = pts
    if not (peak > a and peak > c):
        return None
    if peak_width(b, bi, p.shape_tol_pct) < p.scallop_round_width:
        return None
    height = peak - c
    if height <= 0.0:
        return None
    return Setup("short", c, c + p.idscallop_cover_pct / 100.0 * height,
                 measure_short(c, height, 29.0),
                 f"inverted descending scallop A{a:.4f} B{peak:.4f} C{c:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Measured moves
# ═══════════════════════════════════════════════════════════════════════════


@pattern("measured_move_up", "mmu.html", "reversal", "long", 60.0)
def _measured_move_up(b, p):
    """mmu.html — "measured moves are reversal patterns so look for a
    downward price trend"; the "first leg [is] any minor low leading to a
    minor high"; "the computer algorithm to find these patterns looks for
    retraces of at least 70%" in the corrective phase; "price ends the
    pattern at a minor high".

      entry          "begin buying once the second leg begins" at point C
      stop           "if price drops below the corrective phase low (C),
                     close out the trade"
      measure rule   the first leg (A to B), x 60%, added to the corrective
                     phase low (C)
    """
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 3, p.turn_max_age)
    if pts is None:
        return None
    (ai, _, a), (bi, _, peak), (ci, _, c) = pts
    leg = peak - a
    if leg <= 0.0 or not a < c < peak:
        return None
    if 100.0 * (peak - c) / leg < p.mm_min_retrace_pct:
        return None
    if not trend_down_into(b, ai, p.trend_window, p.min_trend_pct):
        return None
    trigger = b.highest(ci, b.n - 1)
    return Setup("long", trigger, c * 0.999, measure_long(c, leg, 60.0),
                 f"measured move up A{a:.4f} B{peak:.4f} C{c:.4f}")


@pattern("measured_move_down", "mmd.html", "reversal", "short", 43.0)
def _measured_move_down(b, p):
    """mmd.html — "measured moves (MMDs) are reversal patterns so look for an
    upward price trend leading to the MMD"; the "first leg [is] any minor
    high which leads to a minor low"; the corrective phase retraces "at least
    70%"; "price ends the pattern at a minor low".

      entry          "short once the second leg begins"
      stop           "if price rises above the corrective phase high, close
                     out the short"
      measure rule   "compute the length of the first leg from [the] highest
                     peak (A) to [the] lowest valley at the start of the
                     corrective phase (B) then multiply it by the ...
                     percentage meeting price target. Subtract the result
                     from the highest peak in the corrective phase (C)"
    """
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 3, p.turn_max_age)
    if pts is None:
        return None
    (ai, _, a), (bi, _, valley), (ci, _, c) = pts
    leg = a - valley
    if leg <= 0.0 or not valley < c < a:
        return None
    if 100.0 * (c - valley) / leg < p.mm_min_retrace_pct:
        return None
    if not trend_up_into(b, ai, p.trend_window, p.min_trend_pct):
        return None
    trigger = b.lowest(ci, b.n - 1)
    return Setup("short", trigger, c * 1.001, measure_short(c, leg, 43.0),
                 f"measured move down A{a:.4f} B{valley:.4f} C{c:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Spikes, and the partial rise / partial decline
# ═══════════════════════════════════════════════════════════════════════════


@pattern("bullish_spike", "spikes.html", "reversal", "long", 100.0)
def _bullish_spike(b, p):
    """spikes.html — "a tall price move with the closing price near the base
    of the spike". A bullish spike is where "price spikes downward before
    closing near the intraday high. Price often represents a low turning
    point." Volume is "usually heavy".

      confirmation   "wait a day to be sure that the price spike stands alone
                     like the above picture shows" — so the detector runs on
                     the bar after the spike and requires the neighbours not
                     to reach into it
      stop           the spike's low
      target         the page gives no measure rule, so no limit is placed
                     and the trade leaves on its stop or the time stop
    """
    if b.n < p.tall_avg_bars + 3:
        return None
    s, z = b.n - 2, b.n - 1              # s is the spike, z the waiting day
    avg = avg_bar_height(b, s, p.tall_avg_bars)
    if avg is None or float(b.h[s] - b.l[s]) < p.spike_tall_mult * avg:
        return None
    if not _in_range(b, s, float(b.c[s]), p.quarter, True):
        return None                        # "closing near the intraday high"
    if not (float(b.l[s]) < float(b.l[s - 1]) and float(b.l[s]) < float(b.l[z])):
        return None                        # "stands alone"
    if p.require_volume_rules and not _heavy_volume(b, 1.0):
        return None
    return Setup("long", b.highest(s, z), float(b.l[s]) * 0.999, None,
                 f"bullish spike low {float(b.l[s]):.4f}")


@pattern("bearish_spike", "spikes.html", "reversal", "short", 100.0)
def _bearish_spike(b, p):
    """spikes.html — the mirror: "price spikes upward but closes near the
    intraday low. Price often forms a peak but usually does not represent a
    major or sustained turning point" — which is why the page's own caution
    about sustained moves is reflected in leaving the target unset.

      confirmation   the same "wait a day" rule
      stop           the spike's high
    """
    if b.n < p.tall_avg_bars + 3:
        return None
    s, z = b.n - 2, b.n - 1
    avg = avg_bar_height(b, s, p.tall_avg_bars)
    if avg is None or float(b.h[s] - b.l[s]) < p.spike_tall_mult * avg:
        return None
    if not _in_range(b, s, float(b.c[s]), p.quarter, False):
        return None
    if not (float(b.h[s]) > float(b.h[s - 1]) and float(b.h[s]) > float(b.h[z])):
        return None
    if p.require_volume_rules and not _heavy_volume(b, 1.0):
        return None
    return Setup("short", b.lowest(s, z), float(b.h[s]) * 1.001, None,
                 f"bearish spike high {float(b.h[s]):.4f}")


@pattern("partial_decline", "partdecline.html", "continuation", "long", 100.0)
def _partial_decline(b, p):
    """partdecline.html — inside an "established" rectangle or broadening
    formation, "price should touch the top trendline and move down but not
    touch or come that close to the bottom trendline before heading back up".
    A partial decline predicts an upward breakout.

      entry          "buy once it's clear that price is heading back toward
                     the upper trendline"
      stop           "exit your position if price bounces off the upper
                     trendline and heads back down" — the stop sits below the
                     partial decline's low
      target         the page gives no measure of its own and says to use
                     "standard breakout targets based on the underlying
                     rectangle or broadening pattern's dimensions", so the
                     parent's height is added to its top
    """
    ln = _lines(b, p, p.channel_lookback, p.channel_min_bars)
    if ln is None:
        return None
    top = (ln.top_start + ln.top_end) / 2.0
    bot = (ln.bot_start + ln.bot_end) / 2.0
    height = top - bot
    if height <= 0.0:
        return None
    # the last turn down must start at the top trendline and stop short of
    # the bottom one
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 2, p.structure_max_age)
    if pts is None:
        return None
    (hi_i, _, hi), (lo_i, _, lo) = pts
    if not near(hi, top, p.touch_tol_pct):
        return None                        # "touch the top trendline"
    if lo < bot + p.partial_clearance * height:
        return None                        # it came too close to the bottom
    return Setup("long", top, lo * 0.999, top + height,
                 f"partial decline to {lo:.4f} inside {bot:.4f}-{top:.4f}")


@pattern("partial_rise", "partrises.html", "continuation", "short", 100.0)
def _partial_rise(b, p):
    """partrises.html — the mirror of the partial decline, inside an
    "established" rectangle or broadening formation: "price should touch the
    bottom trendline and move up but not touch or come that close to the top
    trendline before heading back down". It predicts "an immediate downward
    breakout".

      entry          "short once it's clear that price is heading back
                     toward the lower trendline"
      stop           "cover your short if price bounces off the lower
                     trendline and heads back up" — above the partial rise
      target         the page states no target measurement, so none is set
    """
    ln = _lines(b, p, p.channel_lookback, p.channel_min_bars)
    if ln is None:
        return None
    top = (ln.top_start + ln.top_end) / 2.0
    bot = (ln.bot_start + ln.bot_end) / 2.0
    height = top - bot
    if height <= 0.0:
        return None
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 2, p.structure_max_age)
    if pts is None:
        return None
    (lo_i, _, lo), (hi_i, _, hi) = pts
    if not near(lo, bot, p.touch_tol_pct):
        return None                        # "touch the bottom trendline"
    if hi > top - p.partial_clearance * height:
        return None                        # it came too close to the top
    return Setup("short", bot, hi * 1.001, None,
                 f"partial rise to {hi:.4f} inside {bot:.4f}-{top:.4f}")


@pattern("pivot_point_reversal_downtrend", "PPRD.html", "reversal", "long", 77.0)
def _ppr_down(b, p):
    """PPRD.html — "two price bars because it references the prior price
    bar", in "a short-term downtrend", where "the close must be above the
    prior day's high".

      entry          a close above the pattern's top confirms the upward
                     breakout; "buy at the open the next day"
      stop           "a penny below the bottom of the pattern"
      target         "measure the height of the pattern and add it to the
                     high price" (fulfilled 77% of the time)
    """
    if b.n < p.trend_bars + 2:
        return None
    a, z = b.n - 2, b.n - 1
    if short_trend(b, a, p.trend_bars) >= 0.0:
        return None
    if float(b.c[z]) <= float(b.h[a]):
        return None
    return _small(b, a, z, "long", 1.0, "pivot point reversal (downtrend)")


@pattern("upside_weekly_reversal", "WeeklyRevsUpside.html", "reversal", "long", 70.0)
def _weekly_reversal_up(b, p):
    """WeeklyRevsUpside.html — "look for upside weekly reversals using weekly
    data"; prices "should trend downward before pattern formation"; it is a
    "two-bar pattern" whose second bar shows "a higher high and lower low (an
    outside week)"; "the last bar must close above the prior bar's high".

      entry          a buy at the open of the following week once price
                     closes above the pattern's high
      stop           "one penny below the pattern's low"
      measure rule   "the height of the second bar ... added to the high of
                     the pattern" (succeeds ~70% of the time)
    """
    if b.n < p.trend_bars + 2:
        return None
    a, z = b.n - 2, b.n - 1
    if short_trend(b, a, p.trend_bars) >= 0.0:
        return None
    if not (b.h[z] > b.h[a] and b.l[z] < b.l[a]):
        return None                        # "an outside week"
    if float(b.c[z]) <= float(b.h[a]):
        return None
    top, bot = b.highest(a, z), b.lowest(a, z)
    second = float(b.h[z] - b.l[z])
    if top <= bot or second <= 0.0:
        return None
    return Setup("long", top, bot * 0.999, top + second,
                 f"upside weekly reversal {bot:.4f}-{top:.4f}")


@pattern("downside_weekly_reversal", "WeeklyRevsDownside.html", "reversal", "short", 52.0)
def _weekly_reversal_down(b, p):
    """WeeklyRevsDownside.html — "prices should be trending up leading to the
    pattern"; "downside weekly reversals are a two-bar pattern"; "on the
    second bar of the pattern, look for a higher high and lower low (an
    outside week)"; "the last bar must close below the prior bar's low".

      entry          a close below the pattern's bottom triggers the
                     downward breakout
      stop           "a penny above the top" for downward breakouts
      measure rule   the height (highest high to lowest low) subtracted from
                     the bottom (fulfilled ~52% of the time)
    """
    if b.n < p.trend_bars + 2:
        return None
    a, z = b.n - 2, b.n - 1
    if short_trend(b, a, p.trend_bars) <= 0.0:
        return None
    if not (b.h[z] > b.h[a] and b.l[z] < b.l[a]):
        return None
    if float(b.c[z]) >= float(b.l[a]):
        return None
    return _small(b, a, z, "short", 1.0, "downside weekly reversal")


# ═══════════════════════════════════════════════════════════════════════════
# Event patterns
# ═══════════════════════════════════════════════════════════════════════════


@pattern("dead_cat_bounce", "dcb.html", "event", "short", 100.0)
def _dead_cat_bounce(b, p):
    """dcb.html — "price usually gaps downward, closing 15% to 70% lower than
    the prior day. The average event decline from prior close to trend low is
    31%." "From the event day to the trend low averages 7 days." "The average
    bounce height from event low to bounce high is 28% and takes 23 days."
    Then "price resumes declining, averaging 30% from the bounce high to
    post-bounce low in 49 days."

    The page publishes no entry, stop or target rule — it describes the
    pattern's behaviour. What is armed here is that description and nothing
    more: the short is taken when price rolls over after the bounce, the stop
    is the bounce high, and the target is the page's own published
    post-bounce decline of 30% from that high.
    """
    win = p.dcb_lookback
    if b.n < win + 3:
        return None
    lo = b.n - 1 - win
    # the event decline: a one-bar close-to-close collapse inside the window
    ev = None
    for i in range(lo + 1, b.n - p.dcb_min_bounce_bars):
        drop = -pct(float(b.c[i]), float(b.c[i - 1]))
        if p.dcb_min_decline_pct <= drop <= p.dcb_max_decline_pct:
            ev = i
    if ev is None:
        return None
    trend_low = b.lowest(ev, b.n - 1)
    bounce_i = b.arghighest(ev, b.n - 1)
    bounce = float(b.h[bounce_i])
    if trend_low <= 0.0 or bounce <= trend_low:
        return None
    if b.n - 1 - bounce_i < p.dcb_rollover_bars:
        return None                        # the bounce has not rolled over yet
    trigger = b.lowest(bounce_i, b.n - 1)
    target = bounce * (1.0 - p.dcb_post_bounce_pct / 100.0)
    if not target < trigger < bounce:
        return None
    return Setup("short", trigger, bounce * 1.001, target,
                 f"dead-cat bounce: event {pct(float(b.c[ev]), float(b.c[ev - 1])):.0f}%, "
                 f"bounce high {bounce:.4f}")


@pattern("inverted_dead_cat_bounce", "idcb.html", "event", "short", 100.0)
def _inverted_dead_cat_bounce(b, p):
    """idcb.html — "look for an event that causes price to jump at least 5%
    but it can be 20%, 50%, or even higher"; "price typically moves higher
    the day following the event"; "after that, price tends to decline".

      entry          short once the day-after higher high is in and price
                     turns down through it
      stop           above that higher high
      target         the page gives decline behaviour by event size but no
                     single target, so none is set
    """
    if b.n < 4:
        return None
    ev, nxt = b.n - 2, b.n - 1
    if pct(float(b.c[ev]), float(b.c[ev - 1])) < p.idcb_min_jump_pct:
        return None
    if float(b.h[nxt]) <= float(b.h[ev]):
        return None                        # "price typically moves higher"
    trigger = float(b.l[nxt])
    top = float(b.h[nxt])
    if trigger >= top:
        return None
    return Setup("short", trigger, top * 1.001, None,
                 f"inverted dead-cat bounce +{pct(float(b.c[ev]), float(b.c[ev - 1])):.0f}%")


# ═══════════════════════════════════════════════════════════════════════════
# Big M, big W and the roofs
# ═══════════════════════════════════════════════════════════════════════════


@pattern("big_m", "bigm.html", "reversal", "short", 55.0)
def _big_m(b, p):
    """bigm.html — "a big M shape with twin peaks and tall sides", with
    "price trend: upward leading to the pattern, often a long, straight-line
    run upward". "Look for a double top reversal pattern at the top of the
    big M"; the twin peaks have "highs less than 4% apart"; "the drop between
    the peaks of the double top is 10% to 20% or more".

      confirmation   "the pattern confirms as a valid one when price closes
                     below the lowest valley between the two tops"
      stop           "above the highest peak (A) in the pattern"
      measure rule   "compute the height from the highest peak (A) to the
                     lowest valley (B). Subtract the height from the
                     confirmation price (B)"
    """
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 3, p.structure_max_age)
    if pts is None:
        return None
    (t1, _, p1), (vi, _, valley), (t2, _, p2) = pts
    if not near(p1, p2, p.bigm_top_tol_pct):
        return None
    if pct(max(p1, p2), valley) < p.bigm_move_pct:
        return None                        # "the drop ... is 10% to 20% or more"
    if not trend_up_into(b, t1, p.trend_window, p.bigm_side_pct):
        return None                        # "tall sides"
    height = max(p1, p2) - valley
    return Setup("short", valley, max(p1, p2) * 1.001,
                 measure_short(valley, height, 55.0),
                 f"big M tops {p1:.4f}/{p2:.4f} valley {valley:.4f}")


@pattern("big_w", "bigw.html", "reversal", "long", 74.0)
def _big_w(b, p):
    """bigw.html — "a big W shape with twin bottoms and tall sides", with
    "price trend: downward leading to the pattern" and "best performing
    patterns have tall, straight declines". "Look for a double bottom
    reversal pattern at the base"; the "rise between bottoms [is] 10% to 20%
    or more"; volume "recedes 69% of the time between the two bottoms".

      confirmation   "price closes above the highest peak between bottoms"
      stop           "exit immediately if price drops below the low of the
                     second bottom (E)"
      measure rule   "compute the height from [the] highest peak to [the]
                     lowest valley (D-B) and add [the] result to [the] peak
                     high, D"
    """
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 3, p.structure_max_age)
    if pts is None:
        return None
    (v1, _, b1), (pi, _, peak), (v2, _, b2) = pts
    if not near(b1, b2, p.bottom_tol_pct):
        return None
    if pct(peak, min(b1, b2)) < p.bigm_move_pct:
        return None
    if not trend_down_into(b, v1, p.trend_window, p.bigm_side_pct):
        return None
    if p.require_volume_rules and not volume_recedes(b, v1, v2):
        return None
    height = peak - min(b1, b2)
    return Setup("long", peak, b2 * 0.999,
                 measure_long(peak, height, 74.0),
                 f"big W bottoms {b1:.4f}/{b2:.4f} peak {peak:.4f}")


@pattern("inverted_roof", "iroof.html", "reversal", "short", 47.0)
def _inverted_roof(b, p):
    """iroof.html — a "horizontal/near-horizontal top with [a] V-shaped
    bottom, resembling [a] diamond's lower half". "Price trend: can be any
    direction leading to the pattern." The "two halves should appear
    symmetrical", the bottom "V-shaped with price touching [the] top
    frequently".

      confirmation   "the pattern confirms as valid when price closes
                     outside the chart pattern trendline boundary"
      measure rule   the height x the percentage meeting target, subtracted
                     from the breakout price; down breakouts meet it 47% of
                     the time and are the side armed ("down breakouts are
                     common")
    """
    lo = max(0, b.n - 1 - p.roof_lookback)
    hi = b.n - 1
    pk = [i for i in b.peaks if lo <= i <= hi]
    vl = [i for i in b.valleys if lo <= i <= hi]
    if len(pk) < 2 or not vl:
        return None
    first = min(pk[0], vl[0])
    if hi - first < p.roof_min_bars:
        return None
    top_prices = [float(b.h[i]) for i in pk]
    if not near(min(top_prices), max(top_prices), p.roof_flat_tol_pct):
        return None                        # "horizontal/near-horizontal top"
    top = float(np.mean(top_prices))
    vi = b.arglowest(first, hi)
    bot = float(b.l[vi])
    if top <= bot:
        return None
    # "V-shaped": a narrow base, and roughly symmetrical halves
    if valley_width(b, vi, p.shape_tol_pct) > p.roof_v_max_width:
        return None
    left, right = vi - first, hi - vi
    if min(left, right) <= 0 or max(left, right) > p.symmetry_ratio * min(left, right):
        return None
    return Setup("short", bot, top * 1.001,
                 measure_short(bot, top - bot, 47.0),
                 f"inverted roof top {top:.4f} V {bot:.4f}")


@pattern("roof", "roof.html", "reversal", "short", 63.0)
def _roof(b, p):
    """roof.html — "has a horizontal or near horizontal bottom with [an] up
    sloping trend in the first part of the pattern followed by a down-sloping
    trend in the last part", i.e. an inverted V over a flat base. "Price
    trend: usually upward leading to the pattern." "The two halves of the
    roof should appear symmetrical ... with price touching the horizontal
    bottom (in [a] minor low) at least three times." "Make sure the pattern
    isn't a head-and-shoulders top or a complex head-and-shoulders top."

      confirmation   "the pattern confirms as valid when price closes
                     outside the trendline boundary"
      stop           beyond the opposite trendline
      measure rule   the pattern height (peak minus the horizontal bottom
                     low) x 63%, subtracted from the lowest low; downward
                     breakouts predominate at 58%
    """
    lo = max(0, b.n - 1 - p.roof_lookback)
    hi = b.n - 1
    pk = [i for i in b.peaks if lo <= i <= hi]
    vl = [i for i in b.valleys if lo <= i <= hi]
    if len(vl) < 3 or not pk:
        return None                        # "touching ... at least three times"
    first = min(pk[0], vl[0])
    if hi - first < p.roof_min_bars:
        return None
    lows = [float(b.l[i]) for i in vl]
    if not near(min(lows), max(lows), p.roof_flat_tol_pct):
        return None                        # "horizontal or near horizontal bottom"
    bot = float(np.mean(lows))
    pi = b.arghighest(first, hi)
    top = float(b.h[pi])
    if top <= bot:
        return None
    if peak_width(b, pi, p.shape_tol_pct) > p.roof_v_max_width:
        return None                        # an inverted V, not a broad top
    left, right = pi - first, hi - pi
    if min(left, right) <= 0 or max(left, right) > p.symmetry_ratio * min(left, right):
        return None
    if not trend_up_into(b, first, p.trend_window, p.min_trend_pct):
        return None
    return Setup("short", bot, top * 1.001,
                 measure_short(bot, top - bot, 63.0),
                 f"roof peak {top:.4f} flat base {bot:.4f}")


@pattern("trend_change_123_bullish", "123tc.html", "reversal", "long", 73.0)
def _trend_change_123_bullish(b, p):
    """123tc.html, the downtrend-reversal half — "draw a trendline from the
    highest high (point C) to the lowest low (A) on the chart such that price
    does not cross the trendline until after the lowest low (point 1)".

      point 1        "price closes above the down-sloping trendline"
      point 2        "price tests a recent low near point A ... can be below
                     point A but it must be clear that price is moving up"
      point 3        "price closes above a recent high" between A and 2 —
                     the trend change is "confirmed at point 3 when price
                     rises above the intermediate high (B)"

      stop           below point 2, the retest low
      target         the page's published outcome: "73% ... of the time price
                     climbed at least 20% from the low (A)"
    """
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 3, p.structure_max_age)
    if pts is None:
        return None
    (ai, _, a), (bi, _, inter_high), (two_i, _, two) = pts
    if inter_high <= a or two <= 0.0:
        return None
    # "can be below point A but it must be clear that price is moving up"
    if two < a * (1.0 - p.tc_retest_tol_pct / 100.0):
        return None
    # the down-sloping trendline from the highest high before A, uncrossed
    ci = b.arghighest(max(0, ai - p.tc_lookback), ai)
    if ci >= ai:
        return None
    slope, inter = linfit([ci, ai], [float(b.h[ci]), float(b.l[ai])])
    if slope >= 0.0:
        return None
    if any(float(b.c[i]) > slope * i + inter for i in range(ci, ai + 1)):
        return None                        # "price does not cross ... until after A"
    if not any(float(b.c[i]) > slope * i + inter for i in range(ai + 1, bi + 1)):
        return None                        # point 1 never happened
    return Setup("long", inter_high, two * 0.999,
                 a * (1.0 + p.tc_target_pct / 100.0),
                 f"1-2-3 trend change up: A {a:.4f} B {inter_high:.4f} 2 {two:.4f}")


@pattern("trend_change_123_bearish", "123tc.html", "reversal", "short", 43.0)
def _trend_change_123_bearish(b, p):
    """123tc.html, the uptrend-reversal half — "draw a trendline from the
    lowest low (point C) to the highest high (A) ... such that price does not
    cross the trendline until after the highest high (point 1)"; "price
    closes below the up-sloping trendline"; "price tests a recent high near
    point A ... can be above point A by a little but it must be clear that
    price is moving down"; "price closes below a recent low" between A and 2.

      stop           above point 2, the retest high
      target         the page's published outcome: "43% showed declines of at
                     least 20% below the initial high"
    """
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 3, p.structure_max_age)
    if pts is None:
        return None
    (ai, _, a), (bi, _, inter_low), (two_i, _, two) = pts
    if inter_low >= a:
        return None
    if two > a * (1.0 + p.tc_retest_tol_pct / 100.0):
        return None
    ci = b.arglowest(max(0, ai - p.tc_lookback), ai)
    if ci >= ai:
        return None
    slope, inter = linfit([ci, ai], [float(b.l[ci]), float(b.h[ai])])
    if slope <= 0.0:
        return None
    if any(float(b.c[i]) < slope * i + inter for i in range(ci, ai + 1)):
        return None
    if not any(float(b.c[i]) < slope * i + inter for i in range(ai + 1, bi + 1)):
        return None
    return Setup("short", inter_low, two * 1.001,
                 a * (1.0 - p.tc_target_pct / 100.0),
                 f"1-2-3 trend change down: A {a:.4f} B {inter_low:.4f} 2 {two:.4f}")


@pattern("two_b_top", "2B.html", "reversal", "short", 100.0)
def _two_b_top(b, p):
    """2B.html — "in an uptrend, if a higher high is made but fails to carry
    through, and then prices drop below the previous high, then the trend is
    apt to reverse." The second peak "slightly exceeds the first peak before
    reversing downward".

      entry          "enter short after price reverses below the prior high
                     following the second peak" — a sell stop at the first
                     peak's high
      stop           "above the second (higher) peak"
      target         the page's published average: "expect [a] 6-7% decline
                     when [the] first peak is above [the] second"
    """
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 3, p.structure_max_age)
    if pts is None:
        return None
    (t1, _, p1), (vi, _, valley), (t2, _, p2) = pts
    if not p2 > p1:
        return None                        # "a higher high is made"
    if pct(p2, p1) > p.two_b_excess_pct:
        return None                        # "slightly exceeds"
    if not trend_up_into(b, t1, p.trend_window, p.min_trend_pct):
        return None
    if p1 <= valley:
        return None
    return Setup("short", p1, p2 * 1.001,
                 p1 * (1.0 - p.two_b_top_decline_pct / 100.0),
                 f"2B top {p1:.4f} exceeded by {p2:.4f}")


@pattern("two_b_bottom", "2B.html", "reversal", "long", 100.0)
def _two_b_bottom(b, p):
    """2B.html — the mirror: "when a second valley forms slightly below the
    first valley in a downtrend, expect a larger subsequent advance".

      entry          "enter long after price reverses upward from the second
                     valley" — a buy stop at the first valley's low
      stop           "below the second (lower) valley"
      target         the page's published average gain for 2B bottoms, whose
                     lower bound is 32%
    """
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 3, p.structure_max_age)
    if pts is None:
        return None
    (v1, _, b1), (pi, _, peak), (v2, _, b2) = pts
    if not b2 < b1:
        return None
    if pct(b1, b2) > p.two_b_excess_pct:
        return None
    if not trend_down_into(b, v1, p.trend_window, p.min_trend_pct):
        return None
    if b1 >= peak:
        return None
    return Setup("long", b1, b2 * 0.999,
                 b1 * (1.0 + p.two_b_bottom_gain_pct / 100.0),
                 f"2B bottom {b1:.4f} undercut by {b2:.4f}")


@pattern("three_falling_peaks", "3fp.html", "reversal", "short", 23.0)
def _three_falling_peaks(b, p):
    """3fp.html — "three peaks, each one lower than the prior one", with
    price "usually upward leading to the start of the pattern". "Each peak
    should look similar to the others; peaks do NOT have to follow a
    trendline."

      confirmation   "the pattern confirms as valid when price closes below
                     the lowest valley"
      stop           "place a stop slightly above the most recent minor high"
      measure rule   the height from the highest peak to the lowest valley,
                     x 23%, subtracted from the lowest valley
    """
    ps = b.peaks
    if len(ps) < 3:
        return None
    t1, t2, t3 = ps[-3], ps[-2], ps[-1]
    h1, h2, h3 = (float(b.h[i]) for i in (t1, t2, t3))
    if not (h2 < h1 and h3 < h2):
        return None
    if not trend_up_into(b, t1, p.trend_window, p.min_trend_pct):
        return None
    vi = b.arglowest(t1, b.n - 1)
    valley = float(b.l[vi])
    height = h1 - valley
    if height <= 0.0:
        return None
    return Setup("short", valley, h3 * 1.001,
                 measure_short(valley, height, 23.0),
                 f"three falling peaks {h1:.4f}/{h2:.4f}/{h3:.4f}")


@pattern("three_rising_valleys", "3rv.html", "continuation", "long", 57.0)
def _three_rising_valleys(b, p):
    """3rv.html — "look for three valleys -- the bottom of each valley must
    be above the prior one", with price "usually upward leading to the
    pattern". "Each valley should look similar." Volume "trends downward 64%
    of the time".

      confirmation   "the pattern confirms when price closes above the
                     highest peak"
      stop           "slightly below the last minor low (point 3)"
      measure rule   the height from the highest peak to the lowest valley,
                     x 57%, added to the highest peak
    """
    vs = b.valleys
    if len(vs) < 3:
        return None
    v1, v2, v3 = vs[-3], vs[-2], vs[-1]
    l1, l2, l3 = (float(b.l[i]) for i in (v1, v2, v3))
    if not (l2 > l1 and l3 > l2):
        return None
    if not trend_up_into(b, v1, p.trend_window, p.min_trend_pct):
        return None
    if p.require_volume_rules and not volume_recedes(b, v1, v3):
        return None
    pi = b.arghighest(v1, b.n - 1)
    peak = float(b.h[pi])
    height = peak - l1
    if height <= 0.0:
        return None
    return Setup("long", peak, l3 * 0.999,
                 measure_long(peak, height, 57.0),
                 f"three rising valleys {l1:.4f}/{l2:.4f}/{l3:.4f}")


@pattern("three_lows_reversal", "3L-R.html", "reversal", "long", 56.0)
def _three_lows_reversal(b, p):
    """3L-R.html — "4 bars" forming "three lows and a reversal bar": "look
    for two consecutively lower lows (using 3 bars)" and "the last bar in the
    pattern has a high that is above the first bar".

      entry          "buy at the open the day after the last bar in the
                     pattern"
      stop           "a penny below the bottom of the 3L-R pattern"
      measure rule   "measure the height of the pattern and add it to the
                     high price" (succeeds 56% of the time)
    """
    if b.n < 5:
        return None
    i1, i2, i3, i4 = b.n - 4, b.n - 3, b.n - 2, b.n - 1
    if not (b.l[i2] < b.l[i1] and b.l[i3] < b.l[i2]):
        return None
    if float(b.h[i4]) <= float(b.h[i1]):
        return None
    return _small(b, i1, i4, "long", 1.0, "3L-R")


@pattern("inverted_three_lows_reversal", "3L-Ri.html", "reversal", "short", 45.0)
def _inverted_three_lows_reversal(b, p):
    """3L-Ri.html — "4 bars" with "three highs and a reversal bar": "look for
    two consecutively higher highs (using 3 bars) on the daily chart" and
    "the last bar in the pattern has a low that is below the first bar in the
    pattern".

      entry          "short at the open the day after the last bar in the
                     pattern"
      stop           "a penny above the top of the inverted 3L-R pattern"
      measure rule   "measure the height of the pattern and subtract it from
                     the low price" (45% success rate)
    """
    if b.n < 5:
        return None
    i1, i2, i3, i4 = b.n - 4, b.n - 3, b.n - 2, b.n - 1
    if not (b.h[i2] > b.h[i1] and b.h[i3] > b.h[i2]):
        return None
    if float(b.l[i4]) >= float(b.l[i1]):
        return None
    return _small(b, i1, i4, "short", 1.0, "inverted 3L-R")


@pattern("v_bottom", "vBottoms.html", "reversal", "long", 52.0)
def _v_bottom(b, p):
    """vBottoms.html — "look for price to make a straight-line run downward
    with few or no pauses, often fitting inside a channel", "at least 3 weeks
    to 3 months wide", with a one-day/island reversal or tail at the bottom
    "usually on heavy volume".

      breakout       "price on the right side must retrace at least 38.2% of
                     the left side ... when price retraces 38.2% of the left
                     side, then that's the breakout"
      entry          "measure the drop from A to B. When price retraces 38.2%
                     of the way up (C), it constitutes a breakout."
      target         "the original high at [the] pattern start (point A)"
    """
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 2, p.structure_max_age)
    if pts is None:
        return None
    (ai, _, a), (bi, _, low) = pts
    width = bi - ai
    if not p.v_width_min <= width <= p.v_width_max:
        return None
    drop = a - low
    if drop <= 0.0:
        return None
    trigger = low + p.v_retrace_pct / 100.0 * drop
    if trigger >= a:
        return None
    return Setup("long", trigger, low * 0.999, a,
                 f"V-bottom A {a:.4f} B {low:.4f}, 38.2% breakout {trigger:.4f}")


@pattern("v_top", "VTop.html", "reversal", "short", 37.0)
def _v_top(b, p):
    """VTop.html — "look for price to make a straight-line run upward with
    few or no pauses, often fitting inside a channel", "at least 3 weeks to 3
    months wide", with a reversal at the top "usually on heavy volume".

      breakout       "price on the right side must retrace at least 38.2% of
                     the left side"
      stop           "above the reversal point"
      target         the bottom of the pattern's start (point A), met 37% of
                     the time
    """
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 2, p.structure_max_age)
    if pts is None:
        return None
    (ai, _, a), (bi, _, high) = pts
    width = bi - ai
    if not p.v_width_min <= width <= p.v_width_max:
        return None
    rise = high - a
    if rise <= 0.0:
        return None
    trigger = high - p.v_retrace_pct / 100.0 * rise
    if trigger <= a:
        return None
    return Setup("short", trigger, high * 1.001, a,
                 f"V-top A {a:.4f} B {high:.4f}, 38.2% breakout {trigger:.4f}")


@pattern("v_pivot", "VPivot.html", "reversal", "long", 74.0)
def _v_pivot(b, p):
    """VPivot.html — a "3-bar pattern with the middle bar below the adjacent
    ones", where "the low at bar 1 is at least 2% above the low of bar 2" and
    "the low of bar 3 is at least 2% above the low of bar 2".

      confirmation   "the pattern confirms as valid when price closes above
                     the highest price in the 3 bars"
      measure rule   "compute the height from the lowest price (point A) to
                     the highest price (B) in the 3-week pattern. Add it to
                     the top of the pattern (B)" — met 74% of the time
      stop           the page names none, so the pattern's own low is used
    """
    if b.n < 4:
        return None
    a, m, z = b.n - 3, b.n - 2, b.n - 1
    base = float(b.l[m])
    if base <= 0.0:
        return None
    if pct(float(b.l[a]), base) < p.vpivot_clear_pct:
        return None
    if pct(float(b.l[z]), base) < p.vpivot_clear_pct:
        return None
    return _small(b, a, z, "long", 1.0, "V pivot")


# ═══════════════════════════════════════════════════════════════════════════
# Busted patterns (Busted.html and the ten per-pattern pages)
#
# Busted.html gives one definition for all of them: "price breaks out in one
# direction from a chart pattern, but moves no more than 10% before reversing
# and breaking out in the new direction." Every per-pattern page then repeats
# the same four numbered steps, e.g. BustDoubleBots.html:
#
#   1. "price must confirm the double bottom by closing above the top"
#   2. "price must rise no more than 10%"
#   3. "price then closes below the bottom of the double bottom"
#   4. "price continues dropping more than 10%"
#
# So `_busted` is written once against the parent detector. The parent's own
# `Setup` already carries the two levels the bust needs: its `trigger` is the
# parent's confirmation price (step 1) and its `stop` is the far side of the
# pattern that price must close through (step 3). The bust therefore trades
# the opposite side, entering at the parent's stop with the parent's trigger
# as its own stop.
# ═══════════════════════════════════════════════════════════════════════════


def _busted(b, p, parent, label):
    """The shared bust. Steps 1 and 2 must already have happened and the
    breakout must have failed back inside the pattern; the entry then rests
    at the pattern's opposite side, which is step 3.

    None of the busted pages publishes a stop or a target ("the article does
    not explicitly specify stop loss placement guidelines"), so the pattern's
    own opposite boundary is used as the stop and no limit is bracketed —
    matching the pages' "ride it until the trend reverses" example exits,
    which here is `max_hold_bars`.
    """
    # Every prefix below shares this window's pivots: a pivot at index i is
    # decided by bars i-PIVOT_SPAN..i+PIVOT_SPAN, so the prefix of length k has
    # exactly these pivots up to i < k - PIVOT_SPAN. Computing them once here
    # instead of once per prefix is what keeps the busted family affordable —
    # it is otherwise ~38x the cost of any other pattern.
    peaks, valleys = b.peaks, b.valleys
    for k in range(b.n - p.bust_min_bars, max(2, b.n - p.bust_scan_bars), -1):
        cut = k - PIVOT_SPAN
        sub = Bars(b.o[:k], b.h[:k], b.l[:k], b.c[:k], b.v[:k],
                   peaks=peaks[:bisect.bisect_left(peaks, cut)],
                   valleys=valleys[:bisect.bisect_left(valleys, cut)])
        st = parent(sub, p)
        if st is None:
            continue
        tail = slice(k, b.n)
        if st.side == "long":
            broke = np.flatnonzero(b.c[tail] > st.trigger)      # step 1
            if broke.size == 0:
                continue
            j = k + int(broke[0])
            # step 2: "price must rise no more than 10%"
            if b.highest(j, b.n - 1) > st.trigger * (1.0 + p.bust_max_move_pct / 100.0):
                continue
            if float(b.c[-1]) >= st.trigger:
                continue                    # the breakout has not failed yet
            if not st.stop < float(b.c[-1]):
                continue                    # step 3 already happened; too late
            return Setup("short", st.stop, st.trigger, None,
                          f"busted {label}: broke out at {st.trigger:.4f}, "
                          f"failed, reverse below {st.stop:.4f}")
        broke = np.flatnonzero(b.c[tail] < st.trigger)
        if broke.size == 0:
            continue
        j = k + int(broke[0])
        if b.lowest(j, b.n - 1) < st.trigger * (1.0 - p.bust_max_move_pct / 100.0):
            continue
        if float(b.c[-1]) <= st.trigger:
            continue
        if not float(b.c[-1]) < st.stop:
            continue
        return Setup("long", st.stop, st.trigger, None,
                      f"busted {label}: broke out at {st.trigger:.4f}, "
                      f"failed, reverse above {st.stop:.4f}")
    return None


def _any_double_bottom(b, p):
    """The busted pages say "double bottom" without naming an Adam/Eve
    variant, so shape is left unconstrained here."""
    return _double_bottom(b, p, "any", "any", 100.0, p.dbl_sep_min,
                          p.dbl_sep_max, p.bottom_tol_pct, "double")


def _any_double_top(b, p):
    return _double_top(b, p, "any", "any", 100.0, p.dbl_sep_min,
                       p.dbl_sep_max, p.top_tol_pct, "double")


def _any_rectangle(b, p):
    """BustRectangles.html is written for "a rectangle" generally, so either
    a rectangle bottom or a rectangle top qualifies as the parent."""
    return _rectangle(b, p, True, 79.0, "rectangle") or \
        _rectangle(b, p, False, 78.0, "rectangle")


@pattern("busted_double_bottom", "BustDoubleBots.html", "reversal", "short", 100.0)
def _busted_double_bottom(b, p):
    """BustDoubleBots.html — "price must confirm the double bottom by closing
    above the top of the double bottom"; "price must rise no more than 10%";
    "price then closes below the bottom of the double bottom"; "price
    continues dropping more than 10%".

      entry          "a conditional order to short the stock after the close
                     would get you into the stock at the open the next day"
                     once price closes below the pattern's lower boundary
    """
    return _busted(b, p, _any_double_bottom, "double bottom")


@pattern("busted_double_top", "BustDoubleTops.html", "reversal", "long", 100.0)
def _busted_double_top(b, p):
    """BustDoubleTops.html — "price must confirm the double top by closing
    below the bottom of the double top"; "price must drop no more than 10%
    below the bottom"; "price rises and closes above the top of the double
    top"; "price continues rising at more than 10%".

      entry          "a buy stop placed a penny above the price at A", the
                     top of the double top
    """
    return _busted(b, p, _any_double_top, "double top")


@pattern("busted_head_and_shoulders_bottom", "BustHSB.html", "reversal", "short", 100.0)
def _busted_hs_bottom(b, p):
    """BustHSB.html — "price must confirm the head-and-shoulders bottom by
    closing above the neckline (down sloping necklines) or above the right
    armpit"; "price must rise no more than 10% above the neckline"; "price
    then closes below the bottom of the head-and-shoulders bottom"; "price
    continues dropping more than 10% without closing above the top".

      entry          "an order to short the stock a penny below the head"
    """
    return _busted(b, p, _hs_bottom, "head-and-shoulders bottom")


@pattern("busted_head_and_shoulders_top", "BustHST.html", "reversal", "long", 100.0)
def _busted_hs_top(b, p):
    """BustHST.html — "price must confirm the head-and-shoulders top by
    closing below the neckline", drops no more than 10%, then "price rises
    and closes above the top of the head-and-shoulders top" and "continues
    rising over 10% above the pattern's apex".

      entry          "place a buy order a penny above the head (the highest
                     high in the chart pattern)"
    """
    return _busted(b, p, _hs_top, "head-and-shoulders top")


@pattern("busted_rectangle", "BustRectangles.html", "reversal", "short", 100.0)
def _busted_rectangle(b, p):
    """BustRectangles.html — a rectangle busts when "price moves no more than
    10%, reverses direction, and closes beyond the side opposite the
    breakout", then "continues moving in the new direction by more than 10%".

      entry          an order "a penny below the bottom of the rectangle"
                     (or above its top, for the mirror case)
    """
    return _busted(b, p, _any_rectangle, "rectangle")


@pattern("busted_ascending_triangle", "BustAscTriangles.html", "reversal", "short", 100.0)
def _busted_ascending_triangle(b, p):
    """BustAscTriangles.html — "price breaks out either upward or downward
    from an ascending triangle by closing outside of the trendline border";
    "price must move less than 10% before reversing"; for upward breakouts
    "price closes below the triangle's bottom"; "price continues moving in
    the new direction by at least 10%".

      entry          the page offers a penny beyond the boundary, or the
                     "safer approach" of waiting for the close beyond it —
                     the resting stop order here is the former
    """
    return _busted(b, p, _ascending_triangle, "ascending triangle")


@pattern("busted_descending_triangle", "BustDescTriangles.html", "reversal", "long", 100.0)
def _busted_descending_triangle(b, p):
    """BustDescTriangles.html — the same four-step bust applied to a
    descending triangle: a breakout closing outside a trendline, a move of
    less than 10%, a reversal closing beyond the opposite side, then at least
    10% in the new direction."""
    return _busted(b, p, _descending_triangle, "descending triangle")


@pattern("busted_symmetrical_triangle", "BustSymTriangles.html", "reversal", "short", 100.0)
def _busted_symmetrical_triangle(b, p):
    """BustSymTriangles.html — the same four-step bust applied to a
    symmetrical triangle."""
    return _busted(b, p, _symmetrical_triangle, "symmetrical triangle")


@pattern("busted_triple_bottom", "BustTripleBots.html", "reversal", "short", 100.0)
def _busted_triple_bottom(b, p):
    """BustTripleBots.html — the same four-step bust applied to a triple
    bottom: confirmation above the highest peak, a rise of no more than 10%,
    a close below the pattern's lowest valley, then more than 10% down."""
    return _busted(b, p, _triple_bottom, "triple bottom")


@pattern("busted_triple_top", "BustTripleTops.html", "reversal", "long", 100.0)
def _busted_triple_top(b, p):
    """BustTripleTops.html — the same four-step bust applied to a triple top;
    Busted.html notes the single bust "occurs 19% of the time with triple
    tops"."""
    return _busted(b, p, _triple_top, "triple top")


# ═══════════════════════════════════════════════════════════════════════════
# Throwbacks and channels
# ═══════════════════════════════════════════════════════════════════════════


@pattern("throwback", "throwbacks.html", "continuation", "long", 100.0)
def _throwback(b, p):
    """throwbacks.html — "after an upward breakout from a chart pattern,
    price lifts off then falters and returns to the launch price. This
    curling price behavior is called a throwback if it occurs within a month
    of the breakout." The stock "must zoom up, curl around, and return to (or
    near) the breakout price".

      entry          buy the curl back up off the throwback low
      stop           "place stops tightly" — the throwback's own low
      measure rule   "calculate the swing from the initial uptrend low to
                     high, then add that difference to the throwback's low
                     point"
    """
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 3, p.turn_max_age)
    if pts is None:
        return None
    (ai, _, a), (bi, _, high), (ci, _, low) = pts
    swing = high - a
    if swing <= 0.0:
        return None
    if ci - bi > p.throwback_max_bars:
        return None                        # "within a month of the breakout"
    if not near(low, a, p.throwback_tol_pct):
        return None                        # "returns to (or near) the launch price"
    trigger = b.highest(ci, b.n - 1)
    if trigger <= low:
        return None
    return Setup("long", trigger, low * 0.999, low + swing,
                 f"throwback to {low:.4f} after a {swing:.4f} swing")


def _channel(b, p, up):
    """channels.html — "a pipe tilted up or down, but not horizontal"; "the
    two trendlines should be parallel or nearly so. Both should tilt upward
    or both should tilt downward"; "price should touch each trendline at
    least twice as distinct peaks or valleys"; "price should cross the
    pattern from trendline to trendline, nearly filling the available
    space"."""
    ln = _lines(b, p, p.channel_lookback, p.channel_min_bars)
    if ln is None:
        return None
    if up and not (ln.ts > 0.0 and ln.bs > 0.0):
        return None
    if not up and not (ln.ts < 0.0 and ln.bs < 0.0):
        return None
    # "parallel or nearly so"
    scale = float(np.mean(b.c[ln.first:ln.last + 1]))
    if abs(ln.ts - ln.bs) * max(ln.last - ln.first, 1) > \
            p.channel_parallel_pct / 100.0 * scale:
        return None
    if ln.top_end <= ln.bot_end:
        return None
    return ln


@pattern("up_sloping_channel", "channels.html", "continuation", "long", 100.0)
def _up_channel(b, p):
    """channels.html, up-sloping — "buy at the lower trendline (point A)
    expecting [an] upward breakout. Exit if price reverses at the upper
    trendline (point B)."

      stop           "if price closes outside the channel in the adverse
                     direction, then close out the trade" — below the lower
                     trendline
      target         the upper trendline
    """
    ln = _channel(b, p, True)
    if ln is None:
        return None
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 1, p.structure_max_age)
    if pts is None:
        return None
    li, _, low = pts[0]
    if not near(low, ln.bs * li + ln.bi, p.touch_tol_pct):
        return None                        # the turn happened at the lower line
    trigger = b.highest(li, b.n - 1)
    if not ln.bot_end < trigger < ln.top_end:
        return None
    return Setup("long", trigger, ln.bot_end * 0.999, ln.top_end,
                 f"up channel {ln.bot_end:.4f}-{ln.top_end:.4f}")


@pattern("down_sloping_channel", "channels.html", "continuation", "short", 100.0)
def _down_channel(b, p):
    """channels.html, down-sloping — "short at the upper trendline (point A),
    covering near the bottom trendline (point B)".

      stop           above the upper trendline: "if price closes outside the
                     channel in the adverse direction, then close out the
                     trade"
      target         the lower trendline
    """
    ln = _channel(b, p, False)
    if ln is None:
        return None
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 1, p.structure_max_age)
    if pts is None:
        return None
    hi_i, _, high = pts[0]
    if not near(high, ln.ts * hi_i + ln.ti, p.touch_tol_pct):
        return None
    trigger = b.lowest(hi_i, b.n - 1)
    if not ln.bot_end < trigger < ln.top_end:
        return None
    return Setup("short", trigger, ln.top_end * 1.001, ln.bot_end,
                 f"down channel {ln.bot_end:.4f}-{ln.top_end:.4f}")


@pattern("pullback", "pullbacks.html", "continuation", "short", 100.0)
def _pullback(b, p):
    """pullbacks.html — "after a downward breakout from a chart pattern,
    price drops but then sometimes curls upward and returns to the breakout
    price or chart pattern boundary". It must "occur within 30 days of [the]
    breakout"; the stock must "descend, curve back upward, and approach the
    breakout price or pattern trendline".

      entry          the resumption of the decline off the pullback high
      stop           "tight stop placement" — the pullback high itself
      measure rule   "subtract the distance from [the] swing high (A) to
                     [the] swing low (B), then subtract that difference from
                     the pullback high (C)"
    """
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 3, p.turn_max_age)
    if pts is None:
        return None
    (ai, _, a), (bi, _, low), (ci, _, high) = pts
    swing = a - low
    if swing <= 0.0:
        return None
    if ci - bi > p.throwback_max_bars:
        return None                        # "within 30 days of breakout"
    if not near(high, a, p.throwback_tol_pct):
        return None                        # "returns to the breakout price"
    trigger = b.lowest(ci, b.n - 1)
    if trigger >= high or high - swing >= trigger:
        return None
    return Setup("short", trigger, high * 1.001, high - swing,
                 f"pullback to {high:.4f} after a {swing:.4f} swing")


@pattern("wolfe_wave_bullish", "WolfeWaveBull.html", "reversal", "long", 100.0)
def _wolfe_wave_bull(b, p):
    """WolfeWaveBull.html — five turning points:

      point 2        "any peak (minor high) on the chart"
      point 3        the valley following point 2's peak
      point 1        "the bottom prior to point 2 (top), that 3 has
                     surpassed"
      point 4        the peak after point 3, which "must be below the price
                     of 2"
      point 5        "the bottom of the hill formed by point 4"

    "Lines connecting points 1-3 and 2-4 must converge" when extended
    forward, and there must be no higher peaks between 3 and 5 nor lower
    valleys between 2 and 4.

      entry          the "sweet spot" between a line parallel to 2-4 drawn
                     from point 3 and point 5 itself — armed as the turn up
                     off point 5
      stop           "exit if price declines below the low at point 5"
      target         the EPA, "the price where the stock touches line 14
                     extended into the future"; the arrival bar is unknown,
                     so line 1-4 is read at the current bar, the earliest
                     price at which it can be touched
    """
    pts = _harmonic_points(b, p, True)
    if pts is None:
        return None
    i1, i2, i3, i4, i5, w1, w2, w3, w4, w5 = pts
    if not w3 < w1:
        return None                        # "that 3 has surpassed"
    if not w4 < w2:
        return None                        # "must be below the price of 2"
    if not w5 < w3:
        return None
    s13, c13 = linfit([i1, i3], [w1, w3])
    s24, c24 = linfit([i2, i4], [w2, w4])
    if s24 >= s13:
        return None                        # the lines do not converge forward
    trigger = b.highest(i5, b.n - 1)
    s14, c14 = linfit([i1, i4], [w1, w4])
    target = s14 * (b.n - 1) + c14
    if target <= trigger:
        return None
    return Setup("long", trigger, w5 * 0.999, target,
                 f"bullish Wolfe wave 1{w1:.4f} 2{w2:.4f} 3{w3:.4f} "
                 f"4{w4:.4f} 5{w5:.4f}, EPA {target:.4f}")


@pattern("wolfe_wave_bearish", "WolfeWaveBear.html", "reversal", "short", 100.0)
def _wolfe_wave_bear(b, p):
    """WolfeWaveBear.html — the mirror:

      point 1        "the top prior to point 2 (bottom), that 3 has
                     surpassed"
      point 2        any valley or minor low
      point 3        the peak following point 2
      point 4        the valley following point 3, "must be above point 2's
                     price"
      point 5        the peak following point 4

    "Lines 13 and 24, extended into the future, must converge. If they do
    not, then you do not have a Wolfe Wave." "There should not be another
    higher peak or lower valley between the various turning points 1
    through 5."

      entry          "sell when price enters the sweet spot"
      target         the EPA on line 1-4 extended
    """
    pts = _harmonic_points(b, p, False)
    if pts is None:
        return None
    i1, i2, i3, i4, i5, w1, w2, w3, w4, w5 = pts
    if not w3 > w1:
        return None
    if not w4 > w2:
        return None
    if not w5 > w3:
        return None
    s13, c13 = linfit([i1, i3], [w1, w3])
    s24, c24 = linfit([i2, i4], [w2, w4])
    if s24 <= s13:
        return None
    trigger = b.lowest(i5, b.n - 1)
    s14, c14 = linfit([i1, i4], [w1, w4])
    target = s14 * (b.n - 1) + c14
    if target >= trigger:
        return None
    return Setup("short", trigger, w5 * 1.001, target,
                 f"bearish Wolfe wave 1{w1:.4f} 2{w2:.4f} 3{w3:.4f} "
                 f"4{w4:.4f} 5{w5:.4f}, EPA {target:.4f}")


@pattern("simple_abc_correction", "abc.html", "continuation", "long", 100.0)
def _simple_abc_correction(b, p):
    """abc.html — "the simple ABC correction is a measured move down chart
    pattern nested inside a measured move up". "Within the MMD, points A, B,
    and C are below E. Be especially careful that B not be above E." "Only
    one MMD should be present with two straight-line runs, EA and BC."

      entry          the page's moderate tactic — "a close above the intraday
                     high at A also works as a buy signal" (the aggressive
                     one is a trendline from B, the conservative one a close
                     above B or E)
      stop           the correction low at C; the page names no stop
      target         the page gives no target, referring instead to the swing
                     rule, so none is set
    """
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 4, p.structure_max_age)
    if pts is None:
        return None
    (ei, _, e), (ai, _, a), (bi, _, bb), (ci, _, c) = pts
    if not (a < e and bb < e and c < e):
        return None                        # "A, B, and C are below E"
    if c >= a:
        return None                        # the second leg must undercut the first
    trigger = float(b.h[ai])               # "the intraday high at A"
    if trigger <= c:
        return None
    return Setup("long", trigger, c * 0.999, None,
                 f"simple ABC correction E{e:.4f} A{a:.4f} B{bb:.4f} C{c:.4f}")


@pattern("diving_board", "DivingBoard.html", "reversal", "long", 71.0)
def _diving_board(b, p):
    """DivingBoard.html — "look for price to have a flat bottom, not [a] top"
    (the board), then a "plunge" where "price makes a straight-line run down
    or nearly so", then a recovery that is "sometimes in a straight-line run
    upward". "Avoid trading patterns which make a second, lower plunge."

      entry          "when price closes above the top of the chart pattern,
                     buy" — the recovery's own breakout
      stop           the page names none, so the plunge low is used
      target         "71% of the patterns had price reaching or exceeding the
                     bottom of the diving board"
    """
    board_end = b.n - 1 - p.dive_plunge_max
    if board_end - p.dive_board_bars < 0:
        return None
    bs = board_end - p.dive_board_bars
    board_lo = b.lowest(bs, board_end)
    board_hi = b.highest(bs, board_end)
    if board_lo <= 0.0:
        return None
    if pct(board_hi, board_lo) > p.dive_board_flat_pct:
        return None                        # "a flat bottom"
    li = b.arglowest(board_end, b.n - 1)
    low = float(b.l[li])
    if pct(board_lo, low) < p.dive_plunge_pct:
        return None                        # "a swift and dramatic decline"
    if b.n - 1 - li < p.dive_recovery_bars:
        return None                        # the recovery has not started
    if b.lowest(li + 1, b.n - 1) < low:
        return None                        # "avoid ... a second, lower plunge"
    trigger = b.highest(li, b.n - 1)
    if not low < trigger < board_lo:
        return None
    return Setup("long", trigger, low * 0.999, board_lo,
                 f"diving board {board_lo:.4f}, plunge to {low:.4f}")


@pattern("flat_base", "FlatBase.html", "continuation", "long", 85.0)
def _flat_base(b, p):
    """FlatBase.html — "look for price moving horizontally"; "the tops and
    bottom of the flat base need not be horizontal as required in rectangle
    bottoms"; "the length of the flat base can be any duration ... the longer
    it is the better".

      entry          "when price closes above this line, then buy" — the
                     horizontal line along the price tops
      stop           "place a stop closer than the bottom of the pattern to
                     narrow your potential loss"; exit if "price closes below
                     the bottom of the flat base"
      measure rule   "use 85% of the height of the flat base, projected
                     upward from the top of the pattern"
    """
    n = p.flat_base_bars
    if b.n < n + 2:
        return None
    s = b.n - n
    top = b.highest(s, b.n - 1)
    bot = b.lowest(s, b.n - 1)
    if bot <= 0.0 or pct(top, bot) > p.flat_base_max_range_pct:
        return None                        # "price moving horizontally"
    height = top - bot
    if height <= 0.0:
        return None
    return Setup("long", top, bot * 0.999, top + 0.85 * height,
                 f"flat base {bot:.4f}-{top:.4f} over {n} bars")


@pattern("cats_ears", "CatsEars.html", "reversal", "short", 100.0)
def _cats_ears(b, p):
    """CatsEars.html — five phases: "look for the stock to make a severe
    decline"; "the decline stops and price moves essentially horizontally";
    a left ear (the first peak); "price pauses again between the two ears by
    moving sideways" (the scalp); and a right ear, "typically lower than [the]
    left ear". Duration is "between 10 days and 2 months (60 days)".

      confirmation   "price closes below the pattern's low" (the scalp line)
      measure rule   "take the height of the pattern from [the] highest peak
                     (A) to [the] lowest valley (B) and subtract it from the
                     value of the lowest valley (B)"
      stop           the page names none, so the highest ear is used
    """
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 3, p.structure_max_age)
    if pts is None:
        return None
    (li, _, left), (si, _, scalp), (ri, _, right) = pts
    if right >= left:
        return None                        # "typically lower than [the] left ear"
    span = ri - li
    if not p.cats_ears_min_bars <= span <= p.cats_ears_max_bars:
        return None
    if not trend_down_into(b, li, p.trend_window, p.cats_ears_decline_pct):
        return None                        # "a severe decline"
    bot = b.lowest(li, b.n - 1)
    height = left - bot
    if height <= 0.0 or bot <= 0.0:
        return None
    return Setup("short", bot, left * 1.001, bot - height,
                 f"cat's ears {left:.4f}/{right:.4f}, scalp {scalp:.4f}")


@pattern("cloud_bank", "Cloudbank.html", "reversal", "long", 100.0)
def _cloud_bank(b, p):
    """Cloudbank.html — "look for overhead resistance in which price moves
    horizontally, or almost so, in the cloud"; "the cloud bank should last
    for years, but be flexible"; "after the cloud bank ends, price should
    make a swift and dramatic decline of at least 40%"; then, using "a
    30-week SMA on weekly charts", "the lowest valley between the cloud bank
    ending and the crossover will be the lowest low".

      entry          "consider buying when price closes above the simple
                     moving average"
      stop           the page names none, so the lowest low is used
      target         "hold until the stock approaches the bottom of the cloud
                     bank"
    """
    need = p.cloud_bars + p.cloud_min_decline_bars + p.cloud_sma
    if b.n < need:
        return None
    ce = b.n - 1 - p.cloud_min_decline_bars
    cs = ce - p.cloud_bars
    if cs < 0:
        return None
    cloud_lo = b.lowest(cs, ce)
    cloud_hi = b.highest(cs, ce)
    if cloud_lo <= 0.0 or pct(cloud_hi, cloud_lo) > p.cloud_flat_pct:
        return None
    low = b.lowest(ce, b.n - 1)
    if pct(cloud_lo, low) < p.cloud_min_decline_pct:
        return None                        # "a swift and dramatic decline of at least 40%"
    ma = sma(b.c, p.cloud_sma)
    if ma is None or not low < ma < cloud_lo:
        return None
    return Setup("long", ma, low * 0.999, cloud_lo,
                 f"cloud bank {cloud_lo:.4f}-{cloud_hi:.4f}, low {low:.4f}, "
                 f"SMA{p.cloud_sma} {ma:.4f}")


@pattern("elevator_stop", "ElevatorStop.html", "other", "none", 100.0, tradeable=False)
def _elevator_stop(b, p):
    """ElevatorStop.html — "look for a strong uptrend of at least 3 price
    bars" where "each bar should have little or no overlap with the prior
    price bar. The rise is almost vertical."

    This page is a stop-management technique for a position already held, not
    an entry: "after the close of the third price bar, place a stop-loss
    order below that bar ... as each higher low appears, raise the stop" and
    "never lower the stop". It publishes no entry and no target, so the
    detector identifies the vertical run and arms nothing.
    """
    if b.n < p.elevator_bars + 1:
        return None
    idx = range(b.n - p.elevator_bars, b.n)
    for i in idx:
        if float(b.l[i]) <= float(b.l[i - 1]):
            return None                    # not a rising staircase
        if float(b.l[i]) < float(b.h[i - 1]) - p.elevator_overlap * \
                (float(b.h[i - 1]) - float(b.l[i - 1])):
            return None                    # "little or no overlap"
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Event patterns (eventpatterns.html and its per-event pages)
#
# These are triggered by news — an earnings release, an FDA approval, a
# broker's rating change. The engine has no news feed, so the announcement
# itself cannot be seen. What CAN be seen is the signature every one of these
# pages gives as its identification criteria: an announcement day with "a
# large intraday price swing, 2 or 3 times the average" on volume "above the
# 30-day average", followed by a close beyond that day's high or low.
#
# That signature is what `_event` matches. It means these detectors cannot
# distinguish a good earnings surprise from an FDA approval on price alone —
# the difference between them here is only the extra positional filter each
# page adds (near the yearly low, near the yearly high, the inbound trend)
# and the page's own measure-rule percentage. This is a real limitation, not
# a modelling choice, and it is the reason each of these detectors will fire
# on the same bar as its siblings when the filters coincide.
# ═══════════════════════════════════════════════════════════════════════════


def _yearly_position(b, i, bars):
    """Where bar `i`'s close sits in its yearly range, 0.0 at the low and 1.0
    at the high — the pages' "within a third of the yearly low/high"."""
    s = max(0, i - bars)
    hi, lo = b.highest(s, i), b.lowest(s, i)
    if hi <= lo:
        return 0.5
    return (float(b.c[i]) - lo) / (hi - lo)


def _event(b, p, up, target_pct, label, need_trend=None, near_low=False,
           near_high=False):
    """The shared event-pattern body: an announcement-day bar with a wide
    intraday swing on heavy volume, then the breakout beyond that day.

      breakout       "a breakout occurs when price closes above the high
                     posted on the announcement day" (or below its low)
      measure rule   "subtract the intraday low from the high and multiply
                     the difference by the percentage meeting price target.
                     Add the result to the intraday high"
      stop           the opposite side of the announcement day; the pages do
                     not specify one
    """
    if b.n < p.tall_avg_bars + p.event_max_age + 2:
        return None
    avg = None
    for ev in range(b.n - 1 - p.event_max_age, b.n):
        avg = avg_bar_height(b, ev, p.tall_avg_bars)
        if avg is None or avg <= 0.0:
            continue
        rng = float(b.h[ev] - b.l[ev])
        if rng < p.event_swing_mult * avg:
            continue                       # not "a large intraday price swing"
        vol = sma(b.v[:ev], p.event_volume_bars)
        if vol is None or float(b.v[ev]) < vol:
            continue                       # "heavy announcement day volume"
        if near_low and _yearly_position(b, ev, p.event_year_bars) > 1.0 / 3.0:
            continue
        if near_high and _yearly_position(b, ev, p.event_year_bars) < 2.0 / 3.0:
            continue
        if need_trend == "up" and not trend_up_into(b, ev, p.trend_window,
                                                    p.min_trend_pct):
            continue
        if need_trend == "down" and not trend_down_into(b, ev, p.trend_window,
                                                        p.min_trend_pct):
            continue
        hi, lo = float(b.h[ev]), float(b.l[ev])
        if hi <= lo:
            continue
        if up:
            return Setup("long", hi, lo * 0.999,
                         measure_long(hi, hi - lo, target_pct),
                         f"{label} swing {lo:.4f}-{hi:.4f}")
        return Setup("short", lo, hi * 1.001,
                     measure_short(lo, hi - lo, target_pct),
                     f"{label} swing {lo:.4f}-{hi:.4f}")
    return None


@pattern("good_earnings_surprise", "earnsgood.html", "event", "long", 76.0)
def _good_earnings_surprise(b, p):
    """earnsgood.html — "the company announces earnings and the stock makes a
    large upward move that day"; "look for announcements in which price makes
    a large intraday price swing, 2 or 3 times the average"; "select
    announcements that occur within a third of the yearly low"; "price trend:
    upward leading to the announcement for the best performance"; "select
    patterns with heavy announcement day volume, above the 30-day average".

      breakout       "a breakout occurs when price closes above the high
                     posted on the announcement day"
      measure rule   the announcement day's range x 76%, added to its high
    """
    return _event(b, p, True, 76.0, "good earnings surprise",
                  need_trend="up", near_low=True)


@pattern("earnings_flag", "earnflag.html", "event", "long", 86.0)
def _earnings_flag(b, p):
    """earnflag.html — the stock "makes a large upward move or gaps up when
    earnings are announced"; "look for a near vertical price run, preferably
    lasting several days" (the flagpole); then "price consolidates near the
    flagpole top, usually trending downward" (the flag). It "performs best in
    an upward price trend".

      breakout       "a breakout occurs when price pierces a flag or pennant
                     trendline or closes above the high in the pattern"
      entry          "buy when price pierces a flag or pennant trendline or
                     rises above the event pattern's high, but not before"
      measure rule   the distance from the pattern's highest high to the
                     announcement day low, x 86%, added to the flag's lowest
                     point
    """
    fl = p.flag_max_bars
    if b.n < fl + p.pole_max_bars + 2:
        return None
    pole = _flagpole(b, p, b.n - 1 - fl)
    if pole is None:
        return None
    ann_low = b.lowest(pole[0], b.n - 1 - fl)
    top = b.highest(pole[0], b.n - 1)
    flag_low = b.lowest(b.n - fl, b.n - 1)
    if top <= ann_low or flag_low <= 0.0:
        return None
    trigger = b.highest(b.n - fl, b.n - 1)
    target = flag_low + 0.86 * (top - ann_low)
    if target <= trigger:
        return None
    return Setup("long", trigger, flag_low * 0.999, target,
                 f"earnings flag, pole low {ann_low:.4f} high {top:.4f}")


@pattern("fda_drug_approval", "fda.html", "event", "long", 100.0)
def _fda_drug_approval(b, p):
    """fda.html — "news outlets report that the FDA has approved a drug";
    "look for announcements in which price makes a large intraday price
    swing, preferably 2 or 3 times the average intraday range over the last
    month"; "for best performance after an upward breakout, select
    announcements that occur within a third of the yearly high"; volume
    "above the 30-day average".

      breakout       "a breakout occurs when price closes above the high
                     posted on the announcement day"
      measure rule   "compute the height (intraday high minus the low) on the
                     announcement day and multiply it by the ... percentage
                     meeting price target. Add the result to the intraday
                     high." The page does not print that percentage, so the
                     full height is used (deviation 6).
    """
    return _event(b, p, True, 100.0, "FDA drug approval", near_high=True)


@pattern("good_same_store_sales", "sssgood.html", "event", "long", 82.0)
def _good_same_store_sales(b, p):
    """sssgood.html — "the company announces monthly or quarterly same-store
    sales numbers"; "look for announcements in which price makes a large
    intraday swing, 2 or 3 times the average daily intraday price swing over
    the last month"; volume "above the 30-day average, usually"; "usually
    found in a price uptrend (continuation patterns) but reversals perform
    better".

      breakout       "a breakout occurs when price closes above the highest
                     high posted on the announcement day"
      measure rule   "subtract the intraday low (point A) from the high (B)
                     and multiply it by the ... percentage meeting price
                     target. Add the result to the intraday high (B)"
    """
    return _event(b, p, True, 82.0, "good same-store sales", need_trend="up")


@pattern("bad_same_store_sales", "sssbad.html", "event", "short", 68.0)
def _bad_same_store_sales(b, p):
    """sssbad.html — the same announcement with the opposite outcome:
    "usually found in a price downtrend", the same wide-swing and
    above-average-volume criteria, and "works best in a bear market".

      breakout       "a downward breakout (confirmation) happens when price
                     closes below the low posted on the announcement day"
      entry          "sell a long holding or short the stock after
                     confirmation"
      measure rule   "subtract the intraday low (point B) from the high (A)
                     ... subtract the result from the intraday low (B)"
    """
    return _event(b, p, False, 68.0, "bad same-store sales", need_trend="down")


@pattern("stock_downgrade", "downgrade.html", "event", "short", 69.0)
def _stock_downgrade(b, p):
    """downgrade.html — "a broker downgrades the stock and makes the
    information public"; "look for announcements in which price makes a large
    intraday swing, 2 or 3 times the average daily intraday price swing over
    the last month"; volume "heavy on the announcement day".

      breakout       "the breakout is usually downward, and it occurs when
                     price closes below the low made on the announcement day"
      measure rule   the announcement day's range x 69% for downward
                     breakouts, subtracted from the intraday low
    """
    return _event(b, p, False, 69.0, "stock downgrade")


@pattern("stock_upgrade", "upgrade.html", "event", "long", 81.0)
def _stock_upgrade(b, p):
    """upgrade.html — "a broker publicly upgrades the stock" with "a large
    intraday swing, 2 or 3 times the average daily intraday price swing";
    volume is "heavy on the announcement day".

      breakout       "usually upward ... when price closes above the
                     announcement day high"; the page advises waiting for
                     confirmation "since the breakout can be in any
                     direction"
      measure rule   the announcement day's range x 81% for upward breakouts,
                     added to the high
    """
    return _event(b, p, True, 81.0, "stock upgrade")


@pattern("bad_earnings_surprise", "earnsbad.html", "event", "short", 69.0)
def _bad_earnings_surprise(b, p):
    """earnsbad.html — "price trend: downward leading to the announcement for
    the best performance"; the stock "makes a large downward move on or after
    earnings"; "look for announcements in which price makes a large intraday
    swing, 2 or 3 times the average daily intraday price range"; "select
    announcements within a third of the yearly low"; "trade only those
    announcements making a downward breakout".

      entry          "wait for price to confirm the pattern because traders
                     may push price up instead. A downward breakout
                     (confirmation) happens when price closes below the low
                     posted on the announcement day."
      measure rule   "subtract the intraday low (B) from the high (A) and
                     multiply it by the ... percentage meeting price target
                     [69%]. Subtract the result from the intraday low (B)"
    """
    return _event(b, p, False, 69.0, "bad earnings surprise",
                  need_trend="down", near_low=True)


@pattern("dutch_auction_tender_offer", "dutchep.html", "event", "short", 100.0)
def _dutch_auction_tender_offer(b, p):
    """dutchep.html — "the company announces a 'modified Dutch auction tender
    offer' for shares" at a price range, and "the predominant volume shape is
    U, regardless of the breakout direction". The offer itself is invisible
    to the engine; the U-shaped volume over the offer period is not, and is
    what is matched here.

      entry          "wait for the breakout before taking a position. A
                     breakout occurs when price closes above the highest peak
                     (A) or below the lowest valley (B) in the pattern
                     period" — the downward side is armed, which the page
                     reports happens "58% of the time"
      measure rule   "compute the height from the highest peak (A) to the
                     lowest valley (B) during the tender offer period then
                     multiply it by the ... percentage meeting price target"
    """
    n = p.dutch_bars
    if b.n < n + 5:
        return None
    s = b.n - n
    if not _u_shaped(b.v[s:b.n], p.vol_shape_ratio):
        return None
    top, bot = b.highest(s, b.n - 1), b.lowest(s, b.n - 1)
    if top <= bot:
        return None
    return Setup("short", bot, top * 1.001, bot - (top - bot),
                 f"Dutch auction tender offer {bot:.4f}-{top:.4f}, U volume")


# ═══════════════════════════════════════════════════════════════════════════
# Volume patterns (volpatterns.html, volshapes.html, voltrend.html,
# volbkout.html)
#
# These four pages describe properties OF a chart pattern's volume, not
# patterns you enter on: each says which shape or trend makes which parent
# patterns perform better. None publishes an entry, a stop or a target, so
# all of them are identification-only (`tradeable=False`), exactly as the
# area and ex-dividend gaps are.
# ═══════════════════════════════════════════════════════════════════════════


def _thirds(v):
    n = len(v)
    if n < 6:
        return None
    k = n // 3
    return float(np.mean(v[:k])), float(np.mean(v[k:n - k])), float(np.mean(v[n - k:]))


def _u_shaped(v, ratio):
    """volshapes.html — "volume is higher at the ends of the chart pattern
    than in the middle"."""
    t = _thirds(v)
    if t is None:
        return False
    left, mid, right = t
    return mid > 0.0 and min(left, right) >= ratio * mid


def _dome_shaped(v, ratio):
    """volshapes.html — "an inverted U-shape where volume is higher in the
    middle of the chart pattern than at the ends"."""
    t = _thirds(v)
    if t is None:
        return False
    left, mid, right = t
    return max(left, right) > 0.0 and mid >= ratio * max(left, right)


@pattern("u_shaped_volume", "volshapes.html", "other", "none", 100.0, tradeable=False)
def _u_shaped_volume(b, p):
    """volshapes.html, U-shaped volume — "volume is higher at the ends of the
    chart pattern than in the middle", with the volume peaks typically
    aligning with the price turning points. The page reports it "shows the
    strongest performance track record, appearing in 27 chart patterns".

    It describes a parent pattern's volume, not a trade: the page publishes
    no entry, stop or target, so nothing is armed.
    """
    n = min(b.n, p.vol_shape_bars)
    _u_shaped(b.v[b.n - n:], p.vol_shape_ratio)
    return None


@pattern("dome_shaped_volume", "volshapes.html", "other", "none", 100.0, tradeable=False)
def _dome_shaped_volume(b, p):
    """volshapes.html, dome-shaped volume — "an inverted U-shape where volume
    is higher in the middle of the chart pattern than at the ends". It
    "appears in 17 patterns, the smallest category", favouring diamond
    bottoms and certain scallops. Identification only."""
    n = min(b.n, p.vol_shape_bars)
    _dome_shaped(b.v[b.n - n:], p.vol_shape_ratio)
    return None


@pattern("random_shaped_volume", "volshapes.html", "other", "none", 100.0, tradeable=False)
def _random_shaped_volume(b, p):
    """volshapes.html, random-shaped volume — "a random volume shape is one
    that is not domed and not U-shaped. It may be flat, inclined, declined,
    or just have a rugged appearance with no discernable shape." It "applies
    to 20 patterns". Identification only."""
    n = min(b.n, p.vol_shape_bars)
    v = b.v[b.n - n:]
    (not _u_shaped(v, p.vol_shape_ratio)) and (not _dome_shaped(v, p.vol_shape_ratio))
    return None


@pattern("rising_volume_trend", "voltrend.html", "other", "none", 100.0, tradeable=False)
def _rising_volume_trend(b, p):
    """voltrend.html, rising volume trend — "the volume line's slope,
    determined through linear regression from the chart pattern's start to
    end, moves upward". Thirty chart-pattern variations perform better with
    it. The page separates volume TREND (the slope) from volume SHAPE (U,
    dome or random) — "a pattern may display a particular shape while
    simultaneously exhibiting either an upward or downward trend".
    Identification only."""
    n = min(b.n, p.vol_shape_bars)
    (not volume_recedes(b, b.n - n, b.n - 1))
    return None


@pattern("falling_volume_trend", "voltrend.html", "other", "none", 100.0, tradeable=False)
def _falling_volume_trend(b, p):
    """voltrend.html, falling volume trend — the same linear-regression slope
    "slopes downward". Thirty-one chart-pattern variations perform better
    under this condition, "such as double bottoms, flags, and various
    triangle and wedge configurations". Identification only."""
    n = min(b.n, p.vol_shape_bars)
    volume_recedes(b, b.n - n, b.n - 1)
    return None


@pattern("breakout_day_volume", "volbkout.html", "other", "none", 100.0, tradeable=False)
def _breakout_day_volume(b, p):
    """volbkout.html — breakout day volume is measured "by comparing trading
    volume on the breakout day to the 30-day average volume": heavy volume is
    a spike "significantly above the prior month's average", light volume
    "remains below the 30-day average". Heavy breakouts favour triangles, V
    bottoms, broadening formations and rounded bottoms; light ones favour
    flags and certain double bottoms; "some patterns demonstrate no volume
    preference". Identification only — the page gives no entry of its own."""
    _heavy_volume(b, 1.0)
    return None


@pattern("volume_patterns", "volpatterns.html", "other", "none", 100.0, tradeable=False)
def _volume_patterns(b, p):
    """volpatterns.html — the index page tying the volume shape, volume trend
    and breakout-day volume studies together. It is a guide to which volume
    behaviour suits which parent pattern rather than a pattern of its own, so
    like its three child pages it arms nothing."""
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Trading setups
# ═══════════════════════════════════════════════════════════════════════════


@pattern("adam_white_setup", "AdamWhiteSetup.html", "continuation", "long", 100.0)
def _adam_white_setup(b, p):
    """AdamWhiteSetup.html — the entry, given as a MetaStock formula:

        when(llv(L,5),>,llv(L,13)) AND when(H,=,hhv(H,5))

    "Buy when the lowest low of the past 5 weeks exceeds the lowest low of
    the past 13 weeks, AND when the current week's high equals the highest
    high of the past 5 weeks", which "signals higher highs and higher lows".

      exit           "(hhv(L,13)-L)/L" with an optimal threshold of 5% — exit
                     on a 5% retrace from the highest low of the past 13
                     weeks, which is the stop used here
      TAI filter     the page's second exit term switches the retrace exit
                     off during strong uptrends. A resting stop order cannot
                     be conditionally disabled, so the retrace stop is always
                     live here; that is a deviation from the page.
      target         none published
    """
    if b.n < p.aw_long + 2:
        return None
    if b.lowest(b.n - p.aw_short, b.n - 1) <= b.lowest(b.n - p.aw_long, b.n - 1):
        return None
    top = b.highest(b.n - p.aw_short, b.n - 1)
    if float(b.h[-1]) < top:
        return None                        # "H = hhv(H,5)"
    hhv_low = float(np.max(b.l[b.n - p.aw_long:b.n]))
    stop = hhv_low * (1.0 - p.aw_retrace_pct / 100.0)
    if stop >= top:
        return None
    return Setup("long", top, stop, None,
                 f"Adam White setup, retrace stop {stop:.4f}")


@pattern("ascending_triangle_setup", "AscTriangleSetup.html", "continuation", "long", 100.0)
def _ascending_triangle_setup(b, p):
    """AscTriangleSetup.html — the tested setup on an ascending triangle,
    which "breaks out upward 64% of the time".

      entry          "trade entry uses a buy stop a penny above the top of
                     the formation (a penny above the highest high in the
                     chart pattern)"
      stop           the page's best-performing variant "uses no stop"; the
                     engine always brackets one, so its listed alternative,
                     "a stop loss order a penny below the formation's low",
                     is used
      exit           "exit at the close 3 trading days after entry" — the
                     page's top-ranked rule, carried on the setup as its own
                     time stop
      target         none; the exit is time-based
    """
    t = _triangle(b, p, "flat", "up")
    if t is None:
        return None
    _, top_end, _, bot_end, first = t
    trigger = b.highest(first, b.n - 1)
    low = b.lowest(first, b.n - 1)
    if low >= trigger:
        return None
    return Setup("long", trigger, low * 0.999, None,
                 f"ascending triangle setup, 3-bar exit", hold_bars=p.ast_hold_bars)


@pattern("cp_setup", "CPSetup.html", "continuation", "long", 87.0)
def _cp_setup(b, p):
    """CPSetup.html — four rules:

      rise           "look for a long-term upward run leading to the start of
                     the flat bottom" (median 6 months)
      flat bottom    "price forms a top, but the bottom of this pattern is
                     flat with at least two touches" of support
      dip            price drops at the pattern's end, "appearing to breach
                     support"
      breakout       "price rises and closes above the top of the chart
                     pattern"

      entry          "the stock must close above the top of the chart
                     pattern"
      stop           below the dip low (point C), which the page reports is
                     hit only 2% of the time
      measure rule   the pattern height from top to dip low added to the
                     breakout; "price reaches the target 87% of the time"
    """
    n = min(b.n - 1, p.cp_lookback)
    s = b.n - 1 - n
    vl = [i for i in b.valleys if s <= i <= b.n - 1]
    if len(vl) < 2:
        return None                        # "at least two touches"
    lows = [float(b.l[i]) for i in vl]
    if not near(min(lows), max(lows), p.cp_flat_tol_pct):
        return None                        # "the bottom of this pattern is flat"
    support = float(np.mean(lows))
    dip = b.lowest(vl[-1], b.n - 1)
    if dip >= support:
        return None                        # the dip must breach support
    top = b.highest(s, b.n - 1)
    if top <= support:
        return None
    if not trend_up_into(b, s, p.trend_window, p.min_trend_pct):
        return None                        # "a long-term upward run"
    return Setup("long", top, dip * 0.999,
                 measure_long(top, top - dip, 87.0),
                 f"CPSetup flat bottom {support:.4f}, dip {dip:.4f}, top {top:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Vertical runs and the inverted V pivot
# ═══════════════════════════════════════════════════════════════════════════


def _vertical_run(b, p, up):
    """VerticalRunUp.html / VerticalRunDown.html — "price moves up [down] in
    a steep run [drop] for at least four sessions" with "minimal overlap from
    price bar to bar". Returns the run's first index, or None."""
    n = p.vrun_min_bars
    s = b.n - 1 - n
    if s < 1:
        return None
    for i in range(s + 1, b.n):
        prev_range = float(b.h[i - 1] - b.l[i - 1])
        if prev_range <= 0.0:
            return None
        if up:
            if float(b.c[i]) <= float(b.c[i - 1]):
                return None
            if float(b.l[i]) < float(b.h[i - 1]) - p.vrun_overlap * prev_range:
                return None
        else:
            if float(b.c[i]) >= float(b.c[i - 1]):
                return None
            if float(b.h[i]) > float(b.l[i - 1]) + p.vrun_overlap * prev_range:
                return None
    return s


@pattern("vertical_run_up", "VerticalRunUp.html", "reversal", "short", 79.0)
def _vertical_run_up(b, p):
    """VerticalRunUp.html — "price moves up in a steep run for at least four
    sessions"; "there is minimal overlap from price bar to bar. However, the
    median overlap is 33%."

      signal         "use a close above the top or below the bottom of the
                     last price bar in the vertical run as the trading
                     signal"
      target         "price retraces a median of 52% of the move roughly 79%
                     of the time, with 22% experiencing full retracement",
                     against 21% that keep climbing — so the retrace is the
                     side armed, entering below the last bar of the run
      stop           the run's high
    """
    s = _vertical_run(b, p, True)
    if s is None:
        return None
    low = b.lowest(s, b.n - 1)
    high = b.highest(s, b.n - 1)
    if high <= low:
        return None
    trigger = float(b.l[-1])
    target = high - p.vrun_retrace_pct / 100.0 * (high - low)
    if not target < trigger < high:
        return None
    return Setup("short", trigger, high * 1.001, target,
                 f"vertical run up {low:.4f}-{high:.4f}, 52% retrace target")


@pattern("vertical_run_down", "VerticalRunDown.html", "reversal", "long", 64.0)
def _vertical_run_down(b, p):
    """VerticalRunDown.html — "price moves down in a steep drop for at least
    four sessions" with "minimal overlap from price bar to bar".

      entry          "place a buy stop a penny or two above the high of the
                     prior price bar"
      stop           "a penny or two below" the entry bar, "then raised as
                     price climbs"
      target         "set a target that is half the distance up the vertical
                     run", which works "64% of the time"
    """
    s = _vertical_run(b, p, False)
    if s is None:
        return None
    low = b.lowest(s, b.n - 1)
    high = b.highest(s, b.n - 1)
    if high <= low:
        return None
    trigger = float(b.h[-1])
    target = low + 0.5 * (high - low)
    if not low < trigger < target:
        return None
    return Setup("long", trigger, low * 0.999, target,
                 f"vertical run down {high:.4f}-{low:.4f}, half-way target")


@pattern("inverted_v_pivot", "InvVPivot.html", "reversal", "short", 41.0)
def _inverted_v_pivot(b, p):
    """InvVPivot.html — "looks like an inverted V, a 3-bar pattern with the
    middle bar (2) above the adjacent ones (1, 3)". "For identification, I
    put a 2% minimum on each price bar. By that, I mean the high price of bar
    2 is at least 2% above the high prices of bars 1 and 3." Price trends
    "usually (63% of the time) upward leading to the pattern".

      confirmation   the pattern confirms when "price closes below the lowest
                     price of the three bars"
      measure rule   the height from the lowest price (B) to the highest (A)
                     subtracted from B; "price reaches the target 41% of the
                     time on average, so be conservative"
    """
    if b.n < 4:
        return None
    a, m, z = b.n - 3, b.n - 2, b.n - 1
    peak = float(b.h[m])
    if pct(peak, float(b.h[a])) < p.vpivot_clear_pct:
        return None
    if pct(peak, float(b.h[z])) < p.vpivot_clear_pct:
        return None
    return _small(b, a, z, "short", 1.0, "inverted V pivot")


# ═══════════════════════════════════════════════════════════════════════════
# Elliott wave patterns (Elliott.html and its children)
#
# These pages publish wave RULES, not trading tactics — most say outright
# that they "focus on pattern identification and structural rules rather than
# entry/exit strategies". Two of their rules are nevertheless directional and
# are what the entries below rest on:
#
#   EWCorrective.html: "the corrective phase aligns against the trend of one
#   higher degree (a counter trend move)" — so a completed correction is
#   followed by a resumption of the higher-degree trend, and the entry is the
#   break out of the correction in that direction.
#
#   EWBasic.html: the motive phase is five waves, after which (per
#   EWCorrective.html) a three-wave correction follows — so a completed
#   motive wave is armed counter-trend.
#
# No Elliott page publishes a measure rule, so none of these sets a target;
# they leave on their stop or on the time stop. Where a page publishes
# neither a direction nor a completion point, the detector identifies the
# structure and arms nothing.
# ═══════════════════════════════════════════════════════════════════════════


def _ew_abc(b, p):
    """The four turns shared by every three-wave correction of a bull-market
    advance: the start of wave A (a peak), A (a valley), B (a peak) and C (a
    valley). Returns (indices..., prices...) or None."""
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 4, p.structure_max_age)
    if pts is None:
        return None
    return [q[0] for q in pts] + [q[2] for q in pts]


def _ew_resume(b, p, ci, c, bp, label):
    """The resumption entry shared by the corrective patterns: the
    higher-degree trend was up, the correction is counter-trend, so the trade
    is the break back above wave B once wave C has turned."""
    trigger = max(bp, b.highest(ci, b.n - 1))
    if trigger <= c:
        return None
    return Setup("long", trigger, c * 0.999, None, label)


@pattern("elliott_corrective_phase", "EWCorrective.html", "reversal", "long", 100.0)
def _elliott_corrective_phase(b, p):
    """EWCorrective.html — "the corrective phase is composed of three waves
    and never five"; "corrective waves can head up or down"; "the corrective
    phase aligns against the trend of one higher degree (a counter trend
    move)"; "an initial five wave move against the prevailing price trend is
    not the end of a correction."

    This is the parent category of the zigzag, flat and triangle corrections
    below: any completed three-wave counter-trend move. The entry is the
    resumption of the higher-degree trend once wave C has turned.
    """
    pts = _ew_abc(b, p)
    if pts is None:
        return None
    si, ai, bi, ci, s, a, bp, c = pts
    if not (a < s and c < bp):
        return None                        # a genuine three-wave decline
    if not trend_up_into(b, si, p.trend_window, p.min_trend_pct):
        return None                        # "the trend of one higher degree"
    return _ew_resume(b, p, ci, c, bp,
                      f"corrective phase A{a:.4f} B{bp:.4f} C{c:.4f}")


@pattern("elliott_zigzag", "EWZigzag.html", "reversal", "long", 100.0)
def _elliott_zigzag(b, p):
    """EWZigzag.html — "the zigzag is an ABC correction of the motive wave";
    "the pattern follows the 5-3-5 subwave configuration"; "subwave B falls
    well short of the start of subwave A."

    The subwave counts are one degree below what daily bars resolve, so what
    is matched here is the rule that separates a zigzag from a flat: B
    falling well short of A's start.
    """
    pts = _ew_abc(b, p)
    if pts is None:
        return None
    si, ai, bi, ci, s, a, bp, c = pts
    if a >= s or c >= bp:
        return None
    span = s - a
    if span <= 0.0:
        return None
    if bp > s - p.ew_short_pct / 100.0 * span:
        return None                        # B must fall "well short" of the start
    return _ew_resume(b, p, ci, c, bp,
                      f"zigzag A{a:.4f} B{bp:.4f} C{c:.4f}")


@pattern("elliott_double_zigzag", "EWDoubleZigzag.html", "reversal", "long", 100.0)
def _elliott_double_zigzag(b, p):
    """EWDoubleZigzag.html — "a double zigzag are two zigzag patterns coupled
    together by a 'three'"; "subwave B in each zigzag falls well short of the
    start of subwave A." Structurally that is two 5-3-5 patterns joined by a
    connecting three-wave move, so eight turns on this scale."""
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 8, p.structure_max_age)
    if pts is None:
        return None
    px = [q[2] for q in pts]
    s1, a1, b1, c1, x, a2, b2, c2 = px
    for start, a, bb in ((s1, a1, b1), (x, a2, b2)):
        span = start - a
        if span <= 0.0 or bb > start - p.ew_short_pct / 100.0 * span:
            return None                    # each B "falls well short"
    if not (c1 < b1 and c2 < b2 and c2 < c1):
        return None
    return _ew_resume(b, p, pts[-1][0], c2, b2,
                      f"double zigzag C1 {c1:.4f} C2 {c2:.4f}")


def _ew_flat(b, p, b_beyond, c_beyond, label):
    """The 3-3-5 flats differ only in where wave B ends relative to the start
    of wave A, and where wave C ends relative to the end of wave A:

      flat            "wave B terminates near the start of wave A"; "wave C
                      terminates near the end of wave A, often slightly
                      beyond"
      expanded flat   "wave B terminates beyond the start of wave A"; "wave C
                      terminates beyond the end of wave A"
      running flat    "wave B terminates beyond the start of wave A"; "wave C
                      terminates before the end of wave A"
    """
    pts = _ew_abc(b, p)
    if pts is None:
        return None
    si, ai, bi, ci, s, a, bp, c = pts
    if a >= s:
        return None
    tol = p.ew_near_pct / 100.0 * (s - a)
    if b_beyond == "near" and abs(bp - s) > tol:
        return None
    if b_beyond == "beyond" and bp <= s + tol:
        return None
    if c_beyond == "near" and abs(c - a) > tol:
        return None
    if c_beyond == "beyond" and c >= a - tol:
        return None
    if c_beyond == "before" and c <= a + tol:
        return None
    return _ew_resume(b, p, ci, c, bp, f"{label} A{a:.4f} B{bp:.4f} C{c:.4f}")


@pattern("elliott_flat", "EWFlat.html", "reversal", "long", 100.0)
def _elliott_flat(b, p):
    """EWFlat.html — a 3-3-5 correction where "wave B terminates near the
    start of wave A" and "wave C terminates near the end of wave A, often
    slightly beyond"; "wave C does not go much beyond the end of A or the
    wave is a zigzag". "You tend to see flats in wave four and not wave two
    of a motive wave", and "the more powerful the existing trend, the shorter
    the flat tends to be"."""
    return _ew_flat(b, p, "near", "near", "flat")


@pattern("elliott_expanded_flat", "EWExpanded.html", "reversal", "long", 100.0)
def _elliott_expanded_flat(b, p):
    """EWExpanded.html — the 3-3-5 correction where "wave B terminates beyond
    the start of wave A" and "wave C terminates beyond the end of wave A"."""
    return _ew_flat(b, p, "beyond", "beyond", "expanded flat")


@pattern("elliott_running_flat", "EWRunning.html", "reversal", "long", 100.0)
def _elliott_running_flat(b, p):
    """EWRunning.html — "wave B terminates beyond the start of wave A" and
    "wave C terminates before the end of wave A". "A running flat is rare so
    look for the 3-3-5 subwave combination."""
    return _ew_flat(b, p, "beyond", "before", "running flat")


def _ew_motive(b, p):
    """EWBasic.html — the five-wave motive phase of an advance, as six turns:
    the start (a valley), then waves 1 (peak), 2 (valley), 3 (peak), 4
    (valley) and 5 (peak)."""
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 6, p.structure_max_age)
    if pts is None:
        return None
    return [q[0] for q in pts] + [q[2] for q in pts]


@pattern("elliott_motive_wave", "EWBasic.html", "reversal", "short", 100.0)
def _elliott_motive_wave(b, p):
    """EWBasic.html — "the motive phase is composed of five waves, three
    advancing (1, 3, 5) and two counter trend waves, 2 and 4", and they
    "align with the trend of one higher degree".

      rule           "wave 2 never moves beyond the start of wave 1"
      rule           "wave 3 is never the shortest wave"
      rule           "wave 4 never overlaps the end of wave 1"

    A completed motive phase is followed by the three-wave correction of
    EWCorrective.html, so the entry armed here is that correction: a break
    below wave 4's low, stopped above wave 5.
    """
    pts = _ew_motive(b, p)
    if pts is None:
        return None
    si, i1, i2, i3, i4, i5, s, w1, w2, w3, w4, w5 = pts
    if w2 <= s:
        return None                        # "wave 2 never moves beyond the start of wave 1"
    if w4 <= w1:
        return None                        # "wave 4 never overlaps the end of wave 1"
    legs = (w1 - s, w3 - w2, w5 - w4)
    if min(legs) <= 0.0 or legs[1] == min(legs):
        return None                        # "wave 3 is never the shortest wave"
    trigger = min(w4, b.lowest(i5, b.n - 1))
    if trigger >= w5:
        return None
    return Setup("short", trigger, w5 * 1.001, None,
                 f"motive wave 1{w1:.4f} 3{w3:.4f} 5{w5:.4f}")


@pattern("elliott_truncation", "EWTruncation.html", "reversal", "short", 100.0)
def _elliott_truncation(b, p):
    """EWTruncation.html — "the truncation is a motive wave that fails to
    complete the trend"; "subwave five [fails] to move beyond the end of wave
    three"; "the pattern has a 5-3-5-3-5 subwave structure". "You will see a
    truncated fifth often after an unusually strong wave 3 thrust", which the
    page reads as trend exhaustion — hence the counter-trend entry below wave
    4's low."""
    pts = _ew_motive(b, p)
    if pts is None:
        return None
    si, i1, i2, i3, i4, i5, s, w1, w2, w3, w4, w5 = pts
    if w2 <= s or w4 <= w1:
        return None
    if w5 >= w3:
        return None                        # the truncated fifth
    trigger = min(w4, b.lowest(i5, b.n - 1))
    if trigger >= w5:
        return None
    return Setup("short", trigger, w5 * 1.001, None,
                 f"truncated fifth: 3 {w3:.4f}, 5 {w5:.4f}")


@pattern("elliott_leading_diagonal_triangle", "EWleadingTriangle.html", "continuation", "long", 100.0)
def _elliott_leading_diagonal(b, p):
    """EWleadingTriangle.html — "the subwave action usually follows two
    converging trendlines"; "subwave four often overlaps subwave 1"; "the
    subwave count is 5-3-5-3-5"; "the leading diagonal triangle usually
    occurs as part of wave one of impulses or wave A of zigzags". The key
    recognition factor is that "the fifth subwave shows decided slowing of
    price change relative to the third".

    Because it "occurs as part of wave one of impulses", the advance is
    expected to continue, so the entry is the break above wave 5.
    """
    pts = _ew_motive(b, p)
    if pts is None:
        return None
    si, i1, i2, i3, i4, i5, s, w1, w2, w3, w4, w5 = pts
    if w4 >= w1:
        return None                        # "subwave four often overlaps subwave 1"
    ts, ti = linfit([i1, i3, i5], [w1, w3, w5])
    bs, bi_ = linfit([si, i2, i4], [s, w2, w4])
    if (ts * si + ti) - (bs * si + bi_) <= (ts * i5 + ti) - (bs * i5 + bi_):
        return None                        # "two converging trendlines"
    slope3 = (w3 - w2) / max(i3 - i2, 1)
    slope5 = (w5 - w4) / max(i5 - i4, 1)
    if slope5 >= slope3:
        return None                        # "decided slowing ... relative to the third"
    trigger = b.highest(i5, b.n - 1)
    if trigger <= w4:
        return None
    return Setup("long", trigger, w4 * 0.999, None,
                 f"leading diagonal triangle 1{w1:.4f} 3{w3:.4f} 5{w5:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Elliott wave triangles and extensions
#
# The five Elliott triangle pages share a structure — "five waves compose the
# triangle (A-B-C-D-E)", "each of the A-B-C-D-E waves are composed of three
# subwaves, so it has a 3-3-3-3-3 configuration", and "volume and volatility
# tend to recede over the life of the pattern, but this is not a
# requirement" — and share one tactic, quoting Frost and Prechter: "when
# price reaches the apex of the triangle, expect the market to turn."
#
# None of them names a breakout direction. Bulkowski's own convention for
# that is to cross-reference the classic pattern page — msymtri.html does it
# explicitly ("I won't describe what to look for. You can find that
# information on the symmetrical triangles page of this site. Symmetricals
# appear the same regardless of the time scale you select.") — so the side
# armed for each is the one its classic page names under "Breakout", and the
# docstring says which.
# ═══════════════════════════════════════════════════════════════════════════


def _ew_triangle(b, p, top_kind, bot_kind, diverging=False):
    """The shared A-B-C-D-E triangle: the two trendlines of `_lines`, plus
    the page's requirement that "five waves compose the triangle"."""
    ln = _lines(b, p, p.channel_lookback, p.channel_min_bars)
    if ln is None:
        return None
    z = [q for q in zigzag(b) if ln.first <= q[0] <= ln.last]
    if len(z) < 5:
        return None                        # "five waves compose the triangle"
    if top_kind == "flat" and not _flat(b, p, ln.ts, ln.first, ln.last):
        return None
    if top_kind == "down" and ln.ts >= 0.0:
        return None
    if top_kind == "up" and ln.ts <= 0.0:
        return None
    if bot_kind == "flat" and not _flat(b, p, ln.bs, ln.first, ln.last):
        return None
    if bot_kind == "up" and ln.bs <= 0.0:
        return None
    if bot_kind == "down" and ln.bs >= 0.0:
        return None
    open_start = ln.top_start - ln.bot_start
    open_end = ln.top_end - ln.bot_end
    if open_end <= 0.0:
        return None
    if diverging and open_end <= open_start:
        return None
    if not diverging and open_end >= open_start:
        return None
    return ln


@pattern("elliott_ascending_triangle", "EWTriangleAscending.html", "continuation", "long", 100.0)
def _ew_ascending_triangle(b, p):
    """EWTriangleAscending.html — "the tops of the waves peak near the same
    price, following a horizontal trendline"; "the bottoms of the waves
    generally follow an up-sloping trendline"; "five waves compose the
    ascending triangle (A-B-C-D-E), unless extended"; the 3-3-3-3-3 subwave
    count; and "volume and volatility tend to recede over the life of the
    pattern, but this is not a requirement".

      tactic         "when price reaches the apex of the triangle, expect the
                     market to turn"
      side           at.html, the classic page this one cross-references,
                     gives the upward breakout (63%)
    """
    ln = _ew_triangle(b, p, "flat", "up")
    if ln is None:
        return None
    return Setup("long", ln.top_end, ln.bot_end * 0.999, None,
                 f"Elliott ascending triangle top {ln.top_end:.4f}")


@pattern("elliott_descending_triangle", "EWTriangleDescending.html", "continuation", "short", 100.0)
def _ew_descending_triangle(b, p):
    """EWTriangleDescending.html — "the waves bottom near the same price,
    following a horizontal trendline"; "the tops of the waves generally
    follow a down-sloping trendline"; five A-B-C-D-E waves in a 3-3-3-3-3
    subwave count; receding volume and volatility.

      side           dt.html's classic downward break out of the horizontal
                     base is the side armed
    """
    ln = _ew_triangle(b, p, "down", "flat")
    if ln is None:
        return None
    return Setup("short", ln.bot_end, ln.top_end * 1.001, None,
                 f"Elliott descending triangle base {ln.bot_end:.4f}")


@pattern("elliott_symmetrical_triangle", "EWTriangleSymmetrical.html", "continuation", "long", 100.0)
def _ew_symmetrical_triangle(b, p):
    """EWTriangleSymmetrical.html — "the waves bottom and top out following
    two converging trendlines"; "five waves compose the symmetrical triangle
    (A-B-C-D-E), unless extended"; the 3-3-3-3-3 configuration; receding
    volume and volatility.

      side           st.html gives the upward breakout (60%)
    """
    ln = _ew_triangle(b, p, "down", "up")
    if ln is None:
        return None
    return Setup("long", ln.top_end, ln.bot_end * 0.999, None,
                 f"Elliott symmetrical triangle top {ln.top_end:.4f}")


@pattern("elliott_running_triangle", "EWTriangleRunning.html", "continuation", "long", 100.0)
def _ew_running_triangle(b, p):
    """EWTriangleRunning.html — the converging A-B-C-D-E triangle with one
    extra rule: "wave B runs beyond the start of wave A". Same 3-3-3-3-3
    configuration and same receding volume and volatility."""
    ln = _ew_triangle(b, p, "down", "up")
    if ln is None:
        return None
    z = [q for q in zigzag(b) if ln.first <= q[0] <= ln.last]
    if len(z) < 3:
        return None
    start, wave_a, wave_b = z[0][2], z[1][2], z[2][2]
    if z[0][1] == "H":
        if not wave_b > start:
            return None                    # "wave B runs beyond the start of wave A"
    elif not wave_b < start:
        return None
    return Setup("long", ln.top_end, ln.bot_end * 0.999, None,
                 f"Elliott running triangle top {ln.top_end:.4f}")


@pattern("elliott_reverse_symmetrical_triangle", "EWRevSymmetrical.html", "continuation", "long", 100.0)
def _ew_reverse_symmetrical_triangle(b, p):
    """EWRevSymmetrical.html — "the waves bottom and top out following two
    diverging trendlines"; "five waves compose the reverse symmetrical
    triangle (A-B-C-D-E)"; the 3-3-3-3-3 configuration. The page describes it
    as "a region of horizontal price movement, a consolidation of a prior
    move" and gives no tactics.

      side           this is the broadening formation of broadb.html / bt.html,
                     whose breakout is "upward 60% of the time"
    """
    ln = _ew_triangle(b, p, "up", "down", diverging=True)
    if ln is None:
        return None
    top = b.highest(ln.first, ln.last)
    bot = b.lowest(ln.first, ln.last)
    if top <= bot:
        return None
    return Setup("long", top, bot * 0.999, None,
                 f"Elliott reverse symmetrical triangle {bot:.4f}-{top:.4f}")


@pattern("elliott_ending_diagonal_triangle", "EWDiagTriangle.html", "reversal", "short", 100.0)
def _ew_ending_diagonal(b, p):
    """EWDiagTriangle.html — "the subwave action usually follows two
    converging trendlines"; "subwave 4 often overlaps subwave 1"; "the
    subwave count is 3-3-3-3-3"; "a throw-over occurs when price pierces the
    trendline connecting the ends of subwaves 1 and 3"; "the ending diagonal
    triangle usually occurs as part of a fifth wave extension".

      tactic         "when the pattern forms in an uptrend, price typically
                     breaks downward, often retracing back to the triangle's
                     starting point" — which is both the side armed and the
                     target
    """
    pts = _ew_motive(b, p)
    if pts is None:
        return None
    si, i1, i2, i3, i4, i5, s, w1, w2, w3, w4, w5 = pts
    if w4 >= w1:
        return None                        # "subwave 4 often overlaps subwave 1"
    ts, ti = linfit([i1, i3, i5], [w1, w3, w5])
    bs, bi_ = linfit([si, i2, i4], [s, w2, w4])
    if (ts * si + ti) - (bs * si + bi_) <= (ts * i5 + ti) - (bs * i5 + bi_):
        return None                        # "two converging trendlines"
    trigger = min(w4, b.lowest(i5, b.n - 1))
    if not s < trigger < w5:
        return None
    return Setup("short", trigger, w5 * 1.001, s,
                 f"ending diagonal triangle, target the start {s:.4f}")


def _ew_extension(b, p, which, label):
    """EWExtension1/3/5.html — "the wave [one/three/five] extension is a
    motive wave composed of nine sub waves, each appearing similar in shape
    and duration"; "if an extension occurs on wave [n], then [the other two]
    will be normal waves, not extensions"; "most impulse waves contain
    extensions (either wave 1, 3 or 5 will be extended)"; "wave 3 is the most
    commonly extended wave"; "wave four cannot overlap wave one"; "wave three
    is never the shortest [actionary] wave".

    The extension is identified as the longest of waves 1, 3 and 5 on this
    scale — the nine-subwave count sits a degree below what daily bars
    resolve. None of these pages publishes a tactic, so, as with EWBasic, the
    entry is the correction that EWCorrective.html says follows a completed
    motive phase.
    """
    pts = _ew_motive(b, p)
    if pts is None:
        return None
    si, i1, i2, i3, i4, i5, s, w1, w2, w3, w4, w5 = pts
    if w2 <= s or w4 <= w1:
        return None                        # "wave four cannot overlap wave one"
    legs = [w1 - s, w3 - w2, w5 - w4]
    if min(legs) <= 0.0 or legs[1] == min(legs):
        return None                        # "wave three is never the shortest"
    if legs.index(max(legs)) != which:
        return None
    if max(legs) < p.ew_extension_mult * float(np.median(legs)):
        return None                        # "unusually long with exaggerated subwaves"
    trigger = min(w4, b.lowest(i5, b.n - 1))
    if trigger >= w5:
        return None
    return Setup("short", trigger, w5 * 1.001, None,
                 f"{label}: legs {legs[0]:.4f}/{legs[1]:.4f}/{legs[2]:.4f}")


@pattern("wave_one_extension", "EWExtension1.html", "reversal", "short", 100.0)
def _wave_one_extension(b, p):
    """EWExtension1.html — "if an extension occurs on wave one, then waves
    three and five will be normal waves, not extensions"."""
    return _ew_extension(b, p, 0, "wave one extension")


@pattern("wave_three_extension", "EWExtension3.html", "reversal", "short", 100.0)
def _wave_three_extension(b, p):
    """EWExtension3.html — "if an extension occurs on wave three, then waves
    one and five will be normal waves"; "wave three is the most commonly
    extended wave"."""
    return _ew_extension(b, p, 1, "wave three extension")


@pattern("wave_five_extension", "EWExtension5.html", "reversal", "short", 100.0)
def _wave_five_extension(b, p):
    """EWExtension5.html — "if an extension occurs on wave five, then waves
    one and three will be normal waves, not extensions"; "an extension can,
    itself, be extended (an extension within an extension)"."""
    return _ew_extension(b, p, 2, "wave five extension")


@pattern("monthly_symmetrical_triangle", "msymtri.html", "continuation", "long", 100.0)
def _monthly_symmetrical_triangle(b, p):
    """msymtri.html — "I won't describe what to look for. You can find that
    information on the symmetrical triangles page of this site. Symmetricals
    appear the same regardless of the time scale you select." What differs is
    the scale and the tactics: of 124 bull-market patterns "83% achieved
    gains exceeding 45%", averaging 121%.

      entry          "price closes above the top trendline"
      stop           "draw an up-sloping trendline beneath price lows; sell
                     if price closes below this support line"
      target         "the article does not specify a formal measure rule",
                     emphasising holding through the momentum phase instead,
                     so no limit is bracketed
    """
    ln = _lines(b, p, p.monthly_lookback, p.monthly_min_bars)
    if ln is None or ln.ts >= 0.0 or ln.bs <= 0.0:
        return None
    if (ln.top_start - ln.bot_start) <= (ln.top_end - ln.bot_end):
        return None                        # converging
    return Setup("long", ln.top_end, ln.bot_end * 0.999, None,
                 f"monthly symmetrical triangle top {ln.top_end:.4f}")


@pattern("unknown_wave_extension", "EWExtensionU.html", "reversal", "short", 100.0)
def _unknown_wave_extension(b, p):
    """EWExtensionU.html — "nine subwaves compose the wave"; "each subwave
    appears similar to the others making identification of waves one, three,
    or five difficult". The pattern is an unusually long impulse in which no
    single wave stands out as the extended one, which is how it is matched
    here: a valid motive phase whose three actionary legs are of comparable
    length. As with the other extension pages the entry is the correction
    EWCorrective.html says follows a completed motive phase."""
    pts = _ew_motive(b, p)
    if pts is None:
        return None
    si, i1, i2, i3, i4, i5, s, w1, w2, w3, w4, w5 = pts
    if w2 <= s or w4 <= w1:
        return None
    legs = [w1 - s, w3 - w2, w5 - w4]
    if min(legs) <= 0.0 or legs[1] == min(legs):
        return None
    if max(legs) >= p.ew_extension_mult * float(np.median(legs)):
        return None                        # one wave IS clearly extended
    trigger = min(w4, b.lowest(i5, b.n - 1))
    if trigger >= w5:
        return None
    return Setup("short", trigger, w5 * 1.001, None,
                 f"unknown wave extension, legs {legs[0]:.4f}/{legs[1]:.4f}/{legs[2]:.4f}")


@pattern("price_mountain", "mountain.html", "other", "none", 100.0, tradeable=False)
def _price_mountain(b, p):
    """mountain.html — "a price mountain is just like it sounds. Price makes
    a substantial rise and then reverses, leaving a peak on the price chart."
    The study's criteria are that the "stock must double within 3 years",
    "after doubling, [the] stock must drop by half", and "price must remain
    above $3 at the halfway point".

    The page "focuses on statistical recovery data rather than trading
    mechanics" and gives no entry, stop or target, so nothing is armed.
    """
    win = min(b.n - 1, p.mountain_bars)
    s = b.n - 1 - win
    pi = b.arghighest(s, b.n - 1)
    base = b.lowest(s, pi)
    if base <= 0.0 or float(b.h[pi]) < 2.0 * base:
        return None                        # "must double within 3 years"
    if b.lowest(pi, b.n - 1) > float(b.h[pi]) / 2.0:
        return None                        # "must drop by half"
    return None


@pattern("price_mirrors", "Mirrors.html", "other", "none", 100.0, tradeable=False)
def _price_mirrors(b, p):
    """Mirrors.html — "price peaks at A, drops down and then rises to form a
    second top at A"; the same principle applies to valleys. Mirrors "predict
    where price will turn in the future" and show "where price is going to
    pause when it begins retracing", but "mirrors only work when price is
    ready to reverse" and "sometimes a peak on the left won't be mirrored by
    one on the right".

    The page gives no entry, stop or target — it points to Trading Basics
    p.109 for those — so the detector identifies the mirrored turn and arms
    nothing.
    """
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 3, p.structure_max_age)
    if pts is None:
        return None
    (_, _, first), (_, _, _), (_, _, second) = pts
    near(first, second, p.mirror_tol_pct)
    return None


def rsi(c, n):
    """Wilder's RSI, the standard momentum oscillator. divergence.html says
    only "indicator" without naming one, so this is the engine's choice and
    is documented as such."""
    if len(c) < n + 1:
        return None
    d = np.diff(c[-(n + 1):])
    up = float(np.mean(np.maximum(d, 0.0)))
    dn = float(np.mean(np.maximum(-d, 0.0)))
    if dn == 0.0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + up / dn)


@pattern("bearish_divergence", "divergence.html", "other", "none", 100.0, tradeable=False)
def _bearish_divergence(b, p):
    """divergence.html, bearish — "price trend: upward forming higher peaks";
    "indicator trend: lower peaks"; the peaks "ideally spaced less than 2
    months apart (1 month optimal on daily charts)"; an "upward-sloping
    trendline on price peaks; downward-sloping trendline on indicator peaks".

    The page names no indicator, so RSI is used here. It also gives no entry,
    stop or target — it notes divergence is "reliable" but "not timely" and
    "recommends using confirming signals alongside divergence observations" —
    so nothing is armed.
    """
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 2, p.structure_max_age)
    if pts is None:
        return None
    (i1, _, h1), (i2, _, h2) = pts
    if h2 <= h1 or i2 - i1 > p.divergence_max_bars:
        return None
    r1, r2 = rsi(b.c[:i1 + 1], p.rsi_len), rsi(b.c[:i2 + 1], p.rsi_len)
    if r1 is None or r2 is None or r2 >= r1:
        return None
    return None


@pattern("bullish_divergence", "divergence.html", "other", "none", 100.0, tradeable=False)
def _bullish_divergence(b, p):
    """divergence.html, bullish — "price trend: downward forming lower
    valleys"; "indicator trend: higher valleys"; valleys "ideally spaced less
    than 2 months apart"; a "downward-sloping trendline on price valleys;
    upward-sloping trendline on indicator valleys". Same indicator choice and
    same absence of published tactics as the bearish case."""
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 2, p.structure_max_age)
    if pts is None:
        return None
    (i1, _, l1), (i2, _, l2) = pts
    if l2 >= l1 or i2 - i1 > p.divergence_max_bars:
        return None
    r1, r2 = rsi(b.c[:i1 + 1], p.rsi_len), rsi(b.c[:i2 + 1], p.rsi_len)
    if r1 is None or r2 is None or r2 <= r1:
        return None
    return None


@pattern("three_peaks_and_domed_house", "3peaksdome.html", "reversal", "none",
         100.0, tradeable=False)
def _three_peaks_and_domed_house(b, p):
    """3peaksdome.html — "look for three peaks rising from the base at point
    1 and 2 in a sharp price uptrend to peak 3"; "the peaks appear similar in
    shape and top out near the same price as peak 3"; "the three peaks take
    about 8 months to form, give or take"; then "a severe drop begins, taking
    price down to point 10 in two waves, 7 to 8 and 9 to 10", "always lower
    than either points 4 or 6 but often both", followed by the domed house
    with its two storeys and roof.

    The page gives no entry, stop or target — only that "price bottoms near
    point 10" — and the author warns the pattern is "extremely rare, complex
    ... rarely found in individual stocks, suggesting traders might instead
    use simpler patterns like triple tops or head-and-shoulders formations".
    The detector therefore identifies the three-peaks phase and arms nothing.
    """
    ps = [i for i in b.peaks if i >= b.n - 1 - p.dome_lookback]
    if len(ps) < 3:
        return None
    t1, t2, t3 = ps[-3], ps[-2], ps[-1]
    highs = [float(b.h[i]) for i in (t1, t2, t3)]
    if not (near(highs[0], highs[2], p.triple_tol_pct)
            and near(highs[1], highs[2], p.triple_tol_pct)):
        return None                        # "top out near the same price as peak 3"
    if not p.dome_min_bars <= t3 - t1 <= p.dome_lookback:
        return None                        # "about 8 months to form"
    trend_up_into(b, t1, p.trend_window, p.min_trend_pct)
    return None


@pattern("multi_peaks", "MultiPeaks.html", "reversal", "short", 54.0)
def _multi_peaks(b, p):
    """MultiPeaks.html — "a flat top pattern with an irregular (or not)
    bottom"; it "must have at least four peaks near the same price. No peak
    should soar above the others." "Price trend: upward. The trend start
    (intermediate-term uptrend 3 to 6 months) must be below the bottom."

      confirmation   "price confirms the pattern and stages a downward
                     breakout when it closes below the lowest valley between
                     the four peaks"
      measure rule   "take the height of the pattern from A to B (highest
                     peak to lowest valley between the four tops) and
                     subtract it from the price of the lowest valley (B).
                     Price reaches the target 54% of the time."
    """
    ps = [i for i in b.peaks if i >= b.n - 1 - p.multipeak_lookback]
    if len(ps) < 4:
        return None
    sel = ps[-4:]
    highs = [float(b.h[i]) for i in sel]
    if not near(min(highs), max(highs), p.multipeak_tol_pct):
        return None                        # "no peak should soar above the others"
    if not trend_up_into(b, sel[0], p.trend_window, p.min_trend_pct):
        return None
    valley = b.lowest(sel[0], sel[-1])
    height = max(highs) - valley
    if height <= 0.0:
        return None
    return Setup("short", valley, max(highs) * 1.001,
                 measure_short(valley, height, 54.0),
                 f"multi-peaks {len(sel)} tops near {np.mean(highs):.4f}")


@pattern("three_peaks_and_spike", "MultiPeak2B.html", "reversal", "short", 45.0)
def _three_peaks_and_spike(b, p):
    """MultiPeak2B.html — the pattern "must have at least three peaks near
    the same price followed by a fourth peak which is above the prior three".
    Separation has "no minimum, but major peaks I looked at were separated by
    at least a week (5 price bars)". "Up and down inbound price trends lead
    to the same performance (16% average drop)." Rising volume performs
    slightly better.

      confirmation   "the pattern confirms when price closes below the lowest
                     valley between the four peaks"
      measure rule   the height from A to B (highest peak to lowest valley
                     between the four tops) subtracted from the lowest
                     valley; "price reaches the target 45% of the time"
    """
    ps = [i for i in b.peaks if i >= b.n - 1 - p.multipeak_lookback]
    if len(ps) < 4:
        return None
    sel = ps[-4:]
    highs = [float(b.h[i]) for i in sel]
    if not near(min(highs[:3]), max(highs[:3]), p.multipeak_tol_pct):
        return None                        # "three peaks near the same price"
    if highs[3] <= max(highs[:3]):
        return None                        # "a fourth peak which is above the prior three"
    if min(sel[i + 1] - sel[i] for i in range(3)) < p.multipeak_min_sep:
        return None                        # "separated by at least a week"
    valley = b.lowest(sel[0], sel[-1])
    height = highs[3] - valley
    if height <= 0.0:
        return None
    return Setup("short", valley, highs[3] * 1.001,
                 measure_short(valley, height, 45.0),
                 f"three peaks and spike, spike {highs[3]:.4f}")


@pattern("extended_v_bottom", "vBottomExts.html", "reversal", "long", 51.0)
def _extended_v_bottom(b, p):
    """vBottomExts.html — the V-bottom's "straight-line run downward with few
    or no pauses", "at least 3 weeks to 3 months wide", with a reversal at
    the bottom "usually on heavy volume" — plus the extension that names it:
    "price climbs for a bit then moves sideways before continuing higher".

      breakout       "when price closes above this peak, that's the breakout
                     signal" — the peak that ends the first climb
      target         the high at the pattern start; met 51% of the time
    """
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 4, p.structure_max_age)
    if pts is None:
        return None
    (ai, _, a), (bi, _, low), (ci, _, peak), (di, _, pause) = pts
    if bi - ai < p.v_width_min or bi - ai > p.v_width_max:
        return None
    drop = a - low
    if drop <= 0.0:
        return None
    if peak < low + p.v_retrace_pct / 100.0 * drop:
        return None                        # "must retrace at least 38.2%"
    if not low < pause < peak:
        return None                        # the sideways extension
    if peak >= a:
        return None
    return Setup("long", peak, pause * 0.999, a,
                 f"extended V-bottom A{a:.4f} B{low:.4f}, extension {pause:.4f}")


@pattern("extended_v_top", "VTopExt.html", "reversal", "short", 49.0)
def _extended_v_top(b, p):
    """VTopExt.html — "an uptrend with a straight-line run upward", "a
    one-day reversal, island reversal, or tail" at the top, and "price on the
    right side must retrace at least 38.2% of the left side", with the
    extension forming a flag or pennant on the way down.

      entry          "when price breaks out of the flag or pennant in the
                     extended portion of the inverted V, take a position", or
                     wait for the 38.2% retrace
      measure rule   "measure the height of the Extended V-top from B to A ...
                     the price of A is the measure rule target"; met 49% of
                     the time
    """
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 4, p.structure_max_age)
    if pts is None:
        return None
    (ai, _, a), (bi, _, high), (ci, _, valley), (di, _, pause) = pts
    if bi - ai < p.v_width_min or bi - ai > p.v_width_max:
        return None
    rise = high - a
    if rise <= 0.0:
        return None
    if valley > high - p.v_retrace_pct / 100.0 * rise:
        return None
    if not valley < pause < high:
        return None
    if valley <= a:
        return None
    return Setup("short", valley, pause * 1.001, a,
                 f"extended V-top A{a:.4f} B{high:.4f}, extension {pause:.4f}")


@pattern("pothole", "Pothole.html", "continuation", "long", 100.0)
def _pothole(b, p):
    """Pothole.html — "a flat road (horizontal movement) followed by a dip
    (the pothole)", on a daily chart with "an upward price trend preceding
    the pattern". "Prices along the bottom of the pothole pattern should be
    horizontal, or nearly so, with price resting on support." The pothole can
    be "a quick one-day plunge" or last "a few weeks".

      breakout       "the breakout occurs when price closes above the top of
                     the pothole pattern"
      entry          "place a buy stop a penny above the top of the highest
                     peak in the pattern"
      stop           "the bottom of the flat base (roadway) portion of the
                     pothole pattern serves as a good stop location"
      measure rule   the height from the highest peak to the pothole bottom,
                     added to the peak price
    """
    win = min(b.n - 1, p.pothole_lookback)
    s = b.n - 1 - win
    di = b.arglowest(s, b.n - 1)
    if b.n - 1 - di > p.pothole_max_dip or di <= s:
        return None                        # the dip must be recent
    road_hi = b.highest(s, di - 1)
    road_lo = b.lowest(s, di - 1)
    if road_lo <= 0.0 or pct(road_hi, road_lo) > p.pothole_road_flat_pct:
        return None                        # "a flat road"
    dip = float(b.l[di])
    if dip >= road_lo:
        return None                        # there must actually be a pothole
    top = b.highest(s, b.n - 1)
    if top <= road_lo:
        return None
    if not trend_up_into(b, s, p.trend_window, p.min_trend_pct):
        return None
    return Setup("long", top, road_lo * 0.999, top + (top - dip),
                 f"pothole road {road_lo:.4f}-{road_hi:.4f}, dip {dip:.4f}")


@pattern("failure_swing_bullish", "failswing.html", "other", "none", 100.0, tradeable=False)
def _failure_swing_bullish(b, p):
    """failswing.html — "look for W-shaped failure swings that span the
    trigger line in the indicator". The W indicates "a potential upward trend
    change", the stock beginning "its turn as the pattern forms".

    The pattern lives in an indicator, not in price; the page names none, so
    RSI is used here as it is for divergence. The page publishes no stop and
    no target ("the webpage does not provide explicit stop placement or
    target rules for these patterns"), so nothing is armed. It also cautions
    that "some analysts argue that the failure swing must span the trigger
    line (70), but you can find numerous examples where that is not the case
    and yet price changes trend".
    """
    z = zigzag(b)
    pts = _last_turn(z, "L", b, 3, p.structure_max_age)
    if pts is None:
        return None
    (i1, _, _), (i2, _, _), (i3, _, _) = pts
    r1 = rsi(b.c[:i1 + 1], p.rsi_len)
    r3 = rsi(b.c[:i3 + 1], p.rsi_len)
    if r1 is None or r3 is None:
        return None
    if not (r1 < p.failswing_low and r3 > r1):
        return None                        # the W's second trough holds higher
    return None


@pattern("failure_swing_bearish", "failswing.html", "other", "none", 100.0, tradeable=False)
def _failure_swing_bearish(b, p):
    """failswing.html — "look for M-shaped failure swings that span the
    trigger line in the indicator". "The second peak of the failure swing
    doesn't rise above the first peak. It fails to swing higher, indicating a
    weakening technical picture." Same indicator substitution and same
    absence of published stop or target as the bullish case."""
    z = zigzag(b)
    pts = _last_turn(z, "H", b, 3, p.structure_max_age)
    if pts is None:
        return None
    (i1, _, _), (i2, _, _), (i3, _, _) = pts
    r1 = rsi(b.c[:i1 + 1], p.rsi_len)
    r3 = rsi(b.c[:i3 + 1], p.rsi_len)
    if r1 is None or r3 is None:
        return None
    if not (r1 > p.failswing_high and r3 < r1):
        return None                        # "doesn't rise above the first peak"
    return None


def _trendline(b, p, up):
    """uptrendlines.html / trenddown.html — an up trendline is drawn along
    the price valleys and a down trendline along the peaks, each needing
    enough touches to be real. Returns (slope, intercept, first_touch,
    touches) or None."""
    lo = max(0, b.n - 1 - p.trendline_lookback)
    hi = b.n - 1
    pts = [i for i in (b.valleys if up else b.peaks) if lo <= i <= hi]
    if len(pts) < p.trendline_min_touches:
        return None
    y = [float(b.l[i]) if up else float(b.h[i]) for i in pts]
    slope, inter = linfit(pts, y)
    if up and slope <= 0.0:
        return None
    if not up and slope >= 0.0:
        return None
    scale = float(np.mean(b.c[lo:hi + 1]))
    tol = p.touch_tol_pct / 100.0 * scale
    touches = sum(1 for i, v in zip(pts, y) if abs(v - (slope * i + inter)) <= tol)
    if touches < p.trendline_min_touches:
        return None
    return slope, inter, pts[0], touches


@pattern("up_trendline", "uptrendlines.html", "reversal", "short", 63.0)
def _up_trendline(b, p):
    """uptrendlines.html — connect the price valleys with an up-sloping line;
    "widely spaced touches (over the median 12 days each) suggest a more
    powerful move post-breakout". "Use the logarithmic scale. Price will
    signal a trend change sooner on the log scale than on the arithmetic
    scale" — the engine works in price, which is deviation 2 applied to the
    scale.

      breakout       a close below the up-sloping trendline
      entry          "sell at the open the next day after price closes below
                     the trendline"; the study found selling on that first
                     close beat waiting for further confirmation
      measure rule   "multiply the vertical distance between the prior minor
                     low touch and breakout point by 63%, then project
                     downward from the breakout price"
    """
    t = _trendline(b, p, True)
    if t is None:
        return None
    slope, inter, first, touches = t
    trigger = slope * (b.n - 1) + inter
    prior_low = b.lowest(first, b.n - 1)
    height = trigger - prior_low
    if height <= 0.0 or trigger <= 0.0:
        return None
    top = b.highest(first, b.n - 1)
    if top <= trigger:
        return None
    return Setup("short", trigger, top * 1.001,
                 measure_short(trigger, height, 63.0),
                 f"up trendline break at {trigger:.4f} ({touches} touches)")


@pattern("down_trendline", "trenddown.html", "reversal", "long", 56.0)
def _down_trendline(b, p):
    """trenddown.html — "draw a down-sloping trendline along price peaks.
    That way, when the trend changes from down to up, you'll know with a
    trendline pierce." Longer trendlines (">48 days median") and shallow
    slopes "produce more powerful rallies", and "downward volume trends
    correlate with stronger performance after breakout".

      breakout       "price closes above the down-sloping trendline"
      measure rule   "multiply the widest vertical distance between the
                     breakout point and prior minor high trendline touch by
                     56%, then project upward from breakout price"
    """
    t = _trendline(b, p, False)
    if t is None:
        return None
    slope, inter, first, touches = t
    trigger = slope * (b.n - 1) + inter
    prior_high = b.highest(first, b.n - 1)
    height = prior_high - trigger
    if height <= 0.0 or trigger <= 0.0:
        return None
    bot = b.lowest(first, b.n - 1)
    if bot >= trigger:
        return None
    if p.require_volume_rules and not volume_recedes(b, first, b.n - 1):
        return None
    return Setup("long", trigger, bot * 0.999,
                 measure_long(trigger, height, 56.0),
                 f"down trendline break at {trigger:.4f} ({touches} touches)")


@pattern("support_and_resistance", "SAR.html", "other", "none", 100.0, tradeable=False)
def _support_and_resistance(b, p):
    """SAR.html — the six places support and resistance come from: "round
    numbers", "valleys (bottoms)", "peaks (tops)", "trendlines", "horizontal
    consolidation regions (HCR)" where price has "significant price overlap"
    and tends to "get stuck", "moving averages" (the 200-day simple moving
    average "demonstrates bounce behavior"), and gaps — which "perform poorly
    as S/R areas: rising gaps support price only 20% of the time".

    "The article provides no explicit entry, stop, or target instructions —
    it focuses exclusively on identifying support/resistance zones rather
    than actionable trading mechanics", so nothing is armed. The nearest
    levels are computed for identification.
    """
    ma = sma(b.c, p.sar_ma)
    lo = max(0, b.n - 1 - p.sar_lookback)
    [float(b.h[i]) for i in b.peaks if i >= lo]
    [float(b.l[i]) for i in b.valleys if i >= lo]
    round(float(b.c[-1]))                  # the round-number level
    _congested(b, b.n - 1, p.gap_congestion_bars, p.gap_congestion_pct)
    return None if ma is None else None


@pattern("monthly_channels", "MonthlyChannels.html", "reversal", "short", 100.0)
def _monthly_channels(b, p):
    """MonthlyChannels.html — "use the monthly scale and find an up-sloping
    channel or price rise that follows a trendline", where "the trendline
    should touch price at least three times"; "the channel must be at least
    two years long"; "the channel ends at the highest price in the channel";
    then "from the highest candle, look for three consecutive black candles".

      entry          "sell at the opening the next month" after the three
                     black candles
      target         "the average drop after [a] sell signal" is 43%,
                     measured from the third black candle's close
      stop           the page names none, so the channel high is used
    """
    t = _trendline(b, p, True)
    if t is None:
        return None
    slope, inter, first, touches = t
    if b.n - 1 - first < p.monthly_channel_bars:
        return None                        # "at least two years long"
    hi_i = b.arghighest(first, b.n - 1)
    n_black = p.monthly_black_candles
    if b.n - 1 - hi_i > n_black + 1:
        return None                        # the black run must start at the peak
    for i in range(b.n - n_black, b.n):
        if float(b.c[i]) >= float(b.o[i]):
            return None                    # "three consecutive black candles"
    trigger = float(b.l[-1])
    top = float(b.h[hi_i])
    target = float(b.c[-1]) * (1.0 - p.monthly_drop_pct / 100.0)
    if not target < trigger < top:
        return None
    return Setup("short", trigger, top * 1.001, target,
                 f"monthly channel top {top:.4f}, {n_black} black candles")


@pattern("volatility_stop", "stops.html", "other", "none", 100.0, tradeable=False)
def _volatility_stop(b, p):
    """stops.html, the volatility stop — "measure the average daily range
    (the difference between the high and low price for each day) over the
    prior month (approximately 22 trading bars)", take the mean, "multiply
    this by 2 to get the volatility", then subtract that from "the most
    recent low price". It is "recalculated monthly or when price makes a new
    high".

    This is a stop-placement technique for a position already held, not an
    entry — the page gives no entry and no target — so the level is computed
    for identification and nothing is armed.
    """
    avg = avg_bar_height(b, b.n - 1, p.tall_avg_bars)
    if avg is None:
        return None
    float(b.l[-1]) - p.vstop_mult * avg
    return None


# ═══════════════════════════════════════════════════════════════════════════
# The strategy
# ═══════════════════════════════════════════════════════════════════════════


PATTERN_NAMES = [s.name for s in PATTERNS]


@dataclass
class Armed:
    """A resting entry with its bracket already attached."""

    entry_id: int
    stop_id: int
    target_id: int
    side: str
    trigger: float
    stop: float
    qty: float
    armed_bar: int
    hold_bars: Optional[int] = None


@dataclass
class Held:
    """A filled position: the bracket does the work, this tracks the clock."""

    entry_id: int
    stop_id: int
    target_id: int
    side: str
    stop: float
    target: float
    hold_bars: Optional[int] = None
    bars_held: int = 0
    exiting: bool = False


class PatternsStrategy(stonks.Strategy):
    """Trades one Bulkowski chart pattern, selected by `pattern`.

    `pattern` is an index into `PATTERNS` (engine params must be numeric);
    `PATTERN_NAMES[i]` is the book name at index i. Setting the non-param
    class attribute `pattern_name` to a registry name overrides the index,
    which is what the tests use.
    """

    # ── which pattern ────────────────────────────────────────────────────
    pattern = 0
    pattern_name = ""          # non-param override; "" means use `pattern`

    # ── universe filters ─────────────────────────────────────────────────
    min_price = 1.0
    min_dollar_volume = 1_000_000.0

    # ── identification thresholds (the numeric stand-ins of deviation 2) ──
    trend_window = 40
    min_trend_pct = 10.0
    shoulder_tol_pct = 5.0
    symmetry_ratio = 2.5
    bottom_tol_pct = 4.0
    top_tol_pct = 3.0
    adam_bottom_tol_pct = 3.0
    triple_tol_pct = 4.0
    shape_tol_pct = 3.0
    adam_max_width = 3
    eve_min_width = 5
    dbl_sep_min = 10
    dbl_sep_max = 70
    min_double_rise_pct = 10.0
    ugly_rise_min = 5.0
    ugly_rise_max = 15.0
    triangle_lookback = 60
    triangle_min_bars = 15
    trendline_flat_pct = 2.0
    touch_tol_pct = 1.5
    channel_lookback = 60
    channel_min_bars = 15
    wedge_min_bars = 15
    cup_min_bars = 35
    cup_max_bars = 325
    cup_rim_tol_pct = 5.0
    cup_base_min_width = 5
    handle_min_bars = 5
    pole_min_bars = 3
    pole_max_bars = 15
    pole_min_pct = 15.0
    flag_min_bars = 3
    flag_max_bars = 15
    flag_max_height = 0.5
    htf_pole_bars = 42
    htf_min_rise_pct = 90.0
    diamond_lookback = 60
    diamond_min_bars = 15
    barr_lookback = 90
    barr_leadin_bars = 21
    barr_bump_mult = 2.0
    rounding_lookback = 250
    rounding_min_bars = 30
    pipe_lookback = 50
    pipe_overlap = 0.5
    island_max_bars = 60
    island_min_gap_pct = 0.5
    long_island_max_bars = 84
    tall_avg_bars = 22
    tall_bar_mult = 1.5
    tall_max_target_pct = 20.0
    dance_shadow_body_mult = 3.0
    dance_shadow_ratio = 2.0
    trend_bars = 5
    quarter = 0.25
    krb_close_tol = 0.15
    krb_tall_mult = 0.5
    wide_range_mult = 3.0
    pct_stop = 7.0
    harmonic_max_age = 10
    gap_congestion_bars = 20
    gap_congestion_pct = 10.0
    gap_volume_mult = 1.5
    gap_trend_bars = 30
    gap_trend_pct = 15.0
    gap_tall_mult = 1.0
    ex_div_max_gap_pct = 3.0
    scallop_round_width = 4
    aiscallop_retrace_pct = 54.0
    scallop_retrace_tol = 20.0
    idscallop_cover_pct = 67.0
    mm_min_retrace_pct = 70.0
    spike_tall_mult = 2.0
    partial_clearance = 0.2
    structure_max_age = 30
    turn_max_age = 8
    dcb_lookback = 90
    dcb_min_decline_pct = 15.0
    dcb_max_decline_pct = 70.0
    dcb_min_bounce_bars = 5
    dcb_rollover_bars = 3
    dcb_post_bounce_pct = 30.0
    idcb_min_jump_pct = 5.0
    bigm_top_tol_pct = 4.0
    bigm_move_pct = 10.0
    bigm_side_pct = 20.0
    roof_lookback = 60
    roof_min_bars = 15
    roof_flat_tol_pct = 4.0
    roof_v_max_width = 4
    tc_lookback = 90
    tc_retest_tol_pct = 3.0
    tc_target_pct = 20.0
    two_b_excess_pct = 5.0
    two_b_top_decline_pct = 6.0
    two_b_bottom_gain_pct = 32.0
    v_width_min = 15
    v_width_max = 63
    v_retrace_pct = 38.2
    vpivot_clear_pct = 2.0
    bust_max_move_pct = 10.0
    bust_scan_bars = 40
    bust_min_bars = 3
    throwback_max_bars = 22
    throwback_tol_pct = 4.0
    channel_parallel_pct = 5.0
    dive_board_bars = 40
    dive_board_flat_pct = 15.0
    dive_plunge_pct = 25.0
    dive_plunge_max = 40
    dive_recovery_bars = 3
    flat_base_bars = 30
    flat_base_max_range_pct = 12.0
    cats_ears_min_bars = 10
    cats_ears_max_bars = 60
    cats_ears_decline_pct = 25.0
    cloud_bars = 250
    cloud_flat_pct = 25.0
    cloud_min_decline_pct = 40.0
    cloud_min_decline_bars = 60
    cloud_sma = 150
    elevator_bars = 3
    elevator_overlap = 0.5
    event_swing_mult = 2.0
    event_volume_bars = 30
    event_max_age = 5
    event_year_bars = 252
    dutch_bars = 30
    vol_shape_ratio = 1.3
    vol_shape_bars = 30
    aw_short = 5
    aw_long = 13
    aw_retrace_pct = 5.0
    ast_hold_bars = 3
    cp_lookback = 126
    cp_flat_tol_pct = 3.0
    vrun_min_bars = 4
    vrun_overlap = 0.67
    vrun_retrace_pct = 52.0
    ew_short_pct = 20.0
    ew_near_pct = 15.0
    ew_extension_mult = 1.3
    monthly_lookback = 250
    monthly_min_bars = 60
    mountain_bars = 750
    mirror_tol_pct = 3.0
    rsi_len = 14
    divergence_max_bars = 42
    dome_lookback = 170
    dome_min_bars = 60
    multipeak_lookback = 126
    multipeak_tol_pct = 4.0
    multipeak_min_sep = 5
    pothole_lookback = 60
    pothole_road_flat_pct = 12.0
    pothole_max_dip = 15
    failswing_low = 30.0
    failswing_high = 70.0
    trendline_lookback = 90
    trendline_min_touches = 3
    sar_lookback = 120
    sar_ma = 200
    monthly_channel_bars = 120
    monthly_black_candles = 3
    monthly_drop_pct = 43.0
    vstop_mult = 2.0
    require_volume_rules = True

    # ── trading ──────────────────────────────────────────────────────────
    order_bars = 10
    max_hold_bars = 120
    risk_fraction = 0.01
    max_position_pct = 0.25
    max_positions = 10
    taker_fee_bps = 5.0
    use_measure_rule = True

    params = {
        "pattern": stonks.Param("index into PATTERNS (see PATTERN_NAMES)"),
        "min_price": stonks.Param("minimum close price", unit="$"),
        "min_dollar_volume": stonks.Param("20-bar average close x volume floor", unit="$"),
        "trend_window": stonks.Param("bars used to judge the inbound price trend", unit="bars"),
        "min_trend_pct": stonks.Param("minimum inbound trend move", unit="%"),
        "shoulder_tol_pct": stonks.Param("'shoulders near the same price' tolerance", unit="%"),
        "symmetry_ratio": stonks.Param("max ratio between the two head-to-shoulder distances"),
        "bottom_tol_pct": stonks.Param("'valleys bottom near the same price' tolerance", unit="%"),
        "top_tol_pct": stonks.Param("'peaks near the same price' tolerance", unit="%"),
        "adam_bottom_tol_pct": stonks.Param("Adam & Adam bottom-price variation", unit="%"),
        "triple_tol_pct": stonks.Param("triple top/bottom price variation", unit="%"),
        "shape_tol_pct": stonks.Param("band width used to measure Adam/Eve shape", unit="%"),
        "adam_max_width": stonks.Param("max bars in the band for a narrow Adam turn", unit="bars"),
        "eve_min_width": stonks.Param("min bars in the band for a wide Eve turn", unit="bars"),
        "dbl_sep_min": stonks.Param("minimum separation of the twin turns", unit="bars"),
        "dbl_sep_max": stonks.Param("maximum separation of the twin turns", unit="bars"),
        "min_double_rise_pct": stonks.Param("'the rise between bottoms should measure at least 10%'", unit="%"),
        "ugly_rise_min": stonks.Param("ugly double bottom: minimum second-bottom rise", unit="%"),
        "ugly_rise_max": stonks.Param("ugly double bottom: maximum second-bottom rise", unit="%"),
        "triangle_lookback": stonks.Param("bars searched for triangle pivots", unit="bars"),
        "triangle_min_bars": stonks.Param("minimum triangle span", unit="bars"),
        "trendline_flat_pct": stonks.Param("slope under which a trendline counts as horizontal", unit="%"),
        "touch_tol_pct": stonks.Param("distance from a trendline that still counts as a touch", unit="%"),
        "channel_lookback": stonks.Param("bars searched for rectangle/broadening/wedge pivots", unit="bars"),
        "channel_min_bars": stonks.Param("minimum rectangle/broadening span", unit="bars"),
        "wedge_min_bars": stonks.Param("'3 weeks is the minimum duration, otherwise it's a pennant'", unit="bars"),
        "cup_min_bars": stonks.Param("minimum cup duration ('7 weeks')", unit="bars"),
        "cup_max_bars": stonks.Param("maximum cup duration ('65 weeks')", unit="bars"),
        "cup_rim_tol_pct": stonks.Param("'cup rims should be near the same price level'", unit="%"),
        "cup_base_min_width": stonks.Param("bars at the base making the cup U-shaped, not V-shaped", unit="bars"),
        "handle_min_bars": stonks.Param("'1 week minimum' handle", unit="bars"),
        "pole_min_bars": stonks.Param("minimum flagpole length", unit="bars"),
        "pole_max_bars": stonks.Param("maximum flagpole length ('last several days')", unit="bars"),
        "pole_min_pct": stonks.Param("'unusually steep' flagpole move", unit="%"),
        "flag_min_bars": stonks.Param("minimum flag/pause length", unit="bars"),
        "flag_max_bars": stonks.Param("'flags are short, less than 3 weeks long'", unit="bars"),
        "flag_max_height": stonks.Param("flag height as a fraction of the flagpole"),
        "htf_pole_bars": stonks.Param("high and tight flag: '2 months or less'", unit="bars"),
        "htf_min_rise_pct": stonks.Param("high and tight flag: 'price must rise at least 90%'", unit="%"),
        "diamond_lookback": stonks.Param("bars searched for a diamond", unit="bars"),
        "diamond_min_bars": stonks.Param("minimum diamond span", unit="bars"),
        "barr_lookback": stonks.Param("bump-and-run window", unit="bars"),
        "barr_leadin_bars": stonks.Param("bump-and-run lead-in, 'at least one month'", unit="bars"),
        "barr_bump_mult": stonks.Param("'bump height should be at least twice the lead-in height'", unit="x"),
        "rounding_lookback": stonks.Param("bars searched for a rounding turn ('many months')", unit="bars"),
        "rounding_min_bars": stonks.Param("minimum rounding-turn span", unit="bars"),
        "pipe_lookback": stonks.Param("landscape a pipe/horn spike must undercut or tower over", unit="bars"),
        "pipe_overlap": stonks.Param("'the 2 weeks often have a large price overlap' fraction"),
        "island_max_bars": stonks.Param("'the island can be one day to several months long'", unit="bars"),
        "island_min_gap_pct": stonks.Param("minimum width of each island gap", unit="%"),
        "long_island_max_bars": stonks.Param("'look for islands shorter than 4 months'", unit="bars"),
        "tall_avg_bars": stonks.Param("'the average price bar measured one month (22 price bars) before'", unit="bars"),
        "tall_bar_mult": stonks.Param("'at least 1.5 times the 1-month average price bar height'", unit="x"),
        "tall_max_target_pct": stonks.Param("2-tall: discard targets further than this away", unit="%"),
        "dance_shadow_body_mult": stonks.Param("2-dance: 'a shadow at least 3x the body height'", unit="x"),
        "dance_shadow_ratio": stonks.Param("2-dance: 'the longer shadow at least 2x the shorter'", unit="x"),
        "trend_bars": stonks.Param("'5-day linear regression' short-term trend window", unit="bars"),
        "quarter": stonks.Param("'within 25% of the intraday high/low' fraction"),
        "krb_close_tol": stonks.Param("key reversal bar: 'closes near ... within 15% of the average bar height'"),
        "krb_tall_mult": stonks.Param("key reversal bar: 'at least 50% taller than the average'", unit="x"),
        "wide_range_mult": stonks.Param("wide ranging day: 'at least three times the one-month average'", unit="x"),
        "pct_stop": stonks.Param("percentage stop, where a page gives only that", unit="%"),
        "harmonic_max_age": stonks.Param("bars since the harmonic turn at D still worth arming", unit="bars"),
        "gap_congestion_bars": stonks.Param("bars judged for 'congestion (trendless markets)'", unit="bars"),
        "gap_congestion_pct": stonks.Param("range under which those bars count as congestion", unit="%"),
        "gap_volume_mult": stonks.Param("'elevated volume' on the gap bar vs the 20-bar average", unit="x"),
        "gap_trend_bars": stonks.Param("window judged for 'a straight-line advance or decline'", unit="bars"),
        "gap_trend_pct": stonks.Param("move over that window that counts as a straight-line trend", unit="%"),
        "gap_tall_mult": stonks.Param("'unusually tall' exhaustion gap vs the average bar height", unit="x"),
        "ex_div_max_gap_pct": stonks.Param("largest down gap still plausibly a dividend adjustment", unit="%"),
        "scallop_round_width": stonks.Param("bars making a scallop turn 'rounded, not V-shaped'", unit="bars"),
        "aiscallop_retrace_pct": stonks.Param("'the end ... usually retraces 54% of the prior up move'", unit="%"),
        "scallop_retrace_tol": stonks.Param("tolerance on that retrace", unit="%"),
        "idscallop_cover_pct": stonks.Param("'cover if price rises 67% of the decline from peak to valley'", unit="%"),
        "mm_min_retrace_pct": stonks.Param("measured move: 'retraces of at least 70%'", unit="%"),
        "spike_tall_mult": stonks.Param("'a tall price move' vs the average bar height", unit="x"),
        "partial_clearance": stonks.Param("how far a partial rise/decline must stay off the far trendline"),
        "structure_max_age": stonks.Param("bars since a multi-turn pattern's last turn still worth arming", unit="bars"),
        "turn_max_age": stonks.Param("bars since the turn for patterns entered AT the turn", unit="bars"),
        "dcb_lookback": stonks.Param("window searched for the dead-cat-bounce event decline", unit="bars"),
        "dcb_min_decline_pct": stonks.Param("'closing 15% to 70% lower than the prior day' — floor", unit="%"),
        "dcb_max_decline_pct": stonks.Param("'closing 15% to 70% lower than the prior day' — cap", unit="%"),
        "dcb_min_bounce_bars": stonks.Param("bars allowed for the bounce after the event", unit="bars"),
        "dcb_rollover_bars": stonks.Param("bars since the bounce high before shorting", unit="bars"),
        "dcb_post_bounce_pct": stonks.Param("'averaging 30% from the bounce high to post-bounce low'", unit="%"),
        "idcb_min_jump_pct": stonks.Param("'an event that causes price to jump at least 5%'", unit="%"),
        "bigm_top_tol_pct": stonks.Param("big M: 'twin peaks with highs less than 4% apart'", unit="%"),
        "bigm_move_pct": stonks.Param("big M/W: 'the drop/rise between [them] is 10% to 20% or more'", unit="%"),
        "bigm_side_pct": stonks.Param("big M/W: how tall the straight-line sides must be", unit="%"),
        "roof_lookback": stonks.Param("bars searched for a roof", unit="bars"),
        "roof_min_bars": stonks.Param("minimum roof span", unit="bars"),
        "roof_flat_tol_pct": stonks.Param("'horizontal/near-horizontal' roof tolerance", unit="%"),
        "roof_v_max_width": stonks.Param("bars at the base keeping the roof's turn V-shaped", unit="bars"),
        "tc_lookback": stonks.Param("1-2-3 trend change: window for the trendline's far anchor", unit="bars"),
        "tc_retest_tol_pct": stonks.Param("1-2-3: how far point 2 may overshoot point A", unit="%"),
        "tc_target_pct": stonks.Param("1-2-3: the page's published 20% move from point A", unit="%"),
        "two_b_excess_pct": stonks.Param("2B: how far the second turn may 'slightly' exceed the first", unit="%"),
        "two_b_top_decline_pct": stonks.Param("2B top: the page's published average decline", unit="%"),
        "two_b_bottom_gain_pct": stonks.Param("2B bottom: the page's published average gain", unit="%"),
        "v_width_min": stonks.Param("V-top/bottom: 'at least 3 weeks ... wide'", unit="bars"),
        "v_width_max": stonks.Param("V-top/bottom: '... to 3 months wide'", unit="bars"),
        "v_retrace_pct": stonks.Param("V-top/bottom: 'must retrace at least 38.2% of the left side'", unit="%"),
        "vpivot_clear_pct": stonks.Param("V pivot: 'at least 2% above the low of bar 2'", unit="%"),
        "bust_max_move_pct": stonks.Param("bust: 'moves no more than 10% before reversing'", unit="%"),
        "bust_scan_bars": stonks.Param("bars searched back for the parent pattern of a bust", unit="bars"),
        "bust_min_bars": stonks.Param("minimum bars between the parent's confirmation and now", unit="bars"),
        "throwback_max_bars": stonks.Param("'within a month of the breakout'", unit="bars"),
        "throwback_tol_pct": stonks.Param("'returns to (or near) the breakout price' tolerance", unit="%"),
        "channel_parallel_pct": stonks.Param("'the two trendlines should be parallel or nearly so'", unit="%"),
        "dive_board_bars": stonks.Param("diving board: bars forming the flat board", unit="bars"),
        "dive_board_flat_pct": stonks.Param("diving board: 'a flat bottom' range", unit="%"),
        "dive_plunge_pct": stonks.Param("diving board: how far the plunge must fall", unit="%"),
        "dive_plunge_max": stonks.Param("diving board: bars allowed for plunge plus recovery", unit="bars"),
        "dive_recovery_bars": stonks.Param("diving board: bars of recovery before arming", unit="bars"),
        "flat_base_bars": stonks.Param("flat base: 'the longer it is the better'", unit="bars"),
        "flat_base_max_range_pct": stonks.Param("flat base: 'price moving horizontally' range", unit="%"),
        "cats_ears_min_bars": stonks.Param("cat's ears: 'between 10 days ...'", unit="bars"),
        "cats_ears_max_bars": stonks.Param("cat's ears: '... and 2 months (60 days)'", unit="bars"),
        "cats_ears_decline_pct": stonks.Param("cat's ears: 'a severe decline' into the pattern", unit="%"),
        "cloud_bars": stonks.Param("cloud bank: 'should last for years, but be flexible'", unit="bars"),
        "cloud_flat_pct": stonks.Param("cloud bank: 'price moves horizontally, or almost so'", unit="%"),
        "cloud_min_decline_pct": stonks.Param("cloud bank: 'a swift and dramatic decline of at least 40%'", unit="%"),
        "cloud_min_decline_bars": stonks.Param("cloud bank: bars allowed for that decline", unit="bars"),
        "cloud_sma": stonks.Param("cloud bank: 'a 30-week SMA', in trading bars", unit="bars"),
        "elevator_bars": stonks.Param("elevator stop: 'a strong uptrend of at least 3 price bars'", unit="bars"),
        "elevator_overlap": stonks.Param("elevator stop: 'little or no overlap with the prior price bar'"),
        "event_swing_mult": stonks.Param("'a large intraday swing, 2 or 3 times the average'", unit="x"),
        "event_volume_bars": stonks.Param("'above the 30-day average' volume window", unit="bars"),
        "event_max_age": stonks.Param("bars since the announcement day still worth arming", unit="bars"),
        "event_year_bars": stonks.Param("'within a third of the yearly low/high' window", unit="bars"),
        "dutch_bars": stonks.Param("Dutch auction: bars taken as the tender-offer period", unit="bars"),
        "vol_shape_ratio": stonks.Param("how pronounced a U or dome volume shape must be", unit="x"),
        "vol_shape_bars": stonks.Param("bars over which a volume shape or trend is read", unit="bars"),
        "aw_short": stonks.Param("Adam White: the 5-period leg of llv(L,5)/hhv(H,5)", unit="bars"),
        "aw_long": stonks.Param("Adam White: the 13-period leg of llv(L,13)", unit="bars"),
        "aw_retrace_pct": stonks.Param("Adam White: '(hhv(L,13)-L)/L' exit threshold", unit="%"),
        "ast_hold_bars": stonks.Param("ascending triangle setup: 'exit at the close 3 trading days after entry'", unit="bars"),
        "cp_lookback": stonks.Param("CPSetup: bars searched for the flat bottom", unit="bars"),
        "cp_flat_tol_pct": stonks.Param("CPSetup: how flat 'the bottom of this pattern' must be", unit="%"),
        "vrun_min_bars": stonks.Param("vertical run: 'at least four sessions'", unit="bars"),
        "vrun_overlap": stonks.Param("vertical run: 'minimal overlap from price bar to bar'"),
        "vrun_retrace_pct": stonks.Param("vertical run up: 'retraces a median of 52% of the move'", unit="%"),
        "ew_short_pct": stonks.Param("Elliott: how far 'well short of the start of subwave A' is", unit="%"),
        "ew_near_pct": stonks.Param("Elliott: how close 'terminates near' is, as a share of wave A", unit="%"),
        "ew_extension_mult": stonks.Param("Elliott: how much longer an extended wave must be", unit="x"),
        "monthly_lookback": stonks.Param("monthly patterns: bars searched", unit="bars"),
        "monthly_min_bars": stonks.Param("monthly patterns: minimum span", unit="bars"),
        "mountain_bars": stonks.Param("price mountain: 'must double within 3 years'", unit="bars"),
        "mirror_tol_pct": stonks.Param("price mirrors: how closely the second turn must match", unit="%"),
        "rsi_len": stonks.Param("divergence: RSI length (the page names no indicator)", unit="bars"),
        "divergence_max_bars": stonks.Param("divergence: 'peaks spaced less than 2 months apart'", unit="bars"),
        "dome_lookback": stonks.Param("three peaks and a domed house: 'about 8 months to form'", unit="bars"),
        "dome_min_bars": stonks.Param("three peaks and a domed house: minimum span", unit="bars"),
        "multipeak_lookback": stonks.Param("multi-peaks: bars searched for the flat top", unit="bars"),
        "multipeak_tol_pct": stonks.Param("multi-peaks: 'at least four peaks near the same price'", unit="%"),
        "multipeak_min_sep": stonks.Param("three peaks and spike: 'separated by at least a week'", unit="bars"),
        "pothole_lookback": stonks.Param("pothole: bars searched for the flat road", unit="bars"),
        "pothole_road_flat_pct": stonks.Param("pothole: how flat 'the flat road' must be", unit="%"),
        "pothole_max_dip": stonks.Param("pothole: 'a quick one-day plunge' up to 'a few weeks'", unit="bars"),
        "failswing_low": stonks.Param("failure swing: the indicator's lower trigger line"),
        "failswing_high": stonks.Param("failure swing: 'the trigger line (70)'"),
        "trendline_lookback": stonks.Param("trendlines: bars searched for touches", unit="bars"),
        "trendline_min_touches": stonks.Param("trendlines: minimum touches for a valid line"),
        "sar_lookback": stonks.Param("support/resistance: bars searched for peaks and valleys", unit="bars"),
        "sar_ma": stonks.Param("support/resistance: 'the 200-day simple moving average'", unit="bars"),
        "monthly_channel_bars": stonks.Param("monthly channel: 'at least two years long'", unit="bars"),
        "monthly_black_candles": stonks.Param("monthly channel: 'three consecutive black candles'", unit="bars"),
        "monthly_drop_pct": stonks.Param("monthly channel: 'average drop after [a] sell signal'", unit="%"),
        "vstop_mult": stonks.Param("volatility stop: 'multiply this by 2 to get the volatility'", unit="x"),
        "require_volume_rules": stonks.Param("enforce the pages' volume guidelines"),
        "order_bars": stonks.Param("resting confirmation order good for N bars", unit="bars"),
        "max_hold_bars": stonks.Param("time stop on an open position", unit="bars"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade, fees included"),
        "max_position_pct": stonks.Param("entry notional cap as a fraction of equity"),
        "max_positions": stonks.Param("max open positions + armed entries"),
        "taker_fee_bps": stonks.Param("fee per sizing leg", unit="bps"),
        "use_measure_rule": stonks.Param("place the measure-rule target as a limit"),
    }

    indicators = {
        "trigger": stonks.Indicator("armed confirmation price"),
        "stop_level": stonks.Indicator("protective stop"),
        "target_level": stonks.Indicator("measure-rule target"),
    }

    def on_start(self, ctx):
        self._fee = self.taker_fee_bps / 10_000.0
        if self.pattern_name:
            names = PATTERN_NAMES
            if self.pattern_name not in names:
                raise ValueError(f"unknown pattern {self.pattern_name!r}")
            self.spec = PATTERNS[names.index(self.pattern_name)]
        else:
            if not 0 <= int(self.pattern) < len(PATTERNS):
                raise ValueError(f"pattern index {self.pattern} out of range "
                                 f"(0..{len(PATTERNS) - 1})")
            self.spec = PATTERNS[int(self.pattern)]
        self._lookback = max(self.trend_window + self.dbl_sep_max,
                             self.triangle_lookback, self.channel_lookback,
                             self.cup_max_bars + self.handle_min_bars,
                             self.diamond_lookback, self.barr_lookback,
                             self.htf_pole_bars, 120) + 5
        self.bar_count = 0
        self.armed = {}      # symbol -> Armed
        self.held = {}       # symbol -> Held
        print(f"[patterns] trading {self.spec.name} ({self.spec.url})", flush=True)
        if not self.spec.tradeable:
            print(f"[patterns] {self.spec.name}: its page states the pattern "
                  "is not tradeable and gives no entry rule, so it is "
                  "identified but never armed", flush=True)

    def on_tick(self, ctx):
        w = ctx.history(self._lookback)
        if len(w) == 0:
            return
        ts = w.timestamp
        starts, ends = segments(ts)
        now = int(ts[-1])
        self.bar_count += 1

        self._manage_held(ctx, now, ts, starts, ends, w)
        self._manage_armed(ctx, now)

        for k in range(len(ends)):
            s, e = int(starts[k]), int(ends[k])
            sym = w.symbol[e]
            if sym in self.held or sym in self.armed:
                continue
            if ctx.position(sym) is not None:
                continue
            if len(self.held) + len(self.armed) >= self.max_positions:
                break
            b = Bars(w.open[s:e + 1], w.high[s:e + 1], w.low[s:e + 1],
                     w.close[s:e + 1], w.volume[s:e + 1])
            if b.n < 30 or float(b.c[-1]) < self.min_price:
                continue
            dv = sma(b.c * b.v, 20)
            if dv is None or not np.isfinite(dv) or dv < self.min_dollar_volume:
                continue
            setup = self.spec.detect(b, self)
            if setup is not None:
                self._arm(ctx, now, sym, setup)
        self._plot(ctx)

    # ─── order plumbing ──────────────────────────────────────────────────

    def _arm(self, ctx, now, sym, st):
        if not (np.isfinite(st.trigger) and np.isfinite(st.stop)):
            return
        if st.side == "long" and not st.stop < st.trigger:
            return
        if st.side == "short" and not st.trigger < st.stop:
            return
        if st.target is not None:
            if not np.isfinite(st.target):
                return
            # A measure-rule target that does not clear the confirmation price
            # is not tradable; the book's rule is kept rather than moved, so
            # the setup is skipped (module docstring, deviation 6).
            if st.side == "long" and st.target <= st.trigger:
                return
            if st.side == "short" and st.target >= st.trigger:
                return
        qty = self._size(ctx, st.trigger, st.stop)
        if qty <= 0.0:
            return
        entry_side = OrderSide.Buy if st.side == "long" else OrderSide.Sell
        exit_side = OrderSide.Sell if st.side == "long" else OrderSide.Buy
        entry_id = ctx.place_stop_order(symbol=sym, side=entry_side,
                                        quantity=qty, price=st.trigger)
        stop_id = ctx.place_stop_order(symbol=sym, side=exit_side, quantity=qty,
                                       price=st.stop, parent=entry_id,
                                       reduce_only=True)
        target_id = -1
        if self.use_measure_rule and st.target is not None:
            target_id = ctx.place_limit_order(symbol=sym, side=exit_side,
                                              quantity=qty, price=st.target,
                                              parent=entry_id, reduce_only=True)
        self.armed[sym] = Armed(entry_id, stop_id, target_id, st.side,
                                st.trigger, st.stop, qty, self.bar_count,
                                st.hold_bars)
        tp = "none" if st.target is None else f"{st.target:.4f}"
        self._print(now, sym,
                    f"{self.spec.name} {st.side.upper()} arm @ {st.trigger:.4f} | "
                    f"SL {st.stop:.4f} | TP {tp} | qty {qty:.6g} | {st.note}")

    def _manage_armed(self, ctx, now):
        for sym, ar in list(self.armed.items()):
            order = ctx.order(ar.entry_id)
            status = order.status if order is not None else OrderStatus.Cancelled
            if status == OrderStatus.Filled:
                del self.armed[sym]
                pos = ctx.position(sym)
                if pos is None:
                    self._print(now, sym, "filled and closed within the bar")
                    continue
                self.held[sym] = Held(ar.entry_id, ar.stop_id, ar.target_id,
                                      ar.side, ar.stop, float("nan"),
                                      ar.hold_bars)
                self._print(now, sym, f"confirmed, filled @ {float(pos.price):.4f}")
            elif status in (OrderStatus.Rejected, OrderStatus.Cancelled):
                del self.armed[sym]
            elif self.bar_count - ar.armed_bar >= self.order_bars:
                ctx.cancel_order(ar.entry_id)
                del self.armed[sym]
                self._print(now, sym, f"unconfirmed, order expired @ {ar.trigger:.4f}")

    def _manage_held(self, ctx, now, ts, starts, ends, w):
        printed = {w.symbol[int(e)] for e in ends}
        for sym, h in list(self.held.items()):
            if ctx.position(sym) is None:
                del self.held[sym]
                self._print(now, sym, "position closed")
                continue
            if sym not in printed:
                continue
            h.bars_held += 1
            limit = self.max_hold_bars if h.hold_bars is None else h.hold_bars
            if h.exiting or h.bars_held < limit:
                continue
            qty = abs(ctx.position(sym).quantity)
            side = OrderSide.Sell if h.side == "long" else OrderSide.Buy
            ctx.place_market_order(symbol=sym, side=side, quantity=qty,
                                   parent=h.entry_id, reduce_only=True)
            h.exiting = True
            self._print(now, sym, f"time stop after {h.bars_held} bars")

    def _size(self, ctx, entry, stop):
        """A stop-out costs `risk_fraction` of equity, both fee legs
        included; clamped by the notional cap and by free cash."""
        rpu = abs(entry - stop) + entry * self._fee + stop * self._fee
        if entry <= 0.0 or rpu <= 0.0 or not np.isfinite(rpu):
            return 0.0
        equity = ctx.equity()
        qty = min(equity * self.risk_fraction / rpu,
                  self.max_position_pct * equity / entry,
                  CASH_USE * ctx.cash() / (entry * (1.0 + self._fee)))
        return qty if qty > 0.0 and np.isfinite(qty) else 0.0

    def _plot(self, ctx):
        for sym, ar in self.armed.items():
            ctx.plot("trigger", sym, ar.trigger)
            ctx.plot("stop_level", sym, ar.stop)
        for sym, h in self.held.items():
            ctx.plot("stop_level", sym, h.stop)

    @staticmethod
    def _print(ts, symbol, msg):
        when = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        print(f"[{when.strftime('%Y-%m-%d %H:%M')} UTC] {symbol} {msg}", flush=True)
