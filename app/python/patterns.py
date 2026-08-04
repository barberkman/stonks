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
 7. No pyramiding and one position per symbol: the broker rejects same-side
    adds, and a symbol already holding a position is skipped by the scan.

Execution timeline: decisions on bar close, fills from the next bar on.
"""

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

    def __init__(self, o, h, l, c, v):
        self.o, self.h, self.l, self.c, self.v = o, h, l, c, v
        self.n = len(c)
        self._peaks = None
        self._valleys = None

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
    pivot, not several."""
    n = len(series)
    out = []
    for i in range(PIVOT_SPAN, n - PIVOT_SPAN):
        lo, hi = i - PIVOT_SPAN, i + PIVOT_SPAN + 1
        w = series[lo:hi]
        if want_high:
            if series[i] < np.max(w) or np.argmax(w) != PIVOT_SPAN:
                continue
        else:
            if series[i] > np.min(w) or np.argmin(w) != PIVOT_SPAN:
                continue
        out.append(i)
    return out


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
    """"Price trend: downward leading to the pattern." The close `window`
    bars before index `i` must sit at least `min_drop_pct` above the low
    at `i`."""
    j = i - window
    if j < 0:
        return False
    return pct(float(b.c[j]), float(b.l[i])) >= min_drop_pct


def trend_up_into(b, i, window, min_rise_pct):
    """"Price trend: upward leading to the pattern."""
    j = i - window
    if j < 0:
        return False
    return pct(float(b.h[i]), float(b.c[j])) >= min_rise_pct


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
    target: float      # measure-rule target
    note: str = ""     # what was matched, for the run log


@dataclass
class Spec:
    """Registry entry: a named pattern, its source page, and its detector."""

    name: str
    url: str
    kind: str          # "reversal" | "continuation" | "event" | "other"
    side: str          # the breakout direction the page names
    detect: Callable[["Bars", "PatternsStrategy"], Optional[Setup]]
    target_pct: float = 100.0   # page's "percentage meeting price target"


PATTERNS: List[Spec] = []


def pattern(name, url, kind, side, target_pct=100.0):
    """Register a detector under its book name."""

    def wrap(fn):
        PATTERNS.append(Spec(name, url, kind, side, fn, target_pct))
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


@dataclass
class Held:
    """A filled position: the bracket does the work, this tracks the clock."""

    entry_id: int
    stop_id: int
    target_id: int
    side: str
    stop: float
    target: float
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
        if not (np.isfinite(st.trigger) and np.isfinite(st.stop)
                and np.isfinite(st.target)):
            return
        if st.side == "long" and not st.stop < st.trigger < st.target:
            return
        if st.side == "short" and not st.target < st.trigger < st.stop:
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
        if self.use_measure_rule:
            target_id = ctx.place_limit_order(symbol=sym, side=exit_side,
                                              quantity=qty, price=st.target,
                                              parent=entry_id, reduce_only=True)
        self.armed[sym] = Armed(entry_id, stop_id, target_id, st.side,
                                st.trigger, st.stop, qty, self.bar_count)
        self._print(now, sym,
                    f"{self.spec.name} {st.side.upper()} arm @ {st.trigger:.4f} | "
                    f"SL {st.stop:.4f} | TP {st.target:.4f} | qty {qty:.6g} | "
                    f"{st.note}")

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
                                      ar.side, ar.stop, float("nan"))
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
            if h.exiting or h.bars_held < self.max_hold_bars:
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
