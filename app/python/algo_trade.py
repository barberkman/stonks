"""AlgoTrade — the bist manipulation model as a stonks strategy.

`ManipulationModel` is three XGBoost regressors on a drawdown-gated forward
excess-return target, ported from /Users/macmini-1/bist by way of
/Users/macmini-1/trade_algo. Despite the name it is not a manipulation
classifier: it predicts a sigma-multiple of forward market-excess return and
zeroes the cases you would have been stopped out of first, which makes it a
swing-long alpha ranker.

Client surface is two methods:

    model = ManipulationModel().train(pd.read_parquet("app/data/bist_1d.parquet"),
                                      train_end="2024-12-31")
    sigmas = model.signal(ctx.history(ManipulationModel.LOOKBACK))

plus `save`/`load`, because the strategy loads a pre-trained artifact rather
than training inside the engine.

The invariant everything rests on is that `train` and `signal` run the *same*
feature code, and that code is window-bounded: for any i >= LOOKBACK - 1,

    _features(panel)[i] == _features(panel[i - LOOKBACK + 1 : i + 1])[-1]

That identity is why a model fit on a whole panel can be scored one bar at a
time without the two disagreeing. Every feature here is therefore computable
from LOOKBACK rows and no more, which is why `obv` and
`days_since_past_extreme` are bounded below (see their comments) where bist
lets them run from a symbol's first bar.

Deviations from bist, all forced or deliberate:

 1. 72 features, not 79. bist's 7 intraday features aggregate 15-minute bars;
    this engine's feed is daily only. Unlike bist all three heads share one
    feature set, because bist's DN_FEATURE_EXCLUDE is entirely intraday.
 2. The label is forward-only. bist takes max(forward, centered), where the
    centered window reaches back before the signal bar — so a row can score
    high off a move that already started, which a next-bar entry cannot
    capture. bist's own precision figures overstate tradability because of it.
 3. Returns are cleaned before use. The feed's prices are not split- or
    bonus-adjusted and a bonus issue prints as a 2800x gain, so a bar-to-bar
    move outside the BIST daily band is booked as 0% (`price_limit_pct`, the
    same convention shorttermmomentum.py uses). bist's data is pre-adjusted
    upstream.
 4. Denominators that collapse to zero yield NaN, not bist's `+ EPSILON`. A
    halted or limit-locked name has genuinely undefined dispersion; 1e10 is not
    a large z-score, it is a rendering artifact that swamps the model. NaN is
    also stable, which `_ratio`'s docstring explains at length.
 5. There is no liquidity or universe filter. bist restricts scoring to names
    above its MIN_TURNOVER_PERCENTILE; here every symbol that printed a bar is
    eligible. Turnover is still a model *feature*, so the fit can learn from it,
    but nothing gates on it — which means a thin name that spiked on almost no
    volume can win a slot. That is the caveat shorttermmomentum.py documents for
    the same reason, and it is why this strategy's fills should be read as
    optimistic on the small end.

Requires xgboost in the venv (`app/python/.venv/bin/pip install xgboost`; on
macOS also `brew install libomp`). A failed import makes this module invisible
to strategy discovery, so check it before wondering where the strategy went.

Train the artifact from the project root:

    app/python/.venv/bin/python app/python/algo_trade.py --train-end 2024-12-31
"""

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import stonks
from stonks import OrderSide

log = logging.getLogger(__name__)

EPS = 1e-9

# Bars of history needed to produce one row of features. The binding constraint
# is obv_slope_20: a 20-bar slope over a 260-bar rolling sum reaches back 279.
LOOKBACK = 300

OBV_WINDOW = 260
EXTREME_GAP_CAP = 200  # <= LOOKBACK - 100; the trigger itself needs 100 bars
NO_EXTREME = 9999.0    # bist's sentinel for "no past extreme on record"

# A bar-to-bar close move this large is a corporate action or a bad tick, not a
# return. BIST equities trade inside a +/-20% band and the exchange rounds the
# band edges to the tick grid, which lets a genuine limit close print a shade
# past it, hence the headroom.
PRICE_LIMIT_PCT = 20.5
# For "did this print *at* the band" we want the band itself, without headroom.
LIMIT_HIT = 0.195

# bist/config.py::XGBOOST_PARAMS, minus the ensemble-only dispatch keys and
# translated to xgboost's native API: n_estimators -> N_ROUNDS, learning_rate ->
# eta, reg_lambda -> lambda, reg_alpha -> alpha, random_state -> seed, n_jobs ->
# nthread. bist's config records the Huber-vs-squared-error A/B that picked this
# objective (K=10 loss 30.9% vs 43.1%) — read it before changing it.
XGB_PARAMS = {
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "alpha": 0.0,
    "objective": "reg:pseudohubererror",
    "eval_metric": "mphe",
    "huber_slope": 1.0,
    "tree_method": "approx",
    "seed": 0,
    "nthread": -1,
}
N_ROUNDS = 400
EARLY_STOPPING = 30
VAL_FRACTION = 0.15
DEFAULT_DAILY_VOL = 0.03  # bist's fallback when excess_return_vol_60 is NaN
MIN_HISTORY = 60          # bist's MIN_HISTORY_DAYS
FEATURE_COVERAGE = 0.8    # a row needs this fraction of its features finite

# Where each head's training-set predictions fell, so `signal` can report a
# prediction as a percentile of what the fit actually produces.
#
# This exists because bist's absolute thresholds (h5 >= 3.0, h10 >= 2.0) do not
# transfer. A pseudo-Huber regression shrinks hard toward the conditional mean,
# so predicted sigma is not on the same scale as the label: this fit's
# predictions reach ~2-3 sigma at the very extreme and sit near 0.4 at the 99th
# percentile, where bist's reach past 3.0. bist trains on 2024+ only and against
# a larger max(forward, centered) target, both of which widen its output. Taking
# its numbers literally here selects nothing at all.
#
# Calibrating on the training set keeps the choice out of the test period.
QUANTILE_GRID = (0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 1.0)

# The 72 features, in the order `_features` stacks them. Explicit because column
# order is part of a fitted booster's contract: a saved model is only meaningful
# against this exact layout. Names are unchanged from bist so the two can be
# compared column by column.
FEATURE_NAMES = (
    "volume_zscore_5", "volume_ratio_5", "volume_zscore_20",
    "volume_ratio_20", "volume_zscore_60", "volume_ratio_60",
    "log_volume_ratio_20", "volume_pct_rank_60", "volume_acceleration",
    "volume_spike_count_20", "volume_cv_20", "volume_skew_20",
    "volume_kurt_20", "amihud_illiquidity_20", "vol_return_corr_20", "obv",
    "obv_slope_20", "turnover", "turnover_zscore_20", "turnover_ratio_20",
    "vwap_deviation_20", "log_return_1d", "cum_return_3", "cum_return_5",
    "cum_return_10", "cum_return_20", "body_ratio", "upper_shadow_ratio",
    "lower_shadow_ratio", "close_location_value", "overnight_gap",
    "intraday_return", "overnight_to_intraday_ratio_20",
    "body_ratio_mean_10", "body_ratio_std_10", "body_ratio_mean_20",
    "body_ratio_std_20", "bullish_candle_ratio_20", "shadow_asymmetry_20",
    "distance_from_ma_20", "distance_from_ma_50", "return_autocorr_30",
    "consecutive_up_days", "consecutive_down_days", "efficiency_ratio_20",
    "realized_vol_5", "realized_vol_20", "realized_vol_60",
    "parkinson_vol_20", "garman_klass_vol_20", "rogers_satchell_vol_20",
    "yang_zhang_vol_20", "vol_ratio_short_long", "vol_of_vol_20", "atr_14",
    "atr_expansion", "parkinson_to_close_vol_ratio_20",
    "corwin_schultz_spread_20", "roll_spread_20", "hl_range_ratio_20",
    "kyle_lambda_20", "volume_zscore_rank", "return_rank",
    "volatility_rank", "market_log_return_1d", "excess_return_vol_60",
    "market_realized_vol_60", "market_cum_return_60",
    "market_vol_ratio_short_long", "days_since_past_extreme",
    "move_concentration", "close_adj",
)
VOL_COLUMN = FEATURE_NAMES.index("excess_return_vol_60")

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class Head:
    """One (target, horizon) configuration. All three share the regressor."""

    name: str
    horizon: int
    direction: str              # "up" | "dn"
    max_drawdown: float | None  # None = unconditional, no drawdown gate


HEADS = (
    Head("up_h5", 5, "up", -0.10),
    Head("up_h10", 10, "up", -0.10),
    Head("dn_h5", 5, "dn", None),
)


# ---------------------------------------------------------------------------
# Rolling helpers. Every one takes a (T, N) frame and reduces down the time
# axis, so bist's per-symbol groupby-transform becomes a single call over all N
# columns. Nothing here looks forward.
# ---------------------------------------------------------------------------

def _std(x, w, mp):
    """Rolling std, with the residue on a flat window snapped to exact zero.

    A window whose values are all identical has zero dispersion. pandas
    computes rolling variance incrementally, so instead of 0.0 it can leave
    ~1e-6 of the level behind — and whether it does depends on how many windows
    were processed before it. That makes every z-score built on top
    path-dependent, which `_ratio`'s `den > 0` test then amplifies into a
    NaN-or-not disagreement between batch and windowed scoring.

    max and min carry no accumulation, so comparing them detects a flat window
    exactly. BIST halts and limit-locks constantly, so this is common, not a
    corner case.
    """
    s = x.rolling(w, min_periods=mp).std()
    r = x.rolling(w, min_periods=mp)
    return s.where(r.max() != r.min(), 0.0)


def _wsum(x, w, mp, chunk=256):
    """Rolling sum computed fresh from each window rather than incrementally.

    The same failure as `_std`, one level down. pandas advances a rolling sum by
    adding the entering value and subtracting the leaving one, so the answer
    carries the cancellation history of everything before it: a window of
    returns summing to ~0 lands on 5e-17 or on 0.0 depending on how much panel
    preceded it. That is far below any tolerance worth caring about until a tree
    split sits at zero, at which point the row goes down the other branch.

    Summing the window itself is order-independent given the same contents,
    which is exactly what the windowing contract needs.
    """
    a = x.to_numpy(dtype=float)
    T, N = a.shape
    padded = np.vstack([np.full((w - 1, N), np.nan), a])
    out = np.full((T, N), np.nan)
    for lo in range(0, T, chunk):
        hi = min(lo + chunk, T)
        v = np.lib.stride_tricks.sliding_window_view(
            padded[lo:hi + w - 1], w, axis=0)
        n = np.isfinite(v).sum(axis=2)
        out[lo:hi] = np.where(n >= mp, np.nansum(v, axis=2), np.nan)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _var(x, w, mp):
    """Rolling variance, flat windows snapped to zero. See `_std`."""
    v = x.rolling(w, min_periods=mp).var()
    r = x.rolling(w, min_periods=mp)
    return v.where(r.max() != r.min(), 0.0)


def _moments(x, w, mp, chunk=256):
    """Exact rolling (skew, kurt), matching pandas' bias-corrected definitions.

    pandas computes these incrementally, which fails on volume: the series spans
    zero to 1e14, so once a symbol's big days age out of the window the running
    power sums have lost the precision to describe what is left. A dead name
    whose window is nineteen zeros and a single one-lot trade has a real skew of
    4.47, and the incremental path returns NaN for it — but only if enough
    history preceded it, which makes it path-dependent as well as wrong.

    Recomputing each window from scratch over a strided view is exact and
    order-independent. Chunked over time because the (T, N, w) intermediates
    would otherwise run to hundreds of MB.
    """
    a = x.to_numpy(dtype=float)
    T, N = a.shape
    padded = np.vstack([np.full((w - 1, N), np.nan), a])
    skew = np.full((T, N), np.nan)
    kurt = np.full((T, N), np.nan)

    for lo in range(0, T, chunk):
        hi = min(lo + chunk, T)
        v = np.lib.stride_tricks.sliding_window_view(
            padded[lo:hi + w - 1], w, axis=0)              # (hi-lo, N, w)
        n = np.isfinite(v).sum(axis=2).astype(float)
        # An all-NaN window is a symbol that was not listed yet; nanmean warns
        # and returns NaN, which is the answer we want. `ok` drops it either way.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = v - np.nanmean(v, axis=2, keepdims=True)
            m2 = np.nanmean(d ** 2, axis=2)
            m3 = np.nanmean(d ** 3, axis=2)
            m4 = np.nanmean(d ** 4, axis=2)

        with np.errstate(invalid="ignore", divide="ignore"):
            ok = (n >= mp) & (n >= 4) & (m2 > 0)
            g1 = m3 / m2 ** 1.5
            skew[lo:hi] = np.where(ok, np.sqrt(n * (n - 1)) / (n - 2) * g1, np.nan)
            g2 = m4 / m2 ** 2 - 3.0
            kurt[lo:hi] = np.where(
                ok, ((n + 1) * g2 + 6.0) * (n - 1) / ((n - 2) * (n - 3)), np.nan)

    return (pd.DataFrame(skew, index=x.index, columns=x.columns),
            pd.DataFrame(kurt, index=x.index, columns=x.columns))


def _ratio(num, den):
    """num / den, NaN where the denominator has collapsed to zero.

    Every denominator this is used on is a dispersion measure — a rolling
    standard deviation, an intraday range, a sum of absolute moves. bist guards
    these with `+ EPSILON`, which is fine until the dispersion is genuinely
    zero: a halted or limit-locked name whose price has not moved for twenty
    bars. Then `x / 1e-9` is not a large z-score, it is an undefined ratio
    rendered as 1e10, and it swamps everything the model sees.

    It is also not reproducible. Whether the collapsed std lands on exactly 0.0
    or on 1.2e-06 depends on floating-point accumulation order inside pandas'
    incremental rolling, so the same bar scores differently depending on how
    much history preceded it. NaN is both the honest answer and a stable one —
    XGBoost handles it natively.
    """
    return (num / den).where(den > 0)


def _slope(x, w, mp):
    """Rolling OLS slope of each column on the time index, NaN-aware.

    bist uses rolling().apply(), which on a wide panel would be T*N Python
    calls. Closed form instead: b = (Stv - St*Sv/n) / (Stt - St^2/n) over the
    valid points only. Slope is invariant to shifting the index, so using the
    global row number in place of a window-local 0..w-1 changes nothing.
    """
    t = pd.DataFrame(
        np.repeat(np.arange(len(x), dtype=float)[:, None], x.shape[1], axis=1),
        index=x.index, columns=x.columns,
    )
    m = x.notna()
    u = x.fillna(0.0)
    t = t.where(m, 0.0)

    n = m.astype(float).rolling(w, min_periods=mp).sum()
    st = t.rolling(w, min_periods=mp).sum()
    stt = (t * t).rolling(w, min_periods=mp).sum()
    sv = u.rolling(w, min_periods=mp).sum()
    stv = (t * u).rolling(w, min_periods=mp).sum()

    denom = stt - st * st / n
    return ((stv - st * sv / n) / denom.where(denom.abs() > EPS)).where(n >= mp)


def _corr(x, y, w, mp):
    """Rolling Pearson correlation via corr = (E[xy] - E[x]E[y]) / (sx*sy).

    The identity bist already uses for vol_return_corr_20, applied to
    return_autocorr_30 as well so both take one code path. Differs from pandas'
    .corr() only in ddof bookkeeping.
    """
    mx = x.rolling(w, min_periods=mp).mean()
    my = y.rolling(w, min_periods=mp).mean()
    mxy = (x * y).rolling(w, min_periods=mp).mean()
    sx = _std(x, w, mp)
    sy = _std(y, w, mp)
    return _ratio(mxy - mx * my, sx * sy)


def _run_length(flag):
    """(T, N) bool -> length of the True run ending at each row.

    One pass down the time axis carrying a per-symbol counter, which is as much
    state as a causal feature may hold.
    """
    out = np.zeros(flag.shape)
    run = np.zeros(flag.shape[1])
    for i in range(len(flag)):
        run = np.where(flag[i], run + 1.0, 0.0)
        out[i] = run
    return out


def _bars_since(trigger, cap):
    """(T, N) bool -> bars since the last True, capped, sentinel beyond it.

    bist counts forever from a symbol's first trigger. That value cannot be
    reproduced from a bounded window, so it is capped here — past `cap` bars
    with no trigger the answer collapses to the same sentinel bist uses for
    "never".
    """
    out = np.full(trigger.shape, NO_EXTREME)
    since = np.full(trigger.shape[1], np.inf)
    for i in range(len(trigger)):
        since = np.where(trigger[i], 0.0, since + 1.0)
        out[i] = np.where(since <= cap, since, NO_EXTREME)
    return out


def _move_concentration(abs_ret, thr):
    """Peers on the same date whose |return| clears this row's own threshold.

    Per bist: for row i, #{j != i on the same date : |r_j| > 2 * vol_i}. The
    threshold varies per row, so there is no single per-date count to
    precompute — sort the date's |returns| once and binary-search each row's own
    threshold into it. The self-subtraction at the end matters: a symbol that
    clears its own bar must not count itself.
    """
    out = np.full(abs_ret.shape, np.nan)
    for i in range(len(abs_ret)):
        a, t = abs_ret[i], thr[i]
        known = np.isfinite(t)
        if not known.any():
            continue
        srt = np.sort(a[np.isfinite(a)])
        count = srt.size - np.searchsorted(srt, t[known], side="right")
        own = a[known]
        count = count - (np.isfinite(own) & (own > t[known]))
        out[i, known] = np.maximum(count, 0)
    return out


# ---------------------------------------------------------------------------
# Panel plumbing. Both entry points converge on {field: (T, N) DataFrame},
# indexed by timestamp and columned by symbol.
# ---------------------------------------------------------------------------

def _returns(C, price_limit_pct=PRICE_LIMIT_PCT):
    """Bar-to-bar close returns with corporate actions booked as 0%.

    The feed's prices are not split- or bonus-adjusted, so a bonus issue prints
    as a 2800x gain and a split as a -50% crash. Neither is a return, and left
    alone either one dominates every window it lands in. bist takes its prices
    pre-adjusted from upstream; this is the substitute, and it is the same rule
    shorttermmomentum.py applies for the same reason.

    Price *levels* still come from close — bist's own close_adj is unadjusted
    too, which caveat 3 in the plan notes.
    """
    r = C / C.shift(1) - 1.0
    # .where's else-branch would swallow NaN into 0.0, turning "no bar" into "no
    # move"; restore it so halted names stay unknown rather than flat.
    return r.where(r.abs() <= price_limit_pct / 100.0, 0.0).where(r.notna())


def _panel(df):
    """Long-format bars -> {field: (T, N) DataFrame} plus the cleaned `ret`.

    `df` carries the columns app/data/bist_1d.parquet does: timestamp, symbol,
    open, high, low, close, volume. Missing (timestamp, symbol) cells — a late
    listing, a halt mid-window — land as NaN, which every feature already
    expects.
    """
    panel = {f: df.pivot(index="timestamp", columns="symbol", values=f).sort_index()
             for f in FIELDS}
    panel["ret"] = _returns(panel["close"])
    return panel


def _panel_from_window(window):
    """`ctx.history(n)` -> the same panel shape.

    The window is a ragged long view: rows contiguous and ascending per symbol,
    each symbol's slice ending at the current timestamp, and only symbols that
    printed at that timestamp present at all. Everything past the current bar is
    absent by construction, so the engine's structural lookahead guarantee
    carries into the model without the model having to promise anything.
    """
    df = pd.DataFrame({
        "symbol": list(window.symbol),
        "timestamp": np.asarray(window.timestamp),
        "open": np.asarray(window.open, dtype=float),
        "high": np.asarray(window.high, dtype=float),
        "low": np.asarray(window.low, dtype=float),
        "close": np.asarray(window.close, dtype=float),
        "volume": np.asarray(window.volume, dtype=float),
    })
    return _panel(df)


def _truncate(panel, end):
    """Panel restricted to its first `end` rows."""
    return {k: v.iloc[:end] for k, v in panel.items()}


# ---------------------------------------------------------------------------
# Feature groups. One function per bist module, same feature names, same
# windows and min_periods. Insertion order must match FEATURE_NAMES.
# ---------------------------------------------------------------------------

def _limit_hit(ret, H, L, V):
    """Bars that printed at the price band or locked with no range.

    Load-bearing, not cosmetic: it masks the candle-geometry, Parkinson /
    Garman-Klass / Rogers-Satchell and Corwin-Schultz features, all of which
    divide by an intraday range that collapses on a limit day.
    """
    locked = H.notna() & L.notna() & (H == L) & (V > 0)
    return (ret.abs() >= LIMIT_HIT) | locked


def _true_range(H, L, prev_C):
    return np.maximum(np.maximum((H - L).abs(), (H - prev_C).abs()),
                      (L - prev_C).abs())


def _volume_features(out, C, H, L, V, log_ret):
    """Port of bist/features/volume_features.py (21)."""
    abs_ret = log_ret.abs()

    for w in (5, 20, 60):
        mp = max(2, w // 2)
        mean = V.rolling(w, min_periods=mp).mean()
        std = _std(V, w, mp)
        out[f"volume_zscore_{w}"] = _ratio(V - mean, std)
        out[f"volume_ratio_{w}"] = _ratio(V, mean)

    ma5 = V.rolling(5, min_periods=3).mean()
    ma20 = V.rolling(20, min_periods=10).mean()
    std20 = _std(V, 20, 10)

    out["log_volume_ratio_20"] = np.log(_ratio(V.where(V > 0), ma20))
    out["volume_pct_rank_60"] = V.rolling(60, min_periods=30).rank(pct=True)
    out["volume_acceleration"] = _ratio(ma5, ma20)
    out["volume_spike_count_20"] = (
        (V > 2.0 * ma20).astype(float).rolling(20, min_periods=10).sum())
    out["volume_cv_20"] = _ratio(std20, ma20.abs())
    out["volume_skew_20"], out["volume_kurt_20"] = _moments(V, 20, 10)
    out["amihud_illiquidity_20"] = (
        _ratio(abs_ret, V).rolling(20, min_periods=10).mean())
    out["vol_return_corr_20"] = _corr(V, abs_ret, 20, 10)

    # bist accumulates OBV from the symbol's first bar. An unbounded cumsum
    # cannot be recovered from a fixed window and its scale depends on how much
    # history happens to exist, so this is net signed volume over OBV_WINDOW.
    signed = np.sign(log_ret.fillna(0.0)) * V
    obv = signed.rolling(OBV_WINDOW, min_periods=OBV_WINDOW // 2).sum()
    out["obv"] = obv
    out["obv_slope_20"] = _slope(obv, 20, 10)

    typical = (H + L + C) / 3.0
    turnover = V * typical
    to_mean = turnover.rolling(20, min_periods=10).mean()
    to_std = _std(turnover, 20, 10)
    out["turnover"] = turnover
    out["turnover_zscore_20"] = _ratio(turnover - to_mean, to_std)
    out["turnover_ratio_20"] = _ratio(turnover, to_mean)

    vwap = _ratio((typical * V).rolling(20, min_periods=10).sum(),
                  V.rolling(20, min_periods=10).sum())
    out["vwap_deviation_20"] = _ratio(C - vwap, _std(typical, 20, 10))


def _price_features(out, O, H, L, C, prev_C, log_ret, limit):
    """Port of bist/features/price_features.py (24)."""
    out["log_return_1d"] = log_ret
    for h in (3, 5, 10, 20):
        out[f"cum_return_{h}"] = _wsum(log_ret, h, max(2, h // 2))

    hl = H - L
    safe_hl = hl.where(~limit & (hl > 0))
    body = C - O
    upper = H - np.maximum(O, C)
    lower = np.minimum(O, C) - L

    out["body_ratio"] = body / safe_hl
    out["upper_shadow_ratio"] = upper / safe_hl
    out["lower_shadow_ratio"] = lower / safe_hl
    out["close_location_value"] = ((C - L) - (H - C)) / safe_hl
    out["overnight_gap"] = _ratio(O - prev_C, prev_C)
    out["intraday_return"] = _ratio(C - O, O)
    out["overnight_to_intraday_ratio_20"] = _ratio(
        out["overnight_gap"].abs().rolling(20, min_periods=10).sum(),
        out["intraday_return"].abs().rolling(20, min_periods=10).sum())

    for w in (10, 20):
        mp = max(3, w // 2)
        out[f"body_ratio_mean_{w}"] = out["body_ratio"].rolling(w, min_periods=mp).mean()
        out[f"body_ratio_std_{w}"] = _std(out["body_ratio"], w, mp)

    out["bullish_candle_ratio_20"] = (
        (C > O).astype(float).rolling(20, min_periods=10).mean())
    out["shadow_asymmetry_20"] = _ratio(
        upper.where(~limit).rolling(20, min_periods=10).sum(),
        lower.where(~limit).rolling(20, min_periods=10).sum())

    tr = _true_range(H, L, prev_C)
    atr14 = tr.rolling(14, min_periods=7).mean()
    for w in (20, 50):
        ma = C.rolling(w, min_periods=max(5, w // 4)).mean()
        out[f"distance_from_ma_{w}"] = _ratio(C - ma, atr14)

    out["return_autocorr_30"] = _corr(log_ret, log_ret.shift(1), 30, 20)
    out["consecutive_up_days"] = pd.DataFrame(
        _run_length((log_ret > 0).to_numpy()), index=C.index, columns=C.columns)
    out["consecutive_down_days"] = pd.DataFrame(
        _run_length((log_ret < 0).to_numpy()), index=C.index, columns=C.columns)

    net = C - C.shift(20)
    out["efficiency_ratio_20"] = _ratio(
        net.abs(), C.diff().abs().rolling(20, min_periods=10).sum())


def _volatility_features(out, O, H, L, C, prev_C, log_ret, limit):
    """Port of bist/features/volatility_features.py (12)."""
    ln2 = np.log(2.0)
    ok = ~limit

    for w in (5, 20, 60):
        out[f"realized_vol_{w}"] = _std(log_ret, w, max(3, w // 2))

    pk = (np.log(H / L) ** 2 / (4.0 * ln2)).where(ok)
    out["parkinson_vol_20"] = np.sqrt(
        pk.rolling(20, min_periods=10).mean().clip(lower=0.0))

    gk = (0.5 * np.log(H / L) ** 2 - (2.0 * ln2 - 1.0) * np.log(C / O) ** 2).where(ok)
    out["garman_klass_vol_20"] = np.sqrt(
        gk.rolling(20, min_periods=10).mean().clip(lower=0.0))

    rs = (np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O)).where(ok)
    out["rogers_satchell_vol_20"] = np.sqrt(
        rs.rolling(20, min_periods=10).mean().clip(lower=0.0))

    # Yang-Zhang: overnight var + k * open-to-close var + (1-k) * Rogers-Satchell
    on_var = _var(np.log(O / prev_C), 20, 10)
    oc_var = _var(np.log(C / O).where(ok), 20, 10)
    rs_mean = rs.rolling(20, min_periods=10).mean()
    k = 0.34 / (1.34 + 21.0 / 19.0)
    out["yang_zhang_vol_20"] = np.sqrt(
        (on_var + k * oc_var + (1.0 - k) * rs_mean).clip(lower=0.0))

    out["vol_ratio_short_long"] = _ratio(out["realized_vol_5"], out["realized_vol_60"])
    out["vol_of_vol_20"] = _std(out["realized_vol_5"], 20, 10)

    tr = _true_range(H, L, prev_C)
    out["atr_14"] = tr.rolling(14, min_periods=7).mean()
    out["atr_expansion"] = _ratio(tr.rolling(5, min_periods=3).mean(),
                                  tr.rolling(20, min_periods=10).mean())
    out["parkinson_to_close_vol_ratio_20"] = _ratio(
        out["parkinson_vol_20"], out["realized_vol_20"])


def _microstructure_features(out, H, L, C, prev_C, V, log_ret, limit):
    """Port of bist/features/microstructure.py (4)."""
    # Corwin-Schultz (2012). Negative alpha implies a negative spread; the
    # estimator is known to misfire when a jump swamps the transaction-cost
    # signal, so those days drop out rather than being clamped to zero.
    prev_H, prev_L = H.shift(1), L.shift(1)
    beta = np.log(H / L) ** 2 + np.log(prev_H / prev_L) ** 2
    gamma = np.log(np.maximum(H, prev_H) / np.minimum(L, prev_L)) ** 2
    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread = spread.where(~(limit | limit.shift(1, fill_value=False)))
    spread = spread.where(spread > 0)
    out["corwin_schultz_spread_20"] = spread.rolling(20, min_periods=10).mean()

    # Roll (1984): S = 2*sqrt(-cov(dp_t, dp_{t-1})). Positive autocovariance is
    # inconsistent with the model, so those windows are NaN rather than forced.
    dp = C.diff()
    dp_lag = dp.shift(1)
    n, mp = 20, 10
    cov = (dp * dp_lag).rolling(n, min_periods=mp).mean() - (
        dp.rolling(n, min_periods=mp).mean() * dp_lag.rolling(n, min_periods=mp).mean())
    out["roll_spread_20"] = 2.0 * np.sqrt(-cov.where(cov < 0))

    out["hl_range_ratio_20"] = (
        _ratio(H - L, C).where(~limit).rolling(20, min_periods=10).mean())

    # Kyle (1985) lambda as the rolling slope of |return| on volume,
    # cov(|r|, V) / var(V).
    abs_ret = log_ret.abs()
    mv = V.rolling(n, min_periods=mp).mean()
    mr = abs_ret.rolling(n, min_periods=mp).mean()
    mvr = (V * abs_ret).rolling(n, min_periods=mp).mean()
    out["kyle_lambda_20"] = _ratio(mvr - mv * mr, _var(V, n, mp))


def _cross_sectional_features(out, C, log_ret):
    """Port of bist/features/cross_sectional.py (10).

    These rank each symbol against its peers on the same date, so they break
    per-symbol isolation by design: adding a symbol changes every other symbol's
    rank for that date. Per-symbol causality is untouched — a rank on date t
    reads only date t.

    Note the consequence at inference: `ctx.history` only returns symbols that
    printed this tick, so the peer set is narrower than the training panel's.
    With ~600 BIST names printing daily the market mean is a fine estimate, but
    it is not the identical computation.
    """
    # Rank the float32-rounded values, which is what actually reaches the model.
    # Two symbols whose underlying feature differs only in the last bits of a
    # float64 would otherwise sort differently depending on accumulation order,
    # moving both by one rank — 1/N, small but not reproducible. Rounding first
    # turns those near-ties into exact ties, which method="average" resolves
    # identically every time.
    def _rank(frame):
        return frame.astype(np.float32).rank(axis=1, pct=True)

    out["volume_zscore_rank"] = _rank(out["volume_zscore_20"])
    out["return_rank"] = _rank(out["log_return_1d"])
    out["volatility_rank"] = _rank(out["realized_vol_20"])

    # Equal-weighted mean log return: the standard market/idiosyncratic split.
    market = log_ret.mean(axis=1)
    out["market_log_return_1d"] = pd.DataFrame(
        np.repeat(market.to_numpy()[:, None], C.shape[1], axis=1),
        index=C.index, columns=C.columns)

    excess = log_ret.sub(market, axis=0)
    vol60 = _std(excess, 60, 30)
    out["excess_return_vol_60"] = vol60

    for name, series in (
        ("market_realized_vol_60", _std(market.to_frame(), 60, 30).iloc[:, 0]),
        ("market_cum_return_60", _wsum(market.to_frame(), 60, 30).iloc[:, 0]),
        ("market_vol_ratio_short_long",
         _ratio(_std(market.to_frame(), 20, 10).iloc[:, 0],
                _std(market.to_frame(), 60, 30).iloc[:, 0])),
    ):
        out[name] = pd.DataFrame(
            np.repeat(series.to_numpy()[:, None], C.shape[1], axis=1),
            index=C.index, columns=C.columns)

    # Past-only mirror of the forward label: has this symbol had an
    # idiosyncratic 3-sigma run over the last 100 bars? Counts from the 0->1
    # transition, not from "the condition still holds" — a spike stays inside
    # the rolling window for ~100 days after the fact.
    past = _wsum(excess, 100, 50)
    hit = (past > 3.0 * vol60 * np.sqrt(100.0)).fillna(False)
    trigger = hit & ~hit.shift(1, fill_value=False)
    out["days_since_past_extreme"] = pd.DataFrame(
        _bars_since(trigger.to_numpy(), EXTREME_GAP_CAP),
        index=C.index, columns=C.columns)

    out["move_concentration"] = pd.DataFrame(
        _move_concentration(log_ret.abs().to_numpy(),
                            (2.0 * out["realized_vol_20"]).to_numpy()),
        index=C.index, columns=C.columns)


def _log_returns(panel):
    """Cleaned returns as log returns, shared by features and labels."""
    ret = panel["ret"]
    return np.log1p(ret.where(ret > -0.99))


def _features(panel):
    """(T, N, 72) float32, aligned to the rows of `panel`.

    Causal, and window-bounded: for any i >= LOOKBACK - 1,

        _features(panel)[i] == _features(_window(panel, i))[-1]

    That identity is what lets `train` run on a whole panel while `signal` runs
    on 300 bars. Every feature is therefore computable from LOOKBACK rows.
    """
    O, H, L, C, V = (panel[f] for f in FIELDS)
    prev_C = C.shift(1)
    log_ret = _log_returns(panel)
    limit = _limit_hit(panel["ret"], H, L, V)

    out = {}
    _volume_features(out, C, H, L, V, log_ret)
    _price_features(out, O, H, L, C, prev_C, log_ret, limit)
    _volatility_features(out, O, H, L, C, prev_C, log_ret, limit)
    _microstructure_features(out, H, L, C, prev_C, V, log_ret, limit)
    _cross_sectional_features(out, C, log_ret)
    out["close_adj"] = C

    if tuple(out) != FEATURE_NAMES:
        raise RuntimeError(
            "feature layout drifted from FEATURE_NAMES; a fitted booster is "
            "only valid against the order it was trained on")

    stacked = np.stack(
        [np.asarray(out[k], dtype=np.float32) for k in FEATURE_NAMES], axis=-1)
    return np.where(np.isfinite(stacked), stacked, np.nan)


def _labels(panel, head):
    """(T, N) target for one head. Reads forward — training only.

    Forward-only port of bist/labels/synthetic_labels.py:

        sigma = cum excess log return over [T+1, T+H] / (excess_vol_60 * sqrt H)
        up:     zeroed where the forward window drew down past max_drawdown
        dn:     sign-flipped, unconditional

    bist takes max(forward, centered) where centered reaches back before T. That
    lets a row score high off a move already under way, which an entry at T+1
    cannot capture, so the centered term is dropped here.

    The drawdown gate is what makes the up target tradable rather than a pure
    move detector: it zeroes the spikes you would have been stopped out of
    before they arrived.
    """
    h = head.horizon
    C = panel["close"]
    log_ret = _log_returns(panel)

    market = log_ret.mean(axis=1)
    excess = log_ret.sub(market, axis=0)
    denom = _std(excess, 60, 30) * np.sqrt(h)

    mp = min(max(2, h // 2), h)
    fwd = _wsum(excess, h, mp).shift(-h)          # [T+1, T+h]

    sign = 1.0 if head.direction == "up" else -1.0
    sigma = _ratio(sign * fwd, denom)

    if head.max_drawdown is None:
        return sigma.to_numpy(dtype=np.float32)

    fwd_min = C.rolling(h, min_periods=mp).min().shift(-h)
    drawdown = _ratio(fwd_min - C, C)
    loss = drawdown < float(head.max_drawdown)
    # A row whose drawdown is unresolved is not "clean", it is unknown.
    gated = sigma.where(~loss, 0.0).where(sigma.notna() & drawdown.notna())
    return gated.to_numpy(dtype=np.float32)


class ManipulationModel:
    """XGBoost regressors on bist's drawdown-gated extreme-move targets.

    Data in, alpha out; knows nothing about the engine, the portfolio, or the
    training schedule. `train` needs a panel's worth of history, `signal` needs
    LOOKBACK bars, and both go through the same feature code — which is what
    makes them the same model.
    """

    LOOKBACK = LOOKBACK

    def __init__(self, heads=HEADS, *, k_sigma=3.0, params=None, rounds=N_ROUNDS,
                 winsorize=(0.005, 0.995), min_history=MIN_HISTORY):
        self.heads = tuple(heads)
        self.k_sigma = float(k_sigma)
        self.params = dict(params or XGB_PARAMS)
        self.rounds = int(rounds)
        self.winsorize = winsorize
        self.min_history = int(min_history)
        self.models = {}
        self.best_iteration = {}
        self.train_end = None   # last timestamp the fit was allowed to see, ms
        self.quantiles = {}     # head -> predicted sigma at each QUANTILE_GRID point
        self._bounds = None     # (lower, upper), each (72,)

    # -- train --------------------------------------------------------------

    def train(self, df, *, train_end=None):
        """Fit every head on bars up to and including `train_end`.

        `df` is long-format pandas with app/data/bist_1d.parquet's columns.
        `train_end` is a date; bars after it are dropped *before* labels are
        built, so a label whose forward window would reach past the cutoff
        resolves to NaN and is dropped. The embargo is structural, not a
        convention the caller has to remember.
        """
        panel = df if isinstance(df, dict) else _panel(df)
        index = panel["close"].index
        end = len(index)
        if train_end is not None:
            cutoff = pd.Timestamp(train_end)
            values = index.to_numpy()
            if np.issubdtype(values.dtype, np.integer):
                cutoff = cutoff.value // 1_000_000     # ms, as the engine stamps
            else:
                cutoff = np.datetime64(cutoff)
            end = int(np.searchsorted(values, cutoff, side="right"))
        if end <= 0:
            raise ValueError(f"train_end={train_end} precedes every bar")

        panel = _truncate(panel, end)
        last = panel["close"].index[-1]
        self.train_end = (int(last) if isinstance(last, (int, np.integer))
                          else int(pd.Timestamp(last).value // 1_000_000))

        F = _features(panel)
        # bist fits per-date winsorize bounds and forward-fills them to any
        # scoring date past the training window. Every date we score is past it,
        # so the ffill always lands on the last training date — keep that row.
        with warnings.catch_warnings():
            # A feature that is all-NaN on some date has no quantile. np.clip
            # against NaN bounds leaves the column untouched, which is correct.
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = np.nanquantile(F, self.winsorize[0], axis=1)   # (T, 72) per date
            hi = np.nanquantile(F, self.winsorize[1], axis=1)
        self._bounds = (lo[-1], hi[-1])
        X = np.clip(F, lo[:, None, :], hi[:, None, :])

        listed = np.isfinite(panel["close"].to_numpy(dtype=float))
        age = np.cumsum(listed, axis=0)
        enough = np.isfinite(X).sum(axis=2) >= FEATURE_COVERAGE * X.shape[2]
        row = np.arange(end)[:, None]

        for head in self.heads:
            y = _labels(panel, head)
            # bist drops the last `horizon` dates from training outright. Here
            # the forward-only label makes those rows NaN anyway, but the filter
            # stays as the explicit statement of the embargo.
            keep = (np.isfinite(y) & enough & (age >= self.min_history)
                    & (row < end - head.horizon))
            if keep.sum() < 1000:
                raise RuntimeError(
                    f"{head.name}: only {int(keep.sum())} usable training rows")

            # Boolean masking walks row-major, so samples come out in date order
            # and the chronological split for early stopping is a slice.
            #
            # NaN is left in place. bist fills it with 0.0 because its
            # IsolationForest and autoencoder cannot take NaN, but XGBoost
            # handles missing natively by learning a default direction per split.
            # Filling would also undo `_ratio`: an undefined z-score on a halted
            # name would come back as 0.0, telling the model "perfectly average"
            # when the truth is "unknown".
            Xh, yh = X[keep], y[keep]
            split = int(len(Xh) * (1.0 - VAL_FRACTION))
            names = list(FEATURE_NAMES)
            dtrain = xgb.DMatrix(Xh[:split], label=yh[:split], feature_names=names)
            dval = xgb.DMatrix(Xh[split:], label=yh[split:], feature_names=names)
            booster = xgb.train(
                self.params, dtrain, num_boost_round=self.rounds,
                evals=[(dval, "val")], early_stopping_rounds=EARLY_STOPPING,
                verbose_eval=False,
            )
            self.models[head.name] = booster
            self.best_iteration[head.name] = int(booster.best_iteration)

            # Where this head's predictions land over every row the strategy
            # would have been willing to score. `signal` turns a raw sigma into
            # a percentile against this, which is what makes the entry rule
            # survive a retrain — see QUANTILE_GRID.
            scored = self._predict(head, X[enough])
            self.quantiles[head.name] = [
                float(q) for q in np.quantile(scored, QUANTILE_GRID)]

            resolved = np.flatnonzero(keep.any(axis=1))
            log.info(
                "%s: %d rows, newest label row %d reads through %d (< end=%d), "
                "above %.1f sigma %.3f%%, best_iter=%d, "
                "pred q99=%.3f q999=%.3f max=%.3f",
                head.name, len(Xh), resolved[-1], resolved[-1] + head.horizon,
                end, self.k_sigma, 100.0 * float((yh > self.k_sigma).mean()),
                self.best_iteration[head.name],
                self.quantiles[head.name][QUANTILE_GRID.index(0.99)],
                self.quantiles[head.name][QUANTILE_GRID.index(0.999)],
                self.quantiles[head.name][-1],
            )
        return self

    def _predict(self, head, rows):
        """Raw predicted sigma for a (rows, 72) design matrix."""
        booster = self.models[head.name]
        matrix = xgb.DMatrix(rows, feature_names=list(FEATURE_NAMES))
        return booster.predict(
            matrix, iteration_range=(0, self.best_iteration[head.name] + 1))

    # -- score --------------------------------------------------------------

    def signal(self, window):
        """Per-head alpha for the current bar, one row per printing symbol.

        Returns a DataFrame indexed by symbol with three columns per head:

            <head>      raw predicted sigma; bist ranks on this and nothing else
            <head>_pct  bist's expected_excess_pct, (exp(sigma*vol*sqrt H)-1)*100
                        — the forward *excess* return in percent, which unlike
                        raw sigma is comparable across horizons
            <head>_q    where that sigma falls in the training-set prediction
                        distribution, in [0, 1]

        bist's caveat on the first two, verbatim: "NOT a calibrated forecast;
        rank by sigma, treat exp% as a magnitude check." `_q` is the column to
        threshold on, because raw sigma's scale depends on what the fit saw —
        see QUANTILE_GRID.

        Rows with too little history or too few finite features score NaN rather
        than a number derived mostly from missing inputs.
        """
        if not self.models:
            raise RuntimeError("train() or load() before signal()")

        panel = window if isinstance(window, dict) else _panel_from_window(window)
        F = _features(panel)
        symbols = panel["close"].columns

        lo, hi = self._bounds
        last = F[-1]                                        # (N, 72)
        X = np.clip(last, lo, hi)

        # bist reads the *unwinsorized* excess_return_vol_60 here, and falls
        # back to a flat 3% daily vol where it is missing.
        vol = last[:, VOL_COLUMN]
        vol = np.where(np.isfinite(vol), vol, DEFAULT_DAILY_VOL)

        listed = np.isfinite(panel["close"].to_numpy(dtype=float)).sum(axis=0)
        usable = ((np.isfinite(last).sum(axis=1) >= FEATURE_COVERAGE * last.shape[1])
                  & (listed >= self.min_history))

        out = pd.DataFrame(index=symbols)
        for head in self.heads:
            sigma = np.where(usable, self._predict(head, X), np.nan)
            out[head.name] = sigma
            out[f"{head.name}_pct"] = np.expm1(
                sigma * vol * np.sqrt(head.horizon)) * 100.0
            # Beyond the grid's ends np.interp clamps, which is what we want: a
            # prediction above anything seen in training is simply "the top".
            out[f"{head.name}_q"] = np.interp(
                sigma, self.quantiles[head.name], QUANTILE_GRID)

        return out

    # -- persistence --------------------------------------------------------

    def save(self, directory):
        """Same layout as bist's configured_alerts cache, one file per head."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for name, booster in self.models.items():
            booster.save_model(str(d / f"{name}.json"))
        np.savez(d / "bounds.npz", lower=self._bounds[0], upper=self._bounds[1])
        (d / "meta.json").write_text(json.dumps({
            "feature_names": list(FEATURE_NAMES),
            "heads": [vars(h) for h in self.heads],
            "k_sigma": self.k_sigma,
            "params": self.params,
            "rounds": self.rounds,
            "winsorize": list(self.winsorize),
            "min_history": self.min_history,
            # best_iteration is carried explicitly rather than read back off the
            # booster: it decides how many trees `signal` uses, and relying on a
            # JSON round-trip to preserve it would make scoring depend on an
            # xgboost implementation detail.
            "best_iteration": self.best_iteration,
            "train_end": self.train_end,
            "quantile_grid": list(QUANTILE_GRID),
            "quantiles": self.quantiles,
        }, indent=2))

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        meta_path = d / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no trained model at {d}. Train one first:\n"
                f"  app/python/.venv/bin/python app/python/algo_trade.py "
                f"--out {d}")
        meta = json.loads(meta_path.read_text())
        if tuple(meta["feature_names"]) != FEATURE_NAMES:
            raise RuntimeError("saved model was fit on a different feature set")

        model = cls(heads=tuple(Head(**h) for h in meta["heads"]),
                    k_sigma=meta["k_sigma"], params=meta["params"],
                    rounds=meta["rounds"], winsorize=tuple(meta["winsorize"]),
                    min_history=meta["min_history"])
        model.best_iteration = dict(meta["best_iteration"])
        model.train_end = meta["train_end"]
        if tuple(meta["quantile_grid"]) != QUANTILE_GRID:
            raise RuntimeError("saved model was calibrated on a different "
                               "quantile grid; retrain it")
        model.quantiles = dict(meta["quantiles"])
        bounds = np.load(d / "bounds.npz")
        model._bounds = (bounds["lower"], bounds["upper"])
        for head in model.heads:
            booster = xgb.Booster()
            booster.load_model(str(d / f"{head.name}.json"))
            model.models[head.name] = booster
        return model


class AlgoTradeStrategy(stonks.Strategy):
    """Long swings on the model's up heads, protected by a fixed bracket.

    Entry needs both up heads to agree and the down head to stay quiet — bist's
    two-head conjunction with its down-head overlay. Held names exit on a -10%
    stop or a +30% target and nothing else; bist's documented production rules
    are "SL -10%, TP +30%, no time limit".

    The thresholds are percentiles of each head's training-set prediction
    distribution, not bist's absolute sigmas. bist publishes h5 >= 3.0 and
    h10 >= 2.0, but predicted sigma is not on the label's scale and this fit's
    output is far tighter than bist's, so those numbers select nothing at all
    here. QUANTILE_GRID has the full reasoning. Percentiles also survive a
    retrain, where a hardcoded sigma silently changes meaning.

    The model is pre-trained to a dated artifact and frozen, so the strategy
    refuses to trade any bar the fit was allowed to see. Run the backtest from
    roughly LOOKBACK bars before that cutoff: the engine's `--start` truncates
    history and there is no pre-window warmup data, so the early stretch is
    spent accumulating the lookback and takes no positions.

    Caveats worth carrying into the results:

     1. `dn_q_max` is our own rule. bist only ever *displays* the down head as
        an avoidance overlay and never subtracts it, so this threshold has no
        upstream justification — an ablation at dn_q_max = 1.0 is the honest
        comparison.
     2. `signal` is only called when a slot is free. Feature building is a
        300 x ~600 frame reduction, far and away the run's dominant cost, and
        there is nothing to do with a signal when the book is full.
     3. Entry size is capped by free cash less `cash_buffer`. A market order
        sized on today's close and filled at tomorrow's open can overdraw, and
        the broker rejects — never queues — an order it cannot fund at fill
        time.
     4. Nothing screens for tradability. Every symbol that printed a bar is
        eligible, so a thin name that spiked on almost no volume can take a
        slot and its fills will be more optimistic than a real order book
        would allow. See deviation 5 in the module docstring.
    """

    artifact = "app/python/artifacts/algotrade"
    h5_q = 0.99           # top 1% of the 5-day head's training predictions
    h10_q = 0.99          # top 1% of the 10-day head's
    dn_q_max = 0.90       # veto a name whose downside is in the top decile
    stop_pct = 10.0       # bist's documented production stop
    target_pct = 30.0     # bist's documented production target
    max_positions = 10
    cash_buffer = 0.02

    params = {
        "h5_q": stonks.Param(
            "minimum 5-day prediction percentile to enter", unit="quantile"),
        "h10_q": stonks.Param(
            "minimum 10-day prediction percentile to enter", unit="quantile"),
        "dn_q_max": stonks.Param(
            "downside prediction percentile above which entries are vetoed",
            unit="quantile"),
        "stop_pct": stonks.Param("protective stop below the entry bar's close", unit="%"),
        "target_pct": stonks.Param("profit target above the entry bar's close", unit="%"),
        "max_positions": stonks.Param("names held at once"),
        "cash_buffer": stonks.Param(
            "fraction of cash held back from entries for fees and gaps"),
    }

    indicators = {
        "up_h10": stonks.Indicator("predicted 10-day excess move, sigma", color="#4c9f70"),
    }

    def on_start(self, ctx):
        self.model = ManipulationModel.load(self.artifact)
        # symbols we believe we hold; brackets close positions behind our back
        self.book = set()

    def on_tick(self, ctx):
        w = ctx.history(LOOKBACK)
        if len(w) == 0:
            return

        # The artifact saw every bar up to train_end during training; trading
        # them is not a backtest result. Warmup lands here too.
        now = int(np.max(w.timestamp))
        if self.model.train_end is not None and now <= self.model.train_end:
            return
        if np.unique(w.timestamp).size < LOOKBACK:
            return

        self.book = {s for s in self.book if ctx.position(s) is not None}
        free = self.max_positions - len(self.book)
        if free <= 0:
            return

        sig = self.model.signal(w)
        picks = self._rank(sig)[:free]
        if not picks:
            return

        latest = self._closes(w)
        budget = min(ctx.equity() / self.max_positions,
                     ctx.cash() * (1.0 - self.cash_buffer) / len(picks))
        if budget <= 0.0 or not np.isfinite(budget):
            return

        for symbol in picks:
            close = latest.get(symbol)
            if close is None or not np.isfinite(close) or close <= 0.0:
                continue
            quantity = budget / close
            if quantity <= 0.0 or not np.isfinite(quantity):
                continue
            entry = ctx.place_market_order(symbol=symbol, side=OrderSide.Buy,
                                           quantity=quantity)
            # Dormant until the entry fills, then eligible from its fill bar, so
            # the stop protects the entry bar itself. reduce_only keeps an
            # orphaned leg from opening a short.
            ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                 quantity=quantity,
                                 price=close * (1.0 - self.stop_pct / 100.0),
                                 parent=entry, reduce_only=True)
            ctx.place_limit_order(symbol=symbol, side=OrderSide.Sell,
                                  quantity=quantity,
                                  price=close * (1.0 + self.target_pct / 100.0),
                                  parent=entry, reduce_only=True)
            self.book.add(symbol)
            ctx.plot("up_h10", symbol, float(sig.at[symbol, "up_h10"]))

    def _rank(self, sig):
        """Symbols clearing every gate, best composite first.

        `composite` is bist's: the mean of the two up sigmas. Ties break on
        symbol so the ordering is deterministic.
        """
        ok = sig.loc[
            (sig["up_h5_q"] >= self.h5_q)
            & (sig["up_h10_q"] >= self.h10_q)
            & (sig["dn_h5_q"] <= self.dn_q_max)
        ]
        ok = ok.loc[[s for s in ok.index if s not in self.book]]
        if ok.empty:
            return []
        composite = (ok["up_h5"] + ok["up_h10"]) / 2.0
        return sorted(composite.index, key=lambda s: (-composite[s], s))

    @staticmethod
    def _closes(w):
        """This tick's close per symbol.

        Every symbol in the window printed at the current timestamp and its rows
        end there, so the rows stamped `ts[-1]` are exactly the segment ends.
        """
        ts = np.asarray(w.timestamp)
        return {w.symbol[i]: float(w.close[i])
                for i in np.flatnonzero(ts == ts[-1])}


# ---------------------------------------------------------------------------
# Offline trainer. The strategy loads what this writes.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="app/data/bist_1d.parquet")
    parser.add_argument("--train-end", default="2024-12-31",
                        help="last bar the fit may see; the backtest starts after it")
    parser.add_argument("--out", default=AlgoTradeStrategy.artifact)
    parser.add_argument("--rounds", type=int, default=N_ROUNDS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frame = pd.read_parquet(args.data)
    log.info("%s: %d rows, %d symbols, %s to %s", args.data, len(frame),
             frame["symbol"].nunique(), frame["timestamp"].min(),
             frame["timestamp"].max())

    model = ManipulationModel(rounds=args.rounds).train(
        frame, train_end=args.train_end)
    model.save(args.out)
    log.info("wrote %s (train_end=%s)", args.out, args.train_end)


if __name__ == "__main__":
    main()
