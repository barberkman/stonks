"""Qullamaggie breakout swing strategy — an actively-managed long-only port.

This is the Qullamaggie method from the four-step video breakdown, not the
fire-and-forget scanner in `qmsignals.py`. The whole point of the method is the
*trade management* (freeroll + trailing scale-outs), so this strategy tracks its
own holdings and works the position bar by bar:

  Step 1 (Identify)   — liquidity ($ volume), volatility (ADR), a stacked
                        10/21-EMA + 50/200-SMA uptrend, a tight continuation
                        base, and relative strength (see adaptations below).
  Step 2 (Enter/stop) — buy the close-confirmed pivot breakout (next-bar market
                        order); the initial stop is a tight ~½–⅔-ADR level.
  Step 3 (Freeroll)   — sell a fraction at +1R to make the trade risk-free while
                        the remaining shares keep the original stop.
  Step 4 (Optimize)   — trail with the 10/21-EMA: scale out on the first daily
                        close below the 10-EMA, exit the rest below the 21-EMA,
                        and dump on a "supply shoot" (wide down-bar on volume).

Adaptations forced by this engine (all verified against the broker):

  * No stop order exists and a resting limit below entry fills instantly, so the
    initial stop is the engine's leverage *liquidation* — leverage is sized so the
    bankruptcy price sits at the intended stop. Because liquidation is fixed at the
    position's entry/leverage, the remaining shares keep that stop after a partial
    exit — exactly Step 3's "keep the stop at the original level".
  * There is no position query, so a per-symbol *shadow ledger* mirrors the
    broker's documented fill rules. A next-bar market entry makes the fill
    deterministic, so the ledger stays exact and never sells while flat.
  * No S&P/benchmark series and no intraday data exist here, so the RS-line and
    sector-leadership tests become a cross-universe momentum rank, and the
    intraday ORB entry is dropped (it already lives in qmsignals).
"""

from typing import Optional

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide


# ─── Stateless TA helpers ────────────────────────────────────────────────────
# Each returns None when there is not enough history, mirroring std::optional.
def sma(a, n: int) -> Optional[float]:
    if n <= 0 or len(a) < n:
        return None
    return float(np.sum(a[-n:]) / n)


def ema(a, n: int) -> Optional[float]:
    # Windowed EMA: seed with the SMA of the first n bars, then recurse. With a
    # long window relative to n (n is 10/21 here) this equals the running EMA.
    if n <= 0 or len(a) < n:
        return None
    alpha = 2.0 / (n + 1.0)
    e = float(np.sum(a[:n]) / n)
    for x in a[n:]:
        e = alpha * float(x) + (1.0 - alpha) * e
    return e


def adr_pct(high, low, n: int) -> Optional[float]:
    if n <= 0 or len(high) < n:
        return None
    h = high[-n:]
    l = low[-n:]
    return float(np.sum(100.0 * (h / l - 1.0)) / n)


def avg_dollar_vol(close, vol, n: int) -> Optional[float]:
    if n <= 0 or len(close) < n:
        return None
    return float(np.mean(close[-n:] * vol[-n:]))


def gain_pct(close, n: int) -> Optional[float]:
    if len(close) < n + 1:
        return None
    base = close[len(close) - 1 - n]
    if base == 0.0:
        return None
    return float(100.0 * (close[-1] / base - 1.0))


def highest(a, n: int) -> Optional[float]:
    if n <= 0 or len(a) < n:
        return None
    return float(np.max(a[-n:]))


def lowest(a, n: int) -> Optional[float]:
    if n <= 0 or len(a) < n:
        return None
    return float(np.min(a[-n:]))


class QBreakoutStrategy(stonks.Strategy):
    # ─── Configurable parameters ─────────────────────────────────────────────
    # Universe & liquidity (Step 1). $ volume defaults off — the video wants
    # >$100M, but crypto volume units differ, so set it per dataset.
    min_price = 0.0
    min_dollar_vol = 0.0
    dv_len = 20
    # Volatility (Step 1): 20-day ADR must clear the video's 2.25%.
    adr_len = 20
    min_adr = 2.25
    # Trend stack (Step 1): price above a rising 10-EMA > 21-EMA > 50-SMA > 200-SMA.
    ema_fast = 10
    ema_mid = 21
    sma_slow = 50
    sma_trend = 200
    # Relative strength / leadership (Step 1) — momentum-rank proxy for the
    # 52-week RS line: the symbol must sit in the top fraction of the traded
    # universe by rs_len-bar momentum.
    rs_len = 63
    rs_top_frac = 0.5
    # Continuation base + breakout (Step 1/2).
    base_max_len = 40
    min_base_days = 3
    max_depth = 25.0        # max base retrace, % of the pivot high
    use_bo_vol = True
    bo_vol_mult = 1.3       # breakout volume vs prior 50-bar average
    wait_close = True       # close-confirm the break (vs an intrabar high touch)
    # Entry & stop (Step 2).
    adr_stop_mult = 0.6     # initial stop distance from entry, in ADRs (½–⅔)
    use_lod_stop = True     # tighten the stop up to the breakout bar's low
    # Sizing (Step 2): risk 0.25–0.75% of equity, cap any name at 25% of equity.
    risk_fraction = 0.005
    pos_cap_frac = 0.25
    max_leverage = 125.0
    # Management (Steps 3–4).
    freeroll_frac = 0.5     # Method A: sell ½ at +1R (Method B: 1/3 at +2R)
    freeroll_r = 1.0
    scale_frac = 0.25       # sell ¼ on the first close below the 10-EMA
    supply_range_mult = 1.5  # "supply shoot": bar range ≥ this × ADR%
    supply_vol_mult = 2.0    #                  on volume ≥ this × avg volume

    MS_PER_DAY = 86_400_000

    def on_start(self, ctx):
        self.pos = {}       # symbol -> shadow-ledger dict (see _try_enter)
        self._last = {}     # symbol -> last action taken (for tests)

    def lookback(self) -> int:
        # Enough bars for the 200-SMA over the base window, the RS momentum, and
        # the 50-bar breakout-volume average.
        return max(self.sma_trend + self.base_max_len, self.rs_len + 1, 51) + 5

    # ─── Tick loop ───────────────────────────────────────────────────────────
    def on_tick(self, ctx):
        w = ctx.history(self.lookback())
        if len(w) == 0:
            return

        df = pd.DataFrame({
            "symbol": w.symbol,
            "timestamp": w.timestamp,
            "open": w.open, "high": w.high, "low": w.low,
            "close": w.close, "volume": w.volume,
        })
        data = {}
        for symbol, sub in df.groupby("symbol", sort=False):
            sub = sub.sort_values("timestamp")
            data[symbol] = (
                sub["open"].to_numpy(), sub["high"].to_numpy(),
                sub["low"].to_numpy(), sub["close"].to_numpy(),
                sub["volume"].to_numpy(), sub["timestamp"].to_numpy(),
            )

        # Leadership gate (Step 1): rank the printing universe by momentum and
        # keep the top fraction. A lone symbol is its own leader.
        moms = {s: gain_pct(d[3], self.rs_len) for s, d in data.items()}
        moms = {s: m for s, m in moms.items() if m is not None}
        leaders = self.leaders(moms)

        committed = 0.0     # cash earmarked by entries placed earlier this tick
        for symbol, (op, hi, lo, cl, vo, ts) in data.items():
            o, h, l, c, v = op[-1], hi[-1], lo[-1], cl[-1], vo[-1]
            prev_low = lo[-2] if len(lo) >= 2 else l

            self._reconcile(symbol, o, h, l)

            if symbol in self.pos:
                e10 = ema(cl, self.ema_fast)
                e21 = ema(cl, self.ema_mid)
                adr = adr_pct(hi, lo, self.adr_len)
                avgvol = sma(vo[:-1], 50) if len(vo) >= 51 else None
                self._manage(ctx, symbol, o, h, l, c, v, prev_low, avgvol, e10, e21, adr)
            elif symbol in leaders:
                committed = self._try_enter(ctx, symbol, op, hi, lo, cl, vo, ts, committed)

    # ─── Shadow ledger: mirror the broker's fills for the current bar ─────────
    # Order matches the broker: resting exits fill at this open, then the market
    # entry fills at this open, then leverage liquidation is checked on this bar.
    def _reconcile(self, symbol, o, h, l):
        pos = self.pos.get(symbol)
        if pos is None:
            return

        # Exits placed last bar fill at this open and reduce the position.
        if pos["pending_exits"] > 0.0:
            pos["remaining"] = max(0.0, pos["remaining"] - pos["pending_exits"])
            pos["pending_exits"] = 0.0
            if pos["remaining"] <= 1e-9:
                del self.pos[symbol]
                return

        # The next-bar market entry fills at this open; finalize the trade levels
        # off the real fill so the freeroll target and stop stay consistent.
        if pos["state"] == "pending":
            lev = pos["order_lev"]
            pos["state"] = "open"
            pos["entry_fill"] = o
            pos["leverage"] = lev
            pos["qty"] = pos["order_qty"]
            pos["remaining"] = pos["order_qty"]
            pos["effective_stop"] = o * (1.0 - 1.0 / lev)
            pos["R"] = o - pos["effective_stop"]

        # Leverage liquidation is the initial stop (Step 2). A resting exit that
        # already flattened this bar would have pre-empted it (handled above).
        if pos["state"] == "open" and l <= pos["effective_stop"]:
            self._last[symbol] = "liquidated"
            del self.pos[symbol]

    # ─── Steps 3–4: work an open position, placing market exits ───────────────
    def _manage(self, ctx, symbol, o, h, l, c, v, prev_low, avgvol, e10, e21, adr):
        pos = self.pos.get(symbol)
        if pos is None or pos["state"] != "open":
            return
        avail = pos["remaining"]
        if avail <= 1e-9:
            return

        to_sell = 0.0
        action = None
        if self._is_supply_shoot(o, h, l, c, v, prev_low, avgvol, adr):
            to_sell, action = avail, "supply_shoot"      # danger: dump it all
        elif e21 is not None and c < e21:
            to_sell, action = avail, "exit_21"           # trail broke: exit rest
        else:
            placed = 0.0
            # Freeroll: sell a fraction once price prints +freeroll_r R.
            if (not pos["freerolled"] and pos["R"] > 0.0
                    and h >= pos["entry_fill"] + self.freeroll_r * pos["R"]):
                want = min(self.freeroll_frac * pos["qty"], avail - placed)
                if want > 1e-9:
                    placed += want
                    pos["freerolled"] = True
                    action = "freeroll"
            # Scale out on the first daily close below the 10-EMA.
            if not pos["scaled_10"] and e10 is not None and c < e10:
                want = min(self.scale_frac * pos["qty"], avail - placed)
                if want > 1e-9:
                    placed += want
                    pos["scaled_10"] = True
                    action = "scale_10" if action is None else action + "+scale_10"
            to_sell = placed

        if to_sell > 1e-9:
            ctx.place_market_order(symbol=symbol, side=OrderSide.Sell, quantity=to_sell)
            pos["pending_exits"] += to_sell
            if action:
                self._last[symbol] = action

    # ─── Steps 1–2: gate a flat symbol and place the breakout entry ───────────
    def _try_enter(self, ctx, symbol, op, hi, lo, cl, vo, ts, committed):
        sig = self.detect_breakout(op, hi, lo, cl, vo, ts)
        if sig is None:
            return committed

        qty, lev = self.size_position(sig["entry"], sig["stop"], ctx.equity())
        if qty <= 0.0:
            return committed
        # Mirror the broker's affordability check so the ledger can't diverge from
        # a rejected order (which would turn a later sell into an accidental short).
        cost = qty * sig["entry"] / lev
        if cost > ctx.cash() - committed:
            return committed

        ctx.place_market_order(symbol=symbol, side=OrderSide.Buy, quantity=qty, leverage=lev)
        self.pos[symbol] = {
            "state": "pending",         # filled next bar in _reconcile
            "order_qty": qty,
            "order_lev": lev,
            "entry_est": sig["entry"],  # pivot used for sizing; fill may differ
            "intended_stop": sig["stop"],
            "qty": 0.0,
            "remaining": 0.0,
            "entry_fill": 0.0,
            "leverage": lev,
            "effective_stop": 0.0,
            "R": 0.0,
            "freerolled": False,
            "scaled_10": False,
            "pending_exits": 0.0,
        }
        self._last[symbol] = "entry"
        return committed + cost

    # ─── Signal & gates ──────────────────────────────────────────────────────
    def detect_breakout(self, op, hi, lo, cl, vo, ts) -> Optional[dict]:
        N = len(cl)
        if N < self.sma_trend + self.base_max_len + 1:
            return None
        if not self.liq_ok(cl, vo):
            return None
        adr = adr_pct(hi, lo, self.adr_len)
        if adr is None or adr < self.min_adr:
            return None
        if not self.ma_stack_ok(cl):
            return None

        # Tight continuation base: find the pivot high and the pullback depth
        # since it (keep the last high on ties, so the loop is explicit).
        base_end = N - 1
        w_start = base_end - self.base_max_len
        mx = hi[w_start]
        last_pos = 0
        for i in range(self.base_max_len):
            if hi[w_start + i] >= mx:
                mx = hi[w_start + i]
                last_pos = i
        since_pk = (self.base_max_len - 1) - last_pos
        pull_n = max(since_pk, 1)
        pull_low = lo[base_end - pull_n]
        for i in range(base_end - pull_n, base_end):
            pull_low = min(pull_low, lo[i])
        retrace = 100.0 * (mx - pull_low) / mx if mx > 0.0 else 1e9
        if not (since_pk >= self.min_base_days and retrace <= self.max_depth):
            return None

        # Breakout confirmation above the pivot, on expanding volume.
        entry = mx
        broke = cl[-1] >= entry if self.wait_close else hi[-1] >= entry
        if not broke:
            return None
        if self.use_bo_vol:
            avg50 = sma(vo[:-1], 50) if N >= 51 else None
            if avg50 is None or vo[-1] < self.bo_vol_mult * avg50:
                return None

        adr_stop = entry * (1.0 - self.adr_stop_mult * adr / 100.0)
        stop = max(adr_stop, lo[-1]) if self.use_lod_stop else adr_stop
        stop = min(stop, entry * 0.999)     # keep the stop strictly below entry
        return {"entry": float(entry), "stop": float(stop), "adr": float(adr)}

    def liq_ok(self, cl, vo) -> bool:
        if cl[-1] < self.min_price:
            return False
        dv = avg_dollar_vol(cl, vo, self.dv_len)
        return dv is not None and dv >= self.min_dollar_vol

    def ma_stack_ok(self, cl) -> bool:
        e10 = ema(cl, self.ema_fast)
        e21 = ema(cl, self.ema_mid)
        s50 = sma(cl, self.sma_slow)
        s200 = sma(cl, self.sma_trend)
        if None in (e10, e21, s50, s200):
            return False
        return cl[-1] > e10 > e21 > s50 > s200

    def leaders(self, moms: dict) -> set:
        # Symbols in the top rs_top_frac of the universe by momentum. With one
        # entry the quantile equals its value, so a lone symbol always leads.
        if not moms:
            return set()
        thresh = float(np.quantile(list(moms.values()), 1.0 - self.rs_top_frac))
        return {s for s, m in moms.items() if m >= thresh}

    def _is_supply_shoot(self, o, h, l, c, v, prev_low, avgvol, adr) -> bool:
        # Wide-range down bar on heavy volume that undercuts the prior bar's low.
        if c >= o or l <= 0.0 or adr is None:
            return False
        if 100.0 * (h / l - 1.0) < self.supply_range_mult * adr:
            return False
        if avgvol is None or v < self.supply_vol_mult * avgvol:
            return False
        return c < prev_low

    # ─── Sizing (Step 2) ─────────────────────────────────────────────────────
    # Risk a fixed fraction of equity to the stop, cap the name at pos_cap_frac
    # of equity, and set leverage so liquidation lands at the (capped) stop.
    def size_position(self, entry, stop, equity):
        risk = entry - stop
        if risk <= 0.0 or equity <= 0.0:
            return 0.0, 1.0
        lev = min(max(entry / risk, 1.0), self.max_leverage)
        eff_stop = entry * (1.0 - 1.0 / lev)    # actual liquidation level
        eff_risk = entry - eff_stop             # == risk unless leverage capped
        qty = self.risk_fraction * equity / eff_risk if eff_risk > 0.0 else 0.0
        cap_qty = self.pos_cap_frac * equity / entry
        return min(qty, cap_qty), lev

    # ─── Test introspection ──────────────────────────────────────────────────
    def position(self, symbol):
        return self.pos.get(symbol)

    def last_action(self, symbol):
        return self._last.get(symbol)
