# app/python — Python strategies for this app

This folder holds Python strategies and the venv the embedded interpreter
runs them in. Thirty-one strategy files ship here — standalone interpretations of
the two pine scripts in `app/pines/` (Qullamaggie momentum swing and Darvas
box), with `qmliteral.py` as the reference port; the sections below teach the
authoring patterns with small inline examples.

## Layout

- `app/python/` (this folder) — the app's Python area: strategies + the
  venv. Sibling of `app/strategies/` (C++ strategies) and `app/data/`.
- `python/` — the framework package: bindings (`_core.so`), `Strategy`
  base class, `FakeContext` for tests. Generic / reusable across apps.

## Install

The package wraps a compiled extension (`stonks/_core*.so`) that the CMake
build emits into this folder. After building C++:

```sh
cmake --preset linux-debug -DSTONKS_PYTHON=ON
cmake --build --preset linux-debug

# Create the app-local venv and install the framework editable into it:
python3 -m venv app/python/.venv
app/python/.venv/bin/pip install -e python/
```

`import stonks` now works inside that venv.

## Write a strategy

```python
import stonks
from stonks import OrderSide

class EMA50Strategy(stonks.Strategy):
    PERIOD = 50
    ALPHA = 2.0 / (PERIOD + 1)
    POSITION_FRACTION = 0.01   # fraction of equity per new position

    def on_start(self, ctx):
        self.states = {}

    def on_tick(self, ctx):
        # One tick per timestamp: loop the symbols that printed today. EMA is
        # incremental, so history(1) (one bar per symbol) is enough.
        window = ctx.history(1)
        for symbol, close in zip(window.symbol, window.close):
            close = float(close)
            state = self.states.setdefault(
                symbol,
                { "ema": None, "seed_sum": 0.0, "seed_count": 0, "held_quantity": 0.0 },
            )

            if state["ema"] is None:
                state["seed_sum"] += close
                state["seed_count"] += 1
                if state["seed_count"] < self.PERIOD:
                    continue
                state["ema"] = state["seed_sum"] / self.PERIOD
            else:
                state["ema"] = self.ALPHA * close + (1.0 - self.ALPHA) * state["ema"]

            if close > state["ema"] and state["held_quantity"] == 0.0:
                qty = ctx.equity() * self.POSITION_FRACTION / close
                if qty <= 0.0:
                    continue
                # held_quantity is tracked optimistically on placement — the broker
                # may still reject the order (e.g. insufficient cash). For managed
                # strategies, query ctx.position(symbol) / ctx.order(id) instead.
                ctx.place_market_order(symbol=symbol, side=OrderSide.Buy, quantity=qty)
                state["held_quantity"] = qty
            elif close < state["ema"] and state["held_quantity"] > 0.0:
                ctx.place_market_order(
                    symbol=symbol, side=OrderSide.Sell, quantity=state["held_quantity"]
                )
                state["held_quantity"] = 0.0
```

A window-based strategy (e.g. an indicator over the last N bars) would instead
ask for `ctx.history(N)`, build a DataFrame, and `groupby("symbol")` — see the
Context API section below.

## Execution timeline — when things happen

The engine ticks once per **distinct timestamp** (not once per symbol). On
tick T, in this order:

1. **The broker settles first.** Every symbol's bar at T is processed: resting
   orders whose trigger is touched fill against bar T's OHLC; a bracket child
   whose parent fills at T can itself fill at T (settlement runs in rounds);
   then the per-position liquidation check runs.
2. **Your `on_tick` runs once.** `ctx.history(n)` now includes bar T — its
   close is the latest value you see. `ctx.position()` / `ctx.order()` reflect
   everything that settled through T, including fills and liquidations that
   just happened.
3. **Orders you place are stamped T** and become eligible from bar T+1 —
   never against bar T itself (the no-lookahead gate).

In short: *decide on the close of T, execute from T+1.* A market order fills
at exactly T+1's open. A stop or limit rests until its trigger is touched —
GTC, forever, so a stale entry must be cancelled by you (`ctx.cancel_order`).
There is no way to act on intrabar information: one bar is one decision point.

## A managed strategy — brackets, gating, and the observability API

The EMA50 sample above is fire-and-forget. A real strategy usually needs the
full toolkit: a stop-entry, protective legs chained under it, one-trade-at-a-
time gating, and cleanup of stale orders. This complete example runs as-is
(drop it in `app/python/` and unit-test it with `FakeContext`); the pattern
mirrors `qmliteral.py`, the reference port:

```python
import stonks
from stonks import OrderSide, OrderStatus


class BreakoutBracket(stonks.Strategy):
    """20-bar-high breakout with a managed bracket: a stop-entry above the
    pivot, a reduce-only stop-loss and take-profit chained under it, and
    one-trade-at-a-time gating with a cooldown after each close."""

    LOOKBACK = 20
    RISK_FRACTION = 0.01      # fraction of equity risked per trade
    COOLDOWN_BARS = 5

    def on_start(self, ctx):
        self.state = {}       # symbol -> {"pending", "was_in", "cooldown"}

    def on_tick(self, ctx):
        w = ctx.history(self.LOOKBACK + 1)
        by_symbol = {}                    # rows of the long frame, per symbol
        for i, sym in enumerate(w.symbol):
            by_symbol.setdefault(sym, []).append(i)

        for sym, rows in by_symbol.items():
            st = self.state.setdefault(
                sym, {"pending": None, "was_in": False, "cooldown": 0})

            # ── gating: one trade at a time, cooldown after each close ──────
            in_position = ctx.position(sym) is not None
            closed = st["was_in"] and not in_position
            if st["pending"] is not None:
                entry = ctx.order(st["pending"])
                status = entry.status if entry else OrderStatus.Cancelled
                if status == OrderStatus.Filled:
                    if not in_position:   # entered and exited within one bar
                        closed = True
                    st["pending"] = None
                elif status != OrderStatus.Open:
                    st["pending"] = None  # rejected or cancelled: forget it
            if closed:
                st["cooldown"] = self.COOLDOWN_BARS
            elif not in_position and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_position
            if in_position or st["cooldown"] > 0:
                continue

            # ── signal: close breaks the prior LOOKBACK-bar high ────────────
            if len(rows) < self.LOOKBACK + 1:
                continue                  # not enough history yet
            pivot = max(float(w.high[i]) for i in rows[:-1])
            close = float(w.close[rows[-1]])
            if close < pivot:
                continue

            stop = pivot * 0.97           # 3% protective stop
            target = pivot + 2.0 * (pivot - stop)
            qty = ctx.equity() * self.RISK_FRACTION / (pivot - stop)

            if st["pending"] is not None:  # stale unfilled entry: replace it
                ctx.cancel_order(st["pending"])

            entry_id = ctx.place_stop_order(symbol=sym, side=OrderSide.Buy,
                                            quantity=qty, price=pivot)
            ctx.place_stop_order(symbol=sym, side=OrderSide.Sell, quantity=qty,
                                 price=stop, parent=entry_id, reduce_only=True)
            ctx.place_limit_order(symbol=sym, side=OrderSide.Sell, quantity=qty,
                                  price=target, parent=entry_id, reduce_only=True)
            st["pending"] = entry_id
```

The load-bearing details, each of which prevents a class of backtest bug:

- **Stop-entry, not market or limit.** A breakout entry as a *stop* fills at
  the signal level when reached (keeping risk math anchored) and never fills
  at all if price reverses; a market order would chase any next open, and a
  limit below the market would wait for a retest instead of the breakout.
- **`parent=entry_id`** keeps the protective legs dormant until the entry
  fills, and OCO-cancels the loser when one side flattens the position.
- **`reduce_only=True`** on every protective leg: if a leg is ever orphaned
  (position closed some other way), it cancels itself instead of opening an
  unmanaged position.
- **Gate on `ctx.position()`, not your own bookkeeping.** The broker can close
  your position without you (liquidation) — shadow ledgers desync; the query
  never does. `ctx.order(id).status` is the only reliable way to learn a fill
  or rejection happened.
- **Cancel stale entries.** GTC means a never-triggered stop-entry from last
  month is still live — and can fill in a regime that has nothing to do with
  the signal that placed it.
- **Optional refinement** (see `qmliteral.py`): compare the actual fill
  (`ctx.position(sym).price`) with the planned entry and re-anchor the legs
  proportionally when the entry gapped.

## GUI-editable parameters

Declare which class attributes the setup screen may override per run. The
attribute stays the single source of truth for the default value and its type;
`params` adds exposure and labels:

```python
class BreakoutBracket(stonks.Strategy):
    LOOKBACK = 20
    RISK_FRACTION = 0.01
    COOLDOWN_BARS = 5

    params = {
        "LOOKBACK": stonks.Param("bars in the breakout pivot", unit="bars"),
        "RISK_FRACTION": stonks.Param("fraction of equity risked per trade"),
        "COOLDOWN_BARS": stonks.Param("bars to sit out after a close", unit="bars"),
    }
```

- Supported types: float, int, bool — inferred from the attribute's value.
- The GUI renders the fields automatically (docs become labels, units become
  suffixes, declaration order is preserved). No UI code, for any strategy, ever.
- Chosen values are applied to the instance **before `on_start` runs** — keep
  reading them via `self.<name>` as usual. Derive dependent values in
  `on_start`, not at class-definition time, or they go stale under an override
  (e.g. an EMA's `alpha` derived from a `PERIOD` param must be computed in
  `on_start`, not at class-definition time, for exactly this reason).
- An override for an undeclared name is a hard error at run start; a broken
  `params` dict (typo'd attribute, unsupported type) silently drops the
  strategy from the discovery list, like an import failure.
- The effective values are stamped into the run's archived report (the JSON's
  `strategy` block) and shown on the report tab — every run records exactly
  what it ran with. `stonks.param_specs(cls)` returns the spec for tests.

## Read-only indicator overlays

The inverse direction of params: params are editable GUI→strategy inputs;
indicators are read-only strategy→GUI outputs, drawn as lines over the
symbol's candles on the chart. Declare the series in `indicators`, then
publish the values you actually computed with `ctx.plot`:

```python
class EMA50Strategy(stonks.Strategy):
    indicators = {
        "ema50": stonks.Indicator("50-bar EMA of close"),   # optional color="#rrggbb"
    }

    def on_tick(self, ctx):
        ...
        state["ema"] = self.alpha * close + (1.0 - self.alpha) * state["ema"]
        ctx.plot("ema50", symbol, state["ema"])   # display-only
```

- `ctx.plot(name, symbol, value)` records one sample at the current tick's
  timestamp (implicit — there is no timestamp argument). Plotting never
  influences trading logic and, unlike params, is never GUI-editable.
- Bars where you don't call `plot` (warmup) render as a gap, never a line to
  zero. Non-finite values (NaN/Inf) are dropped at the source — an unguarded
  pandas warmup NaN also becomes a gap.
- The dict key IS the series name `ctx.plot` must use. `Indicator.color` is
  optional; when unset the GUI assigns a palette color. An undeclared name
  still renders (palette color, no doc) rather than failing the run.
- Values persist in the archived report (the JSON's `indicators` /
  `indicator_specs` blocks), so restored runs redraw their overlays without
  re-running. `stonks.indicator_specs(cls)` returns the spec for tests;
  `FakeContext.plots` records `plot` calls for pure-Python assertions.
- v1 scope: overlays on the price pane only (EMA/SMA/bands — anything in
  price space). Oscillators with their own axis (RSI/MACD) are not yet
  supported.

## Run inside the engine

Drop your strategy in `app/python/<name>.py` and reference it in `app/src/main.cpp`:

```cpp
#include "strategies/pythonstrategy.h"
// ...
PythonStrategy strategy{ "qmliteral", "QMLiteralStrategy" };
```

Run from the project root:

```sh
./build/linux-debug/app/app
```

`PythonStrategy` defaults `STONKS_VENV=app/python/.venv` and
`STONKS_PYTHONPATH=app/python` (set with `overwrite=0`), so the sample runs
with no env-var setup.

To use a different venv (e.g. one that has sklearn/pandas installed) or a
strategy living elsewhere, export the env vars before launch:

```sh
STONKS_VENV=$VIRTUAL_ENV \
STONKS_PYTHONPATH=$HOME/strats \
./build/linux-debug/app/app
```

`STONKS_VENV` adds the venv's `site-packages` via `site.addsitedir()`, which
processes `.pth` files so editable installs work. `STONKS_PYTHONPATH`
(colon-separated) is prepended to `sys.path`. Cwd is also added.

## AlgoTrade — a strategy that loads a pre-trained model

`algo_trade.py` is the one strategy here that does not decide anything from
price rules: `ManipulationModel` is three XGBoost regressors (a port of
`/Users/macmini-1/bist`) and the strategy trades their output. It needs two
things the other strategies do not.

**xgboost in the venv.** It is a hard import, and strategy discovery imports
every `app/python/*.py` to find strategies — a module that fails to import is
*silently* dropped from the GUI and from `--all`, so check this first if the
strategy goes missing:

```sh
brew install libomp                      # macOS: the wheel needs the OpenMP runtime
app/python/.venv/bin/pip install xgboost
app/python/.venv/bin/python -c "import xgboost; print(xgboost.__version__)"
```

**A trained artifact.** Training happens offline, not in the engine; the file is
its own trainer. Roughly a minute over the full BIST panel:

```sh
app/python/.venv/bin/python app/python/algo_trade.py --train-end 2024-12-31
```

That writes `app/python/artifacts/algotrade/` (one booster per head, the
winsorize bounds, and `meta.json`). It is gitignored — regenerate it rather than
committing it.

Then mind the window. The artifact records its `train_end` and the strategy
refuses to trade any bar the fit was allowed to see, so the backtest is only
out-of-sample after that date. It also needs 300 bars of lookback before it can
score anything, and `KLineFeed` truncates history at `--start` — there is no
pre-window warmup data — so `--start` must sit ~300 trading bars *earlier* than
where you want signals to begin:

```sh
./build/macos-debug/app/app --start 2023-10-02 --end 2026-07-24
```

Retrain with a different `--train-end` and both dates move with it. `main.cpp`'s
defaults match the command above.

## Unit-test strategies without the engine

```python
from stonks import OrderSide
from stonks.testing import FakeContext, FakeKLine
from ema50strategy import EMA50Strategy

def test_buys_on_uptrend():
    bars = [FakeKLine(i, "BTCUSDT", p, p, p, p, 1.0)
            for i, p in enumerate(range(100, 160))]
    ctx = FakeContext(bars)
    s = EMA50Strategy()
    s.on_start(ctx)
    for _ in bars:
        ctx.advance()
        s.on_tick(ctx)
    assert any(o.side == OrderSide.Buy for o in ctx.orders)
```

`FakeContext` mirrors the Context surface without simulating a market — orders
never fill on their own. What it gives you for assertions and scenario setup:

- **`ctx.orders`** — every placed order as a `FakeOrder` with `.symbol`,
  `.side`, `.quantity`, `.price`, `.parent`, `.id`, `.leverage`,
  `.order_type` (`"market"` / `"limit"` / `"stop"`), `.reduce_only`, and
  `.status` (a real `OrderStatus`; starts `Open`).
- **`ctx.positions`** — a test-settable dict: assign
  `ctx.positions["BTCUSDT"] = FakePosition(quantity=1.0, price=100.0)` before
  a tick to simulate holding (and delete the key to simulate a close);
  `ctx.position(sym)` reads it. This is how you exercise gating, cooldowns,
  and fill-reaction logic without a broker.
- **`ctx.order(order_id)`** — looks up a placed `FakeOrder`; flip its
  `.status` to `OrderStatus.Filled` in the test to simulate the entry filling.
- **`ctx.cancel_order(order_id)`** — marks a still-Open order (and its
  children, cascading like the real broker) `Cancelled`, returning `True`.

The behavior suites (`test_qm_family.py`, `test_darvas_family.py`,
`test_qmdarvas_family.py`, with shared bar-builders and fill helpers in
`conftest.py`) demonstrate all of these patterns, including simulating fills
(`fill_entry` / `fill_exit`) and driving a run through the `settle()`
mini-broker (`test_strategy_smoke.py`).

Fill *mechanics* (trigger touching, margin, liquidation, fees) are not
simulated here by design — they are pinned by the C++ suite
(`tests/core/*.cpp`) against the real broker, and whole runs are re-verified
by `tools/verify_backtest.py`. Test your *logic* in pytest; trust the engine
for execution.

## Context API

| Method | Returns |
| --- | --- |
| `ctx.now()` | `Timestamp` |
| `ctx.cash()` | `float` |
| `ctx.equity()` | `float` |
| `ctx.history(count: int)` | `MarketWindow` — every symbol that printed this tick, each with its last N bars, as one combined frame |
| `ctx.place_market_order(symbol, side, quantity, leverage=1.0, time_in_force=GTC, parent=None, reduce_only=False)` | `int` (the broker-assigned OrderID) |
| `ctx.place_limit_order(symbol, side, quantity, price, leverage=1.0, time_in_force=GTC, parent=None, reduce_only=False)` | `int` (OrderID) |
| `ctx.place_stop_order(symbol, side, quantity, price, leverage=1.0, time_in_force=GTC, parent=None, reduce_only=False)` | `int` (OrderID) |
| `ctx.position(symbol)` | `Position` (`.quantity` signed, `.price` entry, `.entry_id`, `.leverage`) or `None` when flat |
| `ctx.order(order_id)` | `Order` snapshot (`.status` is an `OrderStatus`: Open / Filled / Rejected / Cancelled) or `None` for an unknown id |
| `ctx.cancel_order(order_id)` | `bool` — `True` if a still-open order (and its dormant bracket children) was cancelled |

`place_*_order` returns an **OrderID, never a success flag** — the id is handed
out even for orders the broker rejects. To learn what actually happened, check
`ctx.order(order_id).status` on a later tick, or `ctx.position(symbol)` for the
resulting position. Pass `parent=<entry OrderID>` to attach an order as a
bracket child (dormant until the parent fills, OCO-cancelled when the position
goes flat), and `reduce_only=True` on protective legs so an orphaned leg can
only ever shrink a position, never open one.

`history(n)` returns a `MarketWindow` — a long frame over every symbol that
printed at the current timestamp. `.symbol` is a per-row column; `.timestamp` is
an `int64` numpy array and `.open/.high/.low/.close/.volume` are `float64` numpy
arrays, all the same length. Build a DataFrame and process symbol by symbol:

```python
import pandas as pd

w = ctx.history(50)                       # each printing symbol's last 50 bars
df = pd.DataFrame({
    "symbol": w.symbol, "timestamp": w.timestamp,
    "open": w.open, "high": w.high, "low": w.low,
    "close": w.close, "volume": w.volume,
})
for symbol, sub in df.groupby("symbol"):
    ...                                   # per-symbol logic; cross-symbol if you want
```

The arrays are valid for the current tick only — re-query each tick rather than
stashing a view.

## Fill mechanics & broker semantics

Same rules as native strategies (the broker is shared):

- **No lookahead.** `history(n)` only sees bars up to the current timestamp, and an
  order placed on a bar never fills against that same bar. Bracket children are
  the one refinement: once their parent fills, they become eligible from the
  parent's **fill bar** onward (so a stop can protect the entry bar itself).
- **Order types.** *Market* fills at the next bar's open. *Limit* fills once a
  bar's range reaches the price, at `min/max(price, open)` — never worse than
  the limit. *Stop* stays dormant until the market touches its trigger, then
  fills at the trigger or worse (a bar gapping through it fills at the open) —
  use it for stop-losses and breakout entries; a "stop" expressed as a limit on
  the wrong side of the market fills immediately instead of waiting.
- **Same-bar ties resolve by policy, not placement order.** When one bar could
  fill several orders, markets go first, then stops, then limits under the
  default Conservative policy (protective stops before profit targets);
  Optimistic reverses stops/limits for sensitivity re-runs.
- **Margin-collateralized, one position per symbol.** Opening posts
  `quantity * fill_price / leverage` of cash as isolated margin plus the entry
  fee (the default `leverage=1.0` is fully cash-secured); an unaffordable or
  below-`min_notional` order is rejected (it does not wait). While you hold a
  position in a symbol a same-side order is rejected (no adding — close first);
  an opposite-side order closes it, clamping an oversized close to what you
  hold. Shorts are allowed. Leverage only matters on the order that opens the
  position — it is ignored on closing legs. Closes that leave float dust snap
  to exactly flat.
- **Liquidation.** A position is force-closed once a bar's adverse extreme
  reaches `entry * (1 ∓ 1/leverage) / (1 ∓ m)` (`m` = the configured
  maintenance-margin rate; `m = 0` gives the bankruptcy price). A bar gapping
  through it fills at the open; the loss beyond the posted margin comes out of
  cash unless the isolated loss cap is enabled. If account equity reaches the
  configured floor (default 0) the broker liquidates everything and rejects
  all further orders. A resting stop that fills on the crash bar pre-empts its
  position's liquidation.
- **Fees.** Every fill pays `notional × rate + flat` where the rate is the
  maker fee for a limit filled at its own price (it rested and was hit) and the
  taker fee for everything that crosses on arrival — markets, stops, crossing
  limits, forced closes. Rates are per-run knobs (set in the GUI's setup screen
  or `BrokerConfig`; the headless default is Binance USDT-M VIP0, 2/5 bps) and
  the whole config is stamped into the report JSON. Win rate in the report is
  net of fees.

Every archived report can be re-audited trade-by-trade against the raw bars
with `tools/verify_backtest.py` (see its docstring).

## What the engine does not support (by design, today)

Don't design a strategy around these — work with the patterns above instead:

- **No order modification.** To move a stop or a limit, `cancel_order()` the
  old one and place a new one (attach it to the same filled parent if it's a
  protective leg).
- **GTC only.** No day orders or expiries — stale resting orders are the
  strategy's responsibility to cancel.
- **No adding to a position.** A same-side order against an open position is
  rejected; an opposite-side order *reduces or closes* it (one-way netting,
  like Binance one-way mode). One position per symbol, no hedging.
- **No partial fills.** An order fills in full (closes clamp to what you
  hold) or not at all.
- **No intrabar decisions.** Strategies see completed bars only; fills inside
  a bar follow the documented worse-of rules, not a simulated tape.
- **Fees and margin are run configuration** (GUI setup screen /
  `BrokerConfig`), not strategy-controlled.

## Live trading (Binance USDⓈ-M futures)

The same strategy that backtests can trade live, unchanged. `BinanceBroker` is a
compile-time drop-in for the backtest broker (both satisfy the C++ `Broker`
concept), so your Python `on_tick` sees the identical `ctx` surface — the only
difference is that positions, cash, orders, and fills are now **read from Binance
every tick** instead of simulated. No local ledger is kept.

```sh
export BINANCE_API_KEY=...                 # from the testnet or mainnet portal
export BINANCE_PRIVATE_KEY_PEM=~/ed25519.pem   # inline PEM text or a path to it
./build/linux-debug/app/app --live --strategy qmliteral --symbols BTCUSDT --interval 1h
```

- **Ed25519 keys only** (Binance's recommended type). Generate a keypair, register
  the public half in the API-key portal, point `BINANCE_PRIVATE_KEY_PEM` at the
  private PEM. `X-MBX-APIKEY` + a signed query are handled for you.
- **Testnet by default.** Create keys at testnet.binancefuture.com and validate
  there first. `--mainnet` trades real funds; `--dry-run` logs intended orders
  without sending them. `BINANCE_BASE_URL` overrides the endpoint if Binance
  rotates the testnet host.
- **Flags:** `--strategy <module>` (or `<module>:Class`), `--symbols A,B,C`,
  `--interval 1m|5m|1h|1d|…`, `--mainnet`, `--dry-run`. Ctrl-C stops cleanly.
- **One tick per closed candle.** A `LiveKlineFeed` polls klines and drives the
  engine when a candle closes; history is seeded at startup so `ctx.history(n)`
  works from the first tick. It never acts on a still-forming candle.

How the managed-bracket pattern maps, faithfully:

- Your entry (market or resting stop/limit) and its `reduce_only` children behave
  as they do in backtest. Each child's `parent=entry_id` is encoded into the
  Binance `clientOrderId`, so bracket linkage is reconstructed from the exchange —
  no shadow map.
- Binance rejects a `reduce_only` order while flat, so a child placed around a
  **resting** entry is held locally and submitted the moment the entry fills
  (market entries fill synchronously, so their children go out immediately). When
  a symbol goes flat, orphaned protection is cancelled — the live equivalent of
  OCO/subtree cleanup.
- Partial take-profits work naturally: the broker reads the real (partially
  reduced) position from Binance, so the "no partial fills" backtest simplification
  does **not** apply live.

Caveats to know:

- Quantities and prices are snapped to each symbol's `LOT_SIZE`/`PRICE_FILTER` and
  checked against `MIN_NOTIONAL` before sending; account is assumed **one-way,
  isolated margin**, with leverage set per symbol from the order's `leverage`.
- The only local state is the transient table of not-yet-armed bracket children;
  it is lost on process restart (live positions/orders are fully recovered from
  Binance, but an in-flight un-armed bracket would need manual re-placement).
- Stop/take-profit legs are sent as `STOP_MARKET` on `POST /fapi/v1/order`. Binance
  is migrating conditional orders to a separate algo service; verify order
  acceptance on testnet for your account before going to mainnet.

## Implementing a new strategy — checklist

1. Create `app/python/<name>.py` with exactly one `stonks.Strategy` subclass
   (that's what strategy discovery keys on; `test_*` files are skipped).
   Filenames without underscores, mirroring the existing strategies.
2. Declare the knobs a user should be able to tune in `params` (see
   "GUI-editable parameters" above) — they appear in the setup screen and the
   report automatically.
3. Unit-test the logic in `app/python/test_<name>.py` with `FakeContext` —
   signals, sizing, gating, order shape (types, parents, reduce_only), and
   the `params` spec. Run: `app/python/.venv/bin/pytest app/python/ -q`.
4. Run it in the engine: pick it in the GUI's setup screen (it appears by
   discovery), or wire it in `app/main.cpp` for headless runs from the repo
   root.
5. Audit the run: `tools/verify_backtest.py <report.json> <parquet> [run.log]`
   must exit CLEAN — it replays every fill independently against the raw bars.
   Pass the `run.log` whenever the strategy cancels orders (gap re-anchoring
   does); log-less verification of a cancelling run reports false violations.

## IDE autocomplete

Generate stubs for `_core` once after a build:

```sh
pip install pybind11-stubgen
pybind11-stubgen stonks._core -o python/
```

This emits `python/stonks/_core.pyi` (the framework package directory).
`python/stonks/` is configured to ship `*.pyi` as package data, so the
editable install in `app/python/.venv` picks it up.

## Limitations

- Per-tick FFI overhead: each `on_tick` is a GIL acquire + Python dispatch.
  Fine for daily/minute bars; would dominate tick-level data.
- `Py_Finalize` is deliberately never called — incompatible with
  numpy/sklearn finalize→init cycles. Process exit reclaims the interpreter.
- Linux/macOS only; Windows untested.
