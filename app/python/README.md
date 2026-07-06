# app/python — Python strategies for this app

This folder holds Python strategies and the venv the embedded interpreter
runs them in. See `ema50strategy.py` for a working example.

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

## Run inside the engine

Drop your strategy in `app/python/<name>.py` and reference it in `app/main.cpp`:

```cpp
#include "strategies/pythonstrategy.h"
// ...
PythonStrategy{ "ema50strategy", "EMA50Strategy" },
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
