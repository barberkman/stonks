# app/python — Python strategies for this app

This folder holds Python strategies and the venv the embedded interpreter
runs them in. See `ema_cross.py` for a working example.

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

class EMACross(stonks.Strategy):
    def on_start(self, ctx):
        self.short = 12
        self.long = 26
        self.holding = False
        self.symbol = "BTCUSDT"

    def on_tick(self, ctx):
        bars = ctx.klines(self.long + 1)
        if len(bars) < self.long + 1:
            return
        closes = [b.close for b in bars]
        short_ema = _ema(closes, self.short)
        long_ema = _ema(closes, self.long)
        if short_ema > long_ema and not self.holding:
            ctx.place_market_order(
                symbol=self.symbol, side=OrderSide.Buy, quantity=0.01)
            self.holding = True
        elif short_ema < long_ema and self.holding:
            ctx.place_market_order(
                symbol=self.symbol, side=OrderSide.Sell, quantity=0.01)
            self.holding = False

def _ema(values, period):
    k = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema
```

## Run inside the engine

Drop your strategy in `app/python/<name>.py` and reference it in `app/main.cpp`:

```cpp
#include "strategies/pythonstrategy.h"
// ...
PythonStrategy{ "<name>", "EMACross" },
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
from ema_cross import EMACross

def test_buys_on_uptrend():
    bars = [FakeKLine(i, "BTCUSDT", p, p, p, p, 1.0)
            for i, p in enumerate(range(100, 160))]
    ctx = FakeContext(bars)
    s = EMACross()
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
| `ctx.klines(count: int)` | `list[KLine]` — last N bars, clamped to `now()` |
| `ctx.klines(start: Timestamp, end: Timestamp = None)` | `list[KLine]` — range query, clamped to `now()` |
| `ctx.place_market_order(symbol, side, quantity, time_in_force=GTC)` | `bool` |
| `ctx.place_limit_order(symbol, side, quantity, price, time_in_force=GTC)` | `bool` |

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
