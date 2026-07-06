# `app/strategies/` — the Python strategy bridge

This folder no longer holds hand-written C++ strategies. Strategies for the
`stonks` backtest runner are authored in **Python**, under `app/python/`, and
run inside the same C++ engine through the wrapper here.

## What's in this folder

- `pythonstrategy.h` — `PythonStrategy`, the C++/Python bridge. It satisfies the
  `stonks::core::Strategy` concept by loading a user-supplied Python class
  (`PythonStrategy{ "<module>", "<Class>" }`) and forwarding `on_start` /
  `on_tick` / `on_stop` into it via a `stonks::python::IContext` adapter. It also
  applies GUI parameter overrides and exposes declared indicator metadata for the
  chart overlay. Defaults `STONKS_VENV=app/python/.venv` and
  `STONKS_PYTHONPATH=app/python` so the shipped strategy runs with no env setup.
- `strategydiscovery.{h,cpp}` — `discover_strategies()`, which globs
  `app/python/*.py`, imports each module, and returns every file's single
  `stonks.Strategy` subclass plus its declared params. This is what populates the
  GUI's strategy dropdown; the app never names a strategy class in C++.

## Writing a strategy

Author it in Python — see **`app/python/README.md`** for the full authoring
guide (Context API, order/bracket mechanics, GUI-editable params, indicator
overlays, and `FakeContext` unit testing). The shipped reference strategy is
`app/python/qmsignals.py`.

New `app/python/<name>.py` files are picked up automatically by discovery — they
appear in the GUI dropdown with no C++ or build wiring. The headless run in
`app/src/main.cpp` selects a strategy by module/class string
(`PythonStrategy{ "qmsignals", "QMSignalsStrategy" }`).
