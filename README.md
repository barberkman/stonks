# stonks

Versatile C++20 backtesting and live-trading system.

## Build & run

Requires Qt 6 (point `CMAKE_PREFIX_PATH` at your Qt install). Run from the project root (the sample data path is relative):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/home/baris/Qt/6.10.1/gcc_64 -DSTONKS_BUILD_TESTS=OFF && cmake --build build && ./build/app/app
```

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/home/baris/Qt/6.10.1/gcc_64 -DSTONKS_BUILD_TESTS=OFF && cmake --build build && ./build/app/app --gui
```

## Test

```sh
ctest --test-dir build --output-on-failure
```

## Audit a run

Every report JSON under `app/reports/` can be re-verified trade-by-trade by an
independent Python replay of the broker rules against the raw bars:

```sh
app/python/.venv/bin/python tools/verify_backtest.py \
    app/reports/report-<ts>.json app/data/binance_1d.parquet [run.log]
```

The optional log capture comes from a `-DSTONKS_LOG=ON` build
(`./build-log/app/app 2> run.log`) and is needed when the strategy cancels
orders mid-run. Exit 0 means the replay reproduced every fill, order status,
and equity-curve point exactly and all no-lookahead/fill-rule invariants held.
