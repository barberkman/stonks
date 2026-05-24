# stonks

Versatile C++20 backtesting and live-trading system.

## Build

```sh
cmake -S . -B build
cmake --build build
```

## Test

```sh
ctest --test-dir build --output-on-failure
```

## Run

```sh
./build/apps/backtest_runner/backtest_runner
```
