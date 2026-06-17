# stonks

Versatile C++20 backtesting and live-trading system.

## Build & run

Requires Qt 6 (point `CMAKE_PREFIX_PATH` at your Qt install). Run from the project root (the sample data path is relative):

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/home/baris/Qt/6.10.1/gcc_64 && cmake --build build && ./build/app/app
```

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/home/baris/Qt/6.10.1/gcc_64 && cmake --build build && ./build/app/app --gui
```

## Test

```sh
ctest --test-dir build --output-on-failure
```
