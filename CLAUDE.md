# stonks

C++20 backtesting and live-trading system, built phase-by-phase.

## Build & test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

VSCode: F5 builds and launches the debugger (requires CodeLLDB extension).

## Structure

- `include/stonks/<module>/` — public headers, all under the `stonks::` namespace (sub-namespaces match folders, e.g. `stonks::core`).
- `src/<module>/` — implementations, mirrors `include/stonks/`.
- `apps/<name>/` — executables (e.g. `backtest_runner`).
- `tests/<module>/` — GoogleTest unit tests, mirrors `include/stonks/`.
- `cmake/` — build helpers (deps pulled via `FetchContent`).
- `.vscode/` — committed workspace configs for build + F5 debug.

## CMake

When adding/removing/renaming source files, update the relevant CMakeLists.txt automatically. The user should never need to edit CMake files. Also update .clangd if new include paths are needed (e.g., adding a new core lib dependency).
