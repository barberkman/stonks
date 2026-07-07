# stonks

C++20 backtesting and live-trading system, built phase-by-phase.

## Build & test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

VSCode: F5 builds and launches the debugger (requires CodeLLDB extension).

### Linux (Ubuntu / WSL)

Prerequisites: `sudo apt install cmake gdb libcurl4-openssl-dev libssl-dev` (build-essential ships gcc/g++/make). libcurl + OpenSSL back the live Binance broker (HTTP + Ed25519 signing); on macOS they come from the system / Homebrew (`brew install openssl`).

Use the `linux-debug` / `linux-release` presets — they build into `build/linux-debug/` and `build/linux-release/`:

```sh
cmake --preset linux-debug
cmake --build --preset linux-debug
ctest --preset linux-debug
```

VSCode launch configs `Linux: Debug backtest_runner (gdb)` and `Linux: Debug core_tests (gdb)` use cppdbg/gdb against the `linux-debug` preset's build dir; they need the `ms-vscode.cpptools` extension.

## Structure

- `include/stonks/<module>/` — public headers, all under the `stonks::` namespace (sub-namespaces match folders, e.g. `stonks::core`).
- `src/<module>/` — implementations, mirrors `include/stonks/`.
- `app/` — the single application (executable target `app`): headless backtest runner by default, Qt Quick GUI with `--gui`, live Binance trading with `--live`. Strategies live in `app/strategies/` (C++) and `app/python/` (Python).
- `include/stonks/binance/` + `src/binance/` (lib `stonks_binance`) — live USDⓈ-M futures: `BinanceBroker` (a drop-in for `BacktestBroker` — same `core::Broker` concept), REST client + Ed25519 signer, exchange-filter rounding, and `LiveKlineFeed`. All exchange state is read from Binance, never shadow-ledgered. See `app/python/README.md` → "Live trading" for env vars and `--live` usage.
- `tests/<module>/` — GoogleTest unit tests, mirrors `include/stonks/`.
- `tools/` — standalone helpers (e.g. `verify_backtest.py`, the trade-by-trade replay audit of a report).
- `cmake/` — build helpers (deps pulled via `FetchContent`).
- `.vscode/` — committed workspace configs for build + F5 debug.

## CMake

When adding/removing/renaming source files, update the relevant CMakeLists.txt automatically. The user should never need to edit CMake files. Also update .clangd if new include paths are needed (e.g., adding a new core lib dependency).

## Style

- Use a single space between tokens in declarations and assignments — do not column-align across consecutive lines. Examples:
  - `using Price = double;` not `using Price    = double;`
  - `Timestamp timestamp;` next to `Symbol symbol;`, not `Symbol    symbol;`
  - `const auto a = foo();` and `const auto longer_name = foo();` — leave them ragged.
- Normal indentation (function bodies, multi-line argument continuations, initializer lists) is unaffected; the rule is specifically about padding identifiers/operators with extra spaces to make adjacent lines line up.
- Prefer brace-initialization `Type{ value }` over `Type(value)` for object construction, including member initializer lists and throw expressions. Function calls keep `(...)`. Cases where `{}` would change semantics (e.g. `std::vector(n, val)` vs `std::vector{ n, val }`) stay as `()`.
- Put a space inside non-empty braces: `id{ id_ }`, `Order{ ... }`, `std::chrono::days{ 1 }` — not `id{id_}`. Empty braces stay tight: `{}`, `Timestamp{}`, `int* tick_count{};`.

## Naming

- Acronyms in identifiers are fully uppercase, not camel-cased: `OrderID`, `URL`, `HTTPClient` — not `OrderId`, `Url`, `HttpClient`. Words that look like acronyms but aren't (e.g. `KLine` — the K isn't an abbreviation) keep title-case.

## Python strategies

Strategies can be authored in Python and run inside the same C++ engine via `PythonStrategy` (`app/strategies/pythonstrategy.h`). Conventions:

- `python/` — framework package (`stonks`): bindings, base class, `FakeContext`.
- `app/python/` — this app's Python content: strategies + venv. Sibling to `app/strategies/` and `app/data/`.
- `PythonStrategy` defaults `STONKS_VENV=app/python/.venv` and `STONKS_PYTHONPATH=app/python` (set with `overwrite=0`), so the sample runs with no env-var setup. Export your own to override.

Build with `-DSTONKS_PYTHON=ON` (default) — requires CPython 3.10+ headers. Full usage is in `app/python/README.md`.

One-time setup of the app-local venv:

```sh
python3 -m venv app/python/.venv
app/python/.venv/bin/pip install -e python/
```

Smoke: `./build/linux-debug/app/app` (from project root) runs the reference strategy at `app/python/qmliteral.py`.

## Tests

Claude owns the test suite. The user will not write or modify tests — that's Claude's responsibility. After every behavior change, ensure the suite is adequate:

- Cover the new behavior and the invariants the change relies on (no-lookahead, determinism, order-stamping, etc.).
- Update or extend existing tests when behavior shifts; don't leave a behavior change without a test that would have caught a regression.
- Add new test files under `tests/<module>/` and wire them into the relevant `CMakeLists.txt` — the user should never have to edit test build files either.
- If a behavior is genuinely untestable (e.g. UI, real network), say so explicitly rather than silently skipping.
