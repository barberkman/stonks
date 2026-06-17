#pragma once

namespace stonks::app {

// Wires up the Engine (EMA50 strategy + KLineFeed + BacktestBroker) and runs it,
// printing the report to the terminal. Shared by the headless and GUI run paths.
void run_backtest();

}  // namespace stonks::app
