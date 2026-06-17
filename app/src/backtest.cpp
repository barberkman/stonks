#include "backtest.h"

#include <iostream>

#include "stonks/core/engine.h"
#include "stonks/broker/backtestbroker.h"
#include "stonks/datafeed/klinefeed.h"

#include "strategies/ema50strategy.h"
#include "strategies/pythonstrategy.h"

namespace stonks::app {

void run_backtest() {
    std::cout << "--- EMA50Strategy ---" << std::endl;

    stonks::core::Engine engine
    {
        // PythonStrategy{ "ema50strategy", "EMA50Strategy" },
        EMA50Strategy{},
        stonks::datafeed::KLineFeed{ "app/data/us_1d_filtered.parquet" },
        stonks::broker::BacktestBroker{ 1000.0 }
    };
    engine.run();
}

}  // namespace stonks::app
