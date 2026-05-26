#include <iostream>

#include "stonks/core/engine.h"
#include "stonks/broker/backtestbroker.h"
#include "stonks/datafeed/klinefeed.h"

#include "strategies/ema50strategy.h"

int main() {
    std::cout << "stonks app v0.0.1\n";

    stonks::core::Engine engine
    {
        EMA50Strategy{},
        stonks::datafeed::KLineFeed{ "app/data/BTCUSDT_1d.parquet" },
        stonks::broker::BacktestBroker{ 1000.0 }
    };
    engine.run();

    return 0;
}
