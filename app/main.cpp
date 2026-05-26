#include <iostream>

#include "stonks/core/engine.h"
#include "stonks/broker/backtestbroker.h"
#include "stonks/datafeed/klinefeed.h"

#include "strategies/placeholderstrategy.h"

int main() {
    std::cout << "stonks app v0.0.1\n";

    stonks::core::Engine engine
    {
        PlaceholderStrategy{},
        stonks::datafeed::KLineFeed{ "app/data/BTCUSDT_1d.parquet" },
        stonks::broker::BacktestBroker{}
    };
    engine.run();

    return 0;
}
