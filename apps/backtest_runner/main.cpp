#include <iostream>

#include "stonks/core/engine.h"

#include "brokers/backtestbroker.h"
#include "datafeeds/klinefeed.h"
#include "strategies/placeholderstrategy.h"

int main() {
    std::cout << "stonks backtest_runner v0.0.1\n";

    stonks::core::Engine engine
    {
        PlaceholderStrategy{},
        KLineFeed{ "apps/backtest_runner/data/BTCUSDT_1d.parquet" },
        BacktestBroker{}
    };
    engine.run();

    return 0;
}
