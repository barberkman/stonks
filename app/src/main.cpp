#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QtQml/qqml.h>

#include "stonks/core/engine.h"
#include "stonks/broker/backtestbroker.h"
#include "stonks/datafeed/klinefeed.h"

#include "backtestcontroller.h"
#include "report.h"
#include "report_json.h"
#include "strategies/pythonstrategy.h"

namespace stonks::app {

// Wires up the Engine (QM signals scanner + KLineFeed + BacktestBroker) and runs it,
// printing the report to the terminal. Shared by the headless and GUI run paths.
// The scanner places no orders — it prints each fired setup's signal to stdout.
void run_backtest() {
    constexpr stonks::core::Balance starting_cash = 1000.0;
    const std::string data_file = "app/data/binance_1d.parquet";
    const std::string start = "2024-01-01";
    const std::string end = "2026-01-30";
    const std::vector<std::string> symbols{ "BTCUSDT", "ETHUSDT", "SOLUSDT" };
    // Binance USDT-M VIP0 fee schedule; stamped into the report JSON.
    const stonks::broker::BrokerConfig broker_config{
        .maker_fee_bps = 2.0,
        .taker_fee_bps = 5.0,
    };
    PythonStrategy strategy{ "qmsignals", "QMSignalsStrategy" };
    // Declared indicator metadata — class-level, read before the move below.
    std::vector<IndicatorSpec> indicator_specs;
    for (const auto& s : strategy.indicator_specs()) {
        indicator_specs.push_back(IndicatorSpec{ s.name, s.doc, s.color });
    }
    stonks::core::Engine engine
    {
        std::move(strategy),
        stonks::datafeed::KLineFeed{ data_file, { .start = start, .end = end, .symbols = symbols } },
        stonks::broker::BacktestBroker{ starting_cash, broker_config }
    };

    const auto t0 = std::chrono::steady_clock::now();
    engine.run();
    const std::chrono::nanoseconds elapsed = std::chrono::steady_clock::now() - t0;

    const ReportInput input{
        starting_cash,
        engine.bars_processed(),
        engine.trades(),
        engine.orders(),
        engine.equity_curve(),
        engine.cash(),
        engine.equity(),
        elapsed,
        broker_config,
        StrategyRunInfo{ "qmsignals", "QMSignalsStrategy", {} },   // headless: no overrides
        RunMeta{ "QMSignalsStrategy", data_file, "binance_1d", start, end, symbols },
        std::move(indicator_specs),
        engine.indicators(),
    };
    const ReportMetrics metrics = compute_metrics(input);
    print_report(std::cout, metrics);

    const std::string path = timestamped_report_path();
    std::filesystem::create_directories(std::filesystem::path{ path }.parent_path());
    std::ofstream out{ path };
    write_report_json(out, input, metrics);
    std::cout << "Report written to " << path << '\n';
}

// Returns true if "--gui" appears among the command-line arguments.
bool wants_gui(int argc, const char* const* argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string_view{ argv[i] } == "--gui") {
            return true;
        }
    }
    return false;
}

}  // namespace stonks::app

int main(int argc, char* argv[]) {
    if (!stonks::app::wants_gui(argc, argv)) {
        stonks::app::run_backtest();
        return 0;
    }

    QGuiApplication app{ argc, argv };
    stonks::app::BacktestController controller;
    qmlRegisterSingletonInstance("Stonks", 1, 0, "Backtest", &controller);
    QQmlApplicationEngine engine;
    engine.load(QUrl{ "qrc:/qt/qml/Stonks/main.qml" });
    if (engine.rootObjects().isEmpty()) {
        return -1;
    }
    return app.exec();
}
