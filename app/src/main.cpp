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
    stonks::core::Engine engine
    {
        PythonStrategy{ "qmsignals", "QMSignalsStrategy" },
        stonks::datafeed::KLineFeed{ "app/data/binance_1m.parquet"
            , { .start = "2025-06-20", .end = "2026-07-01", .symbols = { "SOLUSDT" } }
        },
        stonks::broker::BacktestBroker{ starting_cash }
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
