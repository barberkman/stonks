#include <iostream>
#include <string_view>

#include <QGuiApplication>
#include <QObject>
#include <QQmlApplicationEngine>
#include <QtQml/qqml.h>

#include "stonks/core/engine.h"
#include "stonks/broker/backtestbroker.h"
#include "stonks/datafeed/klinefeed.h"

#include "strategies/ema50strategy.h"

namespace stonks::app {

// Wires up the Engine (EMA50 strategy + KLineFeed + BacktestBroker) and runs it,
// printing the report to the terminal. Shared by the headless and GUI run paths.
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

// QML-exposed handle that runs the backtest when the Run button is clicked.
class EngineRunner : public QObject {
    Q_OBJECT

public:
    using QObject::QObject;

    Q_INVOKABLE void run() { run_backtest(); }
};

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
    qmlRegisterType<stonks::app::EngineRunner>("Stonks", 1, 0, "EngineRunner");
    QQmlApplicationEngine engine;
    engine.load(QUrl{ "qrc:/qt/qml/Stonks/main.qml" });
    if (engine.rootObjects().isEmpty()) {
        return -1;
    }
    return app.exec();
}

#include "main.moc"
