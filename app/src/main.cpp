#include <QGuiApplication>
#include <QQmlApplicationEngine>

#include "cli.h"
#include "backtest.h"

int main(int argc, char* argv[]) {
    if (!stonks::app::wants_gui(argc, argv)) {
        stonks::app::run_backtest();
        return 0;
    }

    QGuiApplication app{ argc, argv };
    QQmlApplicationEngine engine;
    engine.loadFromModule("Stonks", "Main");
    if (engine.rootObjects().isEmpty()) {
        return -1;
    }
    return app.exec();
}
