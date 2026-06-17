#pragma once

#include <QObject>
#include <QtQml/qqmlregistration.h>

#include "backtest.h"

namespace stonks::app {

// QML-exposed handle that runs the backtest when the Run button is clicked.
class EngineRunner : public QObject {
    Q_OBJECT
    QML_ELEMENT

public:
    using QObject::QObject;

    Q_INVOKABLE void run() { run_backtest(); }
};

}  // namespace stonks::app
