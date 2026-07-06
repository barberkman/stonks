#pragma once

#include <memory>

#include <QObject>
#include <QString>
#include <QVariantList>
#include <QVariantMap>

#include "stonks/core/progressbar.h"   // core::ProgressState

namespace stonks::app {

// Runs backtests on a dedicated thread and owns the embedded-Python interpreter
// for the whole session — created lazily on this thread on first use and never
// torn down (re-initializing CPython in-process is unsafe). The heavy engine /
// pybind types live only in the .cpp (PIMPL), so this header — which AUTOMOC
// processes — stays free of them.
class BacktestWorker : public QObject
{
    Q_OBJECT

public:
    explicit BacktestWorker(QObject* parent = nullptr);
    ~BacktestWorker() override;

    // Called directly from the GUI thread. requestCancel() is a lone atomic
    // store; currentProgress() returns a mutex-guarded snapshot. Neither touches
    // Python, so neither needs the GIL.
    void requestCancel();
    core::ProgressState currentProgress() const;

public Q_SLOTS:
    // All run on the worker thread (the only thread that touches CPython).
    void runImpl(QVariantMap params);
    QVariantList listStrategiesImpl();
    // Rebuild result maps for every archived report under app/reports/ (candle
    // drill-down re-harvested when the report records its data provenance);
    // emits archiveLoaded when done. Unreadable/foreign files are skipped.
    void loadArchiveImpl();

Q_SIGNALS:
    void resultReady(QVariantMap result);
    void runFailed(QString message);
    void runCancelled();
    void archiveLoaded(QVariantList runs);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace stonks::app
