#include "backtestcontroller.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <QMetaObject>

#include "stonks/core/types.h"
#include "stonks/datafeed/parquetmeta.h"

#include "backtestworker.h"

namespace stonks::app {
namespace {

QString pretty_label(const QString& stem)
{
    QString s = stem;
    s.replace('_', QStringLiteral(" · "));
    return s;
}

std::string date_from_ms(std::int64_t ms)
{
    std::ostringstream os;
    os << core::Timestamp::from_millis(ms);
    return os.str().substr(0, 10);   // YYYY-MM-DD
}

} // namespace

BacktestController::BacktestController(QObject* parent) : QObject{ parent }
{
    m_worker = new BacktestWorker;   // no parent; owned by the thread, deleted on finish
    m_worker->moveToThread(&m_thread);
    connect(&m_thread, &QThread::finished, m_worker, &QObject::deleteLater);
    connect(m_worker, &BacktestWorker::resultReady, this, &BacktestController::onResult);
    connect(m_worker, &BacktestWorker::runFailed, this, &BacktestController::onFailed);
    connect(m_worker, &BacktestWorker::runCancelled, this, &BacktestController::onCancelled);
    connect(m_worker, &BacktestWorker::archiveLoaded, this, &BacktestController::archiveLoaded);
    m_thread.start();

    m_pollTimer.setInterval(100);
    connect(&m_pollTimer, &QTimer::timeout, this, &BacktestController::poll);
}

BacktestController::~BacktestController()
{
    m_thread.quit();
    m_thread.wait();
}

QString BacktestController::pathFor(const QString& dataKey) const
{
    return m_dataDir + QStringLiteral("/") + dataKey + QStringLiteral(".parquet");
}

QVariantList BacktestController::listStrategies()
{
    if (m_strategiesCache.isEmpty()) {
        QVariantList out;
        QMetaObject::invokeMethod(m_worker, "listStrategiesImpl",
            Qt::BlockingQueuedConnection, Q_RETURN_ARG(QVariantList, out));
        m_strategiesCache = out;
    }
    return m_strategiesCache;
}

QVariantList BacktestController::listDataFiles()
{
    if (!m_dataFilesCache.isEmpty()) { return m_dataFilesCache; }

    QVariantList out;
    std::error_code ec;
    const std::string dir = m_dataDir.toStdString();
    if (std::filesystem::is_directory(dir, ec)) {
        std::vector<std::string> stems;
        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            if (!entry.is_regular_file()) { continue; }
            if (entry.path().extension() != ".parquet") { continue; }
            stems.push_back(entry.path().stem().string());
        }
        std::ranges::sort(stems);
        for (const auto& stem : stems) {
            const QString key = QString::fromStdString(stem);
            QVariantMap entry;
            entry["key"] = key;
            entry["label"] = pretty_label(key);
            entry["source"] = pathFor(key);
            out.append(entry);
        }
    }
    m_dataFilesCache = out;
    return out;
}

QStringList BacktestController::peekSymbols(const QString& dataKey)
{
    QStringList out;
    try {
        const auto meta = datafeed::peek_parquet(pathFor(dataKey).toStdString());
        for (const auto& s : meta.symbols) { out << QString::fromStdString(s); }
    } catch (const std::exception&) {
    }
    return out;
}

QVariantMap BacktestController::peekDateRange(const QString& dataKey)
{
    QVariantMap out;
    try {
        const auto meta = datafeed::peek_parquet(pathFor(dataKey).toStdString());
        if (meta.min_ts_ms) {
            out["start"] = QString::fromStdString(date_from_ms(*meta.min_ts_ms));
        }
        if (meta.max_ts_ms) {
            // The filter's end bound is exclusive — bump a day so the default
            // range includes the last bar.
            out["end"] = QString::fromStdString(date_from_ms(*meta.max_ts_ms + 86400000LL));
        }
    } catch (const std::exception&) {
    }
    return out;
}

void BacktestController::run(const QVariantMap& params)
{
    if (m_running) { return; }
    setProgress(0.0);
    setBarsDone(0);
    setRunning(true);
    QMetaObject::invokeMethod(m_worker, "runImpl", Qt::QueuedConnection,
        Q_ARG(QVariantMap, params));
    m_pollTimer.start();
}

void BacktestController::cancel()
{
    if (!m_running) { return; }
    m_worker->requestCancel();   // atomic store; the run loop stops on its next bar
}

void BacktestController::loadArchive()
{
    QMetaObject::invokeMethod(m_worker, "loadArchiveImpl", Qt::QueuedConnection);
}

bool BacktestController::deleteRun(const QString& reportPath)
{
    // Containment: only ever delete files inside the reports directory.
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path path = fs::weakly_canonical(fs::path{ reportPath.toStdString() }, ec);
    if (ec || path.empty()) { return false; }
    const fs::path reports = fs::weakly_canonical(fs::path{ "app/reports" }, ec);
    if (ec) { return false; }
    const auto rel = path.lexically_relative(reports);
    if (rel.empty() || rel.native().starts_with("..")) { return false; }
    return fs::remove(path, ec) && !ec;
}

void BacktestController::poll()
{
    const auto state = m_worker->currentProgress();
    setProgress(state.percent < 0 ? 0.0 : static_cast<double>(state.percent));
    setBarsDone(static_cast<int>(state.current));
}

void BacktestController::onResult(QVariantMap result)
{
    m_pollTimer.stop();
    setProgress(100.0);
    setRunning(false);
    Q_EMIT finished(result);
}

void BacktestController::onFailed(QString message)
{
    m_pollTimer.stop();
    setRunning(false);
    Q_EMIT failed(message);
}

void BacktestController::onCancelled()
{
    m_pollTimer.stop();
    setProgress(0.0);
    setBarsDone(0);
    setRunning(false);
}

void BacktestController::setProgress(double value)
{
    if (value != m_progress) { m_progress = value; Q_EMIT progressChanged(); }
}

void BacktestController::setBarsDone(int value)
{
    if (value != m_barsDone) { m_barsDone = value; Q_EMIT barsDoneChanged(); }
}

void BacktestController::setRunning(bool value)
{
    if (value != m_running) { m_running = value; Q_EMIT runningChanged(); }
}

} // namespace stonks::app
