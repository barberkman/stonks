#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QThread>
#include <QTimer>
#include <QVariantList>
#include <QVariantMap>

namespace stonks::app {

class BacktestWorker;

// QML-facing singleton (registered as `Backtest`). Owns the worker thread, polls
// its live progress on a timer, and re-emits run completion to QML. All Python /
// engine work happens on the worker thread; this object stays on the GUI thread.
class BacktestController : public QObject
{
    Q_OBJECT
    Q_PROPERTY(double progress READ progress NOTIFY progressChanged)
    Q_PROPERTY(int barsDone READ barsDone NOTIFY barsDoneChanged)
    Q_PROPERTY(bool running READ running NOTIFY runningChanged)

public:
    explicit BacktestController(QObject* parent = nullptr);
    ~BacktestController() override;

    double progress() const { return m_progress; }
    int barsDone() const { return m_barsDone; }
    bool running() const { return m_running; }

    // Setup-screen helpers. listStrategies blocks briefly on the worker thread
    // (the interpreter); the rest read parquet metadata on the GUI thread.
    Q_INVOKABLE QVariantList listStrategies();
    Q_INVOKABLE QVariantList listDataFiles();
    Q_INVOKABLE QStringList peekSymbols(const QString& dataKey);
    Q_INVOKABLE QVariantMap peekDateRange(const QString& dataKey);

    Q_INVOKABLE void run(const QVariantMap& params);
    Q_INVOKABLE void cancel();

Q_SIGNALS:
    void progressChanged();
    void barsDoneChanged();
    void runningChanged();
    void finished(QVariantMap result);
    void failed(QString message);

private Q_SLOTS:
    void onResult(QVariantMap result);
    void onFailed(QString message);
    void onCancelled();
    void poll();

private:
    void setProgress(double value);
    void setBarsDone(int value);
    void setRunning(bool value);
    QString pathFor(const QString& dataKey) const;

    QThread m_thread;
    BacktestWorker* m_worker{ nullptr };
    QTimer m_pollTimer;
    double m_progress{ 0.0 };
    int m_barsDone{ 0 };
    bool m_running{ false };
    QVariantList m_strategiesCache;
    QVariantList m_dataFilesCache;
    QString m_dataDir{ QStringLiteral("app/data") };
};

} // namespace stonks::app
