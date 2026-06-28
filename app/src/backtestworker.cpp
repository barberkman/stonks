#include "backtestworker.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <QStringList>
#include <QVariant>

#include "stonks/broker/backtestbroker.h"
#include "stonks/core/engine.h"
#include "stonks/datafeed/klinefeed.h"
#include "stonks/python/embeddedpython.h"

#include "analytics.h"
#include "report.h"
#include "resultmap.h"
#include "strategies/pythonstrategy.h"
#include "strategies/strategydiscovery.h"

namespace stonks::app {

using EngineT = core::Engine<PythonStrategy, datafeed::KLineFeed, broker::BacktestBroker>;

// Hidden visibility: the engine field transitively holds PythonStrategy, which
// is hidden-visibility (pybind11's py::object). Matches PythonStrategy itself.
struct __attribute__((visibility("hidden"))) BacktestWorker::Impl
{
    std::optional<python::EmbeddedPython> python;   // the session-long interpreter pin
    std::shared_ptr<EngineT> engine;                // alive only during a run
    mutable std::mutex engine_mtx;
    std::atomic<bool> cancel{ false };
    bool py_inited{ false };
};

BacktestWorker::BacktestWorker(QObject* parent)
: QObject{ parent }, m_impl{ std::make_unique<Impl>() }
{}

BacktestWorker::~BacktestWorker() = default;

void BacktestWorker::requestCancel()
{
    m_impl->cancel.store(true, std::memory_order_relaxed);
}

core::ProgressState BacktestWorker::currentProgress() const
{
    std::shared_ptr<EngineT> engine;
    {
        std::lock_guard lock{ m_impl->engine_mtx };
        engine = m_impl->engine;
    }
    return engine ? engine->progress() : core::ProgressState{};
}

namespace {

// Bars-per-year from the median spacing of the run's distinct timestamps, on a
// 24/7 calendar (the data is crypto; daily equities would skew ~365 vs the 252
// trading-day convention — acceptable for a displayed Sharpe). 1.0 (no scaling)
// when the interval cannot be inferred.
double annualization_from(const std::vector<std::int64_t>& group_ts)
{
    if (group_ts.size() < 3) { return 1.0; }
    std::vector<std::int64_t> deltas;
    deltas.reserve(group_ts.size() - 1);
    for (std::size_t i = 1; i < group_ts.size(); ++i) {
        deltas.push_back(group_ts[i] - group_ts[i - 1]);
    }
    std::ranges::sort(deltas);
    const std::int64_t median_ms = deltas[deltas.size() / 2];
    if (median_ms <= 0) { return 1.0; }
    return 365.25 * 24.0 * 3600.0 * 1000.0 / static_cast<double>(median_ms);
}

} // namespace

void BacktestWorker::runImpl(QVariantMap params)
{
    try {
        if (!m_impl->py_inited) {
            // Mirror PythonStrategy::DefaultPaths *before* the pin initializes the
            // interpreter, so sys.path picks up the venv + strategy dir. overwrite=0
            // lets a caller-supplied env win.
            ::setenv("STONKS_VENV", "app/python/.venv", 0);
            ::setenv("STONKS_PYTHONPATH", "app/python", 0);
            m_impl->python.emplace();   // first EmbeddedPython -> inits CPython on THIS thread
            m_impl->py_inited = true;
        }
        m_impl->cancel.store(false, std::memory_order_relaxed);

        const std::string module = params.value("strategyModule").toString().toStdString();
        const std::string cls = params.value("strategyClass").toString().toStdString();
        std::string display = params.value("strategyDisplay").toString().toStdString();
        if (display.empty()) { display = cls; }
        const std::string data_file = params.value("dataFile").toString().toStdString();
        const std::string data_key = params.value("dataKey").toString().toStdString();
        const std::string start = params.value("start").toString().toStdString();
        const std::string end = params.value("end").toString().toStdString();
        const double cash = params.value("startCash").toDouble();

        std::vector<core::Symbol> symbols;
        for (const auto& s : params.value("symbols").toStringList()) {
            symbols.push_back(s.toStdString());
        }
        const datafeed::Filter filter{ start, end, symbols };

        // Build + run the engine. PythonStrategy bumps the interpreter refcount
        // (no re-init thanks to the pin). The engine polls &cancel each timestamp.
        PythonStrategy strategy{ module, cls };
        datafeed::KLineFeed feed{ data_file, filter };
        broker::BacktestBroker broker{ cash };
        auto engine = std::make_shared<EngineT>(
            std::move(strategy), std::move(feed), broker,
            core::ProgressOutput::Silent, &m_impl->cancel);
        {
            std::lock_guard lock{ m_impl->engine_mtx };
            m_impl->engine = engine;
        }

        const auto t0 = std::chrono::steady_clock::now();
        engine->run();
        const std::chrono::nanoseconds elapsed = std::chrono::steady_clock::now() - t0;

        const bool cancelled = m_impl->cancel.load(std::memory_order_relaxed);

        const ReportInput input{
            cash,
            engine->bars_processed(),
            engine->trades(),
            engine->orders(),
            engine->equity_curve(),
            engine->cash(),
            engine->equity(),
            elapsed,
        };
        const ReportMetrics metrics = compute_metrics(input);

        // Harvest candles from a fresh feed with the same filter (the engine
        // consumed its feed and exposes no getter; build() is deterministic so the
        // per-symbol bar indices match what the engine saw).
        std::map<core::Symbol, SymbolSeries> candles;
        std::vector<std::int64_t> group_ts;
        {
            datafeed::KLineFeed feed2{ data_file, filter };
            while (const auto ts = feed2.next_timestamp()) {
                for (const auto& bar : feed2.current_bars()) {
                    candles[bar.symbol].candles.push_back(Candle{
                        bar.open, bar.high, bar.low, bar.close, bar.volume,
                        bar.timestamp.value.time_since_epoch().count() });
                }
                group_ts.push_back(ts->value.time_since_epoch().count());
                feed2.advance();
            }
        }

        {
            std::lock_guard lock{ m_impl->engine_mtx };
            m_impl->engine.reset();
        }

        if (cancelled) {
            Q_EMIT runCancelled();
            return;
        }

        const RunConfig cfg{ std::string{}, display, data_key };
        QVariantMap result = build_result(cfg, input, metrics, candles,
                                          annualization_from(group_ts));
        Q_EMIT resultReady(result);
    } catch (const std::exception& e) {
        {
            std::lock_guard lock{ m_impl->engine_mtx };
            m_impl->engine.reset();
        }
        Q_EMIT runFailed(QString::fromStdString(e.what()));
    }
}

QVariantList BacktestWorker::listStrategiesImpl()
{
    QVariantList out;
    try {
        if (!m_impl->py_inited) {
            ::setenv("STONKS_VENV", "app/python/.venv", 0);
            ::setenv("STONKS_PYTHONPATH", "app/python", 0);
            m_impl->python.emplace();
            m_impl->py_inited = true;
        }
        for (const auto& info : discover_strategies()) {
            QVariantMap entry;
            entry["display"] = QString::fromStdString(info.display);
            entry["module"] = QString::fromStdString(info.module);
            entry["cls"] = QString::fromStdString(info.cls);
            out.append(entry);
        }
    } catch (const std::exception&) {
        // Best-effort: return whatever resolved before the failure.
    }
    return out;
}

} // namespace stonks::app
