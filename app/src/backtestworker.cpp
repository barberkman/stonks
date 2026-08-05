#include "backtestworker.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
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
#include "report_json.h"
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

        // Effective strategy parameters chosen in the setup screen (full map:
        // declared defaults overlaid with edits). One source, two carriers:
        // the std::map drives the PythonStrategy overrides + the JSON stamp,
        // the raw QVariantMap rides to the QML report view.
        const QVariantMap raw_strategy_params = params.value("strategyParams").toMap();
        std::map<std::string, double> strategy_params;
        for (auto it = raw_strategy_params.constBegin(); it != raw_strategy_params.constEnd(); ++it) {
            strategy_params[it.key().toStdString()] = it.value().toDouble();
        }

        // Build + run the engine. PythonStrategy bumps the interpreter refcount
        // (no re-init thanks to the pin). The engine polls &cancel each timestamp.
        PythonStrategy strategy{ module, cls, strategy_params };

        // Declared indicator metadata — class-level, so read it now, before
        // the strategy is moved into the engine. Converted into the report
        // layer's own pybind11-free IndicatorSpec (same glue pattern as the
        // strategy params above).
        std::vector<IndicatorSpec> indicator_specs;
        for (const auto& s : strategy.indicator_specs()) {
            indicator_specs.push_back(IndicatorSpec{ s.name, s.doc, s.color });
        }

        datafeed::KLineFeed feed{ data_file, filter };
        // Fee schedule from the setup screen (bps of notional + flat per fill),
        // defaulting to Binance USDT-M VIP0 when the params omit it; stamped
        // into the report JSON.
        const broker::BrokerConfig broker_config{
            .maker_fee_bps = params.value("makerFeeBps", 2.0).toDouble(),
            .taker_fee_bps = params.value("takerFeeBps", 5.0).toDouble(),
            .fee_per_fill = params.value("feePerFill", 0.0).toDouble(),
        };
        broker::BacktestBroker broker{ cash, broker_config };
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
            broker_config,
            StrategyRunInfo{ module, cls, strategy_params },
            RunMeta{ display, data_file, data_key, start, end, symbols },
            std::move(indicator_specs),
            engine->indicators(),
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

        // Archive the run like the headless path does, so GUI runs are
        // auditable (tools/verify_backtest.py), restorable next session, and
        // the parameter stamp is durable. Cancelled runs never reach this point.
        const std::string report_path = timestamped_report_path();
        {
            std::filesystem::create_directories(std::filesystem::path{ report_path }.parent_path());
            std::ofstream json_out{ report_path };
            write_report_json(json_out, input, metrics);
        }

        const RunConfig cfg{ std::string{}, display, data_key, raw_strategy_params };
        QVariantMap result = build_result(cfg, input, metrics, candles,
                                          annualization_from(group_ts));
        result["reportPath"] = QString::fromStdString(report_path);
        Q_EMIT resultReady(result);
    } catch (const std::exception& e) {
        {
            std::lock_guard lock{ m_impl->engine_mtx };
            m_impl->engine.reset();
        }
        Q_EMIT runFailed(QString::fromStdString(e.what()));
    }
}

void BacktestWorker::loadArchiveImpl()
{
    namespace fs = std::filesystem;
    QVariantList out;
    std::vector<fs::path> paths;
    std::error_code ec;
    if (fs::is_directory("app/reports", ec)) {
        for (const auto& entry : fs::directory_iterator("app/reports", ec)) {
            if (entry.is_regular_file() && entry.path().extension() == ".json") {
                paths.push_back(entry.path());
            }
        }
    }
    std::ranges::sort(paths, std::greater{});   // timestamped names: newest first

    for (const auto& path : paths) {
        try {
            std::ifstream in{ path };
            const nlohmann::json j = nlohmann::json::parse(in);
            const ReportInput input = report_input_from_json(j);
            const ReportMetrics metrics = compute_metrics(input);

            // The equity curve's timestamps are exactly the run's tick groups.
            std::vector<std::int64_t> group_ts;
            group_ts.reserve(input.equity_curve.size());
            for (const auto& p : input.equity_curve) {
                group_ts.push_back(p.timestamp.value.time_since_epoch().count());
            }

            // Candle drill-down: re-harvest from the recorded data provenance.
            // Pre-provenance archives load fine with empty per-symbol charts.
            std::map<core::Symbol, SymbolSeries> candles;
            if (!input.run.data_file.empty() && fs::exists(input.run.data_file, ec)) {
                const datafeed::Filter filter{ input.run.start, input.run.end, input.run.symbols };
                datafeed::KLineFeed feed{ input.run.data_file, filter };
                while (const auto ts = feed.next_timestamp()) {
                    for (const auto& bar : feed.current_bars()) {
                        candles[bar.symbol].candles.push_back(Candle{
                            bar.open, bar.high, bar.low, bar.close, bar.volume,
                            bar.timestamp.value.time_since_epoch().count() });
                    }
                    feed.advance();
                }
            }

            QVariantMap params;
            for (const auto& [name, value] : input.strategy.params) {
                params[QString::fromStdString(name)] = value;
            }
            const std::string display = !input.run.display.empty() ? input.run.display
                : (!input.strategy.cls.empty() ? input.strategy.cls : "archived run");
            const RunConfig cfg{ std::string{}, display, input.run.data_key, params };
            QVariantMap result = build_result(cfg, input, metrics, candles,
                                              annualization_from(group_ts));
            result["reportPath"] = QString::fromStdString(path.string());
            out.append(result);
        } catch (const std::exception&) {
            continue;   // unreadable or foreign JSON: skip, never break startup
        }
    }
    Q_EMIT archiveLoaded(out);
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
            QVariantList params;
            for (const auto& p : info.params) {
                QVariantMap pm;
                pm["name"] = QString::fromStdString(p.name);
                pm["default"] = p.default_value;
                pm["type"] = QString::fromStdString(p.type_name);
                pm["doc"] = QString::fromStdString(p.doc);
                pm["unit"] = QString::fromStdString(p.unit);
                QStringList choices;
                for (const auto& c : p.choices) {
                    choices << QString::fromStdString(c);
                }
                pm["choices"] = choices;
                params.append(pm);
            }
            entry["params"] = params;
            out.append(entry);
        }
    } catch (const std::exception&) {
        // Best-effort: return whatever resolved before the failure.
    }
    return out;
}

} // namespace stonks::app
