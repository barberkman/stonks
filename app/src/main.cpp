#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <QGuiApplication>
#include <QIcon>
#include <QQmlApplicationEngine>
#include <QQuickStyle>
#include <QtQml/qqml.h>

#include "stonks/core/engine.h"
#include "stonks/broker/backtestbroker.h"
#include "stonks/binance/binancebroker.h"
#include "stonks/binance/liveklinefeed.h"
#include "stonks/datafeed/klinefeed.h"
#include "stonks/python/embeddedpython.h"

#include "backtestcontroller.h"
#include "report.h"
#include "report_json.h"
#include "strategies/pythonstrategy.h"
#include "strategies/strategydiscovery.h"

namespace stonks::app {

bool has_flag(int argc, const char* const* argv, std::string_view flag);
std::string flag_value(int argc, const char* const* argv, std::string_view flag,
                       std::string fallback);
std::vector<std::string> split_csv(const std::string& csv);

// Wires up the Engine (the algo_trade strategy + KLineFeed + BacktestBroker)
// and runs it, printing the report to the terminal.
//
//   app [--data app/data/us_1d.parquet] [--symbols MU,NVDA]
//       [--start 2024-06-01] [--end 2026-07-01]
//
// Defaults run the BIST daily set with no symbol filter — an empty --symbols
// list is KLineFeed's "all symbols in the file".
//
// The default window is set by algo_trade's pre-trained artifact, which was fit
// through 2024-12-31: the strategy refuses to trade any bar the fit could see.
// --start is the *beginning of the feed* rather than a lookback's worth earlier,
// because two of the model's features (obv and days_since_past_extreme)
// accumulate from a symbol's first bar. KLineFeed truncates history at --start,
// so a later start silently reseeds both against a shorter history and scores
// bars on values the fit never saw. Retrain with a different --train-end and only
// the trading window moves; --start stays at the front of the data.
void run_backtest(int argc, const char* const* argv) {
    constexpr stonks::core::Balance starting_cash = 1000.0;
    const std::string data_file =
        flag_value(argc, argv, "--data", "app/data/bist_1d.parquet");
    const std::string start = flag_value(argc, argv, "--start", "2020-01-02");
    const std::string end = flag_value(argc, argv, "--end", "2026-07-24");
    const std::vector<std::string> symbols =
        split_csv(flag_value(argc, argv, "--symbols", ""));
    // Binance USDT-M VIP0 fee schedule; stamped into the report JSON.
    const stonks::broker::BrokerConfig broker_config{
        .maker_fee_bps = 2.0,
        .taker_fee_bps = 5.0,
    };
    PythonStrategy strategy{ "algo_trade", "AlgoTradeStrategy" };
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
        StrategyRunInfo{ "algo_trade", "AlgoTradeStrategy", {} },   // headless: no overrides
        RunMeta{ "AlgoTradeStrategy", data_file,
                 std::filesystem::path{ data_file }.stem().string(), start, end, symbols },
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

// Runs every Python strategy discovered in app/python/ sequentially against the
// same data window and broker config as run_backtest(), printing each strategy's
// report and a final leaderboard sorted by return. Each run is also archived as
// its own JSON under one timestamped batch directory. A strategy that fails to
// load or raises mid-run is reported and skipped so the batch still completes.
void run_all_backtests() {
    constexpr stonks::core::Balance starting_cash = 1000.0;
    const std::string data_file = "app/data/bist_1d.parquet";
    const std::string start = "2024-01-01";
    const std::string end = "2026-01-30";
    const std::vector<std::string> symbols{};   // empty = every symbol in the file
    const stonks::broker::BrokerConfig broker_config{
        .maker_fee_bps = 2.0,
        .taker_fee_bps = 5.0,
    };

    // Bring the embedded interpreter up before discovery (which needs it live),
    // mirroring PythonStrategy's app-local path defaults so `./app` from the
    // project root finds the venv + strategy dir with no env setup (overwrite=0
    // lets a caller-supplied env win). This outer handle keeps the interpreter
    // alive across the whole loop; each PythonStrategy just bumps its refcount.
    ::setenv("STONKS_VENV", "app/python/.venv", 0);
    ::setenv("STONKS_PYTHONPATH", "app/python", 0);
    stonks::python::EmbeddedPython python{};

    const std::vector<StrategyInfo> strategies = discover_strategies("app/python");
    if (strategies.empty()) {
        std::cout << "No strategies found in app/python/.\n";
        return;
    }

    // Group this batch's per-strategy reports under one timestamped directory,
    // derived from the same helper the single-run path uses (minus the .json).
    std::string batch_dir = timestamped_report_path();
    batch_dir.erase(batch_dir.size() - std::string_view{ ".json" }.size());
    std::filesystem::create_directories(batch_dir);

    // One leaderboard row per strategy; a failed run carries an error message
    // instead of metrics.
    struct Row
    {
        std::string name;
        std::optional<ReportMetrics> metrics;
        std::string error;
    };
    std::vector<Row> rows;

    for (const auto& info : strategies) {
        std::cout << "\n########## " << info.display << " (" << info.module << ") ##########\n";
        try {
            PythonStrategy strategy{ info.module, info.cls };
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
                StrategyRunInfo{ info.module, info.cls, {} },   // batch: no overrides
                RunMeta{ info.display, data_file, "bist_1d", start, end, symbols },
                std::move(indicator_specs),
                engine.indicators(),
            };
            const ReportMetrics metrics = compute_metrics(input);
            print_report(std::cout, metrics);

            const std::string path = batch_dir + "/" + info.module + ".json";
            std::ofstream out{ path };
            write_report_json(out, input, metrics);

            rows.push_back(Row{ info.module, metrics, {} });
        } catch (const std::exception& e) {
            // One broken strategy must not abort the other runs.
            std::cout << "FAILED: " << e.what() << '\n';
            rows.push_back(Row{ info.module, std::nullopt, e.what() });
        }
    }

    // Leaderboard: successful runs by descending return, failures last.
    std::stable_sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        if (a.metrics.has_value() != b.metrics.has_value()) { return a.metrics.has_value(); }
        if (!a.metrics.has_value()) { return false; }
        return a.metrics->return_pct.value_or(0.0) > b.metrics->return_pct.value_or(0.0);
    });

    std::ios old_state{ nullptr };
    old_state.copyfmt(std::cout);
    std::cout << "\n=== All-strategy leaderboard (" << rows.size()
              << " strategies, sorted by return) ===\n";
    std::cout << std::left << std::setw(20) << "Strategy" << std::right
              << std::setw(10) << "Return%" << std::setw(9) << "Trades"
              << std::setw(9) << "Closed" << std::setw(8) << "Win%"
              << std::setw(9) << "MaxDD%" << std::setw(5) << "Liq"
              << std::setw(12) << "EndEquity" << '\n';
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& r : rows) {
        std::cout << std::left << std::setw(20) << r.name << std::right;
        if (!r.metrics) {
            std::cout << "   failed: " << r.error << '\n';
            continue;
        }
        const ReportMetrics& m = *r.metrics;
        std::cout << std::setw(10) << m.return_pct.value_or(0.0)
                  << std::setw(9) << m.trade_count
                  << std::setw(9) << m.closed_trades
                  << std::setw(8) << m.win_rate_pct.value_or(0.0)
                  << std::setw(9) << m.max_drawdown_pct
                  << std::setw(5) << m.liquidations
                  << std::setw(12) << m.ending_equity << '\n';
    }
    std::cout.copyfmt(old_state);
    std::cout << "\nReports written to " << batch_dir << "/\n";
}

// Returns true if `flag` appears among the command-line arguments.
bool has_flag(int argc, const char* const* argv, std::string_view flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::string_view{ argv[i] } == flag) {
            return true;
        }
    }
    return false;
}

// Returns true if "--gui" appears among the command-line arguments.
bool wants_gui(int argc, const char* const* argv) {
    return has_flag(argc, argv, "--gui");
}

// The value following `flag` (e.g. "--interval" -> "1h"), or `fallback` if the
// flag is absent or has no following argument.
std::string flag_value(int argc, const char* const* argv, std::string_view flag,
                       std::string fallback = {}) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string_view{ argv[i] } == flag) { return argv[i + 1]; }
    }
    return fallback;
}

// Split "BTCUSDT,ETHUSDT" into {"BTCUSDT","ETHUSDT"}.
std::vector<std::string> split_csv(const std::string& csv) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start <= csv.size()) {
        const std::size_t comma = csv.find(',', start);
        const std::size_t end = comma == std::string::npos ? csv.size() : comma;
        if (end > start) { out.push_back(csv.substr(start, end - start)); }
        if (comma == std::string::npos) { break; }
        start = comma + 1;
    }
    return out;
}

// Set by SIGINT so a live run stops cleanly: the feed's blocking wait aborts and
// the engine loop exits after the current tick.
std::atomic<bool> g_live_cancel{ false };
void on_sigint(int) { g_live_cancel.store(true, std::memory_order_relaxed); }

// Live trading against Binance USDⓈ-M Futures. Wires the same Engine with a
// LiveKlineFeed (REST kline polling) and a BinanceBroker (all state read from
// Binance). Defaults to testnet; pass --mainnet to trade real funds.
//
//   app --live --strategy qmmomentumswing --symbols BTCUSDT --interval 1h [--mainnet] [--dry-run]
//
// Credentials come from the environment: BINANCE_API_KEY and
// BINANCE_PRIVATE_KEY_PEM (inline Ed25519 PEM or a path to it).
int run_live(int argc, const char* const* argv) {
    const bool testnet = !has_flag(argc, argv, "--mainnet");
    const bool dry_run = has_flag(argc, argv, "--dry-run");
    const std::string interval = flag_value(argc, argv, "--interval", "1h");
    const std::vector<std::string> symbols =
        split_csv(flag_value(argc, argv, "--symbols", "BTCUSDT"));
    const std::string strategy_arg = flag_value(argc, argv, "--strategy", "qmmomentumswing");

    if (symbols.empty()) {
        std::cerr << "No symbols given (use --symbols BTCUSDT,ETHUSDT).\n";
        return 2;
    }

    try {
        // Bring up the interpreter (PythonStrategy path defaults) before
        // discovery / construction, mirroring the batch runner.
        ::setenv("STONKS_VENV", "app/python/.venv", 0);
        ::setenv("STONKS_PYTHONPATH", "app/python", 0);
        stonks::python::EmbeddedPython python{};

        // Resolve "module" or "module:Class" to a (module, class) pair. A bare
        // module is looked up via strategy discovery.
        std::string module = strategy_arg;
        std::string cls;
        if (const auto colon = strategy_arg.find(':'); colon != std::string::npos) {
            module = strategy_arg.substr(0, colon);
            cls = strategy_arg.substr(colon + 1);
        } else {
            for (const auto& info : discover_strategies("app/python")) {
                if (info.module == module) { cls = info.cls; break; }
            }
            if (cls.empty()) {
                std::cerr << "Strategy module '" << module << "' not found in app/python/.\n";
                return 2;
            }
        }

        stonks::binance::BinanceConfig config =
            stonks::binance::BinanceConfig::from_env(testnet);
        config.dry_run = dry_run;

        std::cout << "Live " << (testnet ? "TESTNET" : "MAINNET")
                  << (dry_run ? " (dry-run)" : "") << ": " << module << ":" << cls
                  << "  symbols=" << flag_value(argc, argv, "--symbols", "BTCUSDT")
                  << "  interval=" << interval << "\n"
                  << "Endpoint: " << config.base_url << "\n"
                  << "Press Ctrl-C to stop.\n";

        std::signal(SIGINT, &on_sigint);

        PythonStrategy strategy{ module, cls };
        stonks::binance::LiveKlineFeed feed{ config, symbols, interval, /*seed_bars=*/200,
                                             &g_live_cancel };
        stonks::binance::BinanceBroker broker{ config };

        std::cout << "Starting equity: " << broker.equity()
                  << "  cash: " << broker.cash() << "\n";

        stonks::core::Engine engine{ std::move(strategy), std::move(feed), std::move(broker),
                                     stonks::core::ProgressOutput::Silent, &g_live_cancel };
        engine.run();

        std::cout << "\nStopped. Final equity: " << engine.equity()
                  << "  cash: " << engine.cash() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Live run failed: " << e.what() << "\n";
        return 1;
    }
}

}  // namespace stonks::app

int main(int argc, char* argv[]) {
    if (!stonks::app::wants_gui(argc, argv)) {
        if (stonks::app::has_flag(argc, argv, "--live")) {
            return stonks::app::run_live(argc, argv);
        }
        if (stonks::app::has_flag(argc, argv, "--all")) {
            stonks::app::run_all_backtests();
        } else {
            stonks::app::run_backtest(argc, argv);
        }
        return 0;
    }

    QGuiApplication app{ argc, argv };
    // Dock / taskbar icon (macOS routes this to NSApp.applicationIconImage).
    app.setWindowIcon(QIcon{ ":/qt/qml/Stonks/qml/icons/appicon.png" });
    QQuickStyle::setStyle("Basic");   // fully customizable base (the native macOS style forbids the ComboBox overrides in StyledSelect.qml)
    stonks::app::BacktestController controller;
    qmlRegisterSingletonInstance("Stonks", 1, 0, "Backtest", &controller);
    QQmlApplicationEngine engine;
    engine.load(QUrl{ "qrc:/qt/qml/Stonks/main.qml" });
    if (engine.rootObjects().isEmpty()) {
        return -1;
    }
    return app.exec();
}
