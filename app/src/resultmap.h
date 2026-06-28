#pragma once

#include <map>
#include <string>

#include <QVariantMap>

#include "stonks/core/types.h"

#include "analytics.h"
#include "report.h"

namespace stonks::app {

// Run identity/config the GUI shows but the engine does not produce.
struct RunConfig
{
    std::string id;
    std::string strategy_display;
    std::string data_key;
};

// Assemble the full QVariantMap the QML views consume, mirroring the mock-data
// schema (mockdata.js): a backtest summary with pre-formatted display strings,
// plus numeric drill-down arrays (equity, drawdown, and per-symbol candles /
// round-trip trades / sparkline). `candles_by_symbol` holds each symbol's full
// candle series (harvested from a fresh feed); `annualization` scales Sharpe.
QVariantMap build_result(const RunConfig& cfg,
                         const ReportInput& input,
                         const ReportMetrics& metrics,
                         const std::map<core::Symbol, SymbolSeries>& candles_by_symbol,
                         double annualization);

} // namespace stonks::app
