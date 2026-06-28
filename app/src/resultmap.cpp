#include "resultmap.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <QString>
#include <QVariant>
#include <QVariantList>

namespace stonks::app {
namespace {

// --- display formatters (mirror app/qml/js/format.js + mockdata.js) ---

std::string group_int(long long value)
{
    const bool neg = value < 0;
    unsigned long long mag = neg ? 0ULL - static_cast<unsigned long long>(value)
                                  : static_cast<unsigned long long>(value);
    std::string digits = std::to_string(mag);
    std::string out;
    int count = 0;
    for (int i = static_cast<int>(digits.size()) - 1; i >= 0; --i) {
        out.push_back(digits[static_cast<std::size_t>(i)]);
        if (++count % 3 == 0 && i > 0) { out.push_back(','); }
    }
    std::reverse(out.begin(), out.end());
    return (neg ? "-" : "") + out;
}

std::string commas(long long value) { return group_int(value); }

std::string usd(double amount)
{
    return "$" + group_int(std::llround(amount));
}

std::string signed_usd(double amount)
{
    const long long rounded = std::llround(amount);
    return (rounded >= 0 ? "+$" : "-$") + group_int(std::llabs(rounded));
}

// Locale-independent fixed-point format (always '.' decimal). std::snprintf
// would honour LC_NUMERIC, which QGuiApplication adopts from the environment and
// can be a comma — wrong for these money/percent strings.
std::string fixed(double value, int decimals)
{
    char buf[64];
    const auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), value,
                                          std::chars_format::fixed, decimals);
    if (ec != std::errc{}) { return "0"; }
    return std::string{ buf, ptr };
}

// Signed percentage, e.g. +63.4% / -18.7%.
std::string pct_signed(double value, int decimals = 1)
{
    return (value >= 0.0 ? "+" : "") + fixed(value, decimals) + "%";
}

// Unsigned percentage, e.g. 57.3%.
std::string pct_plain(double value, int decimals = 1)
{
    return fixed(value, decimals) + "%";
}

std::string compact_usd(double v)
{
    const double a = std::abs(v);
    const std::string sign = v < 0.0 ? "-" : "";
    if (a >= 1e9) { return sign + "$" + fixed(a / 1e9, 2) + "B"; }
    if (a >= 1e6) { return sign + "$" + fixed(a / 1e6, 2) + "M"; }
    if (a >= 1e3) { return sign + "$" + fixed(a / 1e3, 0) + "K"; }
    return sign + "$" + fixed(a, 2);
}

std::string elapsed_str(std::chrono::nanoseconds elapsed)
{
    const double s = std::chrono::duration<double>{ elapsed }.count();
    return fixed(s, 2) + "s";
}

std::string per_bar_str(std::chrono::nanoseconds elapsed, std::size_t bars)
{
    if (bars == 0) { return "—"; }
    const double us = std::chrono::duration<double, std::micro>{ elapsed }.count()
        / static_cast<double>(bars);
    return fixed(us, 1) + "µs";   // µs
}

std::string iso_date(core::Timestamp ts)
{
    std::ostringstream os;
    os << ts;
    return os.str().substr(0, 10);   // YYYY-MM-DD
}

QString qs(const std::string& s) { return QString::fromStdString(s); }

// Downsample to at most `cap` elements by striding; returns the stride used.
std::size_t stride_for(std::size_t n, std::size_t cap)
{
    if (n <= cap || cap == 0) { return 1; }
    return (n + cap - 1) / cap;   // ceil(n / cap)
}

constexpr std::size_t kCandleCap = 1500;   // display candles per symbol
constexpr std::size_t kSparkCap = 64;      // sparkline points per symbol

} // namespace

QVariantMap build_result(const RunConfig& cfg,
                         const ReportInput& input,
                         const ReportMetrics& metrics,
                         const std::map<core::Symbol, SymbolSeries>& candles_by_symbol,
                         double annualization)
{
    const auto round_trips = reconstruct_round_trips(input.trades);
    const auto sharpe = sharpe_ratio(input.equity_curve, annualization);
    const double pf = profit_factor(round_trips);
    const auto by_symbol = per_symbol_breakdown(round_trips);

    QVariantMap result;
    result["id"] = qs(cfg.id);
    result["strategy"] = qs(cfg.strategy_display);
    result["dataKey"] = qs(cfg.data_key);
    result["status"] = "completed";

    // --- top-line summary (pre-formatted display strings) ---
    const double ret = metrics.return_pct.value_or(0.0);
    result["ret"] = qs(pct_signed(ret));
    result["retPos"] = ret >= 0.0;
    result["maxdd"] = qs(pct_signed(-metrics.max_drawdown_pct));   // stored as positive magnitude
    result["win"] = qs(metrics.win_rate_pct ? pct_plain(*metrics.win_rate_pct) : std::string{ "—" });
    result["trades"] = static_cast<int>(metrics.closed_trades);
    result["sharpe"] = qs(sharpe ? fixed(*sharpe, 2) : std::string{ "—" });
    result["pf"] = qs(std::isinf(pf) ? std::string{ "∞" } : fixed(pf, 2));   // ∞
    result["notional"] = qs(compact_usd(metrics.notional));
    result["elapsed"] = qs(elapsed_str(metrics.elapsed));
    result["perbar"] = qs(per_bar_str(metrics.elapsed, metrics.bars_processed));
    result["startEqStr"] = qs(usd(metrics.starting_cash));
    result["endEqStr"] = qs(usd(metrics.ending_equity));
    result["bars"] = qs(commas(static_cast<long long>(metrics.bars_processed)));
    result["orders"] = qs(commas(static_cast<long long>(metrics.orders_placed)));
    result["range"] = qs(metrics.first_ts
        ? iso_date(*metrics.first_ts) + " → " + iso_date(*metrics.last_ts)   // →
        : std::string{ "—" });

    // --- portfolio equity + drawdown numeric series ---
    QVariantList equity;
    equity.reserve(static_cast<int>(input.equity_curve.size()));
    double peak = -1e300;
    QVariantList drawdown;
    drawdown.reserve(static_cast<int>(input.equity_curve.size()));
    for (const auto& point : input.equity_curve) {
        equity.append(point.equity);
        peak = std::max(peak, point.equity);
        drawdown.append(peak > 0.0 ? (point.equity - peak) / peak : 0.0);
    }
    result["equity"] = equity;
    result["drawdown"] = drawdown;

    // --- per-symbol: summary rows + drill-down (candles, trades, spark) ---
    QVariantList symbols;
    QVariantMap per_symbol;

    for (const auto& [sym, series] : candles_by_symbol) {
        // Round-trips for this symbol, annotated against its full-res candles.
        std::vector<TradeRow> rows;
        int n = 0;
        for (const auto& rt : round_trips) {
            if (rt.symbol == sym) { rows.push_back(annotate(rt, series, ++n)); }
        }

        // Display candles: stride down to keep the hand-drawn Canvas responsive.
        const std::size_t stride = stride_for(series.candles.size(), kCandleCap);
        QVariantList candles;
        for (std::size_t i = 0; i < series.candles.size(); i += stride) {
            const Candle& c = series.candles[i];
            QVariantMap cm;
            cm["o"] = c.o; cm["h"] = c.h; cm["l"] = c.l; cm["c"] = c.c; cm["v"] = c.v;
            cm["t"] = static_cast<qlonglong>(c.t);
            candles.append(cm);
        }

        // Trades: re-index entry/exit onto the strided candle array (bars/MFE/MAE
        // stay full-resolution).
        QVariantList trades;
        for (const auto& row : rows) {
            QVariantMap tm;
            tm["n"] = row.n;
            tm["side"] = row.is_long ? "LONG" : "SHORT";
            tm["entryIdx"] = static_cast<int>(row.entry_idx / static_cast<int>(stride));
            tm["exitIdx"] = static_cast<int>(row.exit_idx / static_cast<int>(stride));
            tm["entryTs"] = static_cast<qlonglong>(row.entry_ts);
            tm["exitTs"] = static_cast<qlonglong>(row.exit_ts);
            tm["entryPrice"] = row.entry_price;
            tm["exitPrice"] = row.exit_price;
            tm["qty"] = row.qty;
            tm["pnlNum"] = row.pnl;
            tm["retNum"] = row.ret_pct;
            tm["bars"] = row.bars;
            tm["mfe"] = row.mfe;
            tm["mae"] = row.mae;
            trades.append(tm);
        }

        // Sparkline: strided closes.
        const std::size_t spark_stride = stride_for(series.candles.size(), kSparkCap);
        QVariantList spark;
        for (std::size_t i = 0; i < series.candles.size(); i += spark_stride) {
            spark.append(series.candles[i].c);
        }

        QVariantMap drill;
        drill["candles"] = candles;
        drill["trades"] = trades;
        drill["spark"] = spark;
        per_symbol[qs(sym)] = drill;

        // Summary row.
        const auto it = by_symbol.find(sym);
        const SymbolStat stat = it != by_symbol.end() ? it->second : SymbolStat{};
        const double sym_ret = metrics.starting_cash != 0.0
            ? stat.pnl / metrics.starting_cash * 100.0 : 0.0;
        QVariantMap row;
        row["id"] = qs(sym);
        row["ret"] = qs(pct_signed(sym_ret));
        row["retPos"] = stat.pnl >= 0.0;
        row["pnl"] = qs(signed_usd(stat.pnl));
        row["trades"] = static_cast<int>(stat.closed);
        row["win"] = qs(stat.closed > 0
            ? std::to_string(std::lround(static_cast<double>(stat.winning)
                / static_cast<double>(stat.closed) * 100.0)) + "%"
            : std::string{ "0%" });
        symbols.append(row);
    }

    result["symbols"] = symbols;
    result["perSymbol"] = per_symbol;
    return result;
}

} // namespace stonks::app
