#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "stonks/core/types.h"

// Backtest analytics the GUI needs but the headless reporter (report.h) does
// not compute: round-trip reconstruction, Sharpe, profit factor, per-symbol
// breakdown, and per-trade candle-index annotation with MFE/MAE. Pure free
// functions over plain data (no Qt / Arrow / pybind), so they are directly
// unit-testable, in the same spirit as report.h.
namespace stonks::app {

// One OHLCV bar plus its epoch-ms timestamp (kept so trades can be mapped to
// candle indices and the chart can render real dates).
struct Candle
{
    double o{}, h{}, l{}, c{}, v{};
    std::int64_t t{};   // epoch ms
};

// One symbol's candles, chronological.
struct SymbolSeries
{
    std::vector<Candle> candles;
};

// A closed round-trip position reconstructed from the fill stream. `side` is the
// entry direction (Buy = long, Sell = short). `entry_price` is the blended
// average entry, `exit_price` the volume-weighted average exit, `qty` the peak
// absolute position reached, and `realized` the cycle's P&L.
struct RoundTrip
{
    core::Symbol symbol;
    core::OrderSide side{ core::OrderSide::Buy };
    std::int64_t entry_ts{};
    std::int64_t exit_ts{};
    core::Price entry_price{};
    core::Price exit_price{};
    core::Quantity qty{};
    core::Balance realized{};
};

// A RoundTrip annotated against a symbol's candle series: the QML trade-row
// shape — candle indices, bars held, return %, and MFE/MAE (in %).
struct TradeRow
{
    int n{};
    bool is_long{ true };
    std::int64_t entry_ts{};
    std::int64_t exit_ts{};
    int entry_idx{};
    int exit_idx{};
    double entry_price{};
    double exit_price{};
    double qty{};
    double pnl{};
    double ret_pct{};
    int bars{};
    double mfe{};
    double mae{};
};

struct SymbolStat
{
    core::Balance pnl{ 0.0 };
    std::size_t closed{ 0 };
    std::size_t winning{ 0 };
};

// Reconstruct closed round-trip positions per symbol from the fill stream. Uses
// the same signed-position / blended-average math as compute_metrics (report.h),
// so the closed-trade count and win/loss classification agree — but emits each
// full cycle (entry/exit ts + price, size, realized P&L) for the GUI.
inline std::vector<RoundTrip> reconstruct_round_trips(const std::vector<core::Trade>& fills)
{
    struct Pos
    {
        double qty{ 0.0 };          // signed: +long / -short
        double avg_price{ 0.0 };    // blended average entry
        double realized{ 0.0 };
        core::OrderSide side{ core::OrderSide::Buy };
        std::int64_t entry_ts{ 0 };
        double peak_qty{ 0.0 };
        double close_qty{ 0.0 };
        double close_notional{ 0.0 };
    };
    std::map<core::Symbol, Pos> positions;
    std::vector<RoundTrip> out;

    auto open_cycle = [](Pos& p, const core::Trade& t, double signed_qty) {
        p.qty = signed_qty;
        p.avg_price = t.price;
        p.realized = 0.0;
        p.side = t.side;
        p.entry_ts = t.timestamp.value.time_since_epoch().count();
        p.peak_qty = std::abs(signed_qty);
        p.close_qty = 0.0;
        p.close_notional = 0.0;
    };

    for (const auto& t : fills) {
        const double fill = (t.side == core::OrderSide::Buy ? t.quantity : -t.quantity);
        Pos& p = positions[t.symbol];

        if (p.qty == 0.0) {
            open_cycle(p, t, fill);
            p.realized -= t.fee;   // the entry's fee belongs to this cycle (net-of-cost, like report.h)
            continue;
        }
        if ((p.qty > 0.0) == (fill > 0.0)) {
            // Scale in: blend the average entry.
            const double prev = std::abs(p.qty);
            const double add = std::abs(fill);
            p.avg_price = (p.avg_price * prev + t.price * add) / (prev + add);
            p.qty += fill;
            p.peak_qty = std::max(p.peak_qty, std::abs(p.qty));
            p.realized -= t.fee;
            continue;
        }

        // Opposing fill: close against the open position.
        const double closing = std::min(std::abs(fill), std::abs(p.qty));
        p.realized += (p.qty > 0.0 ? (t.price - p.avg_price) : (p.avg_price - t.price)) * closing;
        p.realized -= t.fee;
        p.close_qty += closing;
        p.close_notional += closing * t.price;
        p.qty += (p.qty > 0.0 ? -closing : closing);

        if (p.qty == 0.0) {
            out.push_back(RoundTrip{
                t.symbol,
                p.side,
                p.entry_ts,
                t.timestamp.value.time_since_epoch().count(),
                p.avg_price,
                p.close_qty > 0.0 ? p.close_notional / p.close_qty : t.price,
                p.peak_qty,
                p.realized,
            });

            const double leftover = std::abs(fill) - closing;
            p = Pos{};
            if (leftover > 0.0) {   // fill flips past flat -> opens a fresh cycle
                open_cycle(p, t, fill > 0.0 ? leftover : -leftover);
            }
        }
    }
    return out;
}

// Sharpe ratio from per-bar equity-curve returns, scaled by `annualization`
// (e.g. bars-per-year). nullopt when there are <2 returns or zero variance.
inline std::optional<double> sharpe_ratio(const std::vector<core::EquityPoint>& curve,
                                          double annualization)
{
    std::vector<double> rets;
    rets.reserve(curve.size());
    for (std::size_t i = 1; i < curve.size(); ++i) {
        const double prev = curve[i - 1].equity;
        if (prev != 0.0) { rets.push_back((curve[i].equity - prev) / prev); }
    }
    if (rets.size() < 2) { return std::nullopt; }

    double mean = 0.0;
    for (const double r : rets) { mean += r; }
    mean /= static_cast<double>(rets.size());

    double var = 0.0;
    for (const double r : rets) { var += (r - mean) * (r - mean); }
    var /= static_cast<double>(rets.size() - 1);   // sample variance
    if (var <= 0.0) { return std::nullopt; }

    return (mean / std::sqrt(var)) * std::sqrt(annualization);
}

// Profit factor: gross win / gross loss over closed round-trips. +inf when there
// are wins but no losses; 0 when there are no wins.
inline double profit_factor(const std::vector<RoundTrip>& trips)
{
    double gross_win = 0.0, gross_loss = 0.0;
    for (const auto& t : trips) {
        if (t.realized > 0.0) { gross_win += t.realized; }
        else if (t.realized < 0.0) { gross_loss += -t.realized; }
    }
    if (gross_loss == 0.0) {
        return gross_win > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
    }
    return gross_win / gross_loss;
}

// Per-symbol P&L / closed-trade / winning-trade tally over round-trips.
inline std::map<core::Symbol, SymbolStat> per_symbol_breakdown(
    const std::vector<RoundTrip>& trips)
{
    std::map<core::Symbol, SymbolStat> out;
    for (const auto& t : trips) {
        SymbolStat& s = out[t.symbol];
        s.pnl += t.realized;
        ++s.closed;
        if (t.realized > 0.0) { ++s.winning; }
    }
    return out;
}

// Map a round-trip onto a symbol's candle series: candle indices, bars held,
// return %, and MFE/MAE (max favourable / adverse excursion, in %, sign-aware
// for long vs short).
inline TradeRow annotate(const RoundTrip& rt, const SymbolSeries& series, int n)
{
    const auto& cs = series.candles;
    auto index_of = [&](std::int64_t ts) -> int {
        const auto it = std::lower_bound(cs.begin(), cs.end(), ts,
            [](const Candle& c, std::int64_t v) { return c.t < v; });
        int idx = static_cast<int>(it - cs.begin());
        if (idx >= static_cast<int>(cs.size())) { idx = static_cast<int>(cs.size()) - 1; }
        if (idx < 0) { idx = 0; }
        return idx;
    };

    TradeRow row;
    row.n = n;
    row.is_long = (rt.side == core::OrderSide::Buy);
    row.entry_ts = rt.entry_ts;
    row.exit_ts = rt.exit_ts;
    row.entry_idx = cs.empty() ? 0 : index_of(rt.entry_ts);
    row.exit_idx = cs.empty() ? 0 : index_of(rt.exit_ts);
    row.bars = row.exit_idx - row.entry_idx;
    row.entry_price = rt.entry_price;
    row.exit_price = rt.exit_price;
    row.qty = rt.qty;
    row.pnl = rt.realized;
    row.ret_pct = (rt.entry_price != 0.0 && rt.qty != 0.0)
        ? rt.realized / (rt.entry_price * rt.qty) * 100.0 : 0.0;

    // MFE/MAE over the holding window [entry_idx, exit_idx].
    double mfe = 0.0, mae = 0.0;
    const double entry = rt.entry_price;
    if (entry != 0.0 && !cs.empty()) {
        bool first = true;
        for (int k = row.entry_idx; k <= row.exit_idx && k < static_cast<int>(cs.size()); ++k) {
            const Candle& c = cs[k];
            double up, down;   // favourable / adverse excursion, %
            if (row.is_long) {
                up = (c.h / entry - 1.0) * 100.0;
                down = (c.l / entry - 1.0) * 100.0;
            } else {
                up = (entry / c.l - 1.0) * 100.0;   // price falling is favourable for a short
                down = (entry / c.h - 1.0) * 100.0;
            }
            if (first) { mfe = up; mae = down; first = false; }
            else { mfe = std::max(mfe, up); mae = std::min(mae, down); }
        }
    }
    row.mfe = mfe;
    row.mae = mae;
    return row;
}

} // namespace stonks::app
