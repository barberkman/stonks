#pragma once

#include <stonks/core/log.h>
#include <stonks/core/types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>

// ════════════════════════════════════════════════════════════════════════════
//  Qullamaggie momentum setups — shared C++ framework
//
//  C++ port of app/python/qm_common.py. Implements Kristjan Qullamäki's setups
//  from app/pines/qullamaggie_momentum_swing.pine. The Pine indicator is
//  intraday-aware and uses resting STOP orders with intrabar fills; this engine
//  has neither, so every setup follows one execution model:
//
//    * Signals are read off the CLOSED bar (the last bar of each window). The
//      feed's window is no-lookahead by construction, so this never peeks ahead.
//    * Entries are MARKET orders, which the broker fills at the NEXT bar's open.
//      Stops, targets and the trailing-MA exit are SIMULATED: each bar we inspect
//      the just-closed bar and emit a market order when a level is crossed. That
//      costs ~1 bar of lag vs the Pine intrabar model, unavoidable with
//      Market/Limit-only orders.
//    * Stops/targets are anchored to the Pine reference price (breakout pivot, EP
//      close), accepting the next-open fill gap.
//
//  The engine offers no built-in indicators and no position query, so indicators
//  are hand-rolled below and each strategy tracks its own per-symbol position.
//  `QMBase` (a CRTP base) holds the shared tick loop, sizing and trade
//  management; each setup header subclasses it and implements only `signal()`.
// ════════════════════════════════════════════════════════════════════════════

namespace qm {

using stonks::core::Balance;
using stonks::core::MarketOrderParams;
using stonks::core::OrderSide;
using stonks::core::SeriesView;
using stonks::core::Symbol;
using stonks::core::TimeInForce;

// The dataset carries no tick size, so the "buffer beyond the pivot" is zero.
inline constexpr double MINTICK = 0.0;

enum class Dir { Long, Short };
enum class Mgmt { R, Para };          // R = stop/partial/BE/trail; Para = parabolic short
enum class TrailType { SMA, EMA };

// ─── Parameters (mirror the Pine inputs) ──────────────────────────────────────
struct Params
{
    // Universe & trend filters
    double min_price = 5.0;
    double min_avg_vol = 0.0;
    int adr_len = 20;
    double min_adr = 0.1;
    int mom_len = 24;
    double min_gain = 0.5;
    bool require_mas = true;
    // Setup 1 — momentum breakout
    int base_max_len = 40;
    int min_base_days = 3;
    double max_depth = 40.0;
    bool use_vol_dry = false;
    double vol_dry_ratio = 1.0;
    int buf_ticks = 0;
    bool use_bo_vol = true;
    double bo_vol_mult = 1.3;
    bool wait_close = true;           // daily close-confirm the break (vs intrabar high)
    // Setup 1b — opening range breakout
    int orb_bars = 1;
    // Setup 2 — episodic pivot
    double ep_min_gap = 0.5;
    double ep_vol_mult = 1.3;
    bool ep_strong_close = true;
    // Setup 3 — parabolic short
    int ps_lookback = 10;
    double ps_min_gain = 8.0;
    int ps_streak = 3;
    int ps_stop_lb = 3;
    int ps_max_hold = 5;
    // Initial stop
    double adr_stop_mult = 1.0;
    bool use_lod_stop = true;
    // Trade management
    double partial_rr = 2.0;
    int partial_days = 6;
    bool move_be = true;
    TrailType trail_type = TrailType::EMA;
    int trail_len = 20;
    // Sizing (no Pine equivalent — the indicator is signal-only)
    double risk_per_trade_pct = 0.5;
    double partial_fraction = 0.5;
};

// ─── Indicators (operate on a column span, return the latest value) ───────────
// Each returns nullopt when there is not enough history, so callers can compose
// them with universe gates without separate length checks.
inline std::optional<double> sma(std::span<const double> a, int n)
{
    if (n <= 0 || std::ssize(a) < n) return std::nullopt;
    const auto last = a.last(n);
    double s = 0.0;
    for (const double x : last) s += x;
    return s / n;
}

// SMA-seeded EMA over the supplied window (seed = SMA of the first n bars).
inline std::optional<double> ema(std::span<const double> a, int n)
{
    if (n <= 0 || std::ssize(a) < n) return std::nullopt;
    const double alpha = 2.0 / (n + 1);
    double e = 0.0;
    for (const double x : a.first(n)) e += x;
    e /= n;
    for (std::size_t i = static_cast<std::size_t>(n); i < a.size(); ++i)
        e = alpha * a[i] + (1.0 - alpha) * e;
    return e;
}

inline std::optional<double> highest(std::span<const double> a, int n)
{
    if (n <= 0 || std::ssize(a) < n) return std::nullopt;
    const auto last = a.last(n);
    return *std::max_element(last.begin(), last.end());
}

inline std::optional<double> lowest(std::span<const double> a, int n)
{
    if (n <= 0 || std::ssize(a) < n) return std::nullopt;
    const auto last = a.last(n);
    return *std::min_element(last.begin(), last.end());
}

// Average per-bar range %, matching Pine's adrPct = sma(100 * (high/low - 1)).
inline std::optional<double> adr_pct(std::span<const double> high, std::span<const double> low, int n)
{
    if (n <= 0 || std::ssize(high) < n) return std::nullopt;
    const auto h = high.last(n);
    const auto l = low.last(n);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += 100.0 * (h[i] / l[i] - 1.0);
    return s / n;
}

// Momentum: 100 * (close[-1] / close[-1-n] - 1).
inline std::optional<double> gain_pct(std::span<const double> close, int n)
{
    if (std::ssize(close) < n + 1) return std::nullopt;
    const double base = close[close.size() - 1 - static_cast<std::size_t>(n)];
    if (base == 0.0) return std::nullopt;
    return 100.0 * (close.back() / base - 1.0);
}

// ─── Universe / trend gates (mirror Pine liqOK / adrOK / trendUp|DnOK) ────────
inline bool liq_ok(const SeriesView& b, const Params& p)
{
    const auto av20 = sma(b.volume, 20);
    return b.close.back() >= p.min_price && av20 && *av20 >= p.min_avg_vol;
}

inline bool adr_ok(const SeriesView& b, const Params& p)
{
    const auto a = adr_pct(b.high, b.low, p.adr_len);
    return a && *a >= p.min_adr;
}

inline bool ma_ok_up(const SeriesView& b, const Params& p)
{
    if (!p.require_mas) return true;
    const auto s10 = sma(b.close, 10);
    const auto s20 = sma(b.close, 20);
    return s10 && s20 && b.close.back() > *s20 && *s10 > *s20;
}

inline bool ma_ok_dn(const SeriesView& b, const Params& p)
{
    if (!p.require_mas) return true;
    const auto s10 = sma(b.close, 10);
    const auto s20 = sma(b.close, 20);
    return s10 && s20 && b.close.back() < *s20 && *s10 < *s20;
}

inline bool universe_long(const SeriesView& b, const Params& p)
{
    const auto g = gain_pct(b.close, p.mom_len);
    return liq_ok(b, p) && adr_ok(b, p) && g && *g >= p.min_gain && ma_ok_up(b, p);
}

inline bool universe_short(const SeriesView& b, const Params& p)
{
    const auto g = gain_pct(b.close, p.mom_len);
    return liq_ok(b, p) && adr_ok(b, p) && g && *g <= -p.min_gain && ma_ok_dn(b, p);
}

// ─── Entry plan: what a setup's signal() returns when it fires ───────────────
struct EntryPlan
{
    double entry_ref;                          // price the stop/target/size anchor to
    std::optional<double> explicit_stop{};     // parabolic short sets this; else ADR/LoD stop
    bool use_target = true;                     // parabolic short has no R target
};

// ─── Breakout volume helpers (shared by long and short breakouts) ─────────────
inline bool vol_dry_ok(std::span<const double> vol, const Params& p)
{
    const auto v5 = sma(vol, 5);
    const auto v50 = sma(vol, 50);
    return v5 && v50 && *v5 < p.vol_dry_ratio * *v50;
}

// Break-bar volume >= boVolMult * avgVol50[1] (the 50-bar average as of the prior bar).
inline bool break_vol_ok(std::span<const double> vol, const Params& p)
{
    if (!p.use_bo_vol) return true;
    const auto prev_avg50 = sma(vol.first(vol.size() - 1), 50);
    return prev_avg50 && vol.back() >= p.bo_vol_mult * *prev_avg50;
}

// ─── Breakout base detection (the pivot excludes the current/break bar) ───────
// A valid base is a prior consolidation: the highest high formed at least
// min_base_days ago, with a shallow pullback since. We return its pivot level if
// the current bar breaks above it on expanding volume, else nullopt.
inline std::optional<EntryPlan> flag_base_long(const SeriesView& bars, const Params& p)
{
    const auto& high = bars.high;
    const auto& low = bars.low;
    const int n = p.base_max_len;
    const int N = static_cast<int>(bars.size());
    if (N < n + 1) return std::nullopt;

    // The base window is the n bars ending one before the current bar; the
    // current bar (index N-1) is the candidate break and must not raise the pivot.
    const int base_end = N - 1;            // exclusive: current bar lives here
    const int w_start = base_end - n;
    double mx = high[w_start];
    int last_pos = 0;
    for (int i = 0; i < n; ++i) {
        if (high[w_start + i] >= mx) { mx = high[w_start + i]; last_pos = i; }
    }
    const int since_pk = (n - 1) - last_pos;               // bars from peak to last base bar
    const int pull_n = std::max(since_pk, 1);
    double pull_low = low[base_end - pull_n];
    for (int i = base_end - pull_n; i < base_end; ++i) pull_low = std::min(pull_low, low[i]);
    const double retrace = mx > 0.0 ? 100.0 * (mx - pull_low) / mx : 1e9;

    const bool vol_dry = !p.use_vol_dry || vol_dry_ok(bars.volume, p);
    if (!(since_pk >= p.min_base_days && retrace <= p.max_depth && vol_dry)) return std::nullopt;

    const double pivot = mx + p.buf_ticks * MINTICK;
    const bool broke = p.wait_close ? bars.close.back() >= pivot : high.back() >= pivot;
    if (!broke || !break_vol_ok(bars.volume, p)) return std::nullopt;
    return EntryPlan{ .entry_ref = pivot };
}

// Mirror of flag_base_long: pivot = the low of a prior base, broken below.
inline std::optional<EntryPlan> flag_base_short(const SeriesView& bars, const Params& p)
{
    const auto& high = bars.high;
    const auto& low = bars.low;
    const int n = p.base_max_len;
    const int N = static_cast<int>(bars.size());
    if (N < n + 1) return std::nullopt;

    const int base_end = N - 1;
    const int w_start = base_end - n;
    double mn = low[w_start];
    int last_pos = 0;
    for (int i = 0; i < n; ++i) {
        if (low[w_start + i] <= mn) { mn = low[w_start + i]; last_pos = i; }
    }
    const int since_tr = (n - 1) - last_pos;               // bars from trough to last base bar
    const int bounce_n = std::max(since_tr, 1);
    double bounce_high = high[base_end - bounce_n];
    for (int i = base_end - bounce_n; i < base_end; ++i) bounce_high = std::max(bounce_high, high[i]);
    const double retrace = mn > 0.0 ? 100.0 * (bounce_high - mn) / mn : 1e9;

    const bool vol_dry = !p.use_vol_dry || vol_dry_ok(bars.volume, p);
    if (!(since_tr >= p.min_base_days && retrace <= p.max_depth && vol_dry)) return std::nullopt;

    const double pivot = mn - p.buf_ticks * MINTICK;
    const bool broke = p.wait_close ? bars.close.back() <= pivot : low.back() <= pivot;
    if (!broke || !break_vol_ok(bars.volume, p)) return std::nullopt;
    return EntryPlan{ .entry_ref = pivot };
}

// ─── Sizing ──────────────────────────────────────────────────────────────────
// Risk a fixed % of equity to the stop, then cap notional at available cash so a
// market buy can't exceed cash (an unfillable buy would linger and fill late).
inline double size_by_risk(Balance equity, Balance cash, double entry, double stop, const Params& p)
{
    const double rps = std::abs(entry - stop);
    if (rps <= 0.0 || entry <= 0.0) return 0.0;
    const double qty = (equity * p.risk_per_trade_pct / 100.0) / rps;
    const double cap = (cash * 0.99) / entry;
    return std::max(0.0, std::min(qty, cap));
}

// ─── Per-symbol open position ─────────────────────────────────────────────────
struct Position
{
    double entry;
    double stop;
    double target;            // NaN for the parabolic short (no R target)
    double qty;               // remaining quantity held
    int bars_held = 0;
    bool partial_done = false;
    bool filled = false;      // false until the bar after entry (the fill bar)
};

// ─── Base strategy: shared tick loop, entry, and management (CRTP) ────────────
// Derived supplies three compile-time traits and a signal():
//   static constexpr Dir  DIR;            // Long / Short
//   static constexpr Mgmt MGMT;           // R / Para
//   static constexpr bool SAME_BAR_BAIL;  // bail on the fill bar if it closes past the stop
//   std::optional<EntryPlan> signal(const SeriesView& bars);
template <class Derived>
struct QMBase
{
    Params p{};
    std::unordered_map<Symbol, Position> positions;     // one entry per held symbol

    // How many bars of history to request: enough for the longest lookback
    // (avgVol50[1] needs 51; base and momentum windows are usually shorter).
    int lookback() const { return std::max({ p.base_max_len, 51, p.mom_len }) + 5; }

    void on_tick(auto& ctx)
    {
        // One tick per timestamp: the window holds every symbol that printed,
        // each with its own last-n bars. Manage held symbols; otherwise look for
        // a fresh entry. Cross-symbol independence falls out of the per-symbol map.
        for (const auto& series : ctx.history(lookback()).series) {
            const auto& bars = series.bars;
            if (bars.size() < 2) continue;
            const Symbol sym{ series.symbol };
            if (auto it = positions.find(sym); it != positions.end()) {
                manage(ctx, sym, bars, it->second);
            } else if (auto plan = static_cast<Derived&>(*this).signal(bars)) {
                enter(ctx, sym, bars, *plan);
            }
        }
    }

    // Compute the stop/target from the entry reference, size by risk, and place
    // the opening market order (fills next bar). Direction is a compile-time trait.
    void enter(auto& ctx, const Symbol& sym, const SeriesView& bars, const EntryPlan& plan)
    {
        const auto adr = adr_pct(bars.high, bars.low, p.adr_len);
        if (!adr) return;
        const double entry = plan.entry_ref;
        double stop = 0.0;
        double target = 0.0;
        OrderSide side_open = OrderSide::Buy;

        if constexpr (Derived::DIR == Dir::Long) {
            if (plan.explicit_stop) {
                stop = *plan.explicit_stop;
            } else {
                const double adr_stop = entry * (1.0 - p.adr_stop_mult * *adr / 100.0);
                stop = p.use_lod_stop ? std::max(adr_stop, bars.low.back()) : adr_stop;
            }
            stop = std::min(stop, entry * 0.999);          // keep the stop strictly below entry
            const double risk = entry - stop;
            if (risk <= 0.0) return;
            target = plan.use_target ? entry + p.partial_rr * risk
                                     : std::numeric_limits<double>::quiet_NaN();
            side_open = OrderSide::Buy;
        } else {
            if (plan.explicit_stop) {
                stop = *plan.explicit_stop;
            } else {
                const double adr_stop = entry * (1.0 + p.adr_stop_mult * *adr / 100.0);
                stop = p.use_lod_stop ? std::min(adr_stop, bars.high.back()) : adr_stop;
            }
            stop = std::max(stop, entry * 1.001);          // keep the stop strictly above entry
            const double risk = stop - entry;
            if (risk <= 0.0) return;
            target = plan.use_target ? entry - p.partial_rr * risk
                                     : std::numeric_limits<double>::quiet_NaN();
            side_open = OrderSide::Sell;                    // sell-to-open the short
        }

        const double qty = size_by_risk(ctx.equity(), ctx.cash(), entry, stop, p);
        if (qty <= 0.0) return;
        const auto order = ctx.make_market_order(MarketOrderParams{
            .symbol = sym, .side = side_open, .quantity = qty, .time_in_force = TimeInForce::GTC });
        if (ctx.place_order(order)) {
            positions[sym] = Position{ .entry = entry, .stop = stop, .target = target, .qty = qty };
            STONKS_LOG("qm", "ENTER sym={} side={} entry={:.4f} stop={:.4f} target={:.4f} qty={:.6f}",
                sym, side_open == OrderSide::Buy ? "Buy" : "Sell", entry, stop, target, qty);
        }
    }

    // Run stop / partial / breakeven / trailing-MA logic against the just-closed
    // bar. Mgmt and direction are compile-time traits, so the dead branches are
    // pruned per strategy.
    void manage(auto& ctx, const Symbol& sym, const SeriesView& bars, Position& pos)
    {
        const double c = bars.close.back();
        const double h = bars.high.back();
        const double l = bars.low.back();

        if (!pos.filled) {
            // The order placed last tick fills at this bar's open; exit management
            // begins next bar (mirrors Pine's bar_index > entryBar), with an
            // optional same-bar bail if this bar already closes past the stop.
            pos.filled = true;
            pos.bars_held = 0;
            if constexpr (Derived::MGMT == Mgmt::R && Derived::SAME_BAR_BAIL) {
                if constexpr (Derived::DIR == Dir::Long) {
                    if (c < pos.stop) close_position(ctx, sym, pos, OrderSide::Sell);
                } else {
                    if (c > pos.stop) close_position(ctx, sym, pos, OrderSide::Buy);
                }
            }
            return;
        }

        ++pos.bars_held;

        if constexpr (Derived::MGMT == Mgmt::Para) {
            // Parabolic short: cover on a stop above the recent highs, on any green
            // close (momentum stalled), or after the max holding period. No partial.
            const double prev_close = bars.close[bars.size() - 2];
            if (h >= pos.stop || c > prev_close || pos.bars_held >= p.ps_max_hold)
                close_position(ctx, sym, pos, OrderSide::Buy);
            return;
        } else {
            const auto trail = (p.trail_type == TrailType::EMA) ? ema(bars.close, p.trail_len)
                                                                : sma(bars.close, p.trail_len);
            if constexpr (Derived::DIR == Dir::Long) {
                if (l <= pos.stop) { close_position(ctx, sym, pos, OrderSide::Sell); return; }
                if (!pos.partial_done && (h >= pos.target || pos.bars_held >= p.partial_days)) {
                    partial(ctx, sym, pos, OrderSide::Sell);
                    if (p.move_be) pos.stop = std::max(pos.stop, pos.entry);
                }
                if (pos.partial_done && trail && c < *trail) {
                    close_position(ctx, sym, pos, OrderSide::Sell);
                    return;
                }
            } else {
                if (h >= pos.stop) { close_position(ctx, sym, pos, OrderSide::Buy); return; }
                if (!pos.partial_done && (l <= pos.target || pos.bars_held >= p.partial_days)) {
                    partial(ctx, sym, pos, OrderSide::Buy);
                    if (p.move_be) pos.stop = std::min(pos.stop, pos.entry);
                }
                if (pos.partial_done && trail && c > *trail) {
                    close_position(ctx, sym, pos, OrderSide::Buy);
                    return;
                }
            }
        }
    }

    // Scale out partial_fraction of the position and arm the trailing stop.
    void partial(auto& ctx, const Symbol& sym, Position& pos, OrderSide side)
    {
        const double part = pos.qty * p.partial_fraction;
        if (part <= 0.0) return;
        const auto order = ctx.make_market_order(MarketOrderParams{
            .symbol = sym, .side = side, .quantity = part, .time_in_force = TimeInForce::GTC });
        if (ctx.place_order(order)) {
            pos.qty -= part;
            pos.partial_done = true;
            STONKS_LOG("qm", "PARTIAL sym={} qty={:.6f} remaining={:.6f}", sym, part, pos.qty);
        }
    }

    // Liquidate the remainder and drop the position. `pos` dangles after the
    // erase, so callers must return immediately.
    void close_position(auto& ctx, const Symbol& sym, Position& pos, OrderSide side)
    {
        if (pos.qty > 0.0) {
            const auto order = ctx.make_market_order(MarketOrderParams{
                .symbol = sym, .side = side, .quantity = pos.qty, .time_in_force = TimeInForce::GTC });
            ctx.place_order(order);
            STONKS_LOG("qm", "CLOSE sym={} side={} qty={:.6f}",
                sym, side == OrderSide::Buy ? "Buy" : "Sell", pos.qty);
        }
        positions.erase(sym);
    }
};

} // namespace qm
