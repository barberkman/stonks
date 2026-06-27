#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <format>
#include <iostream>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <stonks/core/types.h>

// Qullamaggie momentum scanner — all five setups in one strategy, print-only.
//
// Runs every setup against each symbol's closed bar and, whenever one fires, prints
// a line with the timestamp, setup name, and the trade levels (entry / stop / sell).
// It does NOT place orders — it only reports signals via std::cout.
//
//   breakout        — long: break above a tight base's pivot high
//   short_breakout  — short: breakdown below a base's pivot low (downtrend mirror)
//   episodic_pivot  — long: gap up on volume with a strong close
//   parabolic_short — short: first red bar after a parabolic run-up
//   orb             — long: opening-range breakout (intraday sessions only)
//
// Trade levels follow one R-based model: enter at the setup's reference price, stop
// an ADR away (breakout/short/orb/episodic_pivot) or at the recent-high pivot
// (parabolic_short), and take profit ("sell") at target_rr times the risk.
//
// Self-contained: the only dependency is the core bar types; the TA math lives in
// the private section below.
struct QMSignalsStrategy
{
    // ─── Configurable parameters ─────────────────────────────────────────────
    // Universe & trend filters
    double min_price = 5.0;
    double min_avg_vol = 0.0;
    int adr_len = 20;
    double min_adr = 0.1;
    int mom_len = 24;
    double min_gain = 0.5;
    bool require_mas = true;
    // Breakout / short-breakout base
    int base_max_len = 40;
    int min_base_days = 3;
    double max_depth = 40.0;
    bool use_vol_dry = false;
    double vol_dry_ratio = 1.0;
    int buf_ticks = 0;
    bool use_bo_vol = true;
    double bo_vol_mult = 1.3;
    bool wait_close = true;       // close-confirm the break (vs intrabar high/low)
    // Opening range breakout
    int orb_bars = 1;
    // Episodic pivot
    double ep_min_gap = 0.5;
    double ep_vol_mult = 1.3;
    bool ep_strong_close = true;
    // Parabolic short
    int ps_lookback = 10;
    double ps_min_gain = 8.0;
    int ps_streak = 3;
    int ps_stop_lb = 3;
    // Stop & take-profit
    double adr_stop_mult = 1.0;   // stop distance from entry, in ADRs
    bool use_lod_stop = true;     // tighten the stop to the signal bar's extreme if closer
    double target_rr = 2.0;       // take-profit ("sell") target, in R multiples

    // A fired setup on a closed bar, with the trade levels to print.
    struct Signal
    {
        std::string_view setup;
        std::int64_t timestamp;   // ms since epoch (the closed bar)
        double entry;
        double stop;
        double sell;
    };

    int lookback() const
    {
        return std::max({ base_max_len, 51, mom_len, adr_len, ps_lookback, ps_streak + 1, ps_stop_lb }) + 5;
    }

    void on_tick(auto& ctx)
    {
        static constexpr double position_fraction{ 0.02 };

        for (const auto& series : ctx.history(lookback()).series) {
            const stonks::core::Symbol sym{ series.symbol };
            auto sigs = scan(series.bars);

            /*
            for (const auto& s : sigs) {
                std::cout << std::format("[qm] {:%F %T} {} {} entry={:.4f} stop={:.4f} sell={:.4f}\n",
                as_time(s.timestamp), s.setup, sym, s.entry, s.stop, s.sell);
            }
            */

            if (!sigs.empty()) {
                const auto& s = sigs.front();
                const bool is_long = s.stop < s.entry;                 // long setups stop below entry
                const double qty = ctx.equity() * position_fraction / s.entry;
                if (qty > 0.0) {
                    // Enter order
                    auto order_id = ctx.place_order(stonks::core::LimitOrderParams{
                        .symbol = sym,
                        .side = is_long ? stonks::core::OrderSide::Buy
                                        : stonks::core::OrderSide::Sell,
                        .quantity = qty,
                        .price = s.entry
                    });

                    // Stop loss order
                    ctx.place_order(stonks::core::LimitOrderParams{
                        .symbol = sym,
                        .side = is_long ? stonks::core::OrderSide::Sell
                                        : stonks::core::OrderSide::Buy,
                        .quantity = qty,
                        .price = s.stop,
                    }, order_id);

                    // Take profit order
                    ctx.place_order(stonks::core::LimitOrderParams{
                        .symbol = sym,
                        .side = is_long ? stonks::core::OrderSide::Sell
                                        : stonks::core::OrderSide::Buy,
                        .quantity = qty,
                        .price = s.sell
                    }, order_id);
                }
            }

            m_last[sym] = std::move(sigs);
        }
    }

    // The setups that fired on this symbol's last processed bar (for tests).
    // (Named `last_signals`, not `signals`, to dodge Qt's `signals` keyword macro.)
    std::vector<Signal> last_signals(const stonks::core::Symbol& sym) const
    {
        const auto it = m_last.find(sym);
        return it == m_last.end() ? std::vector<Signal>{} : it->second;
    }

    // Run every setup against the closed bar; collect the ones that fired.
    std::vector<Signal> scan(const stonks::core::SeriesView& b) const
    {
        std::vector<Signal> out;
        if (auto s = breakout(b)) out.push_back(*s);
        if (auto s = short_breakout(b)) out.push_back(*s);
        if (auto s = episodic_pivot(b)) out.push_back(*s);
        if (auto s = parabolic_short(b)) out.push_back(*s);
        if (auto s = orb(b)) out.push_back(*s);
        return out;
    }

private:
    // The dataset carries no tick size, so the "buffer beyond the pivot" is zero.
    static constexpr double MINTICK = 0.0;
    static constexpr std::int64_t MS_PER_DAY = 86'400'000;

    // ─── Setup 1 — momentum breakout (long) ──────────────────────────────────
    std::optional<Signal> breakout(const stonks::core::SeriesView& b) const
    {
        const int N = static_cast<int>(b.size());
        if (N < base_max_len + 1) return std::nullopt;

        const auto adr = adr_pct(b.high, b.low, adr_len);
        const auto g = gain_pct(b.close, mom_len);
        const bool universe = liq_ok(b) && adr && *adr >= min_adr
                           && g && *g >= min_gain && ma_ok_up(b);
        if (!universe) return std::nullopt;

        const int base_end = N - 1;
        const int w_start = base_end - base_max_len;
        double mx = b.high[w_start];
        int last_pos = 0;
        for (int i = 0; i < base_max_len; ++i) {
            if (b.high[w_start + i] >= mx) { mx = b.high[w_start + i]; last_pos = i; }
        }
        const int since_pk = (base_max_len - 1) - last_pos;
        const int pull_n = std::max(since_pk, 1);
        double pull_low = b.low[base_end - pull_n];
        for (int i = base_end - pull_n; i < base_end; ++i) pull_low = std::min(pull_low, b.low[i]);
        const double retrace = mx > 0.0 ? 100.0 * (mx - pull_low) / mx : 1e9;

        const bool vd = !use_vol_dry || vol_dry_ok(b.volume);
        if (!(since_pk >= min_base_days && retrace <= max_depth && vd)) return std::nullopt;

        const double entry = mx + buf_ticks * MINTICK;
        const bool broke = wait_close ? b.close.back() >= entry : b.high.back() >= entry;
        if (!broke || !break_vol_ok(b.volume) || !adr) return std::nullopt;

        const auto [stop, sell] = long_levels(entry, *adr, b.low.back());
        return Signal{ "breakout", b.timestamp.back(), entry, stop, sell };
    }

    // ─── Setup 1c — short breakout / breakdown (short, mirror) ────────────────
    std::optional<Signal> short_breakout(const stonks::core::SeriesView& b) const
    {
        const int N = static_cast<int>(b.size());
        if (N < base_max_len + 1) return std::nullopt;

        const auto adr = adr_pct(b.high, b.low, adr_len);
        const auto g = gain_pct(b.close, mom_len);
        const bool universe = liq_ok(b) && adr && *adr >= min_adr
                           && g && *g <= -min_gain && ma_ok_dn(b);
        if (!universe) return std::nullopt;

        const int base_end = N - 1;
        const int w_start = base_end - base_max_len;
        double mn = b.low[w_start];
        int last_pos = 0;
        for (int i = 0; i < base_max_len; ++i) {
            if (b.low[w_start + i] <= mn) { mn = b.low[w_start + i]; last_pos = i; }
        }
        const int since_tr = (base_max_len - 1) - last_pos;
        const int bounce_n = std::max(since_tr, 1);
        double bounce_high = b.high[base_end - bounce_n];
        for (int i = base_end - bounce_n; i < base_end; ++i) bounce_high = std::max(bounce_high, b.high[i]);
        const double retrace_up = mn > 0.0 ? 100.0 * (bounce_high - mn) / mn : 1e9;

        const bool vd = !use_vol_dry || vol_dry_ok(b.volume);
        if (!(since_tr >= min_base_days && retrace_up <= max_depth && vd)) return std::nullopt;

        const double entry = mn - buf_ticks * MINTICK;
        const bool broke = wait_close ? b.close.back() <= entry : b.low.back() <= entry;
        if (!broke || !break_vol_ok(b.volume) || !adr) return std::nullopt;

        const auto [stop, sell] = short_levels(entry, *adr, b.high.back());
        return Signal{ "short_breakout", b.timestamp.back(), entry, stop, sell };
    }

    // ─── Setup 2 — episodic pivot (gap bar, long) ─────────────────────────────
    std::optional<Signal> episodic_pivot(const stonks::core::SeriesView& b) const
    {
        if (b.size() < 2) return std::nullopt;
        if (!liq_ok(b)) return std::nullopt;

        const double o = b.open.back();
        const double h = b.high.back();
        const double l = b.low.back();
        const double c = b.close.back();
        const double prev_close = b.close[b.size() - 2];
        if (prev_close <= 0.0) return std::nullopt;

        if (100.0 * (o / prev_close - 1.0) < ep_min_gap) return std::nullopt;

        const auto prev_avg50 = sma(b.volume.first(b.volume.size() - 1), 50);
        if (!prev_avg50 || b.volume.back() < ep_vol_mult * *prev_avg50) return std::nullopt;

        if (ep_strong_close && !(c > o && c >= (h + l) / 2.0)) return std::nullopt;

        const auto adr = adr_pct(b.high, b.low, adr_len);
        const double risk_ps = c - l;
        if (!adr || !(risk_ps > 0.0 && risk_ps <= adr_stop_mult * *adr / 100.0 * c)) return std::nullopt;

        const auto [stop, sell] = long_levels(c, *adr, l);
        return Signal{ "episodic_pivot", b.timestamp.back(), c, stop, sell };
    }

    // ─── Setup 3 — parabolic short (first red bar) ────────────────────────────
    std::optional<Signal> parabolic_short(const stonks::core::SeriesView& b) const
    {
        const auto& close = b.close;
        const int need = std::max({ ps_lookback, ps_streak + 1, ps_stop_lb }) + 1;
        if (static_cast<int>(close.size()) < need) return std::nullopt;
        if (!liq_ok(b)) return std::nullopt;

        const auto hi = highest(b.high, ps_lookback);
        const auto lo = lowest(b.low, ps_lookback);
        if (!hi || !lo || *lo <= 0.0) return std::nullopt;
        if (100.0 * (*hi / *lo - 1.0) < ps_min_gain) return std::nullopt;

        int up_streak = 0;
        for (std::size_t i = close.size() - 2; i >= 1 && close[i] > close[i - 1]; --i) ++up_streak;
        if (!(close.back() < close[close.size() - 2] && up_streak >= ps_streak)) return std::nullopt;

        const auto ps_stop = highest(b.high, ps_stop_lb);
        if (!ps_stop) return std::nullopt;

        const double entry = close.back();
        const double stop = *ps_stop + buf_ticks * MINTICK;
        const double sell = entry - target_rr * (stop - entry);
        return Signal{ "parabolic_short", b.timestamp.back(), entry, stop, sell };
    }

    // ─── Setup 1b — opening range breakout (intraday, long) ───────────────────
    std::optional<Signal> orb(const stonks::core::SeriesView& b) const
    {
        if (b.size() < 1) return std::nullopt;

        const auto adr = adr_pct(b.high, b.low, adr_len);
        const auto g = gain_pct(b.close, mom_len);
        const bool universe = liq_ok(b) && adr && *adr >= min_adr
                           && g && *g >= min_gain && ma_ok_up(b);
        if (!universe) return std::nullopt;

        const auto& ts = b.timestamp;
        const std::int64_t cur_day = ts.back() / MS_PER_DAY;
        std::size_t start = ts.size() - 1;
        while (start > 0 && ts[start - 1] / MS_PER_DAY == cur_day) --start;
        const int bars_into_session = static_cast<int>(ts.size() - 1 - start);
        if (bars_into_session < orb_bars) return std::nullopt;

        double or_high = b.high[start];
        for (int k = 0; k < orb_bars; ++k) or_high = std::max(or_high, b.high[start + k]);

        const double entry = or_high + buf_ticks * MINTICK;
        const bool broke = wait_close ? b.close.back() >= entry : b.high.back() >= entry;
        if (!broke || !adr) return std::nullopt;

        const auto [stop, sell] = long_levels(entry, *adr, b.low.back());
        return Signal{ "orb", ts.back(), entry, stop, sell };
    }

    // ─── Trade levels ─────────────────────────────────────────────────────────
    std::pair<double, double> long_levels(double entry, double adr, double bar_low) const
    {
        const double adr_stop = entry * (1.0 - adr_stop_mult * adr / 100.0);
        double stop = use_lod_stop ? std::max(adr_stop, bar_low) : adr_stop;
        stop = std::min(stop, entry * 0.999);
        return { stop, entry + target_rr * (entry - stop) };
    }

    std::pair<double, double> short_levels(double entry, double adr, double bar_high) const
    {
        const double adr_stop = entry * (1.0 + adr_stop_mult * adr / 100.0);
        double stop = use_lod_stop ? std::min(adr_stop, bar_high) : adr_stop;
        stop = std::max(stop, entry * 1.001);
        return { stop, entry - target_rr * (stop - entry) };
    }

    // ─── Stateless TA helpers ─────────────────────────────────────────────────
    static std::optional<double> sma(std::span<const double> a, int n)
    {
        if (n <= 0 || std::ssize(a) < n) return std::nullopt;
        double s = 0.0;
        for (const double x : a.last(n)) s += x;
        return s / n;
    }

    static std::optional<double> adr_pct(std::span<const double> high, std::span<const double> low, int n)
    {
        if (n <= 0 || std::ssize(high) < n) return std::nullopt;
        const auto h = high.last(n);
        const auto l = low.last(n);
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += 100.0 * (h[i] / l[i] - 1.0);
        return s / n;
    }

    static std::optional<double> gain_pct(std::span<const double> close, int n)
    {
        if (std::ssize(close) < n + 1) return std::nullopt;
        const double base = close[close.size() - 1 - static_cast<std::size_t>(n)];
        if (base == 0.0) return std::nullopt;
        return 100.0 * (close.back() / base - 1.0);
    }

    static std::optional<double> highest(std::span<const double> a, int n)
    {
        if (n <= 0 || std::ssize(a) < n) return std::nullopt;
        const auto last = a.last(n);
        return *std::max_element(last.begin(), last.end());
    }

    static std::optional<double> lowest(std::span<const double> a, int n)
    {
        if (n <= 0 || std::ssize(a) < n) return std::nullopt;
        const auto last = a.last(n);
        return *std::min_element(last.begin(), last.end());
    }

    // ─── Gate predicates ──────────────────────────────────────────────────────
    bool liq_ok(const stonks::core::SeriesView& b) const
    {
        const auto av20 = sma(b.volume, 20);
        return b.close.back() >= min_price && av20 && *av20 >= min_avg_vol;
    }

    bool ma_ok_up(const stonks::core::SeriesView& b) const
    {
        if (!require_mas) return true;
        const auto s10 = sma(b.close, 10);
        const auto s20 = sma(b.close, 20);
        return s10 && s20 && b.close.back() > *s20 && *s10 > *s20;
    }

    bool ma_ok_dn(const stonks::core::SeriesView& b) const
    {
        if (!require_mas) return true;
        const auto s10 = sma(b.close, 10);
        const auto s20 = sma(b.close, 20);
        return s10 && s20 && b.close.back() < *s20 && *s10 < *s20;
    }

    bool vol_dry_ok(std::span<const double> vol) const
    {
        const auto v5 = sma(vol, 5);
        const auto v50 = sma(vol, 50);
        return v5 && v50 && *v5 < vol_dry_ratio * *v50;
    }

    bool break_vol_ok(std::span<const double> vol) const
    {
        if (!use_bo_vol) return true;
        if (vol.size() < 51) return false;
        const auto prev_avg50 = sma(vol.first(vol.size() - 1), 50);
        return prev_avg50 && vol.back() >= bo_vol_mult * *prev_avg50;
    }

    static std::chrono::sys_time<std::chrono::milliseconds> as_time(std::int64_t ms)
    {
        return std::chrono::sys_time<std::chrono::milliseconds>{ std::chrono::milliseconds{ ms } };
    }

    std::unordered_map<stonks::core::Symbol, std::vector<Signal>> m_last;
};
