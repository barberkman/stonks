#pragma once

#include <stonks/core/types.h>

#include <algorithm>
#include <deque>
#include <unordered_map>

// Long-only trend-pullback strategy, applied independently per symbol:
//   1. Trend filter — only trade while close is above the 200-bar SMA.
//   2. Entry — buy when the latest close is at a 7-bar low.
//   3. Exit  — sell the position when the latest close is at a 7-bar high.
// The trend filter gates both entries and exits: while below the SMA the
// strategy stays passive (this matches the literal reading of the spec).
struct SMA200PullbackStrategy
{
    static constexpr int SMA_PERIOD = 200;
    static constexpr int BREAKOUT_PERIOD = 7;

    struct SymbolState
    {
        std::deque<double> closes;
        double sum{};
        double held_quantity{};
    };

    std::unordered_map<stonks::core::Symbol, SymbolState> states;

    void on_tick(auto& context)
    {
        // klines() interleaves bars across symbols, so route the latest bar
        // to its own symbol's rolling window — same pattern as EMA50Strategy.
        const auto bars = context.klines(1);
        if (bars.empty()) return;
        const auto& bar = bars.back();

        auto& state = states[bar.symbol];

        // Rolling 200-bar window with an O(1) running sum.
        state.closes.push_back(bar.close);
        state.sum += bar.close;
        if (static_cast<int>(state.closes.size()) > SMA_PERIOD)
        {
            state.sum -= state.closes.front();
            state.closes.pop_front();
        }

        // Need a full 200-bar window before the trend filter is meaningful.
        if (static_cast<int>(state.closes.size()) < SMA_PERIOD) return;

        const auto sma = state.sum / SMA_PERIOD;
        if (bar.close <= sma) return;

        // 7-bar window, inclusive of the current bar — `bar.close` is at a
        // 7-bar low/high iff it equals the min/max of this window.
        const auto window_begin = state.closes.end() - BREAKOUT_PERIOD;
        const auto [min_it, max_it] = std::minmax_element(window_begin, state.closes.end());

        if (bar.close <= *min_it && state.held_quantity == 0.0)
        {
            const auto qty = context.cash() / bar.close;
            if (qty <= 0.0) return;
            const auto order = context.make_market_order(stonks::core::MarketOrderParams{
                .symbol = bar.symbol,
                .side = stonks::core::OrderSide::Buy,
                .quantity = qty,
                .time_in_force = stonks::core::TimeInForce::GTC,
            });
            if (context.place_order(order)) state.held_quantity = qty;
        }
        else if (bar.close >= *max_it && state.held_quantity > 0.0)
        {
            const auto order = context.make_market_order(stonks::core::MarketOrderParams{
                .symbol = bar.symbol,
                .side = stonks::core::OrderSide::Sell,
                .quantity = state.held_quantity,
                .time_in_force = stonks::core::TimeInForce::GTC,
            });
            if (context.place_order(order)) state.held_quantity = 0.0;
        }
    }
};
