#pragma once

#include <stonks/core/types.h>

#include <optional>
#include <unordered_map>

// Long-only trend-following strategy, applied independently per symbol: for
// each symbol the feed surfaces, hold while its close is above its own 50-bar
// EMA and stay flat otherwise. Each entry commits 1% of current account equity,
// so many symbols can hold a position at once (one per symbol). No shorting and
// no cross-symbol coupling.
struct EMA50Strategy
{
    static constexpr int PERIOD = 50;
    // Standard EMA smoothing factor: 2/(N+1) gives the latest bar a weight that
    // makes the EMA's responsiveness comparable to an N-period SMA.
    static constexpr double ALPHA = 2.0 / (PERIOD + 1);
    // Fraction of current account equity committed to each new position.
    static constexpr double POSITION_FRACTION = 0.01;

    struct SymbolState
    {
        std::optional<double> ema;
        double seed_sum{};
        int seed_count{};
        double held_quantity{};
    };

    std::unordered_map<stonks::core::Symbol, SymbolState> states;

    void on_tick(auto& context)
    {
        // klines() interleaves bars across every symbol the feed surfaces, so a
        // per-symbol EMA must consume only its own symbol's stream. Treat the
        // latest bar as the one that triggered this tick and route it to its
        // symbol's state.
        const auto bars = context.klines(1);
        if (bars.empty()) return;
        const auto& bar = bars.back();

        auto& state = states[bar.symbol];

        if (!state.ema)
        {
            // Accumulate closes until we have PERIOD samples for this symbol,
            // then seed the EMA with their SMA — standard bootstrap that avoids
            // anchoring the recursion to a single bar.
            state.seed_sum += bar.close;
            ++state.seed_count;
            if (state.seed_count < PERIOD) return;
            state.ema = state.seed_sum / PERIOD;
        }
        else
        {
            state.ema = ALPHA * bar.close + (1.0 - ALPHA) * (*state.ema);
        }

        // Enter long on an upside crossover; flat-only means we never short on the downside.
        if (bar.close > *state.ema && state.held_quantity == 0.0)
        {
            const auto qty = context.equity() * POSITION_FRACTION / bar.close;
            if (qty <= 0.0) return;
            const auto order = context.make_market_order(stonks::core::MarketOrderParams{
                .symbol = bar.symbol,
                .side = stonks::core::OrderSide::Buy,
                .quantity = qty,
                .time_in_force = stonks::core::TimeInForce::GTC,
            });
            if (context.place_order(order)) state.held_quantity = qty;
        }
        else if (bar.close < *state.ema && state.held_quantity > 0.0)
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
