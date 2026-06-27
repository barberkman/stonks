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
        // One tick per timestamp: loop the symbols that printed today and run the
        // per-symbol EMA on each. EMA is incremental, so one bar per symbol
        // suffices; each symbol's stream stays independent via its own state.
        for (const auto& s : context.history(1).series)
        {
            const stonks::core::Symbol symbol{ s.symbol };
            const double close = s.bars.close.back();

            auto& state = states[symbol];

            if (!state.ema)
            {
                // Accumulate closes until we have PERIOD samples for this symbol,
                // then seed the EMA with their SMA — standard bootstrap that avoids
                // anchoring the recursion to a single bar.
                state.seed_sum += close;
                ++state.seed_count;
                if (state.seed_count < PERIOD) continue;
                state.ema = state.seed_sum / PERIOD;
            }
            else
            {
                state.ema = ALPHA * close + (1.0 - ALPHA) * (*state.ema);
            }

            // Enter long on an upside crossover; flat-only means we never short on the downside.
            if (close > *state.ema && state.held_quantity == 0.0)
            {
                const auto qty = context.equity() * POSITION_FRACTION / close;
                if (qty <= 0.0) continue;
                // held_quantity is tracked optimistically on placement, not on a
                // confirmed fill — the broker may still reject (e.g. insufficient cash).
                context.place_order(stonks::core::MarketOrderParams{
                    .symbol = symbol,
                    .side = stonks::core::OrderSide::Buy,
                    .quantity = qty,
                    .time_in_force = stonks::core::TimeInForce::GTC,
                });
                state.held_quantity = qty;
            }
            else if (close < *state.ema && state.held_quantity > 0.0)
            {
                context.place_order(stonks::core::MarketOrderParams{
                    .symbol = symbol,
                    .side = stonks::core::OrderSide::Sell,
                    .quantity = state.held_quantity,
                    .time_in_force = stonks::core::TimeInForce::GTC,
                });
                state.held_quantity = 0.0;
            }
        }
    }
};
