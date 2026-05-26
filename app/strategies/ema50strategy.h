#pragma once

#include <stonks/core/types.h>

#include <optional>

struct EMA50Strategy
{
    static constexpr int PERIOD = 50;
    static constexpr double ALPHA = 2.0 / (PERIOD + 1);

    std::optional<double> ema;
    double held_quantity{};

    void on_tick(auto& context)
    {
        const auto bars = context.klines(PERIOD);
        if (static_cast<int>(bars.size()) < PERIOD) return;

        const auto& last = bars.back();
        const auto close = last.close;

        if (!ema)
        {
            double sum = 0.0;
            for (const auto& b : bars) sum += b.close;
            ema = sum / PERIOD;
        }
        else
        {
            ema = ALPHA * close + (1.0 - ALPHA) * (*ema);
        }

        if (close > *ema && held_quantity == 0.0)
        {
            const auto qty = context.cash() / close;
            if (qty <= 0.0) return;
            const auto order = context.make_market_order(stonks::core::MarketOrderParams{
                .symbol = last.symbol,
                .side = stonks::core::OrderSide::Buy,
                .quantity = qty,
                .time_in_force = stonks::core::TimeInForce::GTC,
            });
            if (context.place_order(order)) held_quantity = qty;
        }
        else if (close < *ema && held_quantity > 0.0)
        {
            const auto order = context.make_market_order(stonks::core::MarketOrderParams{
                .symbol = last.symbol,
                .side = stonks::core::OrderSide::Sell,
                .quantity = held_quantity,
                .time_in_force = stonks::core::TimeInForce::GTC,
            });
            if (context.place_order(order)) held_quantity = 0.0;
        }
    }
};
