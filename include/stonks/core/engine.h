#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "stonks/core/broker.h"
#include "stonks/core/clock.h"
#include "stonks/core/context.h"
#include "stonks/core/datafeed.h"
#include "stonks/core/log.h"
#include "stonks/core/progressbar.h"
#include "stonks/core/strategy.h"
#include "stonks/core/types.h"

namespace stonks::core {

template <class StrategyT, DataFeed DataFeedT, Broker BrokerT>
class Engine
{
    static_assert(Strategy<StrategyT, Context<BrokerT, DataFeedT>>,
        "StrategyT must satisfy the Strategy concept");

public:
    Engine(StrategyT strategy, DataFeedT dataFeed, BrokerT broker,
           ProgressOutput progress_output = ProgressOutput::Console)
    : m_strategy{ std::move(strategy) },
      m_dataFeed{ std::move(dataFeed) },
      m_broker{ std::move(broker) },
      m_progress_output{ progress_output }
    {}

    void run()
    {
        // Create context
        using ContextT = Context<BrokerT, DataFeedT>;
        ContextT context{ m_broker, m_dataFeed, m_clock };

        STONKS_LOG("engine", "run start: starting_cash={:.4f}", m_broker.cash());

        // Start the strategy
        if constexpr (HasOnStart<StrategyT, ContextT>) {
            STONKS_LOG("engine", "on_start invoked");
            m_strategy.on_start(context);
        } else {
            STONKS_LOG("engine", "on_start: not defined by strategy");
        }

        // Create progress bar
        std::optional<std::size_t> total;
        if constexpr (HasSize<DataFeedT>) { total = m_dataFeed.size(); }
        STONKS_LOG("engine", "feed total bars={}", total ? *total : 0);
        m_progress.emplace(total, "bars", m_progress_output);

        // Main loop
        while (auto ts = m_dataFeed.next_timestamp()) {
            // Set the timestamp
            m_clock.set(*ts);
            [[maybe_unused]] const auto ts_ms = ts->value.time_since_epoch().count();

            // Process the broker first
            [[maybe_unused]] std::size_t bars_this_ts = 0;
            for (const auto& bar : m_dataFeed.current_bars()) {
                m_broker.on_tick(bar);
                ++m_bars_processed;
                ++bars_this_ts;
            }
            STONKS_LOG("engine", "tick ts={} bars={} total_bars={}", ts_ms, bars_this_ts, m_bars_processed);

            // Update the progress
            m_progress->update(m_bars_processed);

            const Balance eq = m_broker.equity();
            m_equity_curve.push_back(EquityPoint{ *ts, eq });
            STONKS_LOG("engine", "equity ts={} equity={:.4f} cash={:.4f}", ts_ms, eq, m_broker.cash());

            STONKS_LOG("engine", "strategy on_tick ts={}", ts_ms);
            m_strategy.on_tick(context);
            m_dataFeed.advance();
        }
        m_progress->finish();

        // Stop the strategy
        if constexpr (HasOnStop<StrategyT, ContextT>) {
            STONKS_LOG("engine", "on_stop invoked");
            m_strategy.on_stop(context);
        } else {
            STONKS_LOG("engine", "on_stop: not defined by strategy");
        }

        STONKS_LOG("engine", "run end: total_bars={} trades={} cash={:.4f} equity={:.4f}",
            m_bars_processed, m_broker.trades().size(), m_broker.cash(), m_broker.equity());
    }

    // Run history, for an external reporter to consume after run(). The broker
    // accessors forward to the executing broker; the equity curve and bar count
    // are the engine's own per-run record.
    std::vector<Trade> trades() const
    {
        std::vector<Trade> out;
        out.reserve(m_broker.trades().size());
        for (const auto& [id, t] : m_broker.trades()) { out.push_back(t); }
        std::ranges::sort(out, {}, &Trade::id);
        return out;
    }

    std::vector<Order> orders() const
    {
        std::vector<Order> out;
        out.reserve(m_broker.orders().size());
        for (const auto& [id, o] : m_broker.orders()) { out.push_back(o); }
        std::ranges::sort(out, {}, &Order::id);
        return out;
    }
    Balance cash() const { return m_broker.cash(); }
    Balance equity() const { return m_broker.equity(); }
    const std::vector<EquityPoint>& equity_curve() const { return m_equity_curve; }
    std::size_t bars_processed() const { return m_bars_processed; }

    // Live progress snapshot for an external consumer (e.g. a GUI) to render
    // itself. Zeroed before run() constructs the progress bar.
    ProgressState progress() const
    {
        return m_progress ? m_progress->state() : ProgressState{};
    }

private:
    StrategyT m_strategy;
    DataFeedT m_dataFeed;
    BrokerT m_broker;
    Clock m_clock;
    std::vector<EquityPoint> m_equity_curve;
    std::size_t m_bars_processed{ 0 };
    ProgressOutput m_progress_output;
    std::optional<ProgressBar> m_progress;
};

} // namespace stonks::core
