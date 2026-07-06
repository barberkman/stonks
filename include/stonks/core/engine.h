#pragma once

#include <algorithm>
#include <atomic>
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
    // `cancel`, if non-null, is polled once per timestamp; when it becomes true
    // the run loop stops cleanly (on_stop still runs). Lets a GUI cancel a run
    // from another thread without tearing the engine down mid-bar.
    Engine(StrategyT strategy, DataFeedT dataFeed, BrokerT broker,
           ProgressOutput progress_output = ProgressOutput::Console,
           const std::atomic<bool>* cancel = nullptr)
    : m_strategy{ std::move(strategy) },
      m_dataFeed{ std::move(dataFeed) },
      m_broker{ std::move(broker) },
      m_progress_output{ progress_output },
      m_cancel{ cancel }
    {}

    void run()
    {
        // Create context
        using ContextT = Context<BrokerT, DataFeedT>;
        ContextT context{ m_broker, m_dataFeed, m_clock, &m_indicators };

        STONKS_LOG("engine", "ev=run_start cash={:.4f}", m_broker.cash());

        // Start the strategy
        if constexpr (HasOnStart<StrategyT, ContextT>) {
            m_strategy.on_start(context);
        }

        // Create progress bar
        std::optional<std::size_t> total;
        if constexpr (HasSize<DataFeedT>) { total = m_dataFeed.size(); }
        m_progress.emplace(total, "bars", m_progress_output);

        // Main loop
        while (auto ts = m_dataFeed.next_timestamp()) {
            if (m_cancel && m_cancel->load(std::memory_order_relaxed)) { break; }

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

            // Update the progress
            m_progress->update(m_bars_processed);

            const Balance eq = m_broker.equity();
            m_equity_curve.push_back(EquityPoint{ *ts, eq });
            STONKS_LOG("engine", "ev=tick ts={} bars={} bars_total={} cash={:.4f} equity={:.4f}",
                       ts_ms, bars_this_ts, m_bars_processed, m_broker.cash(), eq);
            m_strategy.on_tick(context);
            m_dataFeed.advance();
        }
        m_progress->finish();
        STONKS_LOG("engine", "ev=run_end bars_total={} cash={:.4f} equity={:.4f} trades={} orders={}",
                   m_bars_processed, m_broker.cash(), m_broker.equity(),
                   m_broker.trades().size(), m_broker.orders().size());

        // Stop the strategy
        if constexpr (HasOnStop<StrategyT, ContextT>) {
            m_strategy.on_stop(context);
        }
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
    const IndicatorStore& indicators() const { return m_indicators; }
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
    IndicatorStore m_indicators;
    std::size_t m_bars_processed{ 0 };
    ProgressOutput m_progress_output;
    const std::atomic<bool>* m_cancel{ nullptr };
    std::optional<ProgressBar> m_progress;
};

} // namespace stonks::core
