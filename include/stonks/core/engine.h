#pragma once

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "stonks/core/broker.h"
#include "stonks/core/clock.h"
#include "stonks/core/context.h"
#include "stonks/core/datafeed.h"
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
    Engine(StrategyT strategy, DataFeedT dataFeed, BrokerT broker)
    : m_strategy{ std::move(strategy) },
      m_dataFeed{ std::move(dataFeed) },
      m_broker{ std::move(broker) }
    {}

    void run()
    {
        using ContextT = Context<BrokerT, DataFeedT>;
        ContextT context{ m_broker, m_dataFeed, m_clock };

        if constexpr (HasOnStart<StrategyT, ContextT>) { m_strategy.on_start(context); }

        std::optional<std::size_t> total;
        if constexpr (HasSize<DataFeedT>) { total = m_dataFeed.size(); }
        ProgressBar progress{ total };

        while (auto ts = m_dataFeed.next_timestamp()) {
            m_clock.set(*ts);

            // Settle the whole timestamp first: fill/mark every symbol that
            // printed, before the strategy reacts once for this timestamp.
            for (const auto& bar : m_dataFeed.current_bars()) {
                m_broker.on_tick(bar);
                ++m_bars_processed;
            }
            progress.update(m_bars_processed);

            m_equity_curve.push_back(EquityPoint{ *ts, m_broker.equity() });

            m_strategy.on_tick(context);
            m_dataFeed.advance();
        }
        progress.finish();

        if constexpr (HasOnStop<StrategyT, ContextT>) { m_strategy.on_stop(context); }
    }

    // Run history, for an external reporter to consume after run(). The broker
    // accessors forward to the executing broker; the equity curve and bar count
    // are the engine's own per-run record.
    const std::vector<Trade>& trades() const { return m_broker.trades(); }
    const std::vector<Order>& orders() const { return m_broker.orders(); }
    Balance cash() const { return m_broker.cash(); }
    Balance equity() const { return m_broker.equity(); }
    const std::vector<EquityPoint>& equity_curve() const { return m_equity_curve; }
    std::size_t bars_processed() const { return m_bars_processed; }

private:
    StrategyT m_strategy;
    DataFeedT m_dataFeed;
    BrokerT m_broker;
    Clock m_clock;
    std::vector<EquityPoint> m_equity_curve;
    std::size_t m_bars_processed{ 0 };
};

} // namespace stonks::core
