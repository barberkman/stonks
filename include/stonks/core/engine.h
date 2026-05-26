#pragma once

#include <optional>
#include <utility>

#include "stonks/core/broker.h"
#include "stonks/core/clock.h"
#include "stonks/core/context.h"
#include "stonks/core/datafeed.h"
#include "stonks/core/strategy.h"
#include "stonks/core/types.h"

namespace stonks::core {

template <class StrategyT, DataFeed DataFeedT, Broker BrokerT>
class Engine
{
    static_assert(Strategy<StrategyT, Context<BrokerT, DataFeedT>>);

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

        while (auto ts = m_dataFeed.next_timestamp()) {
            m_clock.set(*ts);
            m_broker.on_tick(m_dataFeed.current_kline());
            m_strategy.on_tick(context);
            m_dataFeed.advance();
        }

        if constexpr (HasOnStop<StrategyT, ContextT>) { m_strategy.on_stop(context); }
    }

private:
    StrategyT m_strategy;
    DataFeedT m_dataFeed;
    BrokerT m_broker;
    Clock m_clock;
};

} // namespace stonks::core
