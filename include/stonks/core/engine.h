#pragma once

#include <exception>
#include <iostream>
#include <optional>

#include "stonks/core/context.h"
#include "stonks/core/strategy.h"
#include "stonks/core/datafeed.h"
#include "stonks/core/broker.h"
#include "stonks/core/clock.h"
#include "stonks/core/types.h"

namespace stonks::core {

template <class StrategyT, DataFeed DataFeedT, Broker BrokerT>
    requires Strategy<StrategyT, Context<BrokerT, DataFeedT>>
class Engine {
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

        // Start the strategy (optional)
        if constexpr (HasOnStart<StrategyT, ContextT>) { m_strategy.on_start(context); }

        // Main loop
        std::cout << "Entering engine's main loop" << std::endl;
        while (true) {
            try {
                std::cout << "\n";

                // Advance time
                std::optional<Timestamp> next_timestamp = m_dataFeed.peek(m_clock.now());
                if (!next_timestamp) { break; }
                m_clock.advance(*next_timestamp);

                // Call strategy
                m_strategy.on_kline(context);
            } catch (const std::exception& ex) {
                std::cout << "Main loop exception: " << ex.what() << std::endl;
            }
        }

        // Stop the strategy (optional)
        if constexpr (HasOnStop<StrategyT, ContextT>) { m_strategy.on_stop(context); }
    }

private:
    StrategyT m_strategy;
    DataFeedT m_dataFeed;
    BrokerT m_broker;
    Clock m_clock;
};

} // namespace stonks::core
