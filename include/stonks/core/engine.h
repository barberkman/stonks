#pragma once

#include <exception>
#include <iostream>
#include <optional>

#include "stonks/core/strategy.h"
#include "stonks/core/datafeed.h"
#include "stonks/core/broker.h"
#include "stonks/core/context.h"
#include "stonks/core/clock.h"
#include "stonks/core/types.h"

namespace stonks::core {

template <Strategy StrategyT, DataFeed DataFeedT, Broker BrokerT, Context ContextT>
class Engine {
public:
    Engine(StrategyT strategy, DataFeedT dataFeed, BrokerT broker, ContextT context)
    : m_strategy{ std::move(strategy) },
    m_dataFeed{ std::move(dataFeed) },
    m_broker{ std::move(broker) },
    m_context{ std::move(context) }
    {}

    void run()
    {
        // Start the strategy (optional)
        if constexpr (StrategyHasOnStart<StrategyT, ContextT>) { m_strategy.on_start(m_context); }

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
                if constexpr (StrategyHasOnKLine<StrategyT, ContextT>) { m_strategy.on_kline(m_context); }
            } catch (const std::exception& ex) {
                std::cout << "Main loop exception: " << ex.what() << std::endl;
            }
        }

        // Stop the strategy (optional)
        if constexpr (StrategyHasOnStop<StrategyT, ContextT>) { m_strategy.on_stop(m_context); }
    }

private:
    StrategyT m_strategy;
    DataFeedT m_dataFeed;
    BrokerT m_broker;
    ContextT m_context;
    Clock m_clock;
};

} // namespace stonks::core
