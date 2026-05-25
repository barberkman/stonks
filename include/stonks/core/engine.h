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

template <Strategy StrategyT, DataFeed DataFeedT, Broker BrokerT>
class Engine {
public:
    Engine(StrategyT strategy, DataFeedT dataFeed, BrokerT broker)
    : m_strategy{ std::move(strategy) },
    m_dataFeed{ std::move(dataFeed) },
    m_broker{ std::move(broker) }
    {}

    void run()
    {
        Context ctx{};

        // Start the strategy (optional)
        if constexpr (HasOnStart<StrategyT>) { m_strategy.on_start(ctx); }

        // Main loop
        std::cout << "Entering engine's main loop" << std::endl;
        while (true) {
            try {
                // Advance time
                std::optional<Timestamp> next_timestamp = m_dataFeed.peek(m_clock.now());
                if (!next_timestamp) { break; }
                m_clock.advance(*next_timestamp);

                // Call strategy
                m_strategy.on_kline(ctx);
            } catch (const std::exception& ex) {
                std::cout << "Main loop exception: " << ex.what() << std::endl;
            }
        }

        // Stop the strategy (optional)
        if constexpr (HasOnStop<StrategyT>) { m_strategy.on_stop(ctx); }
    }

private:
    StrategyT m_strategy;
    DataFeedT m_dataFeed;
    BrokerT m_broker;
    Clock m_clock;
};

}
