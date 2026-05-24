#pragma once

#include <iostream>

#include "stonks/core/context.h"
#include "stonks/core/strategy.h"
#include "stonks/core/types.h"

namespace stonks::core {

template <Strategy S>
class Engine {
public:
    Engine(S strategy)
    : m_strategy{ std::move(strategy) }
    {}

    void run()
    {
        Context ctx{};

        // Start the strategy (optional)
        if constexpr (HasOnStart<S>) { m_strategy.on_start(ctx); }

        // Main loop
        while (true) {
            std::cout << "Engine main loop" << std::endl;
            break;
        }

        // Stop the strategy (optional)
        if constexpr (HasOnStop<S>) { m_strategy.on_stop(ctx); }
    }

private:
    S m_strategy;
};

}
