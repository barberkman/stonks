#pragma once

#include <iostream>

#include "stonks/core/types.h"

class BacktestBroker
{
public:
    stonks::core::Balance cash() const
    {
        return 0.0;
    }

    stonks::core::Balance equity() const
    {
        return 0.0;
    }

    bool place_order(const stonks::core::Order& order)
    {
        std::cout << "BacktestBroker::place_order: " << order.price << std::endl;
        return true;
    }
};
