#pragma once

#include "stonks/core/types.h"

class Broker
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
};
