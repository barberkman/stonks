#pragma once

#include <iostream>

#include "stonks/core/types.h"

namespace stonks::core {

class Clock
{
public:
    Timestamp now() const 
    {
        return m_timestamp;
    }

    void advance(Timestamp timestamp)
    {
        std::cout << "Clock::advance: time advance request to: " << timestamp << std::endl;
        m_timestamp = timestamp;
    }

private:
    Timestamp m_timestamp{};
};

}
