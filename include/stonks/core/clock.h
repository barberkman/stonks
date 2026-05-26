#pragma once

#include "stonks/core/types.h"

namespace stonks::core {

class Clock
{
public:
    Timestamp now() const { return m_timestamp; }

    void set(Timestamp timestamp) { m_timestamp = timestamp; }

private:
    Timestamp m_timestamp{};
};

} // namespace stonks::core
