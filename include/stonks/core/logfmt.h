#pragma once

// Companion to log.h: type-aware formatters for STONKS_LOG. std::vformat cannot
// print scoped enums (OrderSide/OrderType) or Timestamp directly, so these turn
// them into format-friendly values. Compiled in only when logging is enabled;
// remove alongside log.h and the call sites when no longer needed.

#ifdef STONKS_LOG_ENABLED

#include <cstdint>

#include "stonks/core/types.h"

namespace stonks::log {

inline const char* side_str(core::OrderSide s) { return s == core::OrderSide::Buy ? "Buy" : "Sell"; }
inline const char* type_str(core::OrderType t)
{
    switch (t) {
        case core::OrderType::Market: return "Market";
        case core::OrderType::Limit: return "Limit";
        case core::OrderType::Stop: return "Stop";
    }
    return "Unknown";
}
inline std::int64_t ts_ms(core::Timestamp t) { return t.value.time_since_epoch().count(); }

} // namespace stonks::log

#endif
