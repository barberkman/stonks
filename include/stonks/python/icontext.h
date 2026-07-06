#pragma once

#include <optional>

#include "stonks/core/types.h"

namespace stonks::python {

// Type-erased view of stonks::core::Context exposed to Python. The templated
// Context<Broker, DataFeed> can't be bound directly to pybind11, so the adapter
// (ContextAdapter) implements this interface for a given Broker/DataFeed combo
// and the binding only knows about IContext.
class IContext
{
public:
    virtual ~IContext() = default;

    virtual core::Timestamp now() const = 0;
    virtual core::Balance cash() const = 0;
    virtual core::Balance equity() const = 0;

    // This tick's window: every symbol that printed at the current timestamp,
    // each with its last `count` bars (bound to Python as one combined DataFrame).
    virtual core::MarketWindow history(int count) const = 0;

    // Orders are placed by params, not constructed on the Python side: the
    // adapter forwards params straight to the broker (via Context), so Python
    // never holds an unsubmitted Order. The broker-assigned OrderID is returned;
    // `parent` links this order to an entry (children stay dormant until the
    // parent fills, then OCO-cancel their siblings) — nullopt for a standalone order.
    virtual core::OrderID place_market_order(core::MarketOrderParams params,
                                             std::optional<core::OrderID> parent) = 0;
    virtual core::OrderID place_limit_order(core::LimitOrderParams params,
                                            std::optional<core::OrderID> parent) = 0;
    virtual core::OrderID place_stop_order(core::StopOrderParams params,
                                           std::optional<core::OrderID> parent) = 0;
};

} // namespace stonks::python
