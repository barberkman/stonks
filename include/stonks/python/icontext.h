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

    // The current position on a symbol, or nullopt if flat.
    virtual std::optional<core::Position> position(const core::Symbol& symbol) const = 0;

    // This tick's window: every symbol that printed at the current timestamp,
    // each with its last `count` bars (bound to Python as one combined DataFrame).
    virtual core::MarketWindow history(int count) const = 0;

    // Orders are placed by params, not constructed on the Python side. place_order
    // opens a position (entry) and may carry stop_loss / take_profit levels; the
    // broker-assigned OrderID is returned. close() market-closes the position on a
    // symbol; update_exits() retargets its SL/TP levels. Both return whether they
    // found something to act on.
    virtual core::OrderID place_order(core::OrderParams params) = 0;
    virtual bool close(const core::Symbol& symbol) = 0;
    virtual bool update_exits(const core::Symbol& symbol,
                              std::optional<core::Price> stop_loss,
                              std::optional<core::Price> take_profit) = 0;
};

} // namespace stonks::python
