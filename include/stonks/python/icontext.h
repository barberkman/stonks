#pragma once

#include <optional>
#include <vector>

#include "stonks/core/types.h"

namespace stonks::python {

// Type-erased view of stonks::core::Context exposed to Python. The templated
// Context<Broker, DataFeed> can't be bound directly to pybind11, so the adapter
// (ContextAdapter) implements this interface for a given Broker/DataFeed combo
// and the binding only knows about IContext.
//
// The klines() overloads in the C++ Context are split into distinct names here
// because pybind11 binding of virtual overloads differing only in argument
// count is awkward; the Python __init__.py reassembles them into a single
// klines(...) dispatcher.
class IContext
{
public:
    virtual ~IContext() = default;

    virtual core::Timestamp now() const = 0;
    virtual core::Balance cash() const = 0;
    virtual core::Balance equity() const = 0;

    virtual std::vector<core::KLine> klines_count(int count) const = 0;
    virtual std::vector<core::KLine> klines_range(
        core::Timestamp start,
        std::optional<core::Timestamp> end) const = 0;

    // make_*_order is intentionally not exposed: Order's constructor is
    // private and friended only to Context, so Python can never hold an
    // unsubmitted Order. The adapter does make + place in one step.
    virtual bool place_market_order(core::MarketOrderParams params) = 0;
    virtual bool place_limit_order(core::LimitOrderParams params) = 0;
};

} // namespace stonks::python
