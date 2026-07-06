#pragma once

#include <memory>
#include <optional>
#include <utility>

#include "stonks/core/context.h"
#include "stonks/core/types.h"
#include "stonks/python/icontext.h"

namespace stonks::python {

template <class BrokerT, class DataFeedT>
class ContextAdapter final : public IContext
{
public:
    explicit ContextAdapter(core::Context<BrokerT, DataFeedT>& ctx) : m_ctx{ ctx } {}

    core::Timestamp now() const override { return m_ctx.now(); }
    core::Balance cash() const override { return m_ctx.cash(); }
    core::Balance equity() const override { return m_ctx.equity(); }

    core::MarketWindow history(int count) const override { return m_ctx.history(count); }

    core::OrderID place_market_order(core::MarketOrderParams params,
                                     std::optional<core::OrderID> parent) override
    {
        // Context forwards to the broker, which builds + stamps the Order and
        // returns its id; pass `parent` through for bracket/OCO linkage.
        return m_ctx.place_order(params, parent);
    }

    core::OrderID place_limit_order(core::LimitOrderParams params,
                                    std::optional<core::OrderID> parent) override
    {
        return m_ctx.place_order(params, parent);
    }

    core::OrderID place_stop_order(core::StopOrderParams params,
                                   std::optional<core::OrderID> parent) override
    {
        return m_ctx.place_order(params, parent);
    }

    std::optional<core::Position> position(const core::Symbol& symbol) const override
    {
        return m_ctx.position(symbol);
    }

    std::optional<core::Order> order(core::OrderID id) const override
    {
        return m_ctx.order(id);
    }

    bool cancel_order(core::OrderID id) override
    {
        return m_ctx.cancel_order(id);
    }

private:
    core::Context<BrokerT, DataFeedT>& m_ctx;
};

// Deduction helper so callers that don't know B/F at compile time (e.g.
// PythonStrategy) can still build the right adapter from a templated context
// they receive via auto&.
template <class BrokerT, class DataFeedT>
std::unique_ptr<IContext> make_adapter(core::Context<BrokerT, DataFeedT>& ctx)
{
    return std::make_unique<ContextAdapter<BrokerT, DataFeedT>>(ctx);
}

} // namespace stonks::python
