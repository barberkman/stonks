#pragma once

#include <memory>
#include <optional>

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
    std::optional<core::Position> position(const core::Symbol& symbol) const override { return m_ctx.position(symbol); }

    core::MarketWindow history(int count) const override { return m_ctx.history(count); }

    core::OrderID place_order(core::OrderParams params) override { return m_ctx.place_order(params); }
    core::OrderID place_exit(core::OrderParams params) override { return m_ctx.place_exit(params); }

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
