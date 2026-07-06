#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/test_stubs.h"
#include "stonks/broker/backtestbroker.h"
#include "stonks/core/clock.h"
#include "stonks/core/context.h"
#include "stonks/core/types.h"
#include "stonks/python/embeddedpython.h"

#include "strategies/pythonstrategy.h"

namespace {

// Ensures the embedded interpreter is initialized once and sys.path includes
// both the in-source `python/stonks` package and the fixtures directory.
// Function-local statics give us deterministic init at first test that calls
// this; refcounted EmbeddedPython keeps the interpreter alive across tests.
void ensure_python_setup()
{
    // Point the embedded interpreter at the app venv so numpy/pandas (used by
    // the history-inspecting fixture) are importable; overwrite=0 lets a real
    // env var win. Must run before the interpreter initializes below.
    ::setenv("STONKS_VENV", STONKS_VENV_DIR, 0);
    static stonks::python::EmbeddedPython python_handle;
    static const bool paths_added = []() {
        stonks::python::EmbeddedPython::add_sys_path(STONKS_PYTHON_PACKAGE_DIR);
        stonks::python::EmbeddedPython::add_sys_path(STONKS_TEST_FIXTURES_DIR);
        return true;
    }();
    (void)python_handle;
    (void)paths_added;
}

class PythonStrategyTest : public ::testing::Test
{
protected:
    void SetUp() override { ensure_python_setup(); }
};

using namespace stonks;
using core::test::StubBroker;
using core::test::StubFeed;

TEST_F(PythonStrategyTest, ImportFailureSurfacesAsRuntimeError)
{
    EXPECT_THROW(
        PythonStrategy("definitely_not_a_real_module_xyz", "Anything"),
        std::runtime_error);
}

TEST_F(PythonStrategyTest, AttributeErrorSurfacesAsRuntimeError)
{
    EXPECT_THROW(
        PythonStrategy("fixturestrats", "DoesNotExist"),
        std::runtime_error);
}

TEST_F(PythonStrategyTest, MissingOnTickRejectedAtConstruction)
{
    try {
        PythonStrategy strat{ "fixturestrats", "NoOnTick" };
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("on_tick"), std::string::npos)
            << "error message should mention on_tick; got: " << msg;
    }
}

TEST_F(PythonStrategyTest, LifecycleHooksDispatchInOrder)
{
    PythonStrategy strat{ "fixturestrats", "CallRecording" };

    std::vector<core::Order> placed;
    StubBroker broker;
    broker.placed = &placed;
    StubFeed feed;
    core::Clock clock;
    core::Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    strat.on_start(ctx);
    strat.on_tick(ctx);
    strat.on_tick(ctx);
    strat.on_stop(ctx);

    ASSERT_EQ(placed.size(), 4u);
    EXPECT_EQ(placed[0].quantity, 1.0);  // on_start
    EXPECT_EQ(placed[1].quantity, 2.0);  // first on_tick
    EXPECT_EQ(placed[2].quantity, 2.0);  // second on_tick
    EXPECT_EQ(placed[3].quantity, 3.0);  // on_stop
}

TEST_F(PythonStrategyTest, OptionalHooksSilentlySkipped)
{
    PythonStrategy strat{ "fixturestrats", "BareTickOnly" };

    std::vector<core::Order> placed;
    StubBroker broker;
    broker.placed = &placed;
    StubFeed feed;
    core::Clock clock;
    core::Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    EXPECT_NO_THROW(strat.on_start(ctx));   // no on_start defined
    strat.on_tick(ctx);
    EXPECT_NO_THROW(strat.on_stop(ctx));    // no on_stop defined

    ASSERT_EQ(placed.size(), 1u);
    EXPECT_EQ(placed[0].quantity, 2.0);
}

TEST_F(PythonStrategyTest, PythonExceptionWrappedAsRuntimeError)
{
    PythonStrategy strat{ "fixturestrats", "RaisingStrategy" };

    StubBroker broker;
    StubFeed feed;
    core::Clock clock;
    core::Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    try {
        strat.on_tick(ctx);
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("boom from python"), std::string::npos)
            << "wrapped exception should include the Python error text; got: " << msg;
    }
}

TEST_F(PythonStrategyTest, PythonStrategyMovableForEngineByValueConstruct)
{
    // Engine takes the strategy by value and std::moves it into a member, so
    // this is the lifecycle pattern that must survive without breaking the
    // interpreter handle or the held py::object.
    PythonStrategy source{ "fixturestrats", "BareTickOnly" };
    PythonStrategy moved{ std::move(source) };

    std::vector<core::Order> placed;
    StubBroker broker;
    broker.placed = &placed;
    StubFeed feed;
    core::Clock clock;
    core::Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    moved.on_tick(ctx);

    ASSERT_EQ(placed.size(), 1u);
    EXPECT_EQ(placed[0].quantity, 2.0);
}

TEST_F(PythonStrategyTest, ObservabilityApiRoundTripsThroughBindings)
{
    PythonStrategy strat{ "fixturestrats", "ApiProbe" };

    // A real broker: the probe exercises actual cancel/status transitions.
    broker::BacktestBroker broker{ core::Balance{ 10'000.0 } };
    StubFeed feed;
    core::Clock clock;
    core::Context<broker::BacktestBroker, StubFeed> ctx{ broker, feed, clock };

    ASSERT_NO_THROW(strat.on_tick(ctx));   // the probe asserts internally

    // It cancelled its limit order and left one reduce-only stop marker.
    std::vector<core::Order> all;
    for (const auto& [id, o] : broker.orders()) { all.push_back(o); }
    std::ranges::sort(all, {}, &core::Order::id);
    ASSERT_EQ(all.size(), 2u);
    EXPECT_EQ(all[0].status, core::OrderStatus::Cancelled);
    EXPECT_EQ(all[1].type, core::OrderType::Stop);
    EXPECT_TRUE(all[1].reduce_only);
    EXPECT_DOUBLE_EQ(all[1].quantity, 7.0);
}

TEST_F(PythonStrategyTest, StopAndLimitOrdersRoundTripThroughBindings)
{
    PythonStrategy strat{ "fixturestrats", "BracketPlacing" };

    std::vector<core::Order> placed;
    StubBroker broker;
    broker.placed = &placed;
    StubFeed feed;
    core::Clock clock;
    core::Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    strat.on_tick(ctx);

    ASSERT_EQ(placed.size(), 3u);
    // Stop entry: trigger price and leverage forwarded, no parent.
    EXPECT_EQ(placed[0].type, core::OrderType::Stop);
    EXPECT_EQ(placed[0].side, core::OrderSide::Buy);
    ASSERT_TRUE(placed[0].price.has_value());
    EXPECT_DOUBLE_EQ(*placed[0].price, 110.0);
    EXPECT_DOUBLE_EQ(placed[0].leverage, 5.0);
    EXPECT_FALSE(placed[0].parent_id.has_value());
    // Stop-loss child bracketed under the entry.
    EXPECT_EQ(placed[1].type, core::OrderType::Stop);
    EXPECT_EQ(placed[1].side, core::OrderSide::Sell);
    ASSERT_TRUE(placed[1].price.has_value());
    EXPECT_DOUBLE_EQ(*placed[1].price, 95.0);
    EXPECT_DOUBLE_EQ(placed[1].leverage, 1.0);   // default when not specified
    ASSERT_TRUE(placed[1].parent_id.has_value());
    EXPECT_EQ(*placed[1].parent_id, placed[0].id);
    // Take-profit limit child.
    EXPECT_EQ(placed[2].type, core::OrderType::Limit);
    ASSERT_TRUE(placed[2].price.has_value());
    EXPECT_DOUBLE_EQ(*placed[2].price, 130.0);
    ASSERT_TRUE(placed[2].parent_id.has_value());
    EXPECT_EQ(*placed[2].parent_id, placed[0].id);
}

TEST_F(PythonStrategyTest, PythonCanReadContextStateInOnTick)
{
    PythonStrategy strat{ "fixturestrats", "CashAwareStrategy" };

    std::vector<core::Order> placed;
    StubBroker broker;     // StubBroker.cash() returns 0; CashAwareStrategy will see that.
    broker.placed = &placed;
    StubFeed feed;
    core::Clock clock;
    core::Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    strat.on_tick(ctx);

    ASSERT_EQ(placed.size(), 1u);
    EXPECT_EQ(placed[0].quantity, 0.0);  // cash() == 0 round-tripped through Python
    EXPECT_EQ(placed[0].side, core::OrderSide::Sell);
}

TEST_F(PythonStrategyTest, HistoryExposesCombinedMultiSymbolDataFrame)
{
    PythonStrategy strat{ "fixturestrats", "HistoryInspector" };

    std::vector<core::Order> placed;
    StubBroker broker;
    broker.placed = &placed;
    StubFeed feed;
    feed.bars = {
        core::test::make_bar(1000, "A", 100.0), core::test::make_bar(1000, "B", 200.0),
        core::test::make_bar(2000, "A", 101.0), core::test::make_bar(2000, "B", 201.0),
        core::test::make_bar(3000, "A", 102.0), core::test::make_bar(3000, "B", 202.0),
    };
    core::Clock clock;
    core::Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    feed.advance();
    feed.advance();   // -> timestamp 3000
    clock.set(core::Timestamp::from_millis(3000));

    // The fixture builds a DataFrame, groupby-asserts dtypes/values internally,
    // and places one buy per symbol; a failure there surfaces as runtime_error.
    ASSERT_NO_THROW(strat.on_tick(ctx));
    ASSERT_EQ(placed.size(), 2u);
    EXPECT_EQ(placed[0].symbol, "A");                 // groupby sorts symbols
    EXPECT_DOUBLE_EQ(placed[0].quantity, 102.0);
    EXPECT_EQ(placed[1].symbol, "B");
    EXPECT_DOUBLE_EQ(placed[1].quantity, 202.0);
    for (const auto& o : placed) { EXPECT_EQ(o.side, core::OrderSide::Buy); }
}

} // namespace
