// BinanceBroker behavior against a stateful fake exchange: read mapping, bracket
// deferral + arming, orphan cleanup, parent_id reconstruction, and cancel.

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "stonks/binance/binancebroker.h"
#include "stonks/core/types.h"
#include "fake_binance.h"

namespace stonks::binance {
namespace {

using namespace stonks::core;
using json = nlohmann::json;

constexpr OrderID kSyntheticBase = OrderID{ 1 } << 62;

// A tiny in-memory USDⓈ-M exchange: enough surface for the broker to read state,
// place/cancel orders, and fill market orders synchronously.
struct MockExchange
{
    double available = 1000.0;
    double margin_balance = 1000.0;
    struct Pos { double amt; double entry; double lev; };
    std::map<std::string, Pos> positions;
    struct OO
    {
        std::string symbol, side, type, client_id;
        double price = 0, stop_price = 0, qty = 0;
        bool reduce_only = false;
    };
    std::map<long long, OO> open;
    long long next_id = 1000;
    std::vector<std::pair<std::string, bool>> posts;   // (type, reduceOnly) sent to /order

    int reduce_only_posts() const
    {
        int n = 0;
        for (const auto& [t, ro] : posts) { if (ro) { ++n; } }
        return n;
    }

    HttpResponse serve(const HttpRequest& req)
    {
        using P = test::FakeBinance;
        const std::string path = P::path_of(req);
        if (path == "/fapi/v1/exchangeInfo") { return ok(exchange_info()); }
        if (path == "/fapi/v3/account") { return ok(account()); }
        if (path == "/fapi/v3/positionRisk") { return ok(position_risk()); }
        if (path == "/fapi/v1/openOrders") { return ok(open_orders()); }
        if (path == "/fapi/v1/leverage" || path == "/fapi/v1/marginType") { return ok(json::object()); }
        if (path == "/fapi/v1/userTrades") { return ok(json::array()); }
        if (req.method == "POST" && path == "/fapi/v1/order") { return ok(place(req)); }
        if (req.method == "DELETE" && path == "/fapi/v1/order") { return ok(cancel(req)); }
        return HttpResponse{ 404, R"({"code":-1121,"msg":"unknown path"})" };
    }

private:
    static HttpResponse ok(const json& j) { return HttpResponse{ 200, j.dump() }; }

    static json exchange_info()
    {
        return json{ { "symbols", { {
            { "symbol", "BTCUSDT" }, { "quantityPrecision", 3 }, { "pricePrecision", 1 },
            { "filters", {
                { { "filterType", "LOT_SIZE" }, { "stepSize", "0.001" } },
                { { "filterType", "PRICE_FILTER" }, { "tickSize", "0.1" } },
                { { "filterType", "MIN_NOTIONAL" }, { "notional", "1" } },
            } } } } } };
    }

    json account() const
    {
        return json{ { "availableBalance", std::to_string(available) },
                     { "totalMarginBalance", std::to_string(margin_balance) } };
    }

    json position_risk() const
    {
        json arr = json::array();
        for (const auto& [sym, p] : positions) {
            arr.push_back({ { "symbol", sym }, { "positionAmt", std::to_string(p.amt) },
                            { "entryPrice", std::to_string(p.entry) },
                            { "leverage", std::to_string(static_cast<int>(p.lev)) } });
        }
        return arr;
    }

    json open_orders() const
    {
        json arr = json::array();
        for (const auto& [id, o] : open) {
            arr.push_back({ { "orderId", id }, { "symbol", o.symbol }, { "side", o.side },
                            { "type", o.type }, { "price", std::to_string(o.price) },
                            { "stopPrice", std::to_string(o.stop_price) },
                            { "origQty", std::to_string(o.qty) }, { "reduceOnly", o.reduce_only },
                            { "clientOrderId", o.client_id }, { "status", "NEW" },
                            { "updateTime", 1 } });
        }
        return arr;
    }

    json place(const HttpRequest& req)
    {
        using P = test::FakeBinance;
        const std::string symbol = P::param(req, "symbol").value_or("");
        const std::string side = P::param(req, "side").value_or("BUY");
        const std::string type = P::param(req, "type").value_or("MARKET");
        const bool reduce_only = P::param(req, "reduceOnly").value_or("") == "true";
        const double qty = std::stod(P::param(req, "quantity").value_or("0"));
        posts.emplace_back(type, reduce_only);

        const long long id = next_id++;
        if (type == "MARKET") {
            const double signed_qty = side == "BUY" ? qty : -qty;
            if (reduce_only) {
                positions.erase(symbol);   // test model: a reduce-only market flattens
            } else {
                positions[symbol] = Pos{ signed_qty, 100.0, 1.0 };
            }
            return json{ { "orderId", id }, { "status", "FILLED" } };
        }
        OO o;
        o.symbol = symbol;
        o.side = side;
        o.type = type;
        o.client_id = P::param(req, "newClientOrderId").value_or("");
        o.reduce_only = reduce_only;
        o.qty = qty;
        o.price = std::stod(P::param(req, "price").value_or("0"));
        o.stop_price = std::stod(P::param(req, "stopPrice").value_or("0"));
        open[id] = o;
        return json{ { "orderId", id }, { "status", "NEW" } };
    }

    json cancel(const HttpRequest& req)
    {
        const long long id = std::stoll(test::FakeBinance::param(req, "orderId").value_or("0"));
        open.erase(id);
        return json{ { "orderId", id }, { "status", "CANCELED" } };
    }
};

struct Fixture
{
    MockExchange mock;
    test::FakeBinance fake;

    BinanceBroker make_broker()
    {
        fake.responder = [this](const HttpRequest& r, test::FakeBinance&) { return mock.serve(r); };
        return BinanceBroker{ test::test_config(), fake.transport(), [] { return 1700000000000LL; } };
    }
};

KLine bar(std::int64_t ms, const Symbol& sym)
{
    return KLine{ Timestamp::from_millis(ms), sym, 100.0, 100.0, 100.0, 100.0, 1.0 };
}

TEST(BinanceBroker, MapsCashEquityAndPosition)
{
    Fixture fx;
    fx.mock.available = 800.0;
    fx.mock.margin_balance = 1200.0;
    fx.mock.positions["BTCUSDT"] = { 0.5, 100.0, 3.0 };
    auto broker = fx.make_broker();

    broker.on_tick(bar(1, "BTCUSDT"));
    EXPECT_DOUBLE_EQ(broker.cash(), 800.0);
    EXPECT_DOUBLE_EQ(broker.equity(), 1200.0);

    const auto p = broker.position("BTCUSDT");
    ASSERT_TRUE(p.has_value());
    EXPECT_DOUBLE_EQ(p->quantity, 0.5);
    EXPECT_DOUBLE_EQ(p->price, 100.0);
    EXPECT_DOUBLE_EQ(p->leverage, 3.0);
    EXPECT_FALSE(broker.position("ETHUSDT").has_value());
}

TEST(BinanceBroker, DefersReduceOnlyChildOfRestingEntryThenArmsOnFill)
{
    Fixture fx;
    auto broker = fx.make_broker();
    broker.on_tick(bar(1, "BTCUSDT"));   // flat

    const OrderID entry = broker.place_order(
        StopOrderParams{ .symbol = "BTCUSDT", .side = OrderSide::Buy, .quantity = 0.01, .price = 105.0 });
    EXPECT_LT(entry, kSyntheticBase);    // a real Binance id (submitted)

    const OrderID sl = broker.place_order(
        StopOrderParams{ .symbol = "BTCUSDT", .side = OrderSide::Sell, .quantity = 0.01,
                         .price = 95.0, .reduce_only = true }, entry);
    EXPECT_GE(sl, kSyntheticBase);       // deferred -> synthetic id
    EXPECT_EQ(fx.mock.reduce_only_posts(), 0);   // nothing reduce-only was sent yet

    // The deferred child is still queryable as an open order.
    const auto ords = broker.orders();
    ASSERT_TRUE(ords.contains(sl));
    EXPECT_TRUE(ords.at(sl).reduce_only);
    EXPECT_EQ(ords.at(sl).parent_id, entry);

    // Entry fills: it leaves the open book and a position appears.
    fx.mock.open.erase(entry);
    fx.mock.positions["BTCUSDT"] = { 0.01, 105.0, 1.0 };

    broker.on_tick(bar(2, "BTCUSDT"));   // reconcile arms the child
    EXPECT_GE(fx.mock.reduce_only_posts(), 1);

    // The synthetic id is gone; a real reduce-only order now rests, linked to the entry.
    const auto ords2 = broker.orders();
    EXPECT_FALSE(ords2.contains(sl));
    bool found_linked = false;
    for (const auto& [id, o] : ords2) {
        if (o.reduce_only) {
            EXPECT_EQ(o.parent_id, entry);
            found_linked = true;
        }
    }
    EXPECT_TRUE(found_linked);
}

TEST(BinanceBroker, MarketEntryChildIsSubmittedImmediately)
{
    Fixture fx;
    auto broker = fx.make_broker();
    broker.on_tick(bar(1, "BTCUSDT"));   // flat

    const OrderID entry = broker.place_order(
        MarketOrderParams{ .symbol = "BTCUSDT", .side = OrderSide::Buy, .quantity = 0.01 });
    EXPECT_LT(entry, kSyntheticBase);

    // Position exists after the synchronous market fill, so the reduce-only child
    // is accepted right away — no deferral.
    const OrderID sl = broker.place_order(
        StopOrderParams{ .symbol = "BTCUSDT", .side = OrderSide::Sell, .quantity = 0.01,
                         .price = 95.0, .reduce_only = true }, entry);
    EXPECT_LT(sl, kSyntheticBase);
    EXPECT_EQ(fx.mock.reduce_only_posts(), 1);
}

TEST(BinanceBroker, CancelsOrphanedReduceOnlyOrderWhenFlat)
{
    Fixture fx;
    // A reduce-only stop rests but there is no position (a prior close orphaned it).
    fx.mock.open[500] = MockExchange::OO{ "BTCUSDT", "SELL", "STOP_MARKET", "stk-c-1-1",
                                          0.0, 95.0, 0.01, true };
    auto broker = fx.make_broker();

    broker.on_tick(bar(1, "BTCUSDT"));   // reconcile cleans it up
    EXPECT_FALSE(fx.mock.open.contains(500));
    EXPECT_TRUE(fx.fake.has_request("DELETE", "/fapi/v1/order"));
}

TEST(BinanceBroker, CancelHandlesSyntheticAndRealIds)
{
    Fixture fx;
    auto broker = fx.make_broker();
    broker.on_tick(bar(1, "BTCUSDT"));

    const OrderID entry = broker.place_order(
        StopOrderParams{ .symbol = "BTCUSDT", .side = OrderSide::Buy, .quantity = 0.01, .price = 105.0 });
    const OrderID sl = broker.place_order(
        StopOrderParams{ .symbol = "BTCUSDT", .side = OrderSide::Sell, .quantity = 0.01,
                         .price = 95.0, .reduce_only = true }, entry);

    // Cancelling a deferred child just forgets it (never hit the exchange).
    EXPECT_TRUE(broker.cancel_order(sl));
    EXPECT_FALSE(broker.orders().contains(sl));

    // Cancelling a real order issues a DELETE and removes it from the book.
    EXPECT_TRUE(broker.cancel_order(entry));
    EXPECT_FALSE(fx.mock.open.contains(entry));

    // Unknown id -> false.
    EXPECT_FALSE(broker.cancel_order(999999));
}

} // namespace
} // namespace stonks::binance
