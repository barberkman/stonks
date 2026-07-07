// BinanceRestClient request assembly + error handling, all via a fake transport.

#include <gtest/gtest.h>

#include <string>

#include "stonks/binance/restclient.h"
#include "fake_binance.h"

namespace stonks::binance {
namespace {

constexpr long long kFixedNow = 1700000000000LL;

BinanceRestClient make_client(test::FakeBinance& fake)
{
    return BinanceRestClient{ test::test_config(), fake.transport(), [] { return kFixedNow; } };
}

TEST(RestClient, SignedGetCarriesTimestampRecvWindowSignatureAndApiKey)
{
    test::FakeBinance fake;
    fake.responder = [](const HttpRequest&, test::FakeBinance&) {
        return HttpResponse{ 200, R"({"ok":true})" };
    };
    auto client = make_client(fake);

    client.signed_request("GET", "/fapi/v3/account", {});

    ASSERT_EQ(fake.requests.size(), 1u);
    const HttpRequest& req = fake.requests[0];
    EXPECT_EQ(req.method, "GET");
    EXPECT_EQ(test::FakeBinance::path_of(req), "/fapi/v3/account");
    EXPECT_EQ(test::FakeBinance::param(req, "timestamp").value_or(""), std::to_string(kFixedNow));
    EXPECT_EQ(test::FakeBinance::param(req, "recvWindow").value_or(""), "5000");
    EXPECT_FALSE(test::FakeBinance::param(req, "signature").value_or("").empty());

    bool has_key = false;
    for (const auto& [k, v] : req.headers) {
        if (k == "X-MBX-APIKEY" && v == "TESTKEY") { has_key = true; }
    }
    EXPECT_TRUE(has_key);
}

TEST(RestClient, SignedPostPutsParamsInBodyNotUrl)
{
    test::FakeBinance fake;
    fake.responder = [](const HttpRequest&, test::FakeBinance&) {
        return HttpResponse{ 200, R"({"orderId":42})" };
    };
    auto client = make_client(fake);

    client.signed_request("POST", "/fapi/v1/order",
                          { { "symbol", "BTCUSDT" }, { "side", "BUY" } });

    const HttpRequest& req = fake.requests.at(0);
    EXPECT_EQ(req.method, "POST");
    EXPECT_EQ(req.url.find('?'), std::string::npos);            // no query string
    EXPECT_NE(req.body.find("symbol=BTCUSDT"), std::string::npos);
    EXPECT_NE(req.body.find("signature="), std::string::npos);
    bool form = false;
    for (const auto& [k, v] : req.headers) {
        if (k == "Content-Type" && v == "application/x-www-form-urlencoded") { form = true; }
    }
    EXPECT_TRUE(form);
}

TEST(RestClient, PublicRequestIsUnsigned)
{
    test::FakeBinance fake;
    fake.responder = [](const HttpRequest&, test::FakeBinance&) {
        return HttpResponse{ 200, "[]" };
    };
    auto client = make_client(fake);

    client.public_request("GET", "/fapi/v1/klines",
                          { { "symbol", "BTCUSDT" }, { "interval", "1m" } });

    const HttpRequest& req = fake.requests.at(0);
    EXPECT_FALSE(test::FakeBinance::param(req, "signature").has_value());
    EXPECT_FALSE(test::FakeBinance::param(req, "timestamp").has_value());
    EXPECT_EQ(test::FakeBinance::param(req, "interval").value_or(""), "1m");
}

TEST(RestClient, ThrowsBinanceApiErrorOnErrorBody)
{
    test::FakeBinance fake;
    fake.responder = [](const HttpRequest&, test::FakeBinance&) {
        return HttpResponse{ 400, R"({"code":-2019,"msg":"Margin is insufficient."})" };
    };
    auto client = make_client(fake);

    try {
        client.signed_request("POST", "/fapi/v1/order", { { "symbol", "BTCUSDT" } });
        FAIL() << "expected BinanceApiError";
    } catch (const BinanceApiError& e) {
        EXPECT_EQ(e.code, -2019);
        EXPECT_EQ(e.http_status, 400);
    }
}

TEST(RestClient, ThrowsOnNegativeCodeEvenWithHttp200)
{
    test::FakeBinance fake;
    fake.responder = [](const HttpRequest&, test::FakeBinance&) {
        return HttpResponse{ 200, R"({"code":-1102,"msg":"Mandatory parameter was not sent."})" };
    };
    auto client = make_client(fake);
    EXPECT_THROW(client.signed_request("GET", "/fapi/v3/account", {}), BinanceApiError);
}

} // namespace
} // namespace stonks::binance
