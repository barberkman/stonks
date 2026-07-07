#pragma once

// Shared test scaffolding for the Binance module: an injectable fake transport
// that records requests and returns scripted responses, plus a throwaway
// Ed25519 key so the signer can be exercised with no real credentials.

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "stonks/binance/restclient.h"

namespace stonks::binance::test {

// A test-only Ed25519 private key (PKCS#8 PEM). Never used against real Binance.
inline constexpr const char* kTestPem =
    "-----BEGIN PRIVATE KEY-----\n"
    "MC4CAQAwBQYDK2VwBCIEIIBnOxdaUOmdlxsAVC1TToaaF7cCHAdE5yDwfsbR6YUe\n"
    "-----END PRIVATE KEY-----\n";

// Records every request and answers via a responder callback. Kept alive by the
// test; the Transport it hands out captures `this`.
struct FakeBinance
{
    std::vector<HttpRequest> requests;
    std::function<HttpResponse(const HttpRequest&, FakeBinance&)> responder;

    Transport transport()
    {
        return [this](const HttpRequest& req) -> HttpResponse {
            requests.push_back(req);
            return responder ? responder(req, *this)
                             : HttpResponse{ 200, "{}" };
        };
    }

    // The path portion of a request URL (everything before '?').
    static std::string path_of(const HttpRequest& req)
    {
        const auto q = req.url.find('?');
        const auto scheme = req.url.find("://");
        const std::size_t host_start = scheme == std::string::npos ? 0 : scheme + 3;
        const auto slash = req.url.find('/', host_start);
        if (slash == std::string::npos) { return {}; }
        const std::size_t end = q == std::string::npos ? req.url.size() : q;
        return req.url.substr(slash, end - slash);
    }

    // Look up a query/body parameter's value (searches URL query then POST body).
    static std::optional<std::string> param(const HttpRequest& req, const std::string& key)
    {
        auto find_in = [&key](const std::string& s) -> std::optional<std::string> {
            const std::string needle = key + "=";
            std::size_t pos = 0;
            while (pos < s.size()) {
                const std::size_t amp = s.find('&', pos);
                const std::size_t end = amp == std::string::npos ? s.size() : amp;
                const std::string kv = s.substr(pos, end - pos);
                if (kv.rfind(needle, 0) == 0) { return kv.substr(needle.size()); }
                if (amp == std::string::npos) { break; }
                pos = amp + 1;
            }
            return std::nullopt;
        };
        const auto q = req.url.find('?');
        if (q != std::string::npos) {
            if (auto v = find_in(req.url.substr(q + 1))) { return v; }
        }
        return find_in(req.body);
    }

    bool has_request(const std::string& method, const std::string& path) const
    {
        for (const auto& r : requests) {
            if (r.method == method && path_of(r) == path) { return true; }
        }
        return false;
    }

    int count(const std::string& method, const std::string& path) const
    {
        int n = 0;
        for (const auto& r : requests) {
            if (r.method == method && path_of(r) == path) { ++n; }
        }
        return n;
    }
};

// A minimal config wired to the test key; base_url is irrelevant under a fake
// transport but must be non-empty.
inline BinanceConfig test_config()
{
    BinanceConfig cfg;
    cfg.api_key = "TESTKEY";
    cfg.ed25519_pem = kTestPem;
    cfg.base_url = "https://testnet.example";
    cfg.recv_window_ms = 5000;
    return cfg;
}

} // namespace stonks::binance::test
