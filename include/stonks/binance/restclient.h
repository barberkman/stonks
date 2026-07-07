#pragma once

#include <functional>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "stonks/binance/signer.h"

namespace stonks::binance {

// One HTTP request as handed to the transport. `url` already carries the query
// string for GET/DELETE; `body` carries the form-encoded params for POST. The
// transport only executes — all Binance-specific assembly happens above it.
struct HttpRequest
{
    std::string method;   // "GET" / "POST" / "DELETE"
    std::string url;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;
};

struct HttpResponse
{
    long status = 0;      // HTTP status code
    std::string body;     // response body (JSON text)
};

// The seam that makes the client testable: a fake transport records requests and
// returns canned responses, so every layer above runs with no network. The
// default (curl_transport) performs a real libcurl call.
using Transport = std::function<HttpResponse(const HttpRequest&)>;

// A blocking libcurl-backed transport. Throws std::runtime_error on a transport
// failure (DNS, connection, timeout); HTTP error *statuses* are returned so the
// caller can read Binance's {code,msg} error body.
Transport curl_transport();

// Thrown when Binance returns an error status or an error payload ({code,msg}).
struct BinanceApiError : std::runtime_error
{
    int code;   // Binance error code (e.g. -2019 margin insufficient), or 0
    long http_status;
    BinanceApiError(int code_, long http, const std::string& msg)
    : std::runtime_error{ msg }, code{ code_ }, http_status{ http }
    {}
};

struct BinanceConfig
{
    std::string api_key;
    std::string ed25519_pem;   // PEM text of the Ed25519 private key
    std::string base_url;      // no trailing slash
    int recv_window_ms = 5000;
    bool dry_run = false;      // consumed by the broker: log mutating calls, don't send

    // Default REST endpoints. The testnet host is also overridable via
    // BINANCE_BASE_URL for when Binance rotates it.
    static constexpr const char* mainnet_url = "https://fapi.binance.com";
    static constexpr const char* testnet_url = "https://testnet.binancefuture.com";

    // Build from environment: BINANCE_API_KEY, BINANCE_PRIVATE_KEY_PEM (inline
    // PEM text if it starts with "-----BEGIN", else a path to the PEM file),
    // optional BINANCE_BASE_URL. `testnet` selects the default host. Throws if a
    // required variable is missing or the key file cannot be read.
    static BinanceConfig from_env(bool testnet);
};

// Thin Binance REST client: signs SIGNED requests (timestamp + recvWindow +
// Ed25519 signature, X-MBX-APIKEY header) and parses JSON. Holds no account
// state — it is a stateless request layer the broker calls each tick.
class BinanceRestClient
{
public:
    // `now_ms` supplies the request timestamp; injectable so tests can assert an
    // exact signed payload. Defaults to the system clock.
    explicit BinanceRestClient(BinanceConfig config,
                               Transport transport = curl_transport(),
                               std::function<long long()> now_ms = {});

    // SIGNED endpoint: appends timestamp+recvWindow+signature. `params` are the
    // business parameters only. Throws BinanceApiError on a non-2xx / error body.
    nlohmann::json signed_request(const std::string& method,
                                  const std::string& path,
                                  QueryParams params);

    // Public (MARKET_DATA / NONE) endpoint: no signature. Used for exchangeInfo
    // and klines.
    nlohmann::json public_request(const std::string& method,
                                  const std::string& path,
                                  const QueryParams& params);

    const BinanceConfig& config() const { return m_config; }

private:
    nlohmann::json send_and_parse(const HttpRequest& req);

    BinanceConfig m_config;
    Transport m_transport;
    std::function<long long()> m_now_ms;
    Ed25519Signer m_signer;
};

} // namespace stonks::binance
