#include "stonks/binance/restclient.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

#include <curl/curl.h>

namespace stonks::binance {

namespace {

std::size_t write_cb(char* ptr, std::size_t size, std::size_t nmemb, void* userdata)
{
    const std::size_t n = size * nmemb;
    static_cast<std::string*>(userdata)->append(ptr, n);
    return n;
}

// libcurl's global init is not thread-safe; run it exactly once before any
// easy handle is created.
void ensure_curl_global_init()
{
    static std::once_flag flag;
    std::call_once(flag, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

const char* getenv_or_empty(const char* name)
{
    const char* v = std::getenv(name);
    return v ? v : "";
}

} // namespace

Transport curl_transport()
{
    ensure_curl_global_init();
    return [](const HttpRequest& req) -> HttpResponse {
        CURL* curl = curl_easy_init();
        if (!curl) { throw std::runtime_error{ "curl_easy_init failed" }; }

        std::string body;
        curl_slist* headers = nullptr;
        for (const auto& [k, v] : req.headers) {
            headers = curl_slist_append(headers, (k + ": " + v).c_str());
        }

        curl_easy_setopt(curl, CURLOPT_URL, req.url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);

        if (req.method == "POST") {
            curl_easy_setopt(curl, CURLOPT_POST, 1L);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, req.body.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(req.body.size()));
        } else if (req.method == "DELETE") {
            curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
        }

        const CURLcode res = curl_easy_perform(curl);
        long status = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            throw std::runtime_error{ std::string{ "curl transport error: " } + curl_easy_strerror(res) };
        }
        return HttpResponse{ status, std::move(body) };
    };
}

BinanceConfig BinanceConfig::from_env(bool testnet)
{
    BinanceConfig cfg;
    cfg.api_key = getenv_or_empty("BINANCE_API_KEY");
    if (cfg.api_key.empty()) {
        throw std::runtime_error{ "BINANCE_API_KEY is not set" };
    }

    const std::string key = getenv_or_empty("BINANCE_PRIVATE_KEY_PEM");
    if (key.empty()) {
        throw std::runtime_error{ "BINANCE_PRIVATE_KEY_PEM is not set (inline PEM or a path)" };
    }
    if (key.rfind("-----BEGIN", 0) == 0) {
        cfg.ed25519_pem = key;                       // inline PEM text
    } else {
        std::ifstream f{ key };                      // a path to the PEM file
        if (!f) { throw std::runtime_error{ "cannot read BINANCE_PRIVATE_KEY_PEM file: " + key }; }
        std::stringstream ss;
        ss << f.rdbuf();
        cfg.ed25519_pem = ss.str();
    }

    const std::string base = getenv_or_empty("BINANCE_BASE_URL");
    cfg.base_url = !base.empty() ? base : (testnet ? testnet_url : mainnet_url);
    return cfg;
}

BinanceRestClient::BinanceRestClient(BinanceConfig config, Transport transport,
                                     std::function<long long()> now_ms)
: m_config{ std::move(config) },
  m_transport{ std::move(transport) },
  m_now_ms{ now_ms ? std::move(now_ms)
                   : [] {
                        return std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                            .count();
                     } },
  m_signer{ m_config.ed25519_pem }
{}

nlohmann::json BinanceRestClient::send_and_parse(const HttpRequest& req)
{
    const HttpResponse resp = m_transport(req);

    nlohmann::json j;
    if (!resp.body.empty()) {
        j = nlohmann::json::parse(resp.body, nullptr, /*allow_exceptions=*/false);
    }

    const bool http_ok = resp.status >= 200 && resp.status < 300;
    // Binance signals errors either via HTTP status or a {"code":-x,"msg":...}
    // body (occasionally returned with a 200). Treat a negative code as an error.
    const bool error_body = j.is_object() && j.contains("code") && j.contains("msg")
                            && j["code"].is_number_integer() && j["code"].get<int>() < 0;

    if (!http_ok || error_body) {
        const int code = error_body ? j["code"].get<int>() : 0;
        const std::string msg = error_body ? j["msg"].get<std::string>()
                                            : ("HTTP " + std::to_string(resp.status) + ": " + resp.body);
        throw BinanceApiError{ code, resp.status, msg };
    }
    if (j.is_discarded()) {
        throw BinanceApiError{ 0, resp.status, "unparseable response body: " + resp.body };
    }
    return j;
}

nlohmann::json BinanceRestClient::signed_request(const std::string& method,
                                                 const std::string& path,
                                                 QueryParams params)
{
    params.emplace_back("recvWindow", std::to_string(m_config.recv_window_ms));
    params.emplace_back("timestamp", std::to_string(m_now_ms()));

    const std::string payload = encode_query(params);
    const std::string signature = m_signer.sign(payload);
    const std::string full = payload + "&signature=" + url_encode(signature);

    HttpRequest req;
    req.method = method;
    req.headers.emplace_back("X-MBX-APIKEY", m_config.api_key);
    if (method == "POST") {
        req.url = m_config.base_url + path;
        req.body = full;
        req.headers.emplace_back("Content-Type", "application/x-www-form-urlencoded");
    } else {
        req.url = m_config.base_url + path + "?" + full;
    }
    return send_and_parse(req);
}

nlohmann::json BinanceRestClient::public_request(const std::string& method,
                                                 const std::string& path,
                                                 const QueryParams& params)
{
    const std::string payload = encode_query(params);
    HttpRequest req;
    req.method = method;
    req.url = payload.empty() ? (m_config.base_url + path)
                              : (m_config.base_url + path + "?" + payload);
    if (!m_config.api_key.empty()) {
        req.headers.emplace_back("X-MBX-APIKEY", m_config.api_key);
    }
    return send_and_parse(req);
}

} // namespace stonks::binance
