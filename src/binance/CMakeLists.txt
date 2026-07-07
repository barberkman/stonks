add_library(stonks_binance STATIC
    signer.cpp
    restclient.cpp
    exchangeinfo.cpp
    binancebroker.cpp
    liveklinefeed.cpp
)

# nlohmann_json is PUBLIC: restclient.h returns nlohmann::json, so consumers
# (broker, feed, tests) need its include path. CURL/OpenSSL stay implementation
# details.
target_link_libraries(stonks_binance
    PUBLIC stonks_core nlohmann_json::nlohmann_json
    PRIVATE CURL::libcurl OpenSSL::Crypto
)
