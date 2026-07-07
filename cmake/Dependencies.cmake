include(FetchContent)

find_package(Qt6 REQUIRED COMPONENTS Quick)

# Live trading (BinanceBroker): libcurl for HTTP, OpenSSL libcrypto for Ed25519
# request signing. Both are system packages — on Debian-family systems install
# `libcurl4-openssl-dev libssl-dev`.
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

if(STONKS_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

if(STONKS_PYTHON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Embed)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()
