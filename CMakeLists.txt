cmake_minimum_required(VERSION 3.20)

project(stonks
    VERSION 0.0.1
    DESCRIPTION "Versatile C++20 backtesting and live-trading system"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(STONKS_BUILD_TESTS "Build unit tests" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(cmake/Dependencies.cmake)

add_subdirectory(src/core)
add_subdirectory(apps/backtest_runner)

if(STONKS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
