include(GoogleTest)

add_executable(core_tests
    core/types_test.cpp
    core/lookahead_test.cpp
    core/determinism_test.cpp
    core/order_factory_test.cpp
    core/backtestbroker_test.cpp
    core/scenarios_test.cpp
    core/progressbar_test.cpp
    core/ema50strategy_test.cpp
    core/qm_strategies_test.cpp
    core/report_test.cpp
)

target_include_directories(core_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/app
)

target_link_libraries(core_tests PRIVATE
    stonks_core
    stonks_broker
    GTest::gtest_main
)

gtest_discover_tests(core_tests)

# Datafeed tests link the real KLineFeed (and thus Arrow), kept separate so the
# stub-only core_tests stay free of that dependency.
add_executable(datafeed_tests
    core/klinefeed_test.cpp
)

target_link_libraries(datafeed_tests PRIVATE
    stonks_core
    stonks_datafeed
    GTest::gtest_main
)

gtest_discover_tests(datafeed_tests)

if(STONKS_PYTHON)
    add_subdirectory(python)
endif()
