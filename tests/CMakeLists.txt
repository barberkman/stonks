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

if(STONKS_PYTHON)
    add_subdirectory(python)
endif()
