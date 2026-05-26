include(GoogleTest)

add_executable(core_tests
    core/types_test.cpp
    core/lookahead_test.cpp
    core/determinism_test.cpp
    core/order_factory_test.cpp
    core/backtestbroker_test.cpp
    core/scenarios_test.cpp
)

target_link_libraries(core_tests PRIVATE
    stonks_core
    stonks_broker
    GTest::gtest_main
)

gtest_discover_tests(core_tests)
