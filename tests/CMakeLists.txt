include(GoogleTest)

add_executable(core_tests
    core/types_test.cpp
)

target_link_libraries(core_tests PRIVATE
    stonks_core
    GTest::gtest_main
)

gtest_discover_tests(core_tests)
