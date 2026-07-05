add_executable(app_tests
    resultmap_test.cpp
    ${PROJECT_SOURCE_DIR}/app/src/resultmap.cpp
)

target_include_directories(app_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/app
)

target_link_libraries(app_tests PRIVATE
    stonks_core
    Qt6::Core
    GTest::gtest_main
)

gtest_discover_tests(app_tests)
