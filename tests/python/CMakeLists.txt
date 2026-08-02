add_executable(python_tests
    contextadapter_test.cpp
    embeddedpython_test.cpp
    pythonstrategy_test.cpp
    strategydiscovery_test.cpp
    ${PROJECT_SOURCE_DIR}/app/strategies/strategydiscovery.cpp
)

target_include_directories(python_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/tests
    ${PROJECT_SOURCE_DIR}/app
)

target_link_libraries(python_tests PRIVATE
    stonks_core
    stonks_broker
    stonks_python
    GTest::gtest_main
)

# Fixtures live in tests/python/fixtures; the Python package this build emits
# is in python/. Tests inject both into sys.path so the embedded interpreter
# can `import stonks` and `import fixturestrats`.
target_compile_definitions(python_tests PRIVATE
    STONKS_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    STONKS_PYTHON_PACKAGE_DIR="${PROJECT_SOURCE_DIR}/python"
    STONKS_VENV_DIR="${PROJECT_SOURCE_DIR}/app/python/.venv"
)

gtest_discover_tests(python_tests)
