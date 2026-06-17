qt_standard_project_setup(REQUIRES 6.5)

qt_add_executable(app
    src/main.cpp
    src/backtest.cpp
)

qt_add_qml_module(app
    URI Stonks
    VERSION 1.0
    QML_FILES Main.qml
    SOURCES src/enginerunner.h
)

target_include_directories(app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(app PRIVATE
    stonks_core
    stonks_broker
    stonks_datafeed
    Qt6::Quick
)

if(STONKS_PYTHON)
    target_link_libraries(app PRIVATE stonks_python)
endif()
