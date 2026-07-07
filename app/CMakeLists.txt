qt_standard_project_setup(REQUIRES 6.5)

qt_add_executable(app
    src/main.cpp
    src/backtestcontroller.cpp
    src/backtestworker.cpp
    src/resultmap.cpp
    strategies/strategydiscovery.cpp
)

set_source_files_properties(qml/Theme.qml qml/App.qml PROPERTIES
    QT_QML_SINGLETON_TYPE TRUE
)

qt_add_qml_module(app
    URI Stonks
    VERSION 1.0
    QML_FILES
        main.qml
        qml/Theme.qml
        qml/App.qml
        qml/Header.qml
        qml/components/AccentButton.qml
        qml/components/GhostButton.qml
        qml/components/Page.qml
        qml/components/MetricCell.qml
        qml/components/StatCell.qml
        qml/components/EquityChart.qml
        qml/components/DrawdownChart.qml
        qml/components/Sparkline.qml
        qml/components/CandleChart.qml
        qml/components/LineCrosshair.qml
        qml/components/StyledSelect.qml
        qml/components/ParamField.qml
        qml/components/Spinner.qml
        qml/components/ThinScrollBar.qml
        qml/views/BacktestsView.qml
        qml/views/LogsView.qml
        qml/views/DetailView.qml
        qml/views/ReportView.qml
        qml/views/TradesView.qml
        qml/views/SetupView.qml
        qml/views/RunningView.qml
        qml/js/format.js
        qml/js/chartview.js
        qml/js/mockdata.js
)

target_include_directories(app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Qt's lowercase `slots`/`signals`/`emit` keyword macros collide with CPython's
# object.h (`PyType_Slot *slots;`) once pythonstrategy.h drags pybind11/Python.h
# into this target. Disable them; the app uses only the Q_OBJECT/Q_INVOKABLE forms.
target_compile_definitions(app PRIVATE QT_NO_KEYWORDS)

target_link_libraries(app PRIVATE
    stonks_core
    stonks_broker
    stonks_datafeed
    stonks_binance
    Qt6::Quick
    nlohmann_json::nlohmann_json
)

if(STONKS_PYTHON)
    target_link_libraries(app PRIVATE stonks_python)
endif()
