import QtQuick
import QtQuick.Controls
import QtQuick.Window
import Stonks

ApplicationWindow {
    id: window
    visible: true
    visibility: Window.Maximized
    width: 1440
    height: 900
    minimumWidth: 1120
    minimumHeight: 680
    title: "Backtester"
    color: Theme.bg

    // feed the window size into the Theme singleton's responsive scale
    Binding { target: Theme; property: "windowWidth"; value: window.width }
    Binding { target: Theme; property: "windowHeight"; value: window.height }

    Column {
        anchors.fill: parent

        Header {
            id: header
            width: parent.width
        }

        // content area — swapped by App.view
        Item {
            width: parent.width
            height: parent.height - header.height

            Loader {
                anchors.fill: parent
                sourceComponent: {
                    switch (App.view) {
                    case "backtests": return cBacktests
                    case "setup": return cSetup
                    case "running": return cRunning
                    case "detail": return cDetail
                    case "logs": return cLogs
                    }
                    return cBacktests
                }
            }
            Component { id: cBacktests; BacktestsView {} }
            Component { id: cSetup; SetupView {} }
            Component { id: cRunning; RunningView {} }
            Component { id: cDetail; DetailView {} }
            Component { id: cLogs; LogsView {} }
        }
    }
}
