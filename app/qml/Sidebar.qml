import QtQuick
import Stonks

// 212px fixed sidebar: logo, nav, latest-run footer.
Rectangle {
    id: root
    width: 212
    color: Theme.panel

    // right divider
    Rectangle {
        anchors.right: parent.right
        width: 1
        height: parent.height
        color: Theme.border
    }

    Column {
        id: top
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        spacing: 0

        // logo block
        Item {
            width: parent.width
            height: 67
            Row {
                anchors.left: parent.left
                anchors.leftMargin: 18
                anchors.verticalCenter: parent.verticalCenter
                spacing: 11
                Rectangle {
                    width: 26
                    height: 26
                    radius: 5
                    color: Theme.accent
                    Text {
                        anchors.centerIn: parent
                        text: "B"
                        color: Theme.accentInk
                        font.family: Theme.mono
                        font.weight: Font.Bold
                        font.pixelSize: 15
                    }
                }
                Column {
                    spacing: 2
                    anchors.verticalCenter: parent.verticalCenter
                    Text {
                        text: "Backtester"
                        color: Theme.textPrimary
                        font.family: Theme.sans
                        font.weight: Font.DemiBold
                        font.pixelSize: 14
                    }
                    Text {
                        text: "v0.9 · engine 2.3"
                        color: Theme.t6
                        font.family: Theme.mono
                        font.pixelSize: 11
                    }
                }
            }
            Rectangle {
                anchors.bottom: parent.bottom
                width: parent.width
                height: 1
                color: Theme.border
            }
        }

        Item { width: 1; height: 12 } // nav top padding

        NavButton {
            label: "Backtests"
            active: App.view === "backtests" || App.view === "detail" || App.view === "setup" || App.view === "running"
            onClicked: App.go("backtests")
        }
        NavButton {
            label: "Logs"
            active: App.view === "logs"
            onClicked: App.go("logs")
        }
    }

    // latest-run footer
    Item {
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        height: 84

        Rectangle {
            anchors.top: parent.top
            width: parent.width
            height: 1
            color: Theme.border
        }

        Column {
            anchors.fill: parent
            anchors.leftMargin: 18
            anchors.rightMargin: 18
            anchors.topMargin: 16
            spacing: 6

            Text {
                text: "LATEST RUN"
                color: Theme.t6
                font.family: Theme.mono
                font.weight: Font.Medium
                font.pixelSize: 10
                font.letterSpacing: 1
            }
            Text {
                text: "#" + App.latestRun().id
                color: Theme.t1
                font.family: Theme.mono
                font.weight: Font.Medium
                font.pixelSize: 12
            }
            Row {
                spacing: 7
                Rectangle {
                    width: 6
                    height: 6
                    radius: 3
                    anchors.verticalCenter: parent.verticalCenter
                    color: App.latestRun().status === "completed" ? Theme.positive : Theme.accent
                }
                Text {
                    text: (App.latestRun().status || "").toUpperCase()
                    color: App.latestRun().status === "completed" ? Theme.positive : Theme.accent
                    font.family: Theme.mono
                    font.weight: Font.Medium
                    font.pixelSize: 11
                    anchors.verticalCenter: parent.verticalCenter
                }
            }
        }
    }
}
