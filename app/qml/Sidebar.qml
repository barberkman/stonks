import QtQuick
import Stonks

// Fixed-width sidebar: logo, nav, latest-run footer.
Rectangle {
    id: root
    width: Theme.sidebarW
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
            height: Theme.sp(67)
            Row {
                anchors.left: parent.left
                anchors.leftMargin: Theme.sp(18)
                anchors.verticalCenter: parent.verticalCenter
                spacing: Theme.sp(11)
                Rectangle {
                    width: Theme.sp(26)
                    height: Theme.sp(26)
                    radius: Theme.radiusControl
                    color: Theme.accent
                    Text {
                        anchors.centerIn: parent
                        text: "B"
                        color: Theme.accentInk
                        font.family: Theme.mono
                        font.weight: Font.Bold
                        font.pixelSize: Theme.fontSub
                    }
                }
                Column {
                    spacing: Theme.sp(2)
                    anchors.verticalCenter: parent.verticalCenter
                    Text {
                        text: "Backtester"
                        color: Theme.textPrimary
                        font.family: Theme.sans
                        font.weight: Font.DemiBold
                        font.pixelSize: Theme.fontBodyLg
                    }
                    Text {
                        text: "v0.9 · engine 2.3"
                        color: Theme.t6
                        font.family: Theme.mono
                        font.pixelSize: Theme.fontCaption
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

        Item { width: 1; height: Theme.sp(12) } // nav top padding

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
        height: Theme.sp(84)

        Rectangle {
            anchors.top: parent.top
            width: parent.width
            height: 1
            color: Theme.border
        }

        Column {
            anchors.fill: parent
            anchors.leftMargin: Theme.sp(18)
            anchors.rightMargin: Theme.sp(18)
            anchors.topMargin: Theme.sp(16)
            spacing: Theme.sp(6)

            Text {
                text: "LATEST RUN"
                color: Theme.t6
                font.family: Theme.mono
                font.weight: Font.Medium
                font.pixelSize: Theme.fontMicro
                font.letterSpacing: 1 * Theme.scale
            }
            Text {
                text: "#" + App.latestRun().id
                color: Theme.t1
                font.family: Theme.mono
                font.weight: Font.Medium
                font.pixelSize: Theme.fontSmall
            }
            Row {
                spacing: Theme.sp(7)
                Rectangle {
                    width: Theme.sp(6)
                    height: Theme.sp(6)
                    radius: Theme.sp(3)
                    anchors.verticalCenter: parent.verticalCenter
                    color: App.latestRun().status === "completed" ? Theme.positive : Theme.accent
                }
                Text {
                    text: (App.latestRun().status || "").toUpperCase()
                    color: App.latestRun().status === "completed" ? Theme.positive : Theme.accent
                    font.family: Theme.mono
                    font.weight: Font.Medium
                    font.pixelSize: Theme.fontCaption
                    anchors.verticalCenter: parent.verticalCenter
                }
            }
        }
    }
}
