import QtQuick
import Stonks

// Top bar: nav (left) + section label + status dot/label (right).
Rectangle {
    height: Theme.headerH
    color: Theme.panel

    Rectangle {
        anchors.bottom: parent.bottom
        width: parent.width
        height: 1
        color: Theme.border
    }

    // nav — the app's only view switch (Backtests group vs. Logs)
    Row {
        id: nav
        anchors.verticalCenter: parent.verticalCenter
        anchors.left: parent.left
        anchors.leftMargin: Theme.sp(24)
        spacing: Theme.sp(20)

        Text {
            id: navBacktests
            property bool active: App.view === "backtests" || App.view === "detail"
                || App.view === "setup" || App.view === "running"
            text: "Backtests"
            color: active ? Theme.textPrimary : (hoverB.hovered ? Theme.t3 : Theme.t5)
            font.family: Theme.mono
            font.weight: Font.Medium
            font.pixelSize: Theme.fontBody
            HoverHandler { id: hoverB }
            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: App.go("backtests")
            }
        }
        Text {
            id: navLogs
            property bool active: App.view === "logs"
            text: "Logs"
            color: active ? Theme.textPrimary : (hoverL.hovered ? Theme.t3 : Theme.t5)
            font.family: Theme.mono
            font.weight: Font.Medium
            font.pixelSize: Theme.fontBody
            HoverHandler { id: hoverL }
            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: App.go("logs")
            }
        }
    }

    Text {
        anchors.verticalCenter: parent.verticalCenter
        anchors.left: nav.right
        anchors.leftMargin: Theme.sp(20)
        anchors.right: status.left
        anchors.rightMargin: Theme.sp(16)
        elide: Text.ElideRight
        text: App.sectionLabel()
        color: Theme.t4
        font.family: Theme.mono
        font.weight: Font.DemiBold
        font.pixelSize: Theme.fontBody
    }

    Row {
        id: status
        anchors.verticalCenter: parent.verticalCenter
        anchors.right: parent.right
        anchors.rightMargin: Theme.sp(24)
        spacing: Theme.sp(9)

        Rectangle {
            width: Theme.sp(6)
            height: Theme.sp(6)
            radius: Theme.sp(3)
            anchors.verticalCenter: parent.verticalCenter
            color: App.view === "running" ? Theme.accent : Theme.positive
        }
        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: App.topRight()
            color: Theme.t5
            font.family: Theme.mono
            font.weight: Font.Medium
            font.pixelSize: Theme.fontSmall
        }
    }
}
