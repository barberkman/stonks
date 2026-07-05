import QtQuick
import Stonks

// Top bar: section label (left) + status dot/label (right).
Rectangle {
    height: Theme.headerH
    color: Theme.panel

    Rectangle {
        anchors.bottom: parent.bottom
        width: parent.width
        height: 1
        color: Theme.border
    }

    Text {
        anchors.verticalCenter: parent.verticalCenter
        anchors.left: parent.left
        anchors.leftMargin: Theme.sp(24)
        width: parent.width - Theme.sp(320)
        elide: Text.ElideRight
        text: App.sectionLabel()
        color: Theme.t4
        font.family: Theme.mono
        font.weight: Font.DemiBold
        font.pixelSize: Theme.fontBody
    }

    Row {
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
