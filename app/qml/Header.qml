import QtQuick
import Stonks

// Top bar: section label (left) + status dot/label (right).
Rectangle {
    height: 56
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
        anchors.leftMargin: 24
        width: parent.width - 320
        elide: Text.ElideRight
        text: App.sectionLabel()
        color: Theme.t4
        font.family: Theme.mono
        font.weight: Font.DemiBold
        font.pixelSize: 13
    }

    Row {
        anchors.verticalCenter: parent.verticalCenter
        anchors.right: parent.right
        anchors.rightMargin: 24
        spacing: 9

        Rectangle {
            width: 6
            height: 6
            radius: 3
            anchors.verticalCenter: parent.verticalCenter
            color: App.view === "running" ? Theme.accent : Theme.positive
        }
        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: App.topRight()
            color: Theme.t5
            font.family: Theme.mono
            font.weight: Font.Medium
            font.pixelSize: 12
        }
    }
}
