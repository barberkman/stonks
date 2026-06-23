import QtQuick
import Stonks

// Bordered, transparent button (e.g. "‹ Backtests" back button).
Rectangle {
    id: ctrl
    property alias text: label.text
    property int hpad: 13
    property int fontPx: 12
    signal clicked()

    implicitWidth: label.implicitWidth + hpad * 2
    implicitHeight: 30
    radius: 5
    color: ma.containsMouse ? Theme.input : "transparent"
    border.color: Theme.border
    border.width: 1

    Text {
        id: label
        anchors.centerIn: parent
        color: Theme.t3
        font.family: Theme.mono
        font.weight: Font.Medium
        font.pixelSize: ctrl.fontPx
    }

    MouseArea {
        id: ma
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onClicked: ctrl.clicked()
    }
}
