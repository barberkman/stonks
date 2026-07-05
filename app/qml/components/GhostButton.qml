import QtQuick
import Stonks

// Bordered, transparent button (e.g. "‹ Backtests" back button).
Rectangle {
    id: ctrl
    property alias text: label.text
    property int hpad: Theme.sp(13)
    property int fontPx: Theme.fontSmall
    signal clicked()

    implicitWidth: label.implicitWidth + hpad * 2
    implicitHeight: Theme.controlHSm
    radius: Theme.radiusControl
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
