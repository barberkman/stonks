import QtQuick
import Stonks

// Filled accent button (e.g. "+ New backtest", "▶ Start backtest").
Rectangle {
    id: ctrl
    property alias text: label.text
    property int hpad: Theme.sp(20)
    property int fontPx: Theme.fontBody
    property int fontWeight: Font.DemiBold
    signal clicked()

    implicitWidth: label.implicitWidth + hpad * 2
    implicitHeight: Theme.controlH
    radius: Theme.sp(6)
    color: ma.pressed ? Qt.darker(Theme.accent, 1.12) : (ma.containsMouse ? Qt.lighter(Theme.accent, 1.06) : Theme.accent)

    Text {
        id: label
        anchors.centerIn: parent
        color: Theme.accentInk
        font.family: Theme.sans
        font.weight: ctrl.fontWeight
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
