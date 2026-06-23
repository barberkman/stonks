import QtQuick
import Stonks

// Sidebar nav item: left accent bar + mono label, active/idle colors.
Item {
    id: ctrl
    property string label
    property bool active: false
    signal clicked()

    width: parent ? parent.width : 200
    height: 38

    Rectangle {
        anchors.left: parent.left
        width: 2
        height: parent.height
        color: ctrl.active ? Theme.accent : "transparent"
    }

    Text {
        anchors.verticalCenter: parent.verticalCenter
        anchors.left: parent.left
        anchors.leftMargin: 17
        text: ctrl.label
        color: ctrl.active ? Theme.textPrimary : (hover.hovered ? Theme.t3 : Theme.t5)
        font.family: Theme.mono
        font.weight: Font.Medium
        font.pixelSize: 13
    }

    HoverHandler { id: hover }
    MouseArea {
        anchors.fill: parent
        cursorShape: Qt.PointingHandCursor
        onClicked: ctrl.clicked()
    }
}
