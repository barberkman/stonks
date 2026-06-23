import QtQuick
import Stonks

// Large report metric: small mono label + 26px value, optional right divider.
Item {
    id: cell
    property string label: ""
    property string value: ""
    property color valueColor: Theme.textPrimary
    property bool rightBorder: true

    Rectangle {
        visible: cell.rightBorder
        anchors.right: parent.right
        width: 1
        height: parent.height
        color: Theme.rowSep
    }
    Column {
        anchors.left: parent.left
        anchors.leftMargin: 22
        anchors.top: parent.top
        anchors.topMargin: 18
        spacing: 9
        Text {
            text: cell.label
            color: Theme.t5
            font.family: Theme.mono
            font.weight: Font.Medium
            font.pixelSize: 10
            font.letterSpacing: 1
        }
        Text {
            text: cell.value
            color: cell.valueColor
            font.family: Theme.mono
            font.weight: Font.DemiBold
            font.pixelSize: 26
        }
    }
}
