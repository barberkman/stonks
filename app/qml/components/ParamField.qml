import QtQuick
import Stonks

// Labeled number input with a unit suffix (Strategy Parameters grid).
Item {
    id: root
    property string label: ""
    property string unit: ""
    property string value: ""
    signal edited(string v)
    implicitHeight: Theme.sp(58)

    Column {
        width: parent.width
        spacing: Theme.sp(7)
        Text {
            text: root.label
            color: Theme.t4
            font.family: Theme.sans
            font.pixelSize: Theme.fontCaption
        }
        Rectangle {
            width: parent.width
            height: Theme.controlH
            radius: Theme.radiusControl
            color: Theme.input
            border.color: "#3c3c3c"
            border.width: 1

            TextInput {
                id: ti
                anchors.left: parent.left
                anchors.leftMargin: Theme.sp(11)
                anchors.right: unitT.left
                anchors.rightMargin: Theme.sp(6)
                anchors.verticalCenter: parent.verticalCenter
                text: root.value
                color: Theme.textPrimary
                font.family: Theme.mono
                font.pixelSize: Theme.fontBodyLg
                selectByMouse: true
                clip: true
                onEditingFinished: root.edited(text)
            }
            Text {
                id: unitT
                anchors.right: parent.right
                anchors.rightMargin: Theme.sp(11)
                anchors.verticalCenter: parent.verticalCenter
                text: root.unit
                color: Theme.t6
                font.family: Theme.mono
                font.pixelSize: Theme.fontCaption
            }
        }
    }
}
