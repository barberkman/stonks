import QtQuick
import Stonks

// Diagnostic log for the latest run.
Page {
    id: view
    maxWidth: 100000
    sidePad: Theme.sp(24)
    topPad: Theme.sp(18)
    bottomPad: Theme.sp(40)
    spacing: Theme.sp(14)

    // header row: label + filter chips
    Item {
        width: view.innerWidth
        height: Theme.sp(24)

        Text {
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
            text: "DIAGNOSTIC LOG · RUN #" + App.latestRun().id
            color: Theme.t4
            font.family: Theme.mono
            font.weight: Font.Medium
            font.pixelSize: Theme.fontCaption
            font.letterSpacing: 1 * Theme.scale
        }
        Row {
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            spacing: Theme.sp(6)
            Repeater {
                model: [{ t: "ALL", on: true }, { t: "WARN", on: false }, { t: "ERROR", on: false }]
                delegate: Rectangle {
                    required property var modelData
                    width: chipText.implicitWidth + Theme.sp(20)
                    height: Theme.sp(22)
                    radius: Theme.sp(4)
                    color: modelData.on ? Theme.input : "transparent"
                    Text {
                        id: chipText
                        anchors.centerIn: parent
                        text: modelData.t
                        color: modelData.on ? Theme.t4 : Theme.t6
                        font.family: Theme.mono
                        font.weight: Font.Medium
                        font.pixelSize: Theme.fontCaption
                    }
                }
            }
        }
    }

    // log box
    Rectangle {
        width: view.innerWidth
        radius: Theme.radiusCard
        color: Theme.panel
        border.color: Theme.border
        border.width: 1
        implicitHeight: logCol.implicitHeight + Theme.sp(28)

        Column {
            id: logCol
            y: Theme.sp(14)
            width: parent.width

            Repeater {
                model: App.logs()
                delegate: Item {
                    required property var modelData
                    width: logCol.width
                    height: Theme.sp(23)
                    Row {
                        x: Theme.sp(18)
                        anchors.verticalCenter: parent.verticalCenter
                        width: parent.width - Theme.sp(36)
                        spacing: Theme.sp(14)
                        Text { width: Theme.sp(96); text: modelData.t; color: Theme.t7; font.family: Theme.mono; font.pixelSize: Theme.fontSmall }
                        Text { width: Theme.sp(58); text: modelData.lvl; color: modelData.lc; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontSmall }
                        Text { width: parent.width - Theme.sp(96) - Theme.sp(58) - Theme.sp(28); elide: Text.ElideRight; text: modelData.m; color: Theme.t2; font.family: Theme.mono; font.pixelSize: Theme.fontSmall }
                    }
                }
            }
        }
    }
}
