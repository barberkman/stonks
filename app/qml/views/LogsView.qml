import QtQuick
import Stonks

// Diagnostic log for the latest run.
Page {
    id: view
    maxWidth: 100000
    sidePad: 24
    topPad: 18
    bottomPad: 40
    spacing: 14

    // header row: label + filter chips
    Item {
        width: view.innerWidth
        height: 24

        Text {
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
            text: "DIAGNOSTIC LOG · RUN #" + App.latestRun().id
            color: Theme.t4
            font.family: Theme.mono
            font.weight: Font.Medium
            font.pixelSize: 11
            font.letterSpacing: 1
        }
        Row {
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            spacing: 6
            Repeater {
                model: [{ t: "ALL", on: true }, { t: "WARN", on: false }, { t: "ERROR", on: false }]
                delegate: Rectangle {
                    required property var modelData
                    width: chipText.implicitWidth + 20
                    height: 22
                    radius: 4
                    color: modelData.on ? Theme.input : "transparent"
                    Text {
                        id: chipText
                        anchors.centerIn: parent
                        text: modelData.t
                        color: modelData.on ? Theme.t4 : Theme.t6
                        font.family: Theme.mono
                        font.weight: Font.Medium
                        font.pixelSize: 11
                    }
                }
            }
        }
    }

    // log box
    Rectangle {
        width: view.innerWidth
        radius: 6
        color: Theme.panel
        border.color: Theme.border
        border.width: 1
        implicitHeight: logCol.implicitHeight + 28

        Column {
            id: logCol
            y: 14
            width: parent.width

            Repeater {
                model: App.logs()
                delegate: Item {
                    required property var modelData
                    width: logCol.width
                    height: 23
                    Row {
                        x: 18
                        anchors.verticalCenter: parent.verticalCenter
                        width: parent.width - 36
                        spacing: 14
                        Text { width: 96; text: modelData.t; color: Theme.t7; font.family: Theme.mono; font.pixelSize: 12 }
                        Text { width: 58; text: modelData.lvl; color: modelData.lc; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 12 }
                        Text { width: parent.width - 96 - 58 - 28; elide: Text.ElideRight; text: modelData.m; color: Theme.t2; font.family: Theme.mono; font.pixelSize: 12 }
                    }
                }
            }
        }
    }
}
