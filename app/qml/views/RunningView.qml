import QtQuick
import QtQuick.Controls
import Stonks
import "../js/format.js" as Fmt

// Running backtest: progress + live log.
Page {
    id: view
    maxWidth: 720
    topPad: 90
    spacing: 30

    readonly property var bt: App.currentBacktest()

    // --- title ---
    Row {
        spacing: 14
        Spinner { anchors.verticalCenter: parent.verticalCenter }
        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: "Running backtest…"
            color: Theme.textBright
            font.family: Theme.sans
            font.weight: Font.DemiBold
            font.pixelSize: 20
        }
    }

    // --- progress card ---
    Rectangle {
        width: view.innerWidth
        radius: 8
        color: Theme.card
        border.color: Theme.border
        border.width: 1
        implicitHeight: card.implicitHeight + 52

        Column {
            id: card
            x: 28; y: 26
            width: parent.width - 56
            spacing: 0

            // progress label + pct
            Item {
                width: parent.width
                height: 22
                Text { anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter; text: "PROGRESS"; color: Theme.t4; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 12; font.letterSpacing: 0.7 }
                Text { anchors.right: parent.right; anchors.bottom: parent.bottom; text: Math.round(App.progress) + "%"; color: Theme.accent; font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: 18 }
            }
            Item { width: 1; height: 12 }

            // bar
            Rectangle {
                width: parent.width
                height: 8
                radius: 4
                color: Theme.input
                Rectangle {
                    height: parent.height
                    radius: 4
                    color: Theme.accent
                    width: parent.width * App.progress / 100
                    Behavior on width { NumberAnimation { duration: 110 } }
                }
            }
            Item { width: 1; height: 20 }

            // stats
            Row {
                width: parent.width
                Repeater {
                    model: [
                        { l: "BARS PROCESSED", v: Fmt.commas(App.barsDone) },
                        { l: "SYMBOL", v: (view.bt.symbols && view.bt.symbols[0]) ? view.bt.symbols[0].id : "—" },
                        { l: "ELAPSED", v: (App.progress / 100 * 9.84).toFixed(2) + "s" }
                    ]
                    delegate: Column {
                        required property var modelData
                        width: card.width / 3
                        spacing: 5
                        Text { text: modelData.l; color: Theme.t6; font.family: Theme.mono; font.pixelSize: 10; font.letterSpacing: 0.6 }
                        Text { text: modelData.v; color: Theme.t1; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 15 }
                    }
                }
            }
            Item { width: 1; height: 22 }

            // live log
            Item {
                width: parent.width
                height: 16
                Text { anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter; text: "LIVE LOG"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 10; font.letterSpacing: 1 }
                Text { anchors.right: parent.right; anchors.verticalCenter: parent.verticalCenter; text: App.liveLogList().length + " events"; color: Theme.t7; font.family: Theme.mono; font.pixelSize: 10 }
            }
            Item { width: 1; height: 8 }

            Rectangle {
                width: parent.width
                height: 184
                radius: 5
                color: Theme.bg
                border.color: "#272727"
                border.width: 1
                clip: true

                ListView {
                    id: logView
                    anchors.fill: parent
                    anchors.margins: 10
                    anchors.leftMargin: 14
                    anchors.rightMargin: 14
                    clip: true
                    model: App.liveLogList()
                    onCountChanged: positionViewAtEnd()
                    delegate: Item {
                        required property var modelData
                        width: logView.width
                        height: 21
                        Row {
                            spacing: 12
                            Text { width: 60; text: modelData.t; color: Theme.t7; font.family: Theme.mono; font.pixelSize: 11 }
                            Text { width: 48; text: modelData.lvl; color: modelData.lc; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 11 }
                            Text { width: logView.width - 120; elide: Text.ElideRight; text: modelData.m; color: Theme.t2; font.family: Theme.mono; font.pixelSize: 11 }
                        }
                    }
                }
            }
            Item { width: 1; height: 22 }

            // cancel
            Item {
                width: parent.width
                height: 42
                Rectangle {
                    anchors.right: parent.right
                    width: cancelT.implicitWidth + 44
                    height: 42
                    radius: 6
                    color: cancelMa.containsMouse ? Qt.rgba(0.88, 0.34, 0.30, 0.10) : "transparent"
                    border.color: "#3a2326"
                    border.width: 1
                    Text { id: cancelT; anchors.centerIn: parent; text: "✕ Cancel"; color: Theme.negative; font.family: Theme.sans; font.weight: Font.Medium; font.pixelSize: 13 }
                    MouseArea { id: cancelMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: App.cancelRun() }
                }
            }
        }
    }
}
