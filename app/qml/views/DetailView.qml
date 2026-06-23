import QtQuick
import QtQuick.Controls
import Stonks

// Backtest detail: sub-header (back, title, Report/Trades tabs) + body.
Item {
    id: view
    readonly property var bt: App.currentBacktest()
    readonly property bool saved: bt.status === "saved"

    ScrollView {
        anchors.fill: parent
        contentWidth: availableWidth
        clip: true
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

        Column {
            id: stack
            width: view.width

            // --- sub-header ---
            Item {
                width: stack.width
                height: 70
                Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.border }

                Row {
                    anchors.left: parent.left
                    anchors.leftMargin: 24
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: 16
                    GhostButton {
                        anchors.verticalCenter: parent.verticalCenter
                        text: "‹ Backtests"
                        onClicked: App.goBacktests()
                    }
                    Column {
                        anchors.verticalCenter: parent.verticalCenter
                        spacing: 3
                        Row {
                            spacing: 10
                            Text { text: view.bt.strategy; color: Theme.textBright; font.family: Theme.sans; font.weight: Font.DemiBold; font.pixelSize: 15 }
                            Text { anchors.verticalCenter: parent.verticalCenter; text: "#" + view.bt.id; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 12 }
                        }
                        Text {
                            text: ((App.dataFilesObj()[view.bt.dataKey] || {}).label || view.bt.dataKey) + " · " + view.bt.range
                            color: Theme.t5
                            font.family: Theme.mono
                            font.pixelSize: 11
                        }
                    }
                }

                // Report / Trades tabs
                Rectangle {
                    visible: !view.saved
                    anchors.right: parent.right
                    anchors.rightMargin: 24
                    anchors.verticalCenter: parent.verticalCenter
                    width: tabRow.width + 8
                    height: 34
                    radius: 7
                    color: Theme.card
                    border.color: Theme.border
                    border.width: 1
                    Row {
                        id: tabRow
                        anchors.centerIn: parent
                        spacing: 6
                        Repeater {
                            model: [{ k: "report", t: "Report" }, { k: "trades", t: "Trades" }]
                            delegate: Rectangle {
                                required property var modelData
                                readonly property bool on: App.detailTab === modelData.k
                                width: tabLabel.implicitWidth + 36
                                height: 26
                                radius: 5
                                color: on ? Theme.accent : "transparent"
                                Text {
                                    id: tabLabel
                                    anchors.centerIn: parent
                                    text: modelData.t
                                    color: parent.on ? Theme.accentInk : Theme.t3
                                    font.family: Theme.mono
                                    font.weight: Font.DemiBold
                                    font.pixelSize: 12
                                }
                                MouseArea {
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: modelData.k === "trades" ? App.showTrades() : App.showReport()
                                }
                            }
                        }
                    }
                }
            }

            // --- body ---
            Loader {
                width: stack.width
                sourceComponent: view.saved ? savedComp : (App.detailTab === "trades" ? tradesComp : reportComp)
            }
            Component { id: reportComp; ReportView { width: stack.width } }
            Component { id: tradesComp; TradesView { width: stack.width } }
            Component {
                id: savedComp
                Item {
                    width: stack.width
                    implicitHeight: 360
                    Column {
                        width: Math.min(parent.width - 80, 560)
                        anchors.horizontalCenter: parent.horizontalCenter
                        y: 80
                        spacing: 10
                        Text {
                            width: parent.width
                            horizontalAlignment: Text.AlignHCenter
                            text: "This backtest is saved but hasn't been run"
                            color: Theme.textPrimary
                            font.family: Theme.sans
                            font.weight: Font.DemiBold
                            font.pixelSize: 18
                        }
                        Text {
                            width: parent.width
                            horizontalAlignment: Text.AlignHCenter
                            wrapMode: Text.WordWrap
                            text: view.bt.strategy + " on " + ((App.dataFilesObj()[view.bt.dataKey] || {}).label || view.bt.dataKey)
                                  + " — parameters are saved. Run it to generate the report and trade analysis."
                            color: Theme.t5
                            font.family: Theme.mono
                            font.pixelSize: 13
                            lineHeight: 1.5
                        }
                        Item { width: 1; height: 14 }
                        AccentButton {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: "▸ Run this backtest"
                            onClicked: App.runSaved()
                        }
                    }
                }
            }
        }
    }
}
