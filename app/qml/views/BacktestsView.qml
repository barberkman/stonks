import QtQuick
import Stonks

// Saved & completed backtests list.
Page {
    id: view
    maxWidth: Theme.sp(1180)
    spacing: Theme.sp(26)

    // shared grid metrics (header + rows align to these)
    readonly property real gap: Theme.sp(14)
    readonly property real cRun: Theme.sp(60)
    readonly property real cRange: Theme.sp(150)
    readonly property real cRet: Theme.sp(86)
    readonly property real cDD: Theme.sp(78)
    readonly property real cTrd: Theme.sp(60)
    readonly property real cStatus: Theme.sp(104)
    readonly property real rowW: innerWidth - Theme.sp(44)
    readonly property real flexRemain: rowW - (cRun + cRange + cRet + cDD + cTrd + cStatus) - gap * 7
    readonly property real cStrat: Math.max(Theme.sp(120), flexRemain * 1.4 / 2.4)
    readonly property real cData: Math.max(Theme.sp(110), flexRemain * 1.0 / 2.4)

    function retColor(b) {
        return b.status === "saved" ? Theme.t5 : (b.retPos ? Theme.positive : Theme.negative)
    }
    function statusColor(s) {
        return s === "completed" ? Theme.positive : (s === "running" ? Theme.accent : Theme.t5)
    }

    // --- title row + new button ---
    Item {
        width: view.innerWidth
        height: Theme.sp(58)
        Column {
            anchors.left: parent.left
            anchors.bottom: parent.bottom
            spacing: Theme.sp(10)
            Text {
                text: "SAVED & COMPLETED"
                color: Theme.accent
                font.family: Theme.mono
                font.weight: Font.Medium
                font.pixelSize: Theme.fontCaption
                font.letterSpacing: 1.5 * Theme.scale
            }
            Text {
                text: "Backtests"
                color: Theme.textBright
                font.family: Theme.sans
                font.weight: Font.Bold
                font.pixelSize: Theme.fontTitle
            }
        }
        AccentButton {
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            text: "+ New backtest"
            onClicked: App.goSetup()
        }
    }

    // --- table card ---
    Rectangle {
        width: view.innerWidth
        radius: Theme.sp(7)
        color: Theme.card
        border.color: Theme.border
        border.width: 1
        clip: true
        implicitHeight: tableCol.implicitHeight

        Column {
            id: tableCol
            width: parent.width

            // header row
            Item {
                width: parent.width
                height: Theme.sp(40)
                Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.border }
                Row {
                    x: Theme.sp(22)
                    anchors.verticalCenter: parent.verticalCenter
                    width: view.rowW
                    spacing: view.gap
                    Text { width: view.cRun; text: "RUN"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 1 * Theme.scale }
                    Text { width: view.cStrat; text: "STRATEGY"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 1 * Theme.scale }
                    Text { width: view.cData; text: "DATA"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 1 * Theme.scale }
                    Text { width: view.cRange; text: "RANGE"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 1 * Theme.scale }
                    Text { width: view.cRet; horizontalAlignment: Text.AlignRight; text: "RETURN"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 1 * Theme.scale }
                    Text { width: view.cDD; horizontalAlignment: Text.AlignRight; text: "MAX DD"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 1 * Theme.scale }
                    Text { width: view.cTrd; horizontalAlignment: Text.AlignRight; text: "TRD"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 1 * Theme.scale }
                    Text { width: view.cStatus; horizontalAlignment: Text.AlignRight; text: "STATUS"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 1 * Theme.scale }
                }
            }

            // rows
            Repeater {
                model: App.allBacktests()
                delegate: Rectangle {
                    required property var modelData
                    width: tableCol.width
                    height: Theme.sp(48)
                    color: rowMa.containsMouse ? Qt.rgba(1, 1, 1, 0.02) : "transparent"
                    Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.rowSepDim }

                    Row {
                        x: Theme.sp(22)
                        height: parent.height
                        width: view.rowW
                        spacing: view.gap

                        Item {
                            width: view.cRun; height: parent.height
                            Text { anchors.verticalCenter: parent.verticalCenter; text: "#" + modelData.id; color: Theme.t1; font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: Theme.fontBody }
                        }
                        Item {
                            width: view.cStrat; height: parent.height
                            Column {
                                anchors.verticalCenter: parent.verticalCenter
                                width: parent.width
                                spacing: Theme.sp(3)
                                Text { width: parent.width; elide: Text.ElideRight; text: modelData.strategy; color: Theme.textPrimary; font.family: Theme.sans; font.weight: Font.DemiBold; font.pixelSize: Theme.fontBody }
                                Text { width: parent.width; elide: Text.ElideRight; text: modelData.symbols.map(function (s) { return s.id }).join(" · "); color: Theme.t6; font.family: Theme.mono; font.pixelSize: Theme.fontCaption }
                            }
                        }
                        Item {
                            width: view.cData; height: parent.height
                            Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; elide: Text.ElideRight; text: (App.dataFilesObj()[modelData.dataKey] || {}).label || modelData.dataKey; color: Theme.t2; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontSmall }
                        }
                        Item {
                            width: view.cRange; height: parent.height
                            Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; elide: Text.ElideRight; text: modelData.range; color: Theme.t4; font.family: Theme.mono; font.pixelSize: Theme.fontCaption }
                        }
                        Item {
                            width: view.cRet; height: parent.height
                            Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: modelData.ret; color: view.retColor(modelData); font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: Theme.fontBody }
                        }
                        Item {
                            width: view.cDD; height: parent.height
                            Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: modelData.maxdd; color: Theme.t2; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontBody }
                        }
                        Item {
                            width: view.cTrd; height: parent.height
                            Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: modelData.status === "saved" ? "—" : modelData.trades; color: Theme.t4; font.family: Theme.mono; font.pixelSize: Theme.fontBody }
                        }
                        Item {
                            width: view.cStatus; height: parent.height
                            Row {
                                anchors.right: parent.right
                                anchors.verticalCenter: parent.verticalCenter
                                spacing: Theme.sp(7)
                                Rectangle { width: Theme.sp(6); height: Theme.sp(6); radius: Theme.sp(3); anchors.verticalCenter: parent.verticalCenter; color: view.statusColor(modelData.status) }
                                Text { anchors.verticalCenter: parent.verticalCenter; text: (modelData.status || "").toUpperCase(); color: view.statusColor(modelData.status); font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 0.5 * Theme.scale }
                            }
                        }
                    }

                    MouseArea {
                        id: rowMa
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: App.openBacktest(modelData.id)
                    }

                    // Delete (declared after rowMa so it wins the click): shows
                    // on row hover; removes the run and its archived report file.
                    Item {
                        anchors.right: parent.right
                        anchors.rightMargin: Theme.sp(8)
                        anchors.verticalCenter: parent.verticalCenter
                        width: Theme.sp(28)
                        height: Theme.sp(28)
                        visible: rowMa.containsMouse || delMa.containsMouse
                        Text {
                            anchors.centerIn: parent
                            text: "✕"
                            color: delMa.containsMouse ? Theme.negative : Theme.t5
                            font.family: Theme.mono
                            font.pixelSize: Theme.fontBody
                        }
                        MouseArea {
                            id: delMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: App.deleteBacktest(modelData.id)
                        }
                    }
                }
            }
        }
    }
}
