import QtQuick
import Stonks
import "../js/format.js" as Fmt

// Trades tab: symbol pills, candlestick chart, trades table, trade/summary side panel.
Column {
    id: tv
    readonly property var bt: App.currentBacktest()
    readonly property string sym: App.symbol
    readonly property var trades: App.tradesFor(App.symbol)
    readonly property var sel: App.selectedTrade
    readonly property bool hasSel: sel !== null && sel !== undefined

    function fmtRet1(x) { return (x >= 0 ? "+" : "") + x.toFixed(1) + "%" }
    function fmtRet2(x) { return (x >= 0 ? "+" : "") + x.toFixed(2) + "%" }
    function chartSub() {
        if (hasSel && trades[sel]) { var t = trades[sel]; return "isolated trade · " + Fmt.dShort(t.entryIdx) + " → " + Fmt.dShort(t.exitIdx) }
        return "all trades · 120-bar window"
    }
    function sumStats() {
        var ts = trades, w = [], l = [], total = 0, best = -1e9, worst = 1e9;
        for (var i = 0; i < ts.length; i++) {
            var p = ts[i].pnlNum; total += p;
            if (p >= 0) w.push(p); else l.push(p);
            best = Math.max(best, p); worst = Math.min(worst, p);
        }
        function avg(a) { if (!a.length) return 0; var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
        return { totalStr: Fmt.fmtUsd(total), totalColor: total >= 0 ? Theme.positive : Theme.negative,
                 win: ts.length ? Math.round(w.length / ts.length * 100) + "%" : "0%", count: ts.length,
                 avgWin: Fmt.fmtUsd(avg(w)), avgLoss: Fmt.fmtUsd(avg(l)),
                 best: Fmt.fmtUsd(ts.length ? best : 0), worst: Fmt.fmtUsd(ts.length ? worst : 0) };
    }

    // ===== 1. symbol tabs + selection status =====
    Item {
        width: tv.width
        height: 54
        Row {
            x: 24
            anchors.verticalCenter: parent.verticalCenter
            spacing: 8
            Repeater {
                model: tv.bt.symbols
                delegate: Rectangle {
                    required property var modelData
                    readonly property bool on: tv.sym === modelData.id
                    width: pillT.implicitWidth + 32
                    height: 32
                    radius: 5
                    color: on ? Theme.accent : "transparent"
                    border.width: 1
                    border.color: on ? Theme.accent : Theme.border
                    Text {
                        id: pillT
                        anchors.centerIn: parent
                        text: modelData.id
                        color: parent.on ? Theme.accentInk : Theme.t3
                        font.family: Theme.mono
                        font.weight: Font.DemiBold
                        font.pixelSize: 13
                    }
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: App.selectSymbol(modelData.id) }
                }
            }
        }
        // selection status (right)
        Loader {
            anchors.right: parent.right
            anchors.rightMargin: 24
            anchors.verticalCenter: parent.verticalCenter
            sourceComponent: tv.hasSel ? selBtn : selNone
        }
        Component {
            id: selBtn
            Rectangle {
                width: clrT.implicitWidth + 28
                height: 30
                radius: 5
                color: Theme.accentSoft
                border.width: 1
                border.color: Theme.accentLine
                Text {
                    id: clrT
                    anchors.centerIn: parent
                    text: "‹ All trades · single trade #" + (tv.trades[tv.sel] ? tv.trades[tv.sel].n : "")
                    color: Theme.accent
                    font.family: Theme.mono
                    font.weight: Font.Medium
                    font.pixelSize: 12
                }
                MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: App.clearTrade() }
            }
        }
        Component {
            id: selNone
            Text {
                text: "showing all " + tv.trades.length + " trades · " + tv.sym
                color: Theme.t6
                font.family: Theme.mono
                font.weight: Font.Medium
                font.pixelSize: 12
            }
        }
    }

    // ===== 2. chart card =====
    Item {
        width: tv.width
        height: 18 + 488
        Rectangle {
            x: 24; y: 0
            width: parent.width - 48
            height: 488
            radius: 6
            color: Theme.card
            border.color: Theme.border
            border.width: 1

            // header
            Item {
                x: 18; y: 16
                width: parent.width - 36
                height: 22
                Row {
                    anchors.left: parent.left
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: 12
                    Text { anchors.verticalCenter: parent.verticalCenter; text: tv.sym; color: Theme.textPrimary; font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: 16 }
                    Text { anchors.verticalCenter: parent.verticalCenter; text: tv.chartSub(); color: Theme.t6; font.family: Theme.mono; font.pixelSize: 12 }
                }
                Row {
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: 16
                    Row { spacing: 6; Text { text: "▲"; color: Theme.accent; font.family: Theme.mono; font.pixelSize: 11 } Text { text: "entry"; color: Theme.t4; font.family: Theme.mono; font.pixelSize: 11 } }
                    Row { spacing: 6; Text { text: "▼"; color: Theme.positive; font.family: Theme.mono; font.pixelSize: 11 } Text { text: "exit win"; color: Theme.t4; font.family: Theme.mono; font.pixelSize: 11 } }
                    Row { spacing: 6; Text { text: "▼"; color: Theme.negative; font.family: Theme.mono; font.pixelSize: 11 } Text { text: "exit loss"; color: Theme.t4; font.family: Theme.mono; font.pixelSize: 11 } }
                }
            }

            CandleChart {
                x: 18; y: 50
                width: parent.width - 36
                height: 440
                symbol: tv.sym
                selectedTrade: tv.sel
                accentHex: Theme.accent.toString()
            }
        }
    }

    // ===== 3. trades table + side panel =====
    Item {
        id: tradesBody
        width: tv.width
        implicitHeight: bottomRow.implicitHeight + 40

        readonly property real gap: 18
        readonly property real innerW: width - 48
        readonly property real leftW: (innerW - gap) * 1.7 / 2.7
        readonly property real rightW: (innerW - gap) * 1.0 / 2.7

        // table grid
        readonly property real tInner: leftW - 36
        readonly property real tgap: 12
        readonly property real cNum: 44
        readonly property real cQty: 70
        readonly property real cPnl: 104
        readonly property real cRet: 88
        readonly property real cBars: 56
        readonly property real cFlex: (tInner - cNum - cQty - cPnl - cRet - cBars - tgap * 6) / 2

        Row {
            id: bottomRow
            x: 24
            spacing: tradesBody.gap

            // --- trades table ---
            Rectangle {
                width: tradesBody.leftW
                radius: 6
                color: Theme.card
                border.color: Theme.border
                border.width: 1
                clip: true
                implicitHeight: tableCol.implicitHeight

                Column {
                    id: tableCol
                    width: parent.width

                    // header
                    Item {
                        width: parent.width
                        height: 42
                        Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.border }
                        Row {
                            x: 18
                            anchors.verticalCenter: parent.verticalCenter
                            spacing: tradesBody.tgap
                            Text { width: tradesBody.cNum; text: "#"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 10; font.letterSpacing: 0.8 }
                            Text { width: tradesBody.cFlex; text: "ENTRY"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 10; font.letterSpacing: 0.8 }
                            Text { width: tradesBody.cFlex; text: "EXIT"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 10; font.letterSpacing: 0.8 }
                            Text { width: tradesBody.cQty; horizontalAlignment: Text.AlignRight; text: "QTY"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 10; font.letterSpacing: 0.8 }
                            Text { width: tradesBody.cPnl; horizontalAlignment: Text.AlignRight; text: "P&L"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 10; font.letterSpacing: 0.8 }
                            Text { width: tradesBody.cRet; horizontalAlignment: Text.AlignRight; text: "RETURN"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 10; font.letterSpacing: 0.8 }
                            Text { width: tradesBody.cBars; horizontalAlignment: Text.AlignRight; text: "BARS"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 10; font.letterSpacing: 0.8 }
                        }
                    }

                    // rows
                    Repeater {
                        model: tv.trades
                        delegate: Rectangle {
                            id: tradeRow
                            required property var modelData
                            required property int index
                            readonly property bool on: index === tv.sel
                            readonly property bool win: modelData.pnlNum >= 0
                            width: tableCol.width
                            height: 44
                            color: on ? Theme.accentSoft : (rowMa.containsMouse ? Qt.rgba(1, 1, 1, 0.02) : "transparent")
                            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.rowSep }
                            Rectangle { anchors.left: parent.left; width: 2; height: parent.height; color: tradeRow.on ? Theme.accent : "transparent" }
                            Row {
                                x: 18
                                height: parent.height
                                spacing: tradesBody.tgap
                                Item { width: tradesBody.cNum; height: parent.height
                                    Text { anchors.verticalCenter: parent.verticalCenter; text: modelData.n; color: tradeRow.on ? Theme.accent : Theme.t1; font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: 12 } }
                                Item { width: tradesBody.cFlex; height: parent.height
                                    Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; elide: Text.ElideRight; text: Fmt.dFull(modelData.entryIdx); color: Theme.t2; font.family: Theme.mono; font.pixelSize: 12 } }
                                Item { width: tradesBody.cFlex; height: parent.height
                                    Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; elide: Text.ElideRight; text: Fmt.dFull(modelData.exitIdx); color: Theme.t2; font.family: Theme.mono; font.pixelSize: 12 } }
                                Item { width: tradesBody.cQty; height: parent.height
                                    Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: modelData.qty; color: Theme.t4; font.family: Theme.mono; font.pixelSize: 12 } }
                                Item { width: tradesBody.cPnl; height: parent.height
                                    Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: Fmt.fmtUsd(modelData.pnlNum); color: win ? Theme.positive : Theme.negative; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 12 } }
                                Item { width: tradesBody.cRet; height: parent.height
                                    Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: tv.fmtRet1(modelData.retNum); color: win ? Theme.positive : Theme.negative; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 12 } }
                                Item { width: tradesBody.cBars; height: parent.height
                                    Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: modelData.bars; color: Theme.t4; font.family: Theme.mono; font.pixelSize: 12 } }
                            }
                            MouseArea { id: rowMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: App.selectTrade(index) }
                        }
                    }
                }
            }

            // --- side panel ---
            Loader {
                width: tradesBody.rightW
                sourceComponent: tv.hasSel ? tradeDetail : symSummary
            }

            Component {
                id: tradeDetail
                Rectangle {
                    readonly property var t: tv.trades[tv.sel]
                    readonly property bool win: t.pnlNum >= 0
                    width: tradesBody.rightW
                    radius: 6
                    color: Theme.card
                    border.color: Theme.border
                    border.width: 1
                    implicitHeight: dCol.implicitHeight + 44
                    Column {
                        id: dCol
                        x: 22; y: 20
                        width: parent.width - 44
                        spacing: 0
                        Item {
                            width: parent.width
                            height: 24
                            Text { anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter; text: "Trade #" + t.n + " · " + tv.sym; color: Theme.textPrimary; font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: 14 }
                            Rectangle {
                                anchors.right: parent.right; anchors.verticalCenter: parent.verticalCenter
                                width: badgeT.implicitWidth + 20; height: 22; radius: 4; color: Theme.accentBadge
                                Text { id: badgeT; anchors.centerIn: parent; text: t.side; color: Theme.accent; font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: 11 }
                            }
                        }
                        Item { width: 1; height: 18 }
                        Text { text: Fmt.fmtUsd(t.pnlNum); color: win ? Theme.positive : Theme.negative; font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: 32 }
                        Item { width: 1; height: 4 }
                        Text { text: tv.fmtRet2(t.retNum) + " return"; color: win ? Theme.positive : Theme.negative; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 14 }
                        Item { width: 1; height: 20 }
                        Grid {
                            width: parent.width
                            columns: 2
                            rowSpacing: 14
                            columnSpacing: 16
                            Column { spacing: 2; width: (dCol.width - 16) / 2
                                Text { text: "ENTRY"; color: Theme.t6; font.family: Theme.mono; font.pixelSize: 10; font.letterSpacing: 0.6 }
                                Text { text: "$" + t.entryPrice.toFixed(2); color: Theme.t1; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 13 }
                                Text { text: Fmt.dFull(t.entryIdx); color: Theme.t6; font.family: Theme.mono; font.pixelSize: 10 }
                            }
                            Column { spacing: 2; width: (dCol.width - 16) / 2
                                Text { text: "EXIT"; color: Theme.t6; font.family: Theme.mono; font.pixelSize: 10; font.letterSpacing: 0.6 }
                                Text { text: "$" + t.exitPrice.toFixed(2); color: Theme.t1; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 13 }
                                Text { text: Fmt.dFull(t.exitIdx); color: Theme.t6; font.family: Theme.mono; font.pixelSize: 10 }
                            }
                            StatCell { label: "QUANTITY"; value: String(t.qty); valuePx: 13 }
                            StatCell { label: "BARS HELD"; value: t.bars + " bars"; valuePx: 13 }
                            StatCell { label: "MFE"; value: "+" + t.mfe.toFixed(2) + "%"; valueColor: Theme.positive; valuePx: 13 }
                            StatCell { label: "MAE"; value: t.mae.toFixed(2) + "%"; valueColor: Theme.negative; valuePx: 13 }
                        }
                        Item { width: 1; height: 18 }
                        Rectangle { width: parent.width; height: 1; color: Theme.rowSep }
                        Item { width: 1; height: 16 }
                        Text {
                            width: parent.width
                            wrapMode: Text.WordWrap
                            text: "Chart isolates this trade — entry/exit guides and the P&L band shown. Use ‹ All trades to see every fill on " + tv.sym + "."
                            color: Theme.t5
                            font.family: Theme.mono
                            font.pixelSize: 11
                            lineHeight: 1.5
                        }
                    }
                }
            }

            Component {
                id: symSummary
                Rectangle {
                    readonly property var s: tv.sumStats()
                    width: tradesBody.rightW
                    radius: 6
                    color: Theme.card
                    border.color: Theme.border
                    border.width: 1
                    implicitHeight: sCol.implicitHeight + 44
                    Column {
                        id: sCol
                        x: 22; y: 20
                        width: parent.width - 44
                        spacing: 0
                        Text { text: tv.sym + " · summary"; color: Theme.textPrimary; font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: 14 }
                        Item { width: 1; height: 18 }
                        Grid {
                            width: parent.width
                            columns: 2
                            rowSpacing: 16
                            columnSpacing: 16
                            StatCell { label: "NET P&L"; value: s.totalStr; valueColor: s.totalColor; valuePx: 18 }
                            StatCell { label: "WIN RATE"; value: s.win; valueColor: Theme.textPrimary; valuePx: 18 }
                            StatCell { label: "TRADES"; value: String(s.count); valueColor: Theme.textPrimary; valuePx: 18 }
                            StatCell { label: "AVG WIN / LOSS"; value: s.avgWin + " / " + s.avgLoss; valuePx: 13 }
                            StatCell { label: "BEST"; value: s.best; valueColor: Theme.positive; valuePx: 13 }
                            StatCell { label: "WORST"; value: s.worst; valueColor: Theme.negative; valuePx: 13 }
                        }
                        Item { width: 1; height: 18 }
                        Rectangle { width: parent.width; height: 1; color: Theme.rowSep }
                        Item { width: 1; height: 16 }
                        Text {
                            width: parent.width
                            wrapMode: Text.WordWrap
                            text: "All " + s.count + " fills on " + tv.sym + " are plotted on the chart above. Click any trade in the table to isolate it."
                            color: Theme.t5
                            font.family: Theme.mono
                            font.pixelSize: 11
                            lineHeight: 1.5
                        }
                    }
                }
            }
        }
    }
}
