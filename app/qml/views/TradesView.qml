import QtQuick
import QtQuick.Controls
import Stonks
import "../js/format.js" as Fmt

// Trades tab: symbol pills + candlestick chart stay pinned at the top while the
// trades table scrolls in its own ListView; the trade/summary side panel scrolls
// independently too. Fills the height handed down by DetailView.
Item {
    id: tv
    // Available viewport height handed down by DetailView; drives the
    // near-full-screen chart and the tall trades section below it.
    property real viewportH: 600
    implicitHeight: topSection.height + tradesBody.height
    readonly property var bt: App.currentBacktest()
    readonly property string sym: App.symbol
    readonly property var trades: App.tradesFor(App.symbol)
    readonly property var sel: App.selectedTrade
    readonly property bool hasSel: sel !== null && sel !== undefined

    function fmtRet1(x) { return (x >= 0 ? "+" : "") + x.toFixed(1) + "%" }
    function fmtRet2(x) { return (x >= 0 ? "+" : "") + x.toFixed(2) + "%" }
    function chartSub() {
        if (hasSel && trades[sel]) { var t = trades[sel]; return "isolated trade · " + Fmt.tsShort(t.entryTs) + " → " + Fmt.tsShort(t.exitTs) }
        return "all trades · scroll to zoom · drag to pan"
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

    // ===== pinned top: symbol tabs + chart card =====
    Column {
        id: topSection
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right

        // ----- symbol tabs + selection status -----
        Item {
            width: tv.width
            height: Theme.sp(54)
            // Symbol pills scroll horizontally within a width bounded by the
            // status element on the right, so a long list never overflows or
            // overlaps it.
            Flickable {
                id: symScroll
                anchors.left: parent.left
                anchors.leftMargin: Theme.sp(24)
                anchors.right: statusLoader.left
                anchors.rightMargin: Theme.sp(16)
                anchors.verticalCenter: parent.verticalCenter
                height: Theme.sp(32)
                clip: true
                flickableDirection: Flickable.HorizontalFlick
                boundsBehavior: Flickable.StopAtBounds
                contentWidth: pillsRow.width
                contentHeight: height
                ScrollBar.horizontal: ScrollBar {
                    id: symBar
                    policy: ScrollBar.AsNeeded
                    padding: Theme.sp(2)
                    implicitHeight: Theme.sp(8)
                    contentItem: Rectangle {
                        implicitHeight: Theme.sp(4)
                        radius: height / 2
                        color: symBar.pressed ? Theme.t4 : Theme.t6
                        opacity: symBar.active ? 0.9 : 0.0
                        Behavior on opacity { NumberAnimation { duration: 150 } }
                    }
                }
                Row {
                    id: pillsRow
                    spacing: Theme.sp(8)
                    Repeater {
                        model: tv.bt.symbols
                        delegate: Rectangle {
                            required property var modelData
                            readonly property bool on: tv.sym === modelData.id
                            width: pillT.implicitWidth + Theme.sp(32)
                            height: Theme.sp(32)
                            radius: Theme.radiusControl
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
                                font.pixelSize: Theme.fontBody
                            }
                            MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: App.selectSymbol(modelData.id) }
                        }
                    }
                }
            }
            // selection status (right)
            Loader {
                id: statusLoader
                anchors.right: parent.right
                anchors.rightMargin: Theme.sp(24)
                anchors.verticalCenter: parent.verticalCenter
                sourceComponent: tv.hasSel ? selBtn : selNone
            }
            Component {
                id: selBtn
                Rectangle {
                    width: clrT.implicitWidth + Theme.sp(28)
                    height: Theme.controlHSm
                    radius: Theme.radiusControl
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
                        font.pixelSize: Theme.fontSmall
                    }
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: { App.clearTrade(); chart.fit() } }
                }
            }
            Component {
                id: selNone
                Text {
                    text: "showing all " + tv.trades.length + " trades · " + tv.sym
                    color: Theme.t6
                    font.family: Theme.mono
                    font.weight: Font.Medium
                    font.pixelSize: Theme.fontSmall
                }
            }
        }

        // ----- chart card -----
        Item {
            width: tv.width
            height: Theme.sp(18) + card.height
            Rectangle {
                id: card
                x: Theme.sp(24); y: 0
                width: parent.width - Theme.sp(48)
                height: chart.y + chart.height
                radius: Theme.radiusCard
                color: Theme.card
                border.color: Theme.border
                border.width: 1

                // header
                Item {
                    x: Theme.sp(18); y: Theme.sp(16)
                    width: parent.width - Theme.sp(36)
                    height: Theme.sp(22)
                    Row {
                        anchors.left: parent.left
                        anchors.verticalCenter: parent.verticalCenter
                        spacing: Theme.sp(12)
                        Text { anchors.verticalCenter: parent.verticalCenter; text: tv.sym; color: Theme.textPrimary; font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: Theme.fontSub }
                        Text { anchors.verticalCenter: parent.verticalCenter; text: tv.chartSub(); color: Theme.t6; font.family: Theme.mono; font.pixelSize: Theme.fontSmall }
                    }
                    Row {
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        spacing: Theme.sp(16)
                        Row { spacing: Theme.sp(6); Text { text: "▲"; color: Theme.accent; font.family: Theme.mono; font.pixelSize: Theme.fontCaption } Text { text: "entry"; color: Theme.t4; font.family: Theme.mono; font.pixelSize: Theme.fontCaption } }
                        Row { spacing: Theme.sp(6); Text { text: "▼"; color: Theme.positive; font.family: Theme.mono; font.pixelSize: Theme.fontCaption } Text { text: "exit win"; color: Theme.t4; font.family: Theme.mono; font.pixelSize: Theme.fontCaption } }
                        Row { spacing: Theme.sp(6); Text { text: "▼"; color: Theme.negative; font.family: Theme.mono; font.pixelSize: Theme.fontCaption } Text { text: "exit loss"; color: Theme.t4; font.family: Theme.mono; font.pixelSize: Theme.fontCaption } }
                        GhostButton {
                            anchors.verticalCenter: parent.verticalCenter
                            height: Theme.sp(22)
                            text: "⟲ Reset zoom"
                            fontPx: Theme.fontCaption
                            hpad: Theme.sp(10)
                            onClicked: chart.fit()
                        }
                    }
                }

                CandleChart {
                    id: chart
                    x: Theme.sp(18); y: Theme.sp(50)
                    width: parent.width - Theme.sp(36)
                    // 70% of a near-full-screen fill: shorter chart leaves more
                    // room to scroll down to the trades section below.
                    height: Math.round(0.7 * Math.max(Theme.fill(440), tv.viewportH - Theme.sp(178)))
                    symbol: tv.sym
                    selectedTrade: tv.sel
                    accentHex: Theme.accent.toString()
                }
            }
        }
    }

    // ===== trades table + side panel (generous, self-scrolling section) =====
    Item {
        id: tradesBody
        anchors.top: topSection.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        // Grow to fit whichever is taller: a near-full-screen table viewport or
        // the (now self-scrolling-free) side panel's natural content height.
        readonly property real baseH: Math.max(Theme.sp(440), tv.viewportH - Theme.sp(90))
        readonly property real panelH: panelLoader.item ? panelLoader.item.implicitHeight : 0
        height: bottomRow.height + Theme.sp(40)

        readonly property real gap: Theme.sp(18)
        readonly property real innerW: width - Theme.sp(48)
        readonly property real leftW: (innerW - gap) * 1.7 / 2.7
        readonly property real rightW: (innerW - gap) * 1.0 / 2.7

        // table grid
        readonly property real tInner: leftW - Theme.sp(36)
        readonly property real tgap: Theme.sp(12)
        readonly property real cNum: Theme.sp(44)
        readonly property real cQty: Theme.sp(70)
        readonly property real cPnl: Theme.sp(104)
        readonly property real cRet: Theme.sp(88)
        readonly property real cBars: Theme.sp(56)
        readonly property real cFlex: (tInner - cNum - cQty - cPnl - cRet - cBars - tgap * 6) / 2

        Row {
            id: bottomRow
            x: Theme.sp(24)
            y: Theme.sp(20)
            height: Math.max(tradesBody.baseH - Theme.sp(40), tradesBody.panelH)
            spacing: tradesBody.gap

            // --- trades table ---
            Rectangle {
                width: tradesBody.leftW
                height: bottomRow.height
                radius: Theme.radiusCard
                color: Theme.card
                border.color: Theme.border
                border.width: 1
                clip: true

                // pinned column header
                Item {
                    id: tableHeader
                    anchors.top: parent.top
                    anchors.left: parent.left
                    anchors.right: parent.right
                    height: Theme.sp(42)
                    Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.border }
                    Row {
                        x: Theme.sp(18)
                        anchors.verticalCenter: parent.verticalCenter
                        spacing: tradesBody.tgap
                        Text { width: tradesBody.cNum; text: "#"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 0.8 * Theme.scale }
                        Text { width: tradesBody.cFlex; text: "ENTRY"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 0.8 * Theme.scale }
                        Text { width: tradesBody.cFlex; text: "EXIT"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 0.8 * Theme.scale }
                        Text { width: tradesBody.cQty; horizontalAlignment: Text.AlignRight; text: "QTY"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 0.8 * Theme.scale }
                        Text { width: tradesBody.cPnl; horizontalAlignment: Text.AlignRight; text: "P&L"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 0.8 * Theme.scale }
                        Text { width: tradesBody.cRet; horizontalAlignment: Text.AlignRight; text: "RETURN"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 0.8 * Theme.scale }
                        Text { width: tradesBody.cBars; horizontalAlignment: Text.AlignRight; text: "BARS"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontMicro; font.letterSpacing: 0.8 * Theme.scale }
                    }
                }

                // scrolling rows
                ListView {
                    id: tradesList
                    anchors.top: tableHeader.bottom
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    clip: true
                    boundsBehavior: Flickable.StopAtBounds
                    model: tv.trades
                    ScrollBar.vertical: ThinScrollBar {}

                    // keep the selected row on screen when selection changes elsewhere
                    Connections {
                        target: tv
                        function onSelChanged() {
                            if (tv.sel !== null && tv.sel !== undefined)
                                tradesList.positionViewAtIndex(tv.sel, ListView.Contain)
                        }
                    }

                    delegate: Rectangle {
                        id: tradeRow
                        required property var modelData
                        required property int index
                        readonly property bool on: index === tv.sel
                        readonly property bool win: modelData.pnlNum >= 0
                        width: tradesList.width
                        height: Theme.sp(44)
                        color: on ? Theme.accentSoft : (rowMa.containsMouse ? Qt.rgba(1, 1, 1, 0.02) : "transparent")
                        Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.rowSep }
                        Rectangle { anchors.left: parent.left; width: Theme.sp(2); height: parent.height; color: tradeRow.on ? Theme.accent : "transparent" }
                        Row {
                            x: Theme.sp(18)
                            height: parent.height
                            spacing: tradesBody.tgap
                            Item { width: tradesBody.cNum; height: parent.height
                                Text { anchors.verticalCenter: parent.verticalCenter; text: modelData.n; color: tradeRow.on ? Theme.accent : Theme.t1; font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: Theme.fontSmall } }
                            Item { width: tradesBody.cFlex; height: parent.height
                                Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; elide: Text.ElideRight; text: Fmt.tsFull(modelData.entryTs); color: Theme.t2; font.family: Theme.mono; font.pixelSize: Theme.fontSmall } }
                            Item { width: tradesBody.cFlex; height: parent.height
                                Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; elide: Text.ElideRight; text: Fmt.tsFull(modelData.exitTs); color: Theme.t2; font.family: Theme.mono; font.pixelSize: Theme.fontSmall } }
                            Item { width: tradesBody.cQty; height: parent.height
                                Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: modelData.qty; color: Theme.t4; font.family: Theme.mono; font.pixelSize: Theme.fontSmall } }
                            Item { width: tradesBody.cPnl; height: parent.height
                                Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: Fmt.fmtUsd(modelData.pnlNum); color: win ? Theme.positive : Theme.negative; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontSmall } }
                            Item { width: tradesBody.cRet; height: parent.height
                                Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: tv.fmtRet1(modelData.retNum); color: win ? Theme.positive : Theme.negative; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontSmall } }
                            Item { width: tradesBody.cBars; height: parent.height
                                Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: modelData.bars; color: Theme.t4; font.family: Theme.mono; font.pixelSize: Theme.fontSmall } }
                        }
                        MouseArea { id: rowMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: App.selectTrade(index) }
                    }
                }
            }

            // --- side panel ---
            Loader {
                id: panelLoader
                width: tradesBody.rightW
                height: bottomRow.height
                sourceComponent: tv.hasSel ? tradeDetail : symSummary
            }

            Component {
                id: tradeDetail
                Rectangle {
                    anchors.fill: parent
                    implicitHeight: dCol.implicitHeight + Theme.sp(40)
                    readonly property var t: tv.trades[tv.sel]
                    readonly property bool win: t.pnlNum >= 0
                    radius: Theme.radiusCard
                    color: Theme.card
                    border.color: Theme.border
                    border.width: 1
                    clip: true

                    Column {
                        id: dCol
                        x: Theme.sp(22); y: Theme.sp(20)
                        width: parent.width - Theme.sp(44)
                        spacing: 0
                        Item {
                            width: parent.width
                            height: Theme.sp(24)
                            Text { anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter; text: "Trade #" + t.n + " · " + tv.sym; color: Theme.textPrimary; font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: Theme.fontBodyLg }
                            Rectangle {
                                anchors.right: parent.right; anchors.verticalCenter: parent.verticalCenter
                                width: badgeT.implicitWidth + Theme.sp(20); height: Theme.sp(22); radius: Theme.sp(4); color: Theme.accentBadge
                                Text { id: badgeT; anchors.centerIn: parent; text: t.side; color: Theme.accent; font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: Theme.fontCaption }
                            }
                        }
                        Item { width: 1; height: Theme.sp(18) }
                        Text { text: Fmt.fmtUsd(t.pnlNum); color: win ? Theme.positive : Theme.negative; font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: Theme.fontHero }
                        Item { width: 1; height: Theme.sp(4) }
                        Text { text: tv.fmtRet2(t.retNum) + " return"; color: win ? Theme.positive : Theme.negative; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontBodyLg }
                        Item { width: 1; height: Theme.sp(20) }
                        Grid {
                            width: parent.width
                            columns: 2
                            rowSpacing: Theme.sp(14)
                            columnSpacing: Theme.sp(16)
                            Column { spacing: Theme.sp(2); width: (dCol.width - Theme.sp(16)) / 2
                                Text { text: "ENTRY"; color: Theme.t6; font.family: Theme.mono; font.pixelSize: Theme.fontMicro; font.letterSpacing: 0.6 * Theme.scale }
                                Text { text: "$" + t.entryPrice.toFixed(2); color: Theme.t1; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontBody }
                                Text { text: Fmt.tsFull(t.entryTs); color: Theme.t6; font.family: Theme.mono; font.pixelSize: Theme.fontMicro }
                            }
                            Column { spacing: Theme.sp(2); width: (dCol.width - Theme.sp(16)) / 2
                                Text { text: "EXIT"; color: Theme.t6; font.family: Theme.mono; font.pixelSize: Theme.fontMicro; font.letterSpacing: 0.6 * Theme.scale }
                                Text { text: "$" + t.exitPrice.toFixed(2); color: Theme.t1; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontBody }
                                Text { text: Fmt.tsFull(t.exitTs); color: Theme.t6; font.family: Theme.mono; font.pixelSize: Theme.fontMicro }
                            }
                            StatCell { label: "QUANTITY"; value: String(t.qty); valuePx: Theme.fontBody }
                            StatCell { label: "BARS HELD"; value: t.bars + " bars"; valuePx: Theme.fontBody }
                            StatCell { label: "MFE"; value: "+" + t.mfe.toFixed(2) + "%"; valueColor: Theme.positive; valuePx: Theme.fontBody }
                            StatCell { label: "MAE"; value: t.mae.toFixed(2) + "%"; valueColor: Theme.negative; valuePx: Theme.fontBody }
                        }
                        Item { width: 1; height: Theme.sp(18) }
                        Rectangle { width: parent.width; height: 1; color: Theme.rowSep }
                        Item { width: 1; height: Theme.sp(16) }
                        Text {
                            width: parent.width
                            wrapMode: Text.WordWrap
                            text: "Chart isolates this trade — entry/exit guides and the P&L band shown. Use ‹ All trades to see every fill on " + tv.sym + "."
                            color: Theme.t5
                            font.family: Theme.mono
                            font.pixelSize: Theme.fontCaption
                            lineHeight: 1.5
                        }
                    }
                }
            }

            Component {
                id: symSummary
                Rectangle {
                    anchors.fill: parent
                    implicitHeight: sCol.implicitHeight + Theme.sp(40)
                    readonly property var s: tv.sumStats()
                    radius: Theme.radiusCard
                    color: Theme.card
                    border.color: Theme.border
                    border.width: 1
                    clip: true

                    Column {
                        id: sCol
                        x: Theme.sp(22); y: Theme.sp(20)
                        width: parent.width - Theme.sp(44)
                        spacing: 0
                        Text { text: tv.sym + " · summary"; color: Theme.textPrimary; font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: Theme.fontBodyLg }
                        Item { width: 1; height: Theme.sp(18) }
                        Grid {
                            width: parent.width
                            columns: 2
                            rowSpacing: Theme.sp(16)
                            columnSpacing: Theme.sp(16)
                            StatCell { label: "NET P&L"; value: s.totalStr; valueColor: s.totalColor; valuePx: Theme.fontHeading }
                            StatCell { label: "WIN RATE"; value: s.win; valueColor: Theme.textPrimary; valuePx: Theme.fontHeading }
                            StatCell { label: "TRADES"; value: String(s.count); valueColor: Theme.textPrimary; valuePx: Theme.fontHeading }
                            StatCell { label: "AVG WIN / LOSS"; value: s.avgWin + " / " + s.avgLoss; valuePx: Theme.fontBody }
                            StatCell { label: "BEST"; value: s.best; valueColor: Theme.positive; valuePx: Theme.fontBody }
                            StatCell { label: "WORST"; value: s.worst; valueColor: Theme.negative; valuePx: Theme.fontBody }
                        }
                        Item { width: 1; height: Theme.sp(18) }
                        Rectangle { width: parent.width; height: 1; color: Theme.rowSep }
                        Item { width: 1; height: Theme.sp(16) }
                        Text {
                            width: parent.width
                            wrapMode: Text.WordWrap
                            text: "All " + s.count + " fills on " + tv.sym + " are plotted on the chart above. Click any trade in the table to isolate it."
                            color: Theme.t5
                            font.family: Theme.mono
                            font.pixelSize: Theme.fontCaption
                            lineHeight: 1.5
                        }
                    }
                }
            }
        }
    }
}
