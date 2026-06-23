import QtQuick
import Stonks

// Report tab: metrics, equity + drawdown charts, run stats, per-symbol table.
Column {
    id: report
    readonly property var bt: App.currentBacktest()

    function metrics() {
        var b = bt;
        return [
            { label: "RETURN", value: b.ret, color: b.retPos ? Theme.positive : Theme.negative },
            { label: "MAX DD", value: b.maxdd, color: Theme.negative },
            { label: "WIN RATE", value: b.win, color: Theme.textPrimary },
            { label: "TRADES", value: String(b.trades), color: Theme.textPrimary },
            { label: "SHARPE", value: b.sharpe, color: Theme.textPrimary },
            { label: "ENDING EQUITY", value: b.endEqStr, color: Theme.accent }
        ];
    }
    function reportSymbols() {
        var syms = bt.symbols || [];
        var out = [];
        for (var i = 0; i < syms.length; i++) {
            var s = syms[i];
            if (s.ret === undefined) continue;
            out.push({ id: s.id, seed: App.symMeta(s.id).seed, bias: (s.retPos ? 0.011 : -0.008),
                       ret: s.ret, pnl: s.pnl, trades: s.trades, win: s.win,
                       col: s.retPos ? Theme.positive : Theme.negative });
        }
        return out;
    }

    // --- 1. metric cells ---
    Item {
        width: report.width
        height: 80
        Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.border }
        Row {
            width: parent.width
            Repeater {
                model: report.metrics()
                delegate: MetricCell {
                    required property var modelData
                    required property int index
                    width: report.width / 6
                    height: 80
                    label: modelData.label
                    value: modelData.value
                    valueColor: modelData.color
                    rightBorder: index < 5
                }
            }
        }
    }

    // --- 2. portfolio equity ---
    Item {
        width: report.width
        height: 298
        Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.border }
        Item {
            x: 24; y: 20
            width: parent.width - 48
            height: 16
            Text { anchors.left: parent.left; text: "PORTFOLIO EQUITY"; color: Theme.t4; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 11; font.letterSpacing: 0.8 }
            Text { anchors.right: parent.right; text: report.bt.startEqStr + " → " + report.bt.endEqStr + " · " + report.bt.range; color: Theme.t6; font.family: Theme.mono; font.pixelSize: 11 }
        }
        EquityChart {
            x: 24; y: 68
            width: parent.width - 48
            height: 230
            data: App.equityFor(report.bt)
            lineColor: report.bt.retPos ? Theme.accent : Theme.negative
            grid: true
        }
    }

    // --- 3. drawdown + run stats ---
    Item {
        width: report.width
        height: 180
        Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.border }
        Row {
            width: parent.width
            height: parent.height
            Item {
                width: parent.width * 0.6
                height: parent.height
                Rectangle { anchors.right: parent.right; width: 1; height: parent.height; color: Theme.border }
                Column {
                    x: 24; y: 20
                    width: parent.width - 48
                    spacing: 12
                    Text { text: "DRAWDOWN"; color: Theme.t4; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 11; font.letterSpacing: 0.8 }
                    DrawdownChart {
                        width: parent.width
                        height: 110
                        data: App.drawdownFor(report.bt)
                    }
                }
            }
            Item {
                width: parent.width * 0.4
                height: parent.height
                Grid {
                    x: 24
                    anchors.verticalCenter: parent.verticalCenter
                    columns: 2
                    rowSpacing: 16
                    columnSpacing: 22
                    StatCell { label: "BARS"; value: report.bt.bars }
                    StatCell { label: "ORDERS"; value: report.bt.orders }
                    StatCell { label: "NOTIONAL"; value: report.bt.notional }
                    StatCell { label: "PROFIT FACTOR"; value: report.bt.pf }
                    StatCell { label: "ELAPSED"; value: report.bt.elapsed }
                    StatCell { label: "PER BAR"; value: report.bt.perbar }
                }
            }
        }
    }

    // --- 4. per-symbol performance ---
    Item {
        id: perSym
        width: report.width
        implicitHeight: symCol.implicitHeight + 60

        readonly property real cw: width - 48
        readonly property real gap: 18
        readonly property real cSym: 90
        readonly property real cRet: 110
        readonly property real cPnl: 130
        readonly property real cTrd: 80
        readonly property real cWin: 70
        readonly property real cAnalyze: 110
        readonly property real cEq: cw - cSym - cRet - cPnl - cTrd - cWin - cAnalyze - gap * 6

        Column {
            id: symCol
            x: 24; y: 20
            width: parent.width - 48
            spacing: 0

            Item {
                width: parent.width
                height: 34
                Text { anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter; text: "PER-SYMBOL PERFORMANCE"; color: Theme.t4; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 11; font.letterSpacing: 0.8 }
                Text { anchors.right: parent.right; anchors.verticalCenter: parent.verticalCenter; text: "click a row → trade analysis"; color: Theme.t6; font.family: Theme.mono; font.pixelSize: 11 }
            }

            // table header
            Item {
                width: parent.width
                height: 30
                Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.border }
                Row {
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: perSym.gap
                    Text { width: perSym.cSym; text: "SYMBOL"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 10; font.letterSpacing: 1 }
                    Text { width: perSym.cEq; text: "EQUITY"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 10; font.letterSpacing: 1 }
                    Text { width: perSym.cRet; horizontalAlignment: Text.AlignRight; text: "RETURN"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 10; font.letterSpacing: 1 }
                    Text { width: perSym.cPnl; horizontalAlignment: Text.AlignRight; text: "P&L"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 10; font.letterSpacing: 1 }
                    Text { width: perSym.cTrd; horizontalAlignment: Text.AlignRight; text: "TRADES"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 10; font.letterSpacing: 1 }
                    Text { width: perSym.cWin; horizontalAlignment: Text.AlignRight; text: "WIN"; color: Theme.t6; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 10; font.letterSpacing: 1 }
                    Item { width: perSym.cAnalyze; height: 1 }
                }
            }

            // rows
            Repeater {
                model: report.reportSymbols()
                delegate: Rectangle {
                    required property var modelData
                    width: symCol.width
                    height: 56
                    color: symMa.containsMouse ? Qt.rgba(1, 1, 1, 0.02) : "transparent"
                    Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.rowSep }
                    Row {
                        anchors.verticalCenter: parent.verticalCenter
                        spacing: perSym.gap
                        Item {
                            width: perSym.cSym; height: 56
                            Text { anchors.verticalCenter: parent.verticalCenter; text: modelData.id; color: Theme.textPrimary; font.family: Theme.mono; font.weight: Font.DemiBold; font.pixelSize: 14 }
                        }
                        Sparkline {
                            width: perSym.cEq; height: 30
                            anchors.verticalCenter: parent.verticalCenter
                            data: App.sparkSeries(modelData.seed, modelData.bias)
                        }
                        Item {
                            width: perSym.cRet; height: 56
                            Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: modelData.ret; color: modelData.col; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 14 }
                        }
                        Item {
                            width: perSym.cPnl; height: 56
                            Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: modelData.pnl; color: modelData.col; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 14 }
                        }
                        Item {
                            width: perSym.cTrd; height: 56
                            Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: String(modelData.trades); color: Theme.t4; font.family: Theme.mono; font.pixelSize: 14 }
                        }
                        Item {
                            width: perSym.cWin; height: 56
                            Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: modelData.win; color: Theme.t4; font.family: Theme.mono; font.pixelSize: 14 }
                        }
                        Item {
                            width: perSym.cAnalyze; height: 56
                            Text { anchors.verticalCenter: parent.verticalCenter; width: parent.width; horizontalAlignment: Text.AlignRight; text: "analyze →"; color: Theme.accent; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 12 }
                        }
                    }
                    MouseArea {
                        id: symMa
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: App.openSymbolTrades(modelData.id)
                    }
                }
            }
        }
    }
}
