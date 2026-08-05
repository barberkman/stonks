import QtQuick
import Stonks

// Configure a new backtest: pick a Python strategy + data file, filter it
// (symbols + date range), set starting cash, and run.
Page {
    id: view
    maxWidth: Theme.sp(1080)
    topPad: Theme.sp(36)
    spacing: Theme.sp(20)

    readonly property var strats: App.strategyList()    // [{display, module, cls}]
    readonly property var files: App.dataFileList()      // [{key, label, source}]
    readonly property var stratModules: strats.map(function (s) { return s.module })
    readonly property var fileKeys: files.map(function (f) { return f.key })

    Component.onCompleted: {
        if (strats.length && !App.strategy) { App.strategy = strats[0].module; }
        if (files.length && !App.dataFile) { App.setDataFile(files[0].key); }
    }

    // --- back + title ---
    Row {
        spacing: Theme.sp(14)
        GhostButton { anchors.verticalCenter: parent.verticalCenter; text: "‹ Backtests"; onClicked: App.goBacktests() }
        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: "CONFIGURE RUN"
            color: Theme.accent
            font.family: Theme.mono
            font.weight: Font.Medium
            font.pixelSize: Theme.fontCaption
            font.letterSpacing: 1.6 * Theme.scale
        }
    }
    Text {
        text: "New backtest"
        color: Theme.textBright
        font.family: Theme.sans
        font.weight: Font.Bold
        font.pixelSize: Theme.fontTitle
    }

    // --- strategy + data cards ---
    Row {
        width: view.innerWidth
        spacing: Theme.sp(20)
        readonly property real cardW: (width - Theme.sp(20)) / 2

        // strategy
        Rectangle {
            width: parent.cardW
            height: Theme.sp(158)
            radius: Theme.radiusCard
            color: Theme.card
            border.color: Theme.border
            border.width: 1
            Column {
                x: Theme.sp(22); y: Theme.sp(20)
                width: parent.width - Theme.sp(44)
                spacing: Theme.sp(14)
                Text { text: "STRATEGY"; color: Theme.t5; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontCaption; font.letterSpacing: 1 * Theme.scale }
                StyledSelect {
                    width: parent.width
                    model: view.strats.map(function (s) { return s.display })
                    values: view.stratModules
                    currentIndex: Math.max(0, view.stratModules.indexOf(App.strategy))
                    onPicked: function (v) { App.strategy = v }
                }
                Column {
                    spacing: Theme.sp(4)
                    Text { text: "SOURCE"; color: Theme.t6; font.family: Theme.mono; font.pixelSize: Theme.fontMicro }
                    Text { text: App.strategySource(); color: Theme.t2; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontSmall }
                }
            }
        }

        // data
        Rectangle {
            width: parent.cardW
            height: Theme.sp(158)
            radius: Theme.radiusCard
            color: Theme.card
            border.color: Theme.border
            border.width: 1
            Column {
                x: Theme.sp(22); y: Theme.sp(20)
                width: parent.width - Theme.sp(44)
                spacing: Theme.sp(12)
                Text { text: "DATA"; color: Theme.t5; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontCaption; font.letterSpacing: 1 * Theme.scale }
                StyledSelect {
                    width: parent.width
                    model: view.files.map(function (f) { return f.label })
                    values: view.fileKeys
                    currentIndex: Math.max(0, view.fileKeys.indexOf(App.dataFile))
                    onPicked: function (v) { App.setDataFile(v) }
                }
                Row {
                    width: parent.width
                    spacing: Theme.sp(24)
                    Column {
                        spacing: Theme.sp(4)
                        width: (parent.width - Theme.sp(24)) / 2
                        Text { text: "SYMBOLS"; color: Theme.t6; font.family: Theme.mono; font.pixelSize: Theme.fontMicro }
                        Text { width: parent.width; elide: Text.ElideRight; text: App.symbols.length + " of " + App.availableSymbols.length + " selected"; color: Theme.t2; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontSmall }
                    }
                    Column {
                        spacing: Theme.sp(4)
                        width: (parent.width - Theme.sp(24)) / 2
                        Text { text: "SOURCE"; color: Theme.t6; font.family: Theme.mono; font.pixelSize: Theme.fontMicro }
                        Text { width: parent.width; elide: Text.ElideRight; text: (App.dataFilesObj()[App.dataFile] || {}).source || ""; color: Theme.t2; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontSmall }
                    }
                }
            }
        }
    }

    // --- strategy parameters (generic: rendered from the discovery specs) ---
    Rectangle {
        id: paramCard
        readonly property var paramSpecs: App.paramSpecsFor(App.strategy)
        // A param carrying `choices` is a named selection whose value is the
        // index, so it gets a dropdown rather than a number box — typing "137"
        // to pick a chart pattern out of 212 is not a usable control.
        readonly property var choiceSpecs: paramSpecs.filter(function (s) { return (s.choices || []).length > 0; })
        readonly property var numericSpecs: paramSpecs.filter(function (s) { return s.type !== "bool" && (s.choices || []).length === 0; })
        readonly property var boolSpecs: paramSpecs.filter(function (s) { return s.type === "bool"; })

        width: view.innerWidth
        visible: paramSpecs.length > 0
        radius: Theme.radiusCard
        color: Theme.card
        border.color: Theme.border
        border.width: 1
        implicitHeight: visible ? (paramCol.implicitHeight + Theme.sp(40)) : 0

        Column {
            id: paramCol
            x: Theme.sp(22); y: Theme.sp(20)
            width: parent.width - Theme.sp(44)
            spacing: Theme.sp(16)

            Item {
                width: parent.width
                height: Theme.sp(20)
                Text { anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter; text: "STRATEGY PARAMETERS"; color: Theme.t5; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontCaption; font.letterSpacing: 1 * Theme.scale }
                GhostButton { anchors.right: parent.right; anchors.verticalCenter: parent.verticalCenter; text: "reset to defaults"; onClicked: App.resetParams(App.strategy) }
            }

            Repeater {
                model: paramCard.choiceSpecs
                delegate: Column {
                    required property var modelData
                    width: paramCol.width
                    spacing: Theme.sp(7)
                    Text {
                        text: modelData.doc || modelData.name
                        color: Theme.t4
                        font.family: Theme.sans
                        font.pixelSize: Theme.fontCaption
                    }
                    StyledSelect {
                        width: parent.width
                        readonly property int picked: Math.round(App.paramValue(App.strategy, modelData.name, modelData.default))
                        model: modelData.choices
                        // the parameter travels as the index, so the label list
                        // and the stored value stay in step by position
                        values: modelData.choices
                        currentIndex: (picked >= 0 && picked < modelData.choices.length) ? picked : modelData.default
                        onActivated: function (index) { App.setParamEdit(App.strategy, modelData.name, index) }
                    }
                }
            }

            Flow {
                width: parent.width
                spacing: Theme.sp(18)
                Repeater {
                    model: paramCard.numericSpecs
                    delegate: ParamField {
                        required property var modelData
                        width: (paramCol.width - Theme.sp(36)) / 3
                        label: modelData.doc || modelData.name
                        unit: modelData.unit
                        value: String(App.paramValue(App.strategy, modelData.name, modelData.default))
                        onEdited: function (v) { App.setParamEdit(App.strategy, modelData.name, v) }
                    }
                }
            }

            Flow {
                width: parent.width
                spacing: Theme.sp(8)
                Repeater {
                    model: paramCard.boolSpecs
                    delegate: Rectangle {
                        required property var modelData
                        readonly property bool on: App.paramValue(App.strategy, modelData.name, modelData.default) ? true : false
                        width: boolT.implicitWidth + Theme.sp(28)
                        height: Theme.controlHSm
                        radius: Theme.radiusControl
                        color: on ? Theme.accent : "transparent"
                        border.width: 1
                        border.color: on ? Theme.accent : Theme.border
                        Text {
                            id: boolT
                            anchors.centerIn: parent
                            text: modelData.doc || modelData.name
                            color: parent.on ? Theme.accentInk : Theme.t3
                            font.family: Theme.mono
                            font.weight: Font.DemiBold
                            font.pixelSize: Theme.fontSmall
                        }
                        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: App.setParamEdit(App.strategy, modelData.name, !parent.on) }
                    }
                }
            }
        }
    }

    // --- filter: symbols + date range ---
    Rectangle {
        width: view.innerWidth
        radius: Theme.radiusCard
        color: Theme.card
        border.color: Theme.border
        border.width: 1
        implicitHeight: filterCol.implicitHeight + Theme.sp(40)

        Column {
            id: filterCol
            x: Theme.sp(22); y: Theme.sp(20)
            width: parent.width - Theme.sp(44)
            spacing: Theme.sp(16)

            Text { text: "FILTER"; color: Theme.t5; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontCaption; font.letterSpacing: 1 * Theme.scale }

            // symbol multi-select pills
            Column {
                id: symBox
                width: parent.width
                spacing: Theme.sp(8)
                property string symbolQuery: ""
                readonly property var filteredSymbols: {
                    var q = symbolQuery.trim().toUpperCase();
                    if (q === "") return App.availableSymbols;
                    return App.availableSymbols.filter(function (s) { return s.toUpperCase().indexOf(q) >= 0; });
                }
                Item {
                    width: parent.width
                    height: Theme.controlHSm
                    Text { anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter; text: "SYMBOLS"; color: Theme.t6; font.family: Theme.mono; font.pixelSize: Theme.fontMicro; font.letterSpacing: 0.6 * Theme.scale }
                    Row {
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        spacing: Theme.sp(8)
                        GhostButton { text: "All"; onClicked: App.addSymbols(symBox.filteredSymbols) }
                        GhostButton { text: "None"; onClicked: App.removeSymbols(symBox.filteredSymbols) }
                    }
                }
                // search / filter box
                Rectangle {
                    width: parent.width
                    height: Theme.controlHSm
                    radius: Theme.radiusControl
                    color: Theme.input
                    border.width: 1
                    border.color: symIn.activeFocus ? Theme.accent : Theme.border
                    TextInput {
                        id: symIn
                        anchors.left: parent.left; anchors.leftMargin: Theme.sp(11)
                        anchors.right: parent.right; anchors.rightMargin: Theme.sp(11)
                        anchors.verticalCenter: parent.verticalCenter
                        text: symBox.symbolQuery
                        onTextEdited: symBox.symbolQuery = text
                        color: Theme.textPrimary
                        font.family: Theme.mono
                        font.pixelSize: Theme.fontSmall
                        selectByMouse: true
                        clip: true
                        Text {
                            anchors.verticalCenter: parent.verticalCenter
                            text: "Filter symbols…"
                            visible: symIn.text.length === 0
                            color: Theme.t6
                            font.family: Theme.mono
                            font.pixelSize: Theme.fontSmall
                        }
                    }
                }
                Flow {
                    width: parent.width
                    spacing: Theme.sp(8)
                    Repeater {
                        model: symBox.filteredSymbols
                        delegate: Rectangle {
                            required property var modelData
                            readonly property bool on: App.symbols.indexOf(modelData) >= 0
                            width: pillT.implicitWidth + Theme.sp(28)
                            height: Theme.controlHSm
                            radius: Theme.radiusControl
                            color: on ? Theme.accent : "transparent"
                            border.width: 1
                            border.color: on ? Theme.accent : Theme.border
                            Text {
                                id: pillT
                                anchors.centerIn: parent
                                text: modelData
                                color: parent.on ? Theme.accentInk : Theme.t3
                                font.family: Theme.mono
                                font.weight: Font.DemiBold
                                font.pixelSize: Theme.fontSmall
                            }
                            MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: App.toggleSymbol(modelData) }
                        }
                    }
                }
                Text {
                    visible: symBox.filteredSymbols.length === 0
                    text: "No symbols match “" + symBox.symbolQuery + "”"
                    color: Theme.t6
                    font.family: Theme.mono
                    font.pixelSize: Theme.fontSmall
                }
            }

            // date range
            Row {
                width: parent.width
                spacing: Theme.sp(18)
                ParamField {
                    width: (parent.width - Theme.sp(18)) / 2
                    label: "Start date (YYYY-MM-DD)"
                    unit: ""
                    value: App.startDate
                    onEdited: function (v) { App.startDate = v }
                }
                ParamField {
                    width: (parent.width - Theme.sp(18)) / 2
                    label: "End date (YYYY-MM-DD)"
                    unit: ""
                    value: App.endDate
                    onEdited: function (v) { App.endDate = v }
                }
            }
        }
    }

    // --- fees ---
    Rectangle {
        width: view.innerWidth
        radius: Theme.radiusCard
        color: Theme.card
        border.color: Theme.border
        border.width: 1
        implicitHeight: feeCol.implicitHeight + Theme.sp(40)

        Column {
            id: feeCol
            x: Theme.sp(22); y: Theme.sp(20)
            width: parent.width - Theme.sp(44)
            spacing: Theme.sp(16)

            Text { text: "FEES"; color: Theme.t5; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: Theme.fontCaption; font.letterSpacing: 1 * Theme.scale }

            Row {
                width: parent.width
                spacing: Theme.sp(18)
                ParamField {
                    width: (parent.width - Theme.sp(36)) / 3
                    label: "Maker fee (resting fills)"
                    unit: "bps"
                    value: App.makerFeeBps
                    onEdited: function (v) { App.makerFeeBps = v }
                }
                ParamField {
                    width: (parent.width - Theme.sp(36)) / 3
                    label: "Taker fee (crossing fills)"
                    unit: "bps"
                    value: App.takerFeeBps
                    onEdited: function (v) { App.takerFeeBps = v }
                }
                ParamField {
                    width: (parent.width - Theme.sp(36)) / 3
                    label: "Flat fee per fill"
                    unit: "$"
                    value: App.feePerFill
                    onEdited: function (v) { App.feePerFill = v }
                }
            }
        }
    }

    // --- starting cash + start ---
    Item {
        width: view.innerWidth
        height: Theme.sp(44)
        Row {
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
            spacing: Theme.sp(11)
            Text { anchors.verticalCenter: parent.verticalCenter; text: "Starting cash"; color: Theme.t4; font.family: Theme.sans; font.pixelSize: Theme.fontSmall }
            Rectangle {
                anchors.verticalCenter: parent.verticalCenter
                width: Theme.sp(160)
                height: Theme.controlH
                radius: Theme.radiusControl
                color: Theme.input
                border.color: "#3c3c3c"
                border.width: 1
                Text { id: dollar; anchors.left: parent.left; anchors.leftMargin: Theme.sp(11); anchors.verticalCenter: parent.verticalCenter; text: "$"; color: Theme.t5; font.family: Theme.mono; font.pixelSize: Theme.fontBody }
                TextInput {
                    anchors.left: dollar.right
                    anchors.leftMargin: Theme.sp(4)
                    anchors.right: parent.right
                    anchors.rightMargin: Theme.sp(8)
                    anchors.verticalCenter: parent.verticalCenter
                    text: App.startCash
                    color: Theme.textPrimary
                    font.family: Theme.mono
                    font.pixelSize: Theme.fontBodyLg
                    selectByMouse: true
                    clip: true
                    onEditingFinished: App.startCash = text
                }
            }
            Text {
                anchors.verticalCenter: parent.verticalCenter
                visible: App.runError !== ""
                text: "⚠ " + App.runError
                color: Theme.negative
                font.family: Theme.mono
                font.pixelSize: Theme.fontSmall
                width: Theme.sp(360)
                elide: Text.ElideRight
            }
        }
        AccentButton {
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            text: "▸ Start backtest"
            hpad: Theme.sp(26)
            fontPx: Theme.fontBodyLg
            onClicked: App.runBacktest()
        }
    }
}
