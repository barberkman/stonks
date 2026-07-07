pragma Singleton
import QtQuick

// Visual system ported from the "Backtest Terminal" design. Sizes and fonts
// are scale-aware: main.qml pushes the window dimensions in, and sp()/fill()
// derive every dimension from the clamped scale factors below.
QtObject {
    // surfaces
    readonly property color bg: "#141414"
    readonly property color panel: "#181818"
    readonly property color card: "#1c1c1c"
    readonly property color input: "#232323"
    readonly property color border: "#323232"
    readonly property color rowSep: "#272727"
    readonly property color rowSepDim: "#232323"

    // text tiers (bright -> dim)
    readonly property color textBright: "#f1f2f4"
    readonly property color textPrimary: "#e6e8ea"
    readonly property color t1: "#cfd3d8"
    readonly property color t2: "#a8adb5"
    readonly property color t3: "#9aa0a8"
    readonly property color t4: "#8b929c"
    readonly property color t5: "#6b727c"
    readonly property color t6: "#5b626c"
    readonly property color t7: "#46535f"

    // accent + semantic
    readonly property color accent: "#4eb36e"
    readonly property color accentInk: "#181818"
    readonly property color accentSoft: Qt.rgba(0x4e / 255, 0xb3 / 255, 0x6e / 255, 0.10)
    readonly property color accentLine: Qt.rgba(0x4e / 255, 0xb3 / 255, 0x6e / 255, 0.35)
    readonly property color accentBadge: Qt.rgba(0x4e / 255, 0xb3 / 255, 0x6e / 255, 0.14)
    readonly property color positive: "#4eb36e"
    readonly property color negative: "#e1574c"
    readonly property color warn: "#cbb26a"

    // fonts — bundled under qml/fonts/ and registered by the FontLoaders below,
    // so the app renders IBM Plex regardless of what's installed on the OS. Sans
    // is the variable font (all weights via its wght axis); Mono ships the static
    // weights the UI uses (Regular/Medium/SemiBold/Bold), grouped under one family.
    readonly property string sans: "IBM Plex Sans"
    readonly property string mono: "IBM Plex Mono"

    readonly property FontLoader _sansLoader: FontLoader { source: "fonts/IBMPlexSans-Variable.ttf" }
    readonly property FontLoader _monoRegular: FontLoader { source: "fonts/IBMPlexMono-Regular.ttf" }
    readonly property FontLoader _monoMedium: FontLoader { source: "fonts/IBMPlexMono-Medium.ttf" }
    readonly property FontLoader _monoSemiBold: FontLoader { source: "fonts/IBMPlexMono-SemiBold.ttf" }
    readonly property FontLoader _monoBold: FontLoader { source: "fonts/IBMPlexMono-Bold.ttf" }

    // responsive scale (window size pushed from main.qml via Binding)
    property real windowWidth: 1440
    property real windowHeight: 900

    // fonts/spacing/radii/control sizes: min(w,h) ratio, clamped for legibility.
    // At the enforced minimum window (1120x680) the raw ratio is 0.756.
    readonly property real scale: Math.max(0.75, Math.min(1.4,
        Math.min(windowWidth / 1440, windowHeight / 900)))

    // chart-canvas growth: height-only and a wider clamp — more chart area is
    // strictly useful, unlike bigger text
    readonly property real fillScale: Math.max(0.75, Math.min(2.0, windowHeight / 900))

    function sp(px) { return Math.round(px * scale) }
    function fill(px) { return Math.round(px * fillScale) }

    // type scale
    readonly property int fontMicro: sp(10)      // table headers, tiny caps labels
    readonly property int fontCaption: sp(11)    // meta text, canvas axis labels
    readonly property int fontSmall: sp(12)      // secondary UI text
    readonly property int fontBody: sp(13)       // primary UI text, buttons
    readonly property int fontBodyLg: sp(14)     // data rows, inputs
    readonly property int fontSub: sp(16)        // sub-headings
    readonly property int fontHeading: sp(18)    // section headings
    readonly property int fontTitle: sp(24)      // page H1
    readonly property int fontMetric: sp(26)     // report metric values
    readonly property int fontHero: sp(32)       // trade-detail P&L

    // structural tokens
    readonly property int radiusControl: sp(5)
    readonly property int radiusCard: sp(6)
    readonly property int controlH: sp(40)       // buttons, input rows
    readonly property int controlHSm: sp(30)     // small buttons, pills
    readonly property int headerH: sp(56)
}
