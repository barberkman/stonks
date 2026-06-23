pragma Singleton
import QtQuick

// Static visual system ported from the "Backtest Terminal" design.
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

    // fonts (installed system families)
    readonly property string sans: "IBM Plex Sans"
    readonly property string mono: "IBM Plex Mono"
}
