# Backtest Logic Audit — 2026-07

Deep audit of the backtest engine, broker simulation, strategies, and reporting.
Every finding carries a `file:line` reference into the current tree (commit `78c8211`) and,
where possible, empirical confirmation from real run outputs under `app/reports/`.

**Verdict up front:** the core architecture is sound — deterministic, lookahead-safe,
cleanly layered — but the newest layer (bracket/OCO exits + isolated-margin leverage,
the last two commits) contains one critical and several high-severity logic errors that
compound. **Backtest results produced so far are not meaningful**: every bracket exit
degenerates to "close at the next bar's open," and the leverage the strategies compute
(62–125x) turns that into a same-bar liquidation lottery.

---

## 1. Executive summary

Three smoking guns, straight from the archived runs:

| Evidence | Daily run (`report-20260705-144624.json`) | 1h run (`report-20260705-144306.json`) |
|---|---|---|
| Position lifecycles closed | 440 | 18,508 |
| … closed within ~1 bar | **440 / 440 (100%)** | **18,508 / 18,508 (100%)** |
| Exits via bracket "stop/TP" legs | 332 (all Limit children) | 15,061 |
| Exits via forced liquidation | 108 (median hold: **0 days** — same bar as entry) | 3,447 |
| Entry leverage observed | 62x | 125x (the strategy's cap) |
| Ending equity | 936.15 of 1,000 | **1.08 × 10⁻⁸¹** of 1,000 — and the bankruptcy flag never fired |

Root causes, in one paragraph: the broker supports only **Market and Limit** order types,
so the stop-loss leg of every bracket is a limit order on the *wrong side* of the market and
fills immediately at the next bar's open (E1). Protective legs are barred from acting on the
entry bar while liquidation is not (E3), so at the 62–125x leverage the strategies request,
violent entry bars liquidate before any exit can exist. The leverage itself is mis-calibrated
because sizing is computed against the *signal* price while the market entry fills at the
*next open* (S1). Below those two headliners sit a float-dust bug that can silently block a
symbol from ever re-entering (E4), an "isolated" margin model that leaks gap losses into the
whole account (E5), a bankruptcy check that can never trigger (E6), and a Context API that
gives strategies no way to observe positions, fills, rejections, or to cancel anything (E7).

The metrics/reporting layer, by contrast, **checked out clean**: win rate, max drawdown, and
the PnL⇄cash identity all reproduced exactly from the raw trade data (§5).

---

## 2. System overview (what this audit understood the system to be)

**stonks** is a C++20 backtesting system intended to grow into live trading
(`docs/project-brief.md`): strategies, data feeds, and brokers are compile-time concepts
(`include/stonks/core/{strategy,datafeed,broker}.h`), so a `LiveBroker`/`LiveFeed` can later
replace the backtest implementations without touching the engine.

Per-timestamp event loop (`include/stonks/core/engine.h:60-84`):

```
next_timestamp → clock.set(ts)
  → broker.on_tick(bar)     for every symbol printing at ts   (fills happen here)
  → equity marked           one EquityPoint per ts, after fills, before the strategy
  → strategy.on_tick(ctx)   sees history up to and including ts, places orders stamped ts
  → feed.advance()
```

Orders stamped at `ts` cannot fill on a bar with the same timestamp
(`src/broker/backtestbroker.cpp:125` — the lookahead gate), so all fills are next-bar-or-later.
The feed enforces no-lookahead independently (`KLineFeed::window` can only slice up to the
current row, `src/datafeed/klinefeed.cpp:301-317`). Both layers are regression-tested
(`tests/core/lookahead_test.cpp`, `backtestbroker_test.cpp::OrderDoesNotFillOnItsOwnBarTimestamp`).

The broker (`src/broker/backtestbroker.cpp`) is cash-secured, one-position-per-symbol, never
averages in (same-side orders are rejected), supports bracket/OCO chaining via `parent_id`,
and models isolated-margin leverage with forced liquidation at the bankruptcy price plus an
account-wide bankruptcy sweep. A Qt/QML terminal drives the engine on a worker thread;
Python strategies run in an embedded interpreter through a type-erased `IContext`.

**Strengths worth preserving** (all verified): byte-identical determinism across runs
(`determinism_test.cpp`, and the run order is pinned by stable sorts in the feed);
two-layer no-lookahead defense; concept-driven decoupling; the columnar feed with zero-copy
per-symbol windows; a genuinely useful ~189-test suite; clean GUI threading (single worker
thread owns the engine + interpreter, atomics for progress/cancel, queued result marshaling).

---

## 3. Engine logic errors

Findings ordered by severity. Format: **Where · Evidence · Impact · Fix**.

### E1 — CRITICAL: No stop order type; bracket stop-loss legs fill instantly

**Where:** `include/stonks/core/types.h:32-36` (`OrderType` = `Market`, `Limit` only);
fill logic `src/broker/backtestbroker.cpp:138-156`; SL legs placed as Limit children in
`app/strategies/qmsignals.h:122-129` and `app/python/qmsignals.py:173-175`.

A protective stop for a long position is a *sell below the market*. Expressed as a **limit**
sell, it is immediately marketable: the fill branch is

```cpp
if (bar.high < limit) { continue; }        // never true when limit is below the market
price = std::max(limit, bar.open);         // ⇒ fills at the open, right now
```

The limit semantics themselves are correct — the error is that no order type exists that
*waits* on the stop side, so every bracket's SL leg executes as "sell at next open,
unconditionally." (Mirror case for shorts: the buy-stop-as-limit fills at
`min(limit, open)` = open.)

**Evidence:** in the daily run, **all 440** lifecycles closed within one bar; all 332
non-liquidation exits were Limit children. Since `qmsignals.h:123` places the SL *before*
the TP, the SL is swept first every bar (E2) and the TP never gets a chance. The 1h run
shows the same shape at scale (18,508/18,508).

**Impact:** every bracketed strategy actually trades "enter, then exit at the next open."
Take-profits are dead code; stops protect nothing; all published performance numbers
measure noise.

**Fix:** add `OrderType::Stop` (stop-market): buy-stop triggers on `bar.high >= trigger`,
fills at `max(trigger, bar.open)`; sell-stop triggers on `bar.low <= trigger`, fills at
`min(trigger, bar.open)` — the same worse-of convention the liquidation fill already uses
(`backtestbroker.cpp:284-285`). Thread `StopOrderParams` through the Broker concept, Context,
`IContext`/`ContextAdapter`, pybind11 bindings, and `FakeContext`. StopLimit can wait; nothing
needs it yet. (Roadmap P1.)

### E2 — HIGH: Same-bar multi-trigger ambiguity is resolved by placement order

**Where:** `src/broker/backtestbroker.cpp:110` (sweep iterates `m_open_orders`, a plain
vector in placement order); liquidation checked only *after* the sweep (`:268-288`).

When one bar's range touches both a bracket's stop and its take-profit, whichever leg the
strategy happened to `place_order` first wins; `cancel_subtree(..., keep)` (`:233-237`)
then cancels the sibling. There is no intrabar price-path model and no documented policy —
qmsignals placing SL before TP makes ties "conservatively" favor the stop *by accident*.
Similarly, a resting exit that filled this bar pre-empts a liquidation that also breached
this bar (`:268-270`) — an *optimistic* assumption baked in silently.

**Evidence:** confirmed no test exercises a same-bar SL+TP double-touch (the bracket tests
all trigger legs on separate bars).

**Impact:** results depend on incidental code order; after E1 is fixed this becomes the
next dominant bias, worth deciding deliberately.

**Fix:** an explicit, configurable intrabar fill policy — sort each bar's triggerable
candidates by `fill_priority(type, policy)` with `Conservative` default (Market → Stop →
Limit: protective exits before profit-taking), `Optimistic` available for sensitivity
re-runs. Deterministic, documented, tested. (Roadmap P2.)

### E3 — HIGH: Protective children can't act on the entry bar — but liquidation can

**Where:** child arming rewrites the child's timestamp to the parent's fill bar
(`src/broker/backtestbroker.cpp:256-263`), which the lookahead gate (`:125`) then blocks for
the rest of that bar; the liquidation check (`:271-288`) runs the same bar.

A position opened at this bar's open can be liquidated by this bar's low/high
(`leverage_test.cpp::PositionOpenedAndLiquidatedOnTheSameBar` pins this as intended), yet its
own stop leg is structurally forbidden from filling until the *next* bar. At 62–125x, the
bankruptcy price sits 0.8–1.6% from entry — inside a normal daily bar's range.

**Evidence:** 108 of the daily run's 440 lifecycles ended in liquidation with **median hold
0 days** — entry and liquidation on the same bar, stop never eligible.

**Impact:** exactly when protection matters most (violent bars), only the *forced* exit can
happen, at the worst price. Losses that a working stop would cap at ~risk-per-trade become
full-margin wipeouts.

**Fix:** delete the timestamp rewrite. A child's original placement timestamp is always
strictly earlier than the bar its parent fills on (the parent had to clear the same gate),
so with the rewrite gone, children become naturally eligible from the parent's fill bar
onward — the unchanged gate keeps doing its job for everything else. Combine with E2's
policy loop (re-sweep until no progress) so entry → SL can resolve within one bar,
deterministically. (Roadmap P2.)

### E4 — HIGH: Float dust positions permanently block a symbol

**Where:** `src/broker/backtestbroker.cpp:216` (`filled_qty = min(order.quantity,
|position.quantity|)`), flatness test `:225,233` (`position.quantity == 0.0`, exact float
equality). No epsilon exists anywhere in the broker.

Quantities are equity-fraction-sized doubles. Any partial-exit scheme whose legs don't sum
bit-exactly (`q/3 + q/3 + q/3 ≠ q`, or a close computed from a different equity snapshot)
leaves ~1e-16 residue. The position then never reads as flat, so:

- every subsequent same-side entry on that symbol is rejected (`:204-211`) — **forever**;
- OCO cancel-on-flat (`:233-237`) never fires, leaving zombie bracket legs armed.

**Impact:** a strategy can silently lose the ability to trade a symbol mid-run; results are
then a function of floating-point rounding. (Current shipped strategies close with the same
`qty` object, which is why this hasn't visibly fired yet — `qbreakout.py`'s scale-out design
is one refactor away from tripping it.)

**Fix:** relative-epsilon flat snap after each close:
`|remaining| <= flat_epsilon * max(|qty_before_close|, 1.0) → 0.0`. (Roadmap P3.)

### E5 — MEDIUM: "Isolated margin" isn't isolated, and there's no maintenance margin

**Where:** liquidation trigger `src/broker/backtestbroker.cpp:274-277`
(`bankruptcy_price = entry × (1 ∓ 1/L)`); gap-through fill `:284-287` takes the excess loss
out of account cash (`liquidate_position`, `:388-391`, has no cap).

Two deviations from the documented intent (`app/notes/position-calculator-formulas.md`):

1. **§8 is only half-implemented.** The note's formula is `P_liq = E·(1 ∓ 1/L) / (1 ∓ m)`;
   the code hard-codes `m = 0`, i.e. liquidation only at **100% margin loss** (bankruptcy),
   later and further from entry than any real venue.
2. **Losses beyond margin hit the account.** On a gap through the bankruptcy price the fill
   is the open, and the beyond-margin excess is drained from `m_cash` — cross-margin
   behavior wearing an isolated-margin label. A real isolated position loses at most its
   posted margin.

Also: §§1–7 of that note (fees, fee-aware breakeven/PnL/R:R) have **no code counterpart at
all** — no fee/commission/slippage field exists anywhere in `core::` types or the broker.

**Impact:** liquidations happen too late and cost too much; multi-position accounts can be
sunk by one symbol's gap, which the "isolated" model is supposed to prevent by construction.

**Fix:** `maintenance_margin_rate` in a new `BrokerConfig` (formula reduces to today's at
`m = 0`), plus an `isolated_loss_cap` flag clamping forced-close PnL at `-margin` (forced
closes only — voluntary fills keep real gap risk). (Roadmap P5; fees P10.)

### E6 — MEDIUM: The bankruptcy stop can never fire for fraction-of-equity sizing

**Where:** `src/broker/backtestbroker.cpp:293-294` (`equity() <= 0.0`); no minimum-notional
check on opens; `Engine::run` never consults `bankrupt()` (`include/stonks/core/engine.h` —
by design, it's not in the Broker concept).

With sizing proportional to equity, losses are multiplicative: equity decays geometrically
and never crosses ≤ 0 unless a gap-through (E5) drags cash negative in one step.

**Evidence:** the 1h run ended at equity **1.08 × 10⁻⁸¹** — ~270 consecutive halvings,
never "bankrupt," the engine happily simulating sub-atomic position sizes for months of
bars.

**Impact:** ruined accounts keep producing plausible-looking trades and metrics; runtime is
wasted; `return_pct = -100.00` is the only tell.

**Fix:** configurable `min_equity` floor for the sweep and `min_notional` gate on opening
fills (both default 0 = current behavior). Optionally let the app stop the run early when
`bankrupt()` flips. (Roadmap P4.)

### E7 — HIGH (API): Strategies cannot observe positions, order status, or cancel anything

**Where:** `include/stonks/core/context.h:29-56` — the entire strategy-facing surface is
`now / cash / equity / history / place_order`. The broker itself has no `cancel_order`.
Children may only attach while the parent is still `Open` (`backtestbroker.cpp:330-335`),
so protection cannot be added after a fill. `place_order` returns an always-nonzero
`OrderID` even for rejected orders (`register_order` returns `id` unconditionally, `:359-361`).

Consequences visible in the shipped code:

- `app/python/qbreakout.py` maintains a full shadow ledger (`_reconcile`/`_manage`) that
  re-implements broker fill logic by hand — and desyncs the moment the broker liquidates or
  rejects something the model didn't predict.
- `app/python/ema50strategy.py:50` gates state on `if ctx.place_market_order(...):` — the
  return is an `OrderID` (always truthy), so rejection is undetectable and the strategy's
  `held_quantity` goes wrong silently. The docs actively teach this bug:
  `app/python/README.md:139-140,171` and `docs/architecture.md:190-191,348-349` document the
  return type as `bool`.
- An opposite-side "entry" on a symbol with an open position silently *closes* it
  (`:213-237`) — the strategy has no way to know that's what happened (see S2 for the
  orphan-children cascade this enables).

**Fix:** add `position(symbol) → optional<Position>` and `cancel_order(id) → bool` to the
Broker concept and Context; `Context::order(id) → optional<Order>` (derivable from the
already-required `orders()` map); allow attaching children to `Filled` parents; mirror all
of it through `IContext` → `ContextAdapter` → pybind11 → `FakeContext`; fix the docs
(`bool` → `OrderID`, document `leverage`/`parent`). (Roadmap P6, P11.)

### E8 — Realism gaps (deliberate, but now load-bearing)

No fees, no slippage, no funding. `TimeInForce` is stored on every order and never read
(`GTC` is the only value). Zero-cost fills materially flatter any strategy that trades as
often as the current ones do (the daily run turned over 412k notional on a 1k account).
The project brief (`docs/project-brief.md:97-104`) explicitly wants at least a slippage
assumption as the starting realism level. (Roadmap P10 adds maker/taker fees; slippage is
in the improvement roadmap.)

---

## 4. Strategy logic errors

### S1 — CRITICAL: Risk/leverage calibrated to the signal price, filled at a different one

**Where:** `app/strategies/qmsignals.h:109-119` — `qty = equity × risk_fraction /
|entry − stop|` and `lev = entry_leverage(s.entry, s.stop, …)` are computed from the
**signal** price `s.entry`, then the entry is placed as a **Market** order that fills at the
*next bar's open* `O ≠ s.entry`.

`entry_leverage` (`qmsignals.h:168-178`) faithfully implements the §9 formula so that
liquidation lands just past the stop — *relative to `s.entry`*. The actual position's
liquidation price is `O·(1 − 1/L)`; for a long momentum breakout, `O` gaps above `s.entry`,
dragging liquidation **above the stop**, so the position liquidates before the stop can act
(compounding E3). Neither `qty` nor the risk-fraction promise survives the slippage: with
`use_lod_stop = true` tightening stops to the signal bar's low (`:344`), computed leverage
reaches the 125x cap — 0.8% from liquidation.

**Python is a different strategy entirely:** `app/python/qmsignals.py:169-171` enters via
**Limit at `s.entry`** — which, for a breakout that closed *above* the pivot, is a resting
buy-the-retest order that can sit GTC for months and fill in a context the signal no longer
describes. Both ports share the signal math and are tested for identical leverage numbers,
but their executions are not comparable.

**Fix:** enter via **stop-entry at `s.entry`** (new Stop type) in both ports: fills at
`s.entry` when reached without a gap (calibration exact by construction), stays unfilled on
reversals instead of buying them. (Roadmap P7.)

### S2 — HIGH: Stateless re-fire, silent netting, and phantom-position orphans

**Where:** `app/strategies/qmsignals.h:92-144` — the scanner is stateless per bar; nothing
suppresses a setup that stays true, and nothing checks for an existing position/pending
entry (it can't — E7).

Failure chain: symbol is long from `breakout`; a bar later `parabolic_short` fires on the
same symbol → its "entry" order silently **closes** the long (one-position netting) → the
short's own SL/TP children then arm against a *flat* book → the SL (a buy-limit above
market) is instantly marketable and **opens an unmanaged long at leverage 1** with no owner
(the strategy doesn't know it exists, and nothing will ever close it except an accidental
opposite fire).

**Fix:** per-symbol gate — skip signals while `ctx.position(sym)` exists, a pending entry
is `Open`, or a post-exit cooldown is running (needs the P6 API). A broker-level
"bracket children are close-only" hardening is a worthwhile follow-up defense.
(Roadmap P8.)

### S3 — Port divergences: the "mirrors" aren't mirrors

- `app/python/ema50strategy.py:47` sizes `qty = ctx.cash() / close` — **100% of cash per
  entry** — while the C++ original risks **1% of equity**
  (`app/strategies/ema50strategy.h:20,62`; pinned by
  `ema50strategy_test.cpp::EntrySizedToOnePercentOfEquity`). The Python docstring claims
  it's a mirror. It also has no guard against two symbols crossing on the same tick both
  requesting full cash (qbreakout solves this with its `committed` accumulator;
  ema50 lacks it).
- The `if ctx.place_market_order(...):` truthiness bug (E7) — the C++ port at least
  documents its optimistic tracking honestly (`ema50strategy.h:64-65`).
- qmsignals entry-type divergence (S1). No C++/Python parity test exists for any ported
  strategy — drift is currently unobservable. (Roadmap P9 + test additions.)

---

## 5. Metrics & reporting review — validated correct

Independently recomputed from the daily run's raw trade list (no engine code involved):

| Metric | Reported | Recomputed | Match |
|---|---|---|---|
| Closed trades / winners | 440 / 196 | 440 / 196 | ✅ |
| Win rate | 44.55% | 44.55% | ✅ |
| Max drawdown | 44.8829% | 44.8829% (from the equity curve) | ✅ |
| PnL ⇄ cash identity | — | 1000 + Σpnl(−63.85) = 936.15 = ending cash | ✅ |

So `report.h`'s round-trip reconstruction, win-rate convention (breakeven ≠ win), drawdown,
and the broker's cash accounting are mutually consistent. Caveats worth fixing eventually:

- **Duplicated reconstruction logic.** The same round-trip algorithm lives in
  `app/src/report.h:76-131` and `app/src/analytics.h:83-155`, kept in sync by comment only;
  no cross-check test asserts they agree. Extract one shared implementation.
- **Sharpe annualization is untested and calendar-fixed.** `annualization_from`
  (`app/src/backtestworker.cpp:66-84`) derives bars/year from the *median* timestamp gap on
  a 365.25-day calendar — right for crypto, silently wrong for `us_1d.parquet`
  (252 trading days), and unreachable by any test (anonymous namespace in an untested .cpp).
- **Two drawdown implementations** (scalar metric seeded at `starting_cash`,
  `report.h:140-151`; GUI chart series seeded at −∞ as a signed fraction,
  `app/src/resultmap.cpp:160-168`). Both are individually fine; just know they can disagree
  on runs whose equity never regains the starting level.
- `metrics.trade_count` counts fills (880), `closed_trades` counts round trips (440), and
  liquidation fills ride under synthetic `Filled Market` orders — worth a glossary line in
  the report JSON docs so downstream consumers don't misread.

---

## 6. Data layer & Python boundary notes

- **Ragged symbols are structurally invisible.** A symbol with no bar at a timestamp is
  absent from `current_bars()` and `history()` (`src/datafeed/klinefeed.cpp:280-299,319-333`).
  While it doesn't print: its resting orders are never evaluated, its mark is stale, and its
  **liquidation check does not run** (`backtestbroker.cpp:270-271`) — a leveraged position in
  a gapping symbol is insolvent-but-unliquidatable until it prints again. Documented
  behavior, but a real risk once feeds go beyond top-liquidity perps.
- **`history(n)` is "last n prints," not "last n periods."** Cross-symbol math over it
  (e.g. `qbreakout.py`'s 63-bar momentum rank in `leaders()`) compares different calendar
  windows when symbols gap unevenly.
- **Stale-module trap:** the embedded interpreter never re-imports — editing a `.py`
  strategy requires restarting the app (`sys.modules` cache;
  `include/stonks/python/embeddedpython.h` deliberately never finalizes), and the GUI layers
  its own strategy-list cache on top (`app/src/backtestcontroller.h:68`).
- **CLI path has no exception guard:** `app/src/main.cpp:27-61` runs the engine bare — a
  Python strategy exception there aborts the process (the GUI worker wraps everything,
  `app/src/backtestworker.cpp:88-188`).
- **Test-suite blind spots:** same-bar SL+TP double-touch (E2); the parquet-reading
  `KLineFeed` constructor (only the in-memory Row path is tested); `annualization_from`;
  C++/Python strategy parity; and the two pytest suites under `app/python/` don't run under
  `ctest`, so "the suite" passing says nothing about them.
- **Docs drift:** `CLAUDE.md` describes `apps/<name>/` (e.g. `backtest_runner`) — reality is
  a single `app/` target; plus the `bool` return-type errors (E7) and missing
  `leverage`/`parent` parameters in `app/python/README.md`.

---

## 7. Fix roadmap (phased; each phase independently buildable + ctest-green)

Config philosophy: everything lands behind a new `BrokerConfig` whose defaults reproduce
today's behavior bit-for-bit, so each phase is a pure superset and existing tests keep
meaning something.

```cpp
enum class IntrabarFillPolicy : std::uint8_t { Conservative, Optimistic };

struct BrokerConfig
{
    IntrabarFillPolicy fill_policy = IntrabarFillPolicy::Conservative;
    double flat_epsilon = 1e-9;
    double min_equity = 0.0;
    double min_notional = 0.0;
    double maintenance_margin_rate = 0.0;
    bool isolated_loss_cap = false;
    double maker_fee_bps = 0.0;
    double taker_fee_bps = 0.0;
};
// BacktestBroker(Balance initial_cash, BrokerConfig config = {});  — both call sites unchanged
```

| Phase | Fixes | Key changes | New/updated tests |
|---|---|---|---|
| **P1 Stop orders** | E1 | `OrderType::Stop` + `StopOrderParams` (`types.h`); fill branch mirrors Limit with trigger-on-touch, fill at worse-of(trigger, open); thread through Broker concept, Context, IContext/adapter/bindings, `FakeContext` (`order_type` tag), all four test doubles | new `stoporder_test.cpp`: trigger/gap/reject/entry-usable/dormant-child cases; `contextadapter_test.cpp::PlaceStopOrderBuildsAndForwards` |
| **P2 Fill policy + same-bar children** | E2, E3 | Sweep becomes rounds-until-no-progress; candidates sorted by `fill_priority(type, policy)` (Conservative: Market→Stop→Limit); **delete** the child-timestamp rewrite (`backtestbroker.cpp:256-263`) — the untouched lookahead gate then admits children from the parent's fill bar | new `intrabarfillpolicy_test.cpp`: double-touch conservative/optimistic, same-bar entry+SL, grandchild cascade, OCO-loser-cancelled; two `leverage_test.cpp` tests re-specified (they currently encode E1 in their expected numbers) |
| **P3 Dust epsilon** | E4 | Relative flat-snap before the flatness check | three-way scale-out → re-entry allowed; dust triggers cancel-on-flat; genuine-leftover negative case |
| **P4 Floors** | E6 | `min_equity` in the sweep condition; `min_notional` on opening fills | floor-triggers-bankruptcy; legacy-default regression; min-notional open-only |
| **P5 Margin model** | E5 | Liquidation price gains `/(1∓m)`; `isolated_loss_cap` clamps forced-close PnL at −margin (forced closes only) | m-moves-liq-price; m=0 legacy exact; gap-through capped at margin; voluntary close unaffected; bankrupt sweep never negative |
| **P6 Observability API** | E7 | `position()`, `cancel_order()` into Broker concept + broker; `Context::order/position/cancel_order`; parent may be `Open` **or** `Filled`; Python: bind `Order`, `Position`, `OrderStatus`, new methods; `FakeContext` parity | position query lifecycle; cancel cascades + returns-false cases; attach-to-filled-parent; python round-trip fixture |
| **P7 qmsignals entry** | S1 | Stop-entry at `s.entry` in both ports (unifies them) | entry-type assertions updated; stop-entry never fills on reversal / fills exactly at signal price (through a real `BacktestBroker`) |
| **P8 qmsignals state** | S2 | Per-symbol gate: skip while positioned / pending / cooling down (`cooldown_bars = 5`) | no-refire-while-open/pending; opposite-side suppressed-not-netted; cooldown elapses |
| **P9 EMA50 parity** | S3 | Python sizing → `equity × 1% / close`; drop the truthiness gate; both ports read `ctx.position()` instead of shadow `held_quantity` | parity sizing test; desync-on-reject regression; new `test_ema50strategy.py` |
| **P10 Fees** | E8 (part) | `Trade::fee`; maker bps on Limit fills, taker bps on Market/Stop/liquidation; cash debit separate from PnL; `total_fees` in report/JSON | per-type fee tests; zero-default regression; report sum |
| **P11 Docs** | E7 docs, drift | README/architecture `bool` → `OrderID`, document stop orders + new API + `leverage`/`parent`; fix CLAUDE.md structure section | — |

Explicitly deferred: StopLimit, slippage model, funding, partial fills / order-book realism,
walk-forward harness, `qbreakout.py` rewrite onto the new APIs (it works today by
hand-mirroring the engine; simplify it *after* P1–P6), broker-level close-only bracket
children.

---

## 8. Library improvement roadmap (beyond the bug fixes)

Ordered by leverage-per-effort against the project brief's stated goals:

1. **Execution realism dial** (`docs/project-brief.md:97-104` wants this): slippage
   assumption (bps or spread-based) on taker fills → later partial fills/volume caps →
   much later book realism. Belongs in `BrokerConfig`, off by default.
2. **Risk layer** (brief §risk): a composable component between strategy and broker —
   max positions, per-symbol/gross exposure caps, daily-loss halt — identical in backtest
   and live. Today risk logic is smeared into strategies (`risk_fraction`) and the broker
   (margin gate).
3. **Order management maturity:** honor `TimeInForce` (add `DAY`, `IOC`), order
   modify/replace, close-only child flag, an order-event view for strategies (fills since
   last tick) so shadow ledgers die for good.
4. **Run orchestration** (brief §127-130): parameter sweeps and walk-forward splits, N
   engines in parallel (the engine is already value-semantic and single-threaded — the
   outer harness is straightforward), results aggregated for comparison.
5. **Metrics completion** (brief §123-125): exposure, turnover, Sortino, CAGR, per-trade
   R-multiples, and a buy-and-hold benchmark column; single shared round-trip
   reconstruction consumed by both report and analytics; annualization aware of the feed's
   calendar (365 vs 252).
6. **Feed evolution:** optional calendar-grid alignment / forward-fill policy for ragged
   universes; multi-timeframe access (strategy on 1h reading 1d context); tick/L2 feed
   concepts when live work starts.
7. **Instrument metadata:** tick size, step size, contract multiplier, fee schedule per
   symbol — prerequisite for honest fills and for `buf_ticks` in qmsignals to mean anything
   (today `MINTICK = 0.0`).
8. **Synthetic data generator** (brief §106-114): fat tails, regime switches, gaps — cheap
   robustness testing for exactly the class of bug this audit found (E1 would have been
   obvious on synthetic data with known-outcome brackets).
9. **Live-trading seam:** define `LiveBroker`/`LiveFeed` against the existing concepts
   early — even stub implementations force the API gaps (E7!) to the surface before more
   strategies calcify around workarounds.
10. **CI hygiene:** run the `app/python` pytest suites from ctest (or CI alongside it);
    add the C++/Python parity tests; cover the parquet-path `KLineFeed` constructor with a
    committed fixture file.

---

## 9. Appendix

### A. Empirical evidence detail (daily run, `report-20260705-144624.json`)

- 1,431 orders: 548 `Filled Market root` (= 440 real entries + 108 synthetic liquidation
  orders), 332 `Filled Limit child`, 548 `Cancelled Limit child`, 3 still open at end.
  Exactly two Limit children per entry (SL+TP), zero rejects — every bracket resolved as
  "one child fills at next open, sibling cancelled."
- Hold-time distribution: min 0, median 1, max **1** bar (440 lifecycles).
- First order of the run: `Sell BTCUSDT, market, leverage 62.0` — a `short_breakout` fire.
- Worst single-bar equity drops: −54.44 (2022-07-17), −40.34 (2020-01-29), −39.02
  (2023-06-22) — gap-through liquidations (E5).

### B. `position-calculator-formulas.md` vs implementation

| Formula section | Status in code |
|---|---|
| §1–§7 fees / breakeven / net R:R / ROI | **Absent** — no fee concept anywhere in core types or broker |
| §8 isolated liquidation `E(1∓1/L)/(1∓m)` | **Partial** — implemented with `m` hard-zero (`backtestbroker.cpp:274-277`) |
| §8 cross-margin liquidation | Absent (isolated-only model) |
| §9 max leverage (floor, integer step-down, cap) | **Faithful** — `qmsignals.h:168-178`, 6 unit tests |
| §10 Binance Cost = margin + open loss | Absent |

### C. Test coverage map (~189 tests)

| Area | State |
|---|---|
| Broker fills, brackets, leverage, liquidation, bankruptcy | Strong (33 + 23 tests) — but every bracket test triggers legs on separate bars (E2 gap) |
| Engine scenarios, determinism, lookahead | Strong (13 + 1 + 3) |
| KLineFeed (in-memory rows), filters | Strong (14) — parquet-path constructor untested |
| Analytics, report, report-JSON | Strong (11 + 13 + 9) — `annualization_from` untested |
| qmsignals signals + leverage math | Strong (21) — no execution-level test caught E1/S1 because `StubBroker` never fills |
| Python boundary (adapter, strategy lifecycle) | Good (8 + 9) — `place_limit_order` never called *from Python* in tests; `Timestamp`/`KLine` bindings untouched |
| App layer (controller, worker, main) | None (threading/marshaling unverified by tests) |
| C++ ⇄ Python strategy parity | None |
