# Position Calculator — Formula Reference

All percentages and rates below are **fractions** in the formulas (the app divides the inputs by 100): e.g. risk $2\% \rightarrow p = 0.02$, fee $0.05\% \rightarrow f = 0.0005$, maintenance margin $0.4\% \rightarrow m = 0.004$.

## Notation

- **`E`** — Entry price
- **`S`** — Stop-loss price
- **`T`** — Take-profit price
- **`B`** — Account balance (USDT)
- **`p`** — Sizing percentage (risk or margin), as a fraction
- **`L`** — Leverage
- **`m`** — Maintenance margin rate, as a fraction
- **`f_e`, `f_tp`, `f_sl`** — Commission per leg (entry, take-profit, stop-loss), as fractions
- **`Q`** — Position quantity (base-asset units)
- **`N`** — Notional / position value (USDT)
- **`M`** — Initial margin (USDT)
- **`ε`** — Direction sign: `+1` for long, `−1` for short

Each leg's fee is the **maker rate** for a *limit* order, or the **taker rate** for a *market* order — i.e. `f_leg = f_maker` (limit) or `f_taker` (market).

**Direction validity:** long requires $S < E < T$; short requires $T < E < S$.

## 1. Take-profit input

Take profit is entered either as a price $T$ directly, or derived from a **price-based** reward-to-risk ratio $\text{RR}$:

$$T = E + \varepsilon\,\text{RR}\,|E - S|$$

The reverse (ratio implied by a target price): $\ \text{RR}_{\text{price}} = \dfrac{|T - E|}{|E - S|}$.

## 2. Position size (quantity)

**Risk mode** — size so a stop-out loses exactly $p$ of balance, including the entry and stop fees:

$$Q = \frac{B\,p}{\,|E - S| + E\,f_e + S\,f_{sl}\,}$$

**Margin mode** — allocate $p$ of balance as margin:

$$Q = \frac{B\,p\,L}{E}$$

## 3. Notional and margin

$$N = Q\,E \qquad\qquad M = \frac{N}{L} = \frac{Q\,E}{L}$$

$M$ is the calculator's **"Margin used."**

## 4. Fees (per leg)

Each leg is charged on the notional traded at that leg's price:

$$\text{fee}_{\text{entry}} = Q\,E\,f_e \qquad \text{fee}_{tp} = Q\,T\,f_{tp} \qquad \text{fee}_{sl} = Q\,S\,f_{sl}$$

Round trip to take profit: $\,Q E f_e + Q T f_{tp}\,$. Round trip to stop: $\,Q E f_e + Q S f_{sl}\,$.

## 5. Profit and loss (net of fees)

$$\text{PnL}_{TP} = \varepsilon\,Q\,(T - E)\;-\;\big(Q E f_e + Q T f_{tp}\big)$$

$$\text{PnL}_{SL} = \varepsilon\,Q\,(S - E)\;-\;\big(Q E f_e + Q S f_{sl}\big)$$

In **risk mode** the sizing makes the stop loss exact: $\;|\text{PnL}_{SL}| = B\,p$.

## 6. Breakeven price (fee-inclusive)

The exit price at which the gross move exactly covers the entry fee plus a closing fee $f_c$. By default the close uses the take-profit (limit) leg, $f_c = f_{tp}$:

$$P_{\text{be}}^{\text{long}} = E\cdot\frac{1 + f_e}{1 - f_c} \qquad\qquad P_{\text{be}}^{\text{short}} = E\cdot\frac{1 - f_e}{1 + f_c}$$

*Derivation (long):* setting net P&L to zero at exit price $X$ gives $Q(X-E) = Q E f_e + Q X f_c$, hence $X(1-f_c) = E(1+f_e)$.

## 7. Return on margin, R:R, balance impact

$$\text{ROI}(X) = \frac{\text{PnL}(X)}{M}\times 100\%$$

$$\text{R:R}_{\text{net}} = \frac{|\text{PnL}_{TP}|}{|\text{PnL}_{SL}|}$$

(This is the *after-fee* ratio shown in the app; it is slightly below the price-based $\text{RR}$ used to set the target.)

$$\text{risk}\% = \frac{|\text{PnL}_{SL}|}{B}\times 100 \qquad \text{reward}\% = \frac{\text{PnL}_{TP}}{B}\times 100$$

## 8. Liquidation price

**Isolated** (maintenance-margin aware; from equity $=$ maintenance margin):

$$P_{\text{liq}}^{\text{long}} = E\cdot\frac{1 - \tfrac{1}{L}}{1 - m} \qquad\qquad P_{\text{liq}}^{\text{short}} = E\cdot\frac{1 + \tfrac{1}{L}}{1 + m}$$

**Cross** (whole balance backs the position; simplified, ignores $m$):

$$P_{\text{liq}}^{\text{long}} = E - \frac{B}{Q} \qquad\qquad P_{\text{liq}}^{\text{short}} = E + \frac{B}{Q}$$

**Liquidation-before-stop** (the risk flag) is true when, on the losing side, liquidation is reached first:

$$\text{long: } P_{\text{liq}} \ge S \qquad\qquad \text{short: } P_{\text{liq}} \le S$$

## 9. Maximum isolated leverage

The largest leverage that keeps isolated liquidation just beyond the stop, i.e. solving $P_{\text{liq}} = S$ for $L$:

$$L_{\max}^{\text{long}} = \frac{E}{\,E - S\,(1 - m)\,} \qquad\qquad L_{\max}^{\text{short}} = \frac{E}{\,S\,(1 + m) - E\,}$$

The app then takes $\lfloor L_{\max} \rfloor$ (minus one step if the result is an exact integer) so liquidation sits *below* the stop with a small buffer.

## 10. Binance "Cost" vs. "Margin used"

Binance's **Cost** is the initial margin plus an *open-loss* buffer (it does **not** include the trading fee, which is charged separately):

$$\text{OpenLoss} = Q\cdot\max\!\big(0,\;\varepsilon\,(E - P_{\text{mark}})\big)$$

$$\text{Cost} = M + \text{OpenLoss} = \frac{Q\,E}{L} + Q\cdot\max\!\big(0,\;\varepsilon\,(E - P_{\text{mark}})\big)$$

where $P_{\text{mark}}$ is the mark price at order time. When your order price equals the mark price, $\text{OpenLoss}=0$ and $\text{Cost} = M$ (the calculator's "Margin used").

---

*Notes: liquidation formulas are single-position estimates assuming one isolated position with no other open orders; real Binance liquidation also factors the maintenance-amount deduction and tiered maintenance-margin rates. Funding fees on perpetuals are not included.*
