# EE-364B Stochastic Programming — Tax-Smart Transition

Course project for EE364B (Convex Optimization II) at Stanford. Implements stochastic programming methods to optimize tax-smart portfolio transitions — minimizing tax impact while reallocating assets under uncertainty.

> **Status:** three composable layers are implemented:
>
> 1. **`ForwardOptimizer`** — multi-scenario stochastic program (Gurobi-backed). One solve over a list of price scenarios produces a plan for ALL scenarios.
> 2. **`RMPController`** — robust MPC controller. Generates price scenarios via block bootstrap and runs a single t=0 stochastic solve.
> 3. **`Backtester`** — rolling backtester with a receding horizon (anchored on a fixed target end date) and a rolling 1-year history window. Uses an `RMPController` at each step to compute trades, then executes only the first-period trade against realized prices. At the final step the current price is already realized, so it short-circuits to a single-scenario `ForwardOptimizer` solve.
>
> The single-scenario "prescient" case is recovered by passing a one-element list of realized prices to `ForwardOptimizer`.

---

## Repository Structure

```
.
├── packages/
│   └── quant-oracle/                # Installable library (Gurobi-backed)
│       ├── pyproject.toml
│       └── quant_oracle/
│           ├── __init__.py
│           ├── optimizer.py                 # ForwardOptimizer class
│           ├── RMPController.py             # RMPController class
│           ├── Backtester.py                # Backtester class
│           └── analysis_utils.py            # Plotting / metric helpers
├── run_prescient_case.py                    # Run prescient (perfect-foresight) optimizer, pickle solution
├── analyze_prescient_case.py                # Load prescient pickle, plot prescient case
├── run_mpc.py                               # Run RMPController, pickle solution
├── analyze_mpc.py                           # Load MPC + prescient pickles, plot MPC t=0 plan vs prescient
├── run_backtest.py                          # Run rolling Backtester over 2024, pickle per-step results
├── analyze_backtest.py                      # Load backtest + prescient pickles, plot backtest vs prescient
├── pyproject.toml                           # Root project (application, not a library)
├── poetry.toml                              # Poetry config (in-project virtualenv)
├── poetry.lock
└── .pre-commit-config.yaml                  # black + isort pre-commit hooks
```

---

## Packages

### `quant-oracle` (`packages/quant-oracle/`)

Reusable optimization library that wraps Gurobi. Declared as an editable path dependency of the root project.

**Public API** (`from quant_oracle import ForwardOptimizer, RMPController, Backtester`):

| Class | File | Description |
|---|---|---|
| `ForwardOptimizer` | `optimizer.py` | Multi-scenario stochastic optimizer (Gurobi MIP) |
| `RMPController` | `RMPController.py` | Generates scenarios via block bootstrap, runs a single t=0 solve |
| `Backtester` | `Backtester.py` | Rolling MPC backtest along an actual realized price path |

**`ForwardOptimizer` interface:**

```python
class ForwardOptimizer:
    model: gp.Model                     # Gurobi model
    inputs: dict                        # see "inputs dict keys" below
    T: int                              # number of monthly filtration periods
    n_scenario: int                     # number of price scenarios (1 for prescient)
    n_asset: int                        # size of model universe N
    n_start_pos: int                    # number of starting lots L
    filtration: dict[tuple, dict]       # keyed by (s, f) — per-scenario, per-period state
    portfolio_ub: np.ndarray            # shape (n_scenario, T) — heuristic UB per (s, f)
    objectives: dict                    # name -> [Var, priority] for the lex hierarchy

    def __init__(self, inputs: dict) -> None: ...
    def build(self) -> None: ...        # construct variables, constraints, objectives
    def solve(self) -> dict: ...        # invoke Gurobi, return per-(s, f) values
```

`build()` runs the following pipeline in order:

1. `build_filtration` — for every `(s, f)` with $f \in \{1,\dots,T\}$, create decision variables and cache prices/cost basis. Within each filtration the buy/sell decisions come first and `lot_shr` is the resulting **post-decision** state. `sell_shr_l` (and the holding-level sell vars) are created only on *existing* lots $\mathcal{E}_f$.
2. `build_lot_holding_linking_constraints` — link lot-level shares to ticker-level holdings (per scenario).
3. `build_wash_sales_constraints` — sell aggregation, big-M indicator linking, buy/sell exclusivity, dollar self-financing (per scenario, all $f$).
4. `build_lot_dynamics_constraints` — link consecutive filtrations within each scenario. At $f=1$ the prior shares are the input positions (so this method also anchors the initial holdings — no separate starting-lot constraint).
5. `build_information_pattern_constraints` — non-anticipativity at the root: `lot_shr[(s, 1)]` must be identical across scenarios, which (because $f=1$ holdings are post-decision) directly forces the first-period sells/buys to be identical across scenarios.
6. `build_terminal_deviation_objective` — average across scenarios of total absolute deviation from target weights at the final filtration $f=T$.
7. `build_transitory_deviation_objective` — average across scenarios of total deviation outside the `±tkr_adev` band at intermediate post-decision states $f=1..T-1$.
8. `build_tax_cost_objective` — average across scenarios of total realized gain/loss across all filtrations $f=1..T$; per-(s, f) `tax_cost_f` variables are reserved for downstream plotting.
9. `set_objective_hierarchy` — register the three objectives with `setObjectiveN`. In the multi-scenario case (`n_scenario > 1`) a 5% relative MIP gap is applied to the tax-cost stage only, via Gurobi's multi-objective sub-environment; the deviation stages and the single-scenario case keep Gurobi's default tight gap.

**`inputs` dict keys:**

| Key | Type | Description |
|---|---|---|
| `positions` | `pd.DataFrame` | Starting portfolio. Columns: `tkr`, `amt`, `cost_basis_amt`, `shr` (shares). |
| `tax_rate` | `float` | Flat capital gains tax rate applied to realized gains |
| `model` | `pd.DataFrame` | Target model portfolio (`tkr`, `tgt_wt`) |
| `monthly_prices` | `list[pd.DataFrame]` | One DataFrame per scenario; each has month-start prices (rows = periods, cols = tickers) |
| `tkr_adev` | `float` | Allowed weight deviation band (decimal) around target at intermediate filtrations, e.g. `0.05` = ±5% |

**Dependencies:** `gurobipy >= 11.0.0`, `numpy`, `pandas`

---

### `RMPController` (`packages/quant-oracle/quant_oracle/RMPController.py`)

Robust MPC controller. Single responsibility:
1. Generate price scenarios via block bootstrap from a historical return window.
2. Construct a multi-scenario `ForwardOptimizer`.
3. Run a single solve at $t=0$ and return the solution dict.

The rolling/receding-horizon loop is **not** here — it lives in `Backtester`. `RMPController` just generates the t=0 plan.

**Interface:**

```python
class RMPController:
    inputs: dict
    scenario_prices: list[pd.DataFrame]   # populated by build_price_scenarios()
    all_tkrs: list[str]                   # union of positions["tkr"] and model["tkr"]
    n_period: int                         # forecast horizon (months)
    n_scenario: int                       # number of bootstrapped scenarios
    seed: int                             # RNG seed (default 42)

    def __init__(self, inputs: dict) -> None: ...
    def build_price_scenarios(self) -> None: ...   # populates self.scenario_prices
    def solve(self) -> dict: ...                   # builds ForwardOptimizer and solves once
```

**Inputs** (in addition to the `ForwardOptimizer` base inputs above; `monthly_prices` is generated internally and should not be supplied):

| Key | Type | Description |
|---|---|---|
| `start_date` | ISO string or `pd.Timestamp` | Date of the actual portfolio (anchors $t=0$ for all scenarios) |
| `sim_start_date` | ISO string or `pd.Timestamp` | Start of the historical window used for block bootstrap |
| `sim_end_date` | ISO string or `pd.Timestamp` | End of the historical window |
| `n_period` | `int` | Forecast horizon length (months) |
| `n_scenario` | `int` | Number of price scenarios to generate |
| `block_length` | `int` | Bootstrap block length in months |
| `seed` | `int` *(optional)* | RNG seed for reproducibility (default 42) |

**Block bootstrap.** Let $r_t \in \mathbb{R}^N$ be the monthly return vector at month $t$ in the historical window of length $H$. Form all overlapping blocks of length $B$ (`block_length`):

$$\mathcal{B} = \{ r_{t:t+B-1} : t = 0, 1, \dots, H - B \}.$$

For each scenario $s = 1, \dots, S$, sample $\lceil (T-1)/B \rceil$ blocks i.i.d. with replacement, concatenate, and trim to exactly $T-1$ return vectors. Stitch them with the observed start price $p_0 \in \mathbb{R}^N$:

$$p_t^{(s)} = p_{t-1}^{(s)} \odot (1 + r_t^{(s)}), \quad t = 1, \dots, T-1, \qquad p_0^{(s)} = p_0.$$

The starting price is identical across scenarios (it is observed at $t=0$); divergence happens at $t=1$.

**Solve.** Calls `ForwardOptimizer(opt_inputs)` with `monthly_prices = self.scenario_prices`, then `build()` and `solve()`. Returns the optimizer's solution dict.

---

### `Backtester` (`packages/quant-oracle/quant_oracle/Backtester.py`)

Rolling backtester for the receding-horizon MPC interpretation, anchored on a fixed transition target date. At each period $t$ along an **actual** realized price path:

1. Compute this step's remaining horizon $n^{(t)}_{\text{period}} = T_{\text{steps}} - t$ (shrinks by 1 each step so the target end date stays fixed).
2. Compute this step's rolling history window: $\text{sim\_end} = \text{actual\_prices.index}[t]$, $\text{sim\_start} = \text{sim\_end} - \text{lookback\_months}$.
3. If $n^{(t)}_{\text{period}} \ge 2$: build a fresh `RMPController` with the current positions, `start_date = actual_prices.index[t]`, `n_period = `$n^{(t)}_{\text{period}}$, and the rolling sim window. Generate scenarios, solve once.
4. If $n^{(t)}_{\text{period}} = 1$: the current-period price is already realized, so bootstrapping is skipped — `ForwardOptimizer` is called directly on a single-row, single-scenario price built from `actual_prices.iloc[[t]]`.
5. Execute only the first-period trades from the resulting plan (receding-horizon principle).
6. Mark positions to market at $t+1$ actual prices and advance.

**Interface:**

```python
class Backtester:
    base_inputs: dict
    actual_prices: pd.DataFrame           # rows = realized dates, cols = tickers
    lookback_months: int                  # rolling history window length (default 12)

    def __init__(
        self,
        base_inputs: dict,
        actual_prices: pd.DataFrame,
        lookback_months: int = 12,
    ) -> None: ...
    def run(self) -> list[dict]: ...
```

`run()` loops $t = 0, \dots, T_{\text{actual}} - 2$ where $T_{\text{actual}}$ is the number of rows in `actual_prices`, and returns a list of per-step dicts:

| Key | Description |
|---|---|
| `t` | Time step index |
| `n_period` | Remaining forecast horizon used at this step ($T_{\text{steps}} - t$) |
| `positions_before` | `pd.DataFrame` snapshot entering this step |
| `sol` | Full optimizer solution at this step (multi-scenario `RMPController.solve()` for $n^{(t)}_{\text{period}} \ge 2$; single-scenario `ForwardOptimizer` for $n^{(t)}_{\text{period}} = 1$) |

**Position update** (`Backtester._update_positions`). Given `f0_sol = sol["filtration"][(0, 1)]` (the first filtration of scenario 0):
- For each existing lot $i$: `sell_shr_l[(i, 0)]` shares are removed (j=0 because all current holdings are starting lots in each fresh build). Remaining shares are revalued at $t+1$ prices; cost basis is scaled by the surviving fraction. Lots reduced to (near) zero are dropped.
- For each ticker with `buy_shr_h[tkr] > 0`: a new lot is created with cost basis equal to the purchase price (period $t$ actual) and `amt` marked at $t+1$ actual.
- The DataFrame is `reset_index()`'d so its row indices align with the lot indices the next `ForwardOptimizer.build()` expects.

---

## Optimization Problem

### Overview

We manage a portfolio over $T$ monthly periods. The investor starts with $L$ tax lots — existing stock positions each carrying a per-share cost basis — and wants to transition towards a target model portfolio of $N$ assets while minimizing capital gains taxes incurred along the way.

At each filtration $f$, the optimizer decides how many shares of each lot to sell and how many shares of each model asset to buy, subject to a self-financing budget constraint (sell dollars equal buy dollars within the period). Selling at a profit triggers a capital gain; selling at a loss creates a capital loss. The objectives are minimized lexicographically: first the terminal weight deviation, then the realized tax cost.

---

### Notation

#### Indices and Sets

| Symbol | Definition |
|---|---|
| $f \in \{1, \dots, T\}$ | Filtration index (month). Filtration 1 is the first decision point; filtration $T$ is the terminal. |
| $k$ | Ticker (drawn from `positions["tkr"] ∪ model["tkr"]`) |
| $(i,\, j)$ | Lot identifier: row index $i$, column $j$ = filtration the lot was acquired ($j=0$ for starting lots) |
| $\mathcal{L}_f$ | Set of lots available at filtration $f$ |
| $\mathcal{E}_f$ | Existing lots at filtration $f$ (those carried in from $f-1$, or starting lots at $f=1$); the new lots bought at $f$ are $\mathcal{L}_f \setminus \mathcal{E}_f$ |
| $\mathcal{L}_f(k)$, $\mathcal{E}_f(k)$ | Restrictions of the above sets to ticker $k$ |

#### Parameters

| Symbol | Definition |
|---|---|
| $T$ | Number of filtration periods (`monthly_prices.shape[0]`) |
| $N$ | Number of assets in the model universe |
| $L$ | Number of starting lots (rows of `positions`) |
| $p_{k,f}$ | Market price of ticker $k$ at filtration $f$ |
| $w^{*}_{k}$ | Target portfolio weight for ticker $k$ |
| $\tau$ | Flat capital gains tax rate (`tax_rate`) |
| $\delta$ | Per-ticker weight tolerance band (`tkr_dev`); allows weight to deviate $\pm\delta$ from target at intermediate filtrations |
| $c_{i,j}$ | Per-share cost basis of lot $(i, j)$ |

For starting lots ($j = 0$), $c_{i,0} = \text{cb}_i \,/\, (\text{amt}_i / p_{\text{ticker}(i),\,1})$, where $\text{cb}_i$ and $\text{amt}_i$ are the `cost_basis_amt` and `amt` fields of position $i$. For purchased lots ($j \ge 1$), $c_{i,j} = p_{\text{ticker}(i),\,j}$ — the market price at filtration $j$, when the lot was opened.

---

### Lot Structure

Each lot is identified by $(i, j)$ where $j$ is the filtration the lot was acquired and $i$ is its row index *within that column*:

- **Column $j = 0$** (starting lots): $i \in \{0, \dots, L - 1\}$, ticker = $\text{ticker}(i, 0)$ (the $i$-th starting position).
- **Column $j \ge 1$** (purchased lots): $i \in \{0, \dots, N - 1\}$, ticker = $\text{ticker}(i, j)$ (the $i$-th model asset). A lot in column $j$ is created by the buy decision at filtration $j$ and first appears in the same filtration.

So at filtration $f$ the available lot set is

$$\mathcal{L}_f = \{(i, 0) : i \in \{0,\dots,L-1\}\} \;\cup\; \bigcup_{j=1}^{f}\{(i, j) : i \in \{0,\dots,N-1\}\},$$

with $|\mathcal{L}_f| = L + fN$. The total number of lot variables created over all filtrations is

$$\sum_{f=1}^{T}|\mathcal{L}_f| = LT + N\,\frac{T(T+1)}{2}.$$

The set of **existing** lots at filtration $f$ — those that can be sold at $f$ — is

$$\mathcal{E}_f = \mathcal{L}_{f-1} \cup \{(i, 0) : i \in \{0,\dots,L-1\}\} = \mathcal{L}_f \setminus \{(i, f) : i \in \{0,\dots,N-1\}\},$$

i.e. everything except the just-purchased lots. Note the index range depends on the column: $(0, 0)$ refers to the first starting position, while $(0, 1)$ refers to the first model asset purchased at filtration 1.

---

### Decision Variables

All variables are denominated in **shares**, not portfolio weight. Upper bounds are derived from a heuristic portfolio-value cap $\bar{V}_f$ (starting AUM compounded by the maximum monthly growth observed across tickers up to filtration $f$):

$$\bar{V}_f = V_0 \cdot \prod_{t=1}^{f} \max_k \frac{p_{k,t}}{p_{k,t-1}}, \quad \text{share UB} = \lceil \bar{V}_f / p_{k,f} \rceil.$$

#### Lot-level variables

| Variable | Index set | Bounds | Description |
|---|---|---|---|
| $x_{i,j,f} \in \mathbb{R}_+$ | $(i,j) \in \mathcal{L}_f$ | $[0,\, \bar{V}_f / p_{\text{ticker}(i),f}]$ | Post-decision shares of lot $(i,j)$ at filtration $f$ (`lot_shr`) |
| $s^{l}_{i,j,f} \in \mathbb{R}_+$ | $(i,j) \in \mathcal{E}_f$ | $[0,\, \bar{V}_f / p_{\text{ticker}(i),f}]$ | Shares sold from lot $(i,j)$ at filtration $f$ (`sell_shr_l`). Only created for existing lots, since new lots can't be sold the same period they're bought. |

#### Holding-level variables — indexed over ticker $k$ at filtration $f$

| Variable | Bounds | Description |
|---|---|---|
| $h_{k,f} \in \mathbb{R}_+$ | $[0,\, \bar{V}_f / p_{k,f}]$ | Total shares of ticker $k$ held (`shr_h`) |
| $s^{h}_{k,f} \in \mathbb{R}_+$ | $[0,\, \bar{V}_f / p_{k,f}]$ | Total shares of ticker $k$ sold (`sell_shr_h`) |
| $b^{h}_{k,f} \in \mathbb{R}_+$ | $[0,\, \bar{V}_f / p_{k,f}]$ | Total shares of ticker $k$ bought (`buy_shr_h`) |
| $\delta^{s}_{k,f} \in \{0,1\}$ | binary | 1 iff ticker $k$ has any sell at filtration $f$ (`sell_h`) |
| $\delta^{b}_{k,f} \in \{0,1\}$ | binary | 1 iff ticker $k$ has any buy at filtration $f$ (`buy_h`) |

---

### Constraints

#### Lot ↔ holding linking (`build_lot_holding_linking_constraints`)

Per-ticker holding equals the sum of its lot shares:

$$h_{k,f} = \sum_{(i,j)\,\in\,\mathcal{L}_f(k)} x_{i,j,f} \qquad \forall\, k,\, f.$$

#### Sell aggregation (`build_wash_sales_constraints`)

Holding-level sells aggregate lot-level sells across existing lots only:

$$s^{h}_{k,f} = \sum_{(i,j)\,\in\,\mathcal{E}_f(k)} s^{l}_{i,j,f} \qquad \forall\, k,\, f.$$

#### Big-M indicator linking

Binary indicators activate only when the corresponding share quantity is nonzero. The big-M used is the variable's own upper bound $M_{k,f} = \lceil \bar{V}_f / p_{k,f} \rceil$:

$$s^{h}_{k,f} \le M_{k,f}\, \delta^{s}_{k,f}, \qquad b^{h}_{k,f} \le M_{k,f}\, \delta^{b}_{k,f}.$$

#### No simultaneous buy and sell

For tickers that appear in both the sell and buy universes:

$$\delta^{b}_{k,f} + \delta^{s}_{k,f} \le 1.$$

#### Self-financing (per period)

Dollars sold equal dollars bought within each filtration — the portfolio holds no cash:

$$\sum_k p_{k,f}\, s^{h}_{k,f} \;=\; \sum_k p_{k,f}\, b^{h}_{k,f} \qquad \forall\, f.$$

#### Lot dynamics (`build_lot_dynamics_constraints`)

`lot_shr` at filtration $f$ is the **post-decision** state: it follows the prior shares (input positions for $f=1$, lot_shr at $f-1$ otherwise) and the sell decision at $f$.

For existing lots $(i, j) \in \mathcal{E}_f$ that survive from the prior period:

$$x_{i,j,f} = \tilde{x}_{i,j,f} - s^{l}_{i,j,f}, \qquad \text{where } \tilde{x}_{i,j,f} = \begin{cases} \text{shr}_i & \text{if } f=1 \text{ and } j=0 \\ x_{i,j,f-1} & \text{if } f \ge 2 \end{cases}$$

(so at $f=1$ this also anchors the initial holdings — there is no separate starting-lot constraint).

For lots newly created at filtration $f$ (column $j = f$), the buy at $f$ feeds the lot directly:

$$x_{i,f,f} = b^{h}_{\text{ticker}(i,f),\,f}, \qquad i \in \{0,\dots,N-1\}.$$

No same-period sell appears here — $s^{l}_{i,f,f}$ does not exist as a variable, since a just-purchased lot cannot be sold at the same filtration (wash-sale prevention already enforces this at the holding level).

The inequality $s^{l}_{i,j,f} \le \tilde{x}_{i,j,f}$ on existing lots is enforced *implicitly* by combining the dynamics equation with $x_{i,j,f} \ge 0$.

---

### Objectives

The optimizer registers three objectives via Gurobi's hierarchical multi-objective interface (`setObjectiveN`) and minimizes them lexicographically. Higher priority is optimized first; lower priority is minimized subject to the higher priority being optimal. Objectives at the **same** priority are combined with equal weight (Gurobi default).

#### Priority 1 — Terminal weight deviation (`build_terminal_deviation_objective`)

Let $V_T = \sum_k p_{k,T}\, h_{k,T}$ be the final portfolio value. For each model ticker $k$, an auxiliary $\xi_k \ge 0$ bounds the absolute deviation:

$$\xi_k \ge \lvert p_{k,T}\,h_{k,T} - w^{*}_{k}\, V_T \rvert.$$

Encoded as two linear inequalities. The objective is

$$\min \sum_{k \in \text{model}} \xi_k.$$

#### Priority 1 — Transitory weight deviation (`build_transitory_deviation_objective`)

At each intermediate post-decision state $f \in \{1, \dots, T-1\}$ the portfolio is allowed to deviate up to $\pm\delta$ (in weight) from the target. Only the overshoot beyond the band is penalized. Let $V_f = \sum_k p_{k,f}\,h_{k,f}$. For each model ticker $k$, an auxiliary $\zeta_{k,f} \ge 0$ captures the out-of-band dollar deviation:

$$\zeta_{k,f} \ge p_{k,f}\,h_{k,f} - (w^{*}_{k} + \delta)\,V_f,$$
$$\zeta_{k,f} \ge (w^{*}_{k} - \delta)\,V_f - p_{k,f}\,h_{k,f}.$$

Within the band both right-hand sides are $\le 0$, so $\zeta_{k,f}$ can stay at 0. The objective is

$$\min \sum_{f=1}^{T-1} \sum_{k \in \text{model}} \zeta_{k,f}.$$

This objective shares priority 1 with terminal deviation, so both are minimized jointly before tax cost.

#### Priority 0 — Realized tax cost (`build_tax_cost_objective`)

Total realized gain/loss across all sells, decomposed per filtration and then summed. For each filtration $f \in \{1,\dots,T\}$ a scalar variable $\text{tc}_f$ is defined by

$$\text{tc}_f = \tau \sum_{(i,j)\in\mathcal{E}_f} s^{l}_{i,j,f} \cdot (p_{\text{ticker}(i),f} - c_{i,j}),$$

and stored in `self.filtration[f]["tax_cost"]` for post-solve inspection. The aggregate objective is

$$\min \sum_f \text{tc}_f.$$

Losses contribute negatively, so the optimizer is incentivized to harvest them. The objective variable has $\text{lb} = -\infty$.

---

## Modeling Notes

- **Multi-scenario, equally-likely.** All three objectives are averaged across the supplied scenarios with uniform weight $1/S$. Probability-weighted scenarios are not yet supported.
- **Non-anticipativity at the root only.** The first-period decisions (`sell_shr_l[(s, 1)]`, `buy_shr_h[(s, 1)]`) are forced to be identical across scenarios via `build_information_pattern_constraints`, which pins `lot_shr[(s, 1)]` equal across $s$. Because `lot_shr[(s, 1)]` is the **post-decision** state at $f=1$ and every scenario starts from the same input positions, equal `lot_shr[(s, 1)]` directly forces equal first-period sells and buys. From $f \ge 2$ onward, decisions (sells, buys, holdings) are scenario-specific — this is the full-recourse stage.
- **Decision-then-state ordering within a filtration.** Within each filtration $f$ the buy/sell decisions are made first and `lot_shr[(s, f)]` is the resulting post-decision state. A new lot purchased at $f$ (column $j = f$) appears in the same filtration as the buy that created it. There are $T$ trading periods (one per filtration $f = 1, \dots, T$).
- **Per-(s, f) variable layout.** Each filtration entry carries `lot_shr` (all lots in $\mathcal{L}_f$), `shr_h`, `buy_shr_h`, `buy_h`, and the per-period `tax_cost` scalar (at all $f$); `sell_shr_l` (only on existing lots $\mathcal{E}_f$), and `sell_shr_h`, `sell_h` (only on tickers with existing lots); `trans_dev` (only at $1 \le f \le T-1$, the band-penalty range); and `terminal_dev` (only at $f = T$).
- **Tax cost is dollar-denominated.** The objective is $\tau \cdot \text{realized gains}$ summed over $(s, f)$ and averaged over $s$, so its solved value is the actual expected tax bill in dollars. Multiplying by $\tau$ leaves the optimal trade-off unchanged but makes the objective magnitude interpretable. No short-term vs long-term distinction; loss harvesting is treated 1:1.
- **Self-financing forces full investment.** Cash positions are not modeled; every sell dollar must be matched by a buy dollar in the same period (within each scenario). A zero-action period is allowed (both sums = 0).
- **Wash-sale rule is local.** Only same-period buys and sells of the same ticker are blocked. Cross-period wash sales (real rule: 30 days) are not enforced.
- **MIP gap is per-stage and auto-applied for multi-scenario.** When `n_scenario > 1`, a 5% relative MIP gap is set on the tax-cost stage of the lex hierarchy via `model.getMultiobjEnv(idx).setParam("MIPGap", 0.05)` — the deviation stages and the single-scenario case keep Gurobi's default tight gap. This trades a small optimality gap for a large speedup since the tax-cost stage is the expensive integer stage (binaries × scenarios).
- **Per-stage time limit of 5 minutes.** `ForwardOptimizer` sets `model.Params.TimeLimit = 300` (`ForwardOptimizer.TIME_LIMIT_SECONDS`). Gurobi applies the main-model `TimeLimit` to each multi-objective stage, so each lex stage may run up to 5 minutes before returning the best incumbent. Combined with the 5% MIP gap on the tax-cost stage this caps the worst-case wall time of long solves.

---

## Analysis Metrics

### % Transition

A scalar in $[0, 1]$ that summarizes how close the realized portfolio is to the target model weights at any point in time. Plotted by `plot_transition_pct`, `plot_mpc_t0_transition_pct`, and `plot_backtest_transition_pct`, and computed by `calculate_transition_pct` / `_per_scenario_transition_pct` in `quant_oracle.analysis_utils`.

Let $w_{k,f}$ be the portfolio's realized dollar weight of ticker $k$ at filtration $f$ (computed from `shr_h` and the prevailing price) and let $w^{*}_{k}$ be the target weight. Define per-ticker over- and under-weights:

$$o_{k,f} = \max(0,\, w_{k,f} - w^{*}_{k}), \qquad u_{k,f} = \max(0,\, w^{*}_{k} - w_{k,f}).$$

Then

$$\text{transition\_pct}_f \;=\; 1 \;-\; \max\!\left(\sum_{k} o_{k,f},\; \sum_{k} u_{k,f}\right).$$

**Interpretation.** When the portfolio matches the model exactly, every $o_{k,f} = u_{k,f} = 0$ and the metric is $1.0$ (100% transitioned). At the starting position — a single concentrated lot — both sums equal the total active weight distance from the target, and the metric reflects how far the portfolio has drifted away from $w^{*}$. Because portfolio weights and target weights each sum to $1$, $\sum_k o_{k,f} = \sum_k u_{k,f}$ whenever the portfolio and the model share the same ticker universe, so the $\max$ is a defensive choice that handles the general case (off-model tickers contributing only to $o_{k,f}$, or vice versa).

**Why this metric, not L1 distance.** The total absolute deviation $\sum_k |w_{k,f} - w^{*}_{k}|$ equals $\sum_k o_{k,f} + \sum_k u_{k,f}$. Splitting into over- vs under-weight and taking the $\max$ — rather than the sum — gives a tighter bound that does not double-count the same misallocation on both sides of the balance, so the metric stays in $[0, 1]$ and reads as "fraction of target weight that the portfolio has already reached."

---

## Running the Project

### Prescient (perfect-foresight) benchmark — solve

```bash
poetry run python run_prescient_case.py
```

Builds a single-scenario `ForwardOptimizer` over realized 2024 prices (wrapped as a one-element list) and pickles the solution, realized prices, positions, model, `tax_rate`, and `tkr_adev` to `prescient_output.pkl`. The pickle is consumed both by `analyze_prescient_case.py` and by `analyze_mpc.py` (which uses the prescient `sol` as a perfect-foresight benchmark) so neither script has to re-run the optimizer.

### Prescient analysis

```bash
poetry run python analyze_prescient_case.py
```

Loads `prescient_output.pkl` and produces four PNGs without re-solving:
`cumulative_tax_cost.png`, `portfolio_value.png`, `transition_pct.png`, `AAPL_weight_and_price.png`.

### MPC t=0 plan

```bash
poetry run python run_mpc.py
```

Runs `RMPController`: bootstrap `n_scenario` price paths from a historical window, build the multi-scenario `ForwardOptimizer`, and run a single solve. The solution, scenarios, model, positions, realized prices, and config are pickled to `mpc_output.pkl` for downstream analysis.

### MPC analysis with prescient overlay

```bash
poetry run python analyze_mpc.py
```

Loads `mpc_output.pkl` for the MPC t=0 plan and `prescient_output.pkl` for the prescient benchmark (run `run_prescient_case.py` first if the prescient pickle does not exist), then produces four overlay plots: cumulative tax cost, total portfolio value, % transition, and AAPL weight + price (dual axis). MPC-t=0 paths are translucent steelblue with a thick mean line; the prescient trajectory overlays as dashed crimson. Output PNGs: `mpc_t0_cumulative_tax_cost.png`, `mpc_t0_portfolio_value.png`, `mpc_t0_transition_pct.png`, `mpc_t0_AAPL_weight_and_price.png`.

> The plots show the **t=0 plan** (one stochastic solve at the start), not a rolling MPC trajectory. The rolling case is `Backtester` — wrapped by `run_backtest.py` and `analyze_backtest.py` below.

### Rolling backtest

```bash
poetry run python run_backtest.py
```

Runs `Backtester` over realized monthly prices from 01/2024 through 12/2024 (12 trade dates plus one extra month-end for marking-to-market). At each step the receding horizon shrinks by one ($n_{\text{period}} = 12, 11, \dots, 1$) and the rolling 12-month history window slides forward so the bootstrap reflects the most recently observed year. The final step ($n_{\text{period}} = 1$) bypasses bootstrap and solves a single-scenario `ForwardOptimizer` on the realized price. Per-step results, realized prices, model, positions, and config are pickled to `backtest_output.pkl`.

### Backtest analysis with prescient overlay

```bash
poetry run python analyze_backtest.py
```

Loads `backtest_output.pkl` for the realized backtest trajectory, `prescient_output.pkl` for the prescient benchmark, and `mpc_output.pkl` for the MPC t=0 plan (run `run_prescient_case.py` and `run_mpc.py` first if their pickles do not exist). Produces four overlay plots: cumulative tax cost, total portfolio value, % transition, and AAPL weight + price (dual axis). The backtest trajectory is solid steelblue; the prescient trajectory overlays as dashed crimson; the cross-scenario mean of the MPC t=0 plan overlays as dashed seagreen. The cumulative tax-cost plot also reports the total at horizon end for each series in its legend. Output PNGs: `backtest_cumulative_tax_cost.png`, `backtest_portfolio_value.png`, `backtest_transition_pct.png`, `backtest_AAPL_weight_and_price.png`.

---

## Setup

Requires Python 3.11+ and [Poetry](https://python-poetry.org/) (`pip install poetry`).

```bash
# Install all dependencies (creates .venv/ in project root)
poetry install

# Install dev dependencies (black, pre-commit)
poetry install --with dev

# Activate the virtualenv
poetry shell
```

**First-time pre-commit setup** (only needed after a fresh clone):
```bash
poetry run pre-commit install
```

---

## Development

```bash
# Run a script
poetry run python <script>.py

# Launch Jupyter
poetry run jupyter notebook

# Add a dependency to the root project
poetry add <package>

# Add a dependency to quant-oracle
cd packages/quant-oracle && poetry add <package>

# Format all files manually
poetry run black .
```

### Code Style

[black](https://github.com/psf/black) and [isort](https://pycqa.github.io/isort/) are enforced via pre-commit hooks. They run automatically on every `git commit` and reformat staged `.py` files (isort uses the `--profile black` setting for compatibility).

---

## Dependencies

| Package | Purpose |
|---|---|
| `gurobipy` | Gurobi solver interface (used by `quant-oracle`) |
| `numpy` / `scipy` | Numerical computation |
| `pandas` | Data handling (prices, tax lots, positions) |
| `yfinance` | Download historical price data from Yahoo Finance |
| `matplotlib` | Plotting results |
| `jupyter` | Notebooks for exploration and writeup |
| `lxml` / `requests` | HTML parsing and HTTP (yfinance dependencies) |
| `black` *(dev)* | Code formatter |
| `isort` *(dev)* | Import sorter |
| `pre-commit` *(dev)* | Git hook runner (enforces black + isort on commit) |