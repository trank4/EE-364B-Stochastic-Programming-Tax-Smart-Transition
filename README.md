# EE-364B Stochastic Programming — Tax-Smart Transition

Course project for EE364B (Convex Optimization II) at Stanford. Implements stochastic programming methods to optimize tax-smart portfolio transitions — minimizing tax impact while reallocating assets under uncertainty.

> **Status:** three composable layers are implemented:
>
> 1. **`StoxOptimizer`** — multi-scenario stochastic program (Gurobi-backed). One solve over a list of price scenarios produces a per-scenario plan.
> 2. **`RMPController`** — robust MPC controller. Generates price scenarios via block bootstrap and runs a single t=0 stochastic solve.
> 3. **`Backtester`** — rolling backtester. Uses an `RMPController` at each step to compute trades, then executes only the first-period trade against realized prices.
>
> The single-scenario "prescient" case is recovered by passing a one-element list of realized prices to `StoxOptimizer`.

---

## Repository Structure

```
.
├── packages/
│   └── stochastic-optimizer/                # Installable library (Gurobi-backed)
│       ├── pyproject.toml
│       └── stochastic_optimizer/
│           ├── __init__.py
│           ├── optimizer.py                 # StoxOptimizer class
│           ├── RMPController.py             # RMPController class
│           ├── Backtester.py                # Backtester class
│           └── analysis_utils.py            # Plotting / metric helpers
├── run_prescient_case.py                    # Prescient (perfect-foresight) runner
├── run_mpc.py                               # Run RMPController, pickle solution
├── analyze_mpc.py                           # Load pickle, plot MPC t=0 plan vs prescient
├── pyproject.toml                           # Root project (application, not a library)
├── poetry.toml                              # Poetry config (in-project virtualenv)
├── poetry.lock
└── .pre-commit-config.yaml                  # black + isort pre-commit hooks
```

---

## Packages

### `stochastic-optimizer` (`packages/stochastic-optimizer/`)

Reusable optimization library that wraps Gurobi. Declared as an editable path dependency of the root project.

**Public API** (`from stochastic_optimizer import StoxOptimizer, RMPController, Backtester`):

| Class | File | Description |
|---|---|---|
| `StoxOptimizer` | `optimizer.py` | Multi-scenario stochastic optimizer (Gurobi MIP) |
| `RMPController` | `RMPController.py` | Generates scenarios via block bootstrap, runs a single t=0 solve |
| `Backtester` | `Backtester.py` | Rolling MPC backtest along an actual realized price path |

**`StoxOptimizer` interface:**

```python
class StoxOptimizer:
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

1. `build_filtration` — for every `(s, f)`, create decision variables and cache prices/cost basis.
2. `build_lot_holding_linking_constraints` — link lot-level shares to ticker-level holdings (per scenario).
3. `build_starting_lot_constraints` — anchor lot shares at $f=0$ to the input portfolio in every scenario.
4. `build_wash_sales_constraints` — sell aggregation, big-M indicator linking, buy/sell exclusivity, dollar self-financing (per scenario, $f<T-1$).
5. `build_lot_dynamics_constraints` — link consecutive periods within each scenario via sells (existing lots) and buys (new lots).
6. `build_information_pattern_constraints` — non-anticipativity at $f=1$: $\text{lot\_shr}[(s, 1)]$ must be identical across scenarios, which forces the first-period sells/buys to be identical across scenarios.
7. `build_terminal_deviation_objective` — average across scenarios of total absolute deviation from target weights at the final period.
8. `build_transitory_deviation_objective` — average across scenarios of total deviation outside the `±tkr_adev` band at intermediate periods.
9. `build_tax_cost_objective` — average across scenarios of total realized gain/loss; per-(s, f) `tax_cost_f` variables are reserved for downstream plotting.
10. `set_objective_hierarchy` — register the three objectives with `setObjectiveN`. If `inputs["MIPGap"]` is set, applies that relative gap to the tax cost stage only via Gurobi's multi-objective sub-environment.

**`inputs` dict keys:**

| Key | Type | Description |
|---|---|---|
| `positions` | `pd.DataFrame` | Starting portfolio. Columns: `tkr`, `amt`, `cost_basis_amt`, `shr` (shares). |
| `tax_rate` | `float` | Flat capital gains tax rate applied to realized gains |
| `model` | `pd.DataFrame` | Target model portfolio (`tkr`, `tgt_wt`) |
| `monthly_prices` | `list[pd.DataFrame]` | One DataFrame per scenario; each has month-start prices (rows = periods, cols = tickers) |
| `tkr_adev` | `float` | Allowed weight deviation band (decimal) around target at intermediate filtrations, e.g. `0.05` = ±5% |
| `MIPGap` | `float` *(optional)* | Relative MIP gap applied to the tax-cost stage of the lex hierarchy |

**Dependencies:** `gurobipy >= 11.0.0`, `numpy`, `pandas`

---

### `RMPController` (`packages/stochastic-optimizer/stochastic_optimizer/RMPController.py`)

Robust MPC controller. Single responsibility:
1. Generate price scenarios via block bootstrap from a historical return window.
2. Construct a multi-scenario `StoxOptimizer`.
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
    MIPGap: float                         # default 0.05, forwarded to StoxOptimizer

    def __init__(self, inputs: dict) -> None: ...
    def build_price_scenarios(self) -> None: ...   # populates self.scenario_prices
    def solve(self) -> dict: ...                   # builds StoxOptimizer and solves once
```

**Inputs** (in addition to the `StoxOptimizer` base inputs above; `monthly_prices` is generated internally and should not be supplied):

| Key | Type | Description |
|---|---|---|
| `start_date` | ISO string or `pd.Timestamp` | Date of the actual portfolio (anchors $t=0$ for all scenarios) |
| `sim_start_date` | ISO string or `pd.Timestamp` | Start of the historical window used for block bootstrap |
| `sim_end_date` | ISO string or `pd.Timestamp` | End of the historical window |
| `n_period` | `int` | Forecast horizon length (months) |
| `n_scenario` | `int` | Number of price scenarios to generate |
| `block_length` | `int` | Bootstrap block length in months |
| `seed` | `int` *(optional)* | RNG seed for reproducibility (default 42) |
| `MIPGap` | `float` *(optional)* | Relative MIP gap for the tax-cost stage (default 0.05) |

**Block bootstrap.** Let $r_t \in \mathbb{R}^N$ be the monthly return vector at month $t$ in the historical window of length $H$. Form all overlapping blocks of length $B$ (`block_length`):

$$\mathcal{B} = \{ r_{t:t+B-1} : t = 0, 1, \dots, H - B \}.$$

For each scenario $s = 1, \dots, S$, sample $\lceil (T-1)/B \rceil$ blocks i.i.d. with replacement, concatenate, and trim to exactly $T-1$ return vectors. Stitch them with the observed start price $p_0 \in \mathbb{R}^N$:

$$p_t^{(s)} = p_{t-1}^{(s)} \odot (1 + r_t^{(s)}), \quad t = 1, \dots, T-1, \qquad p_0^{(s)} = p_0.$$

The starting price is identical across scenarios (it is observed at $t=0$); divergence happens at $t=1$.

**Solve.** Calls `StoxOptimizer(opt_inputs)` with `monthly_prices = self.scenario_prices` and `MIPGap = self.MIPGap`, then `build()` and `solve()`. Returns the optimizer's solution dict.

---

### `Backtester` (`packages/stochastic-optimizer/stochastic_optimizer/Backtester.py`)

Rolling backtester for the receding-horizon MPC interpretation. At each period $t$ along an **actual** realized price path:

1. Build a fresh `RMPController` with the current positions and `start_date = actual_prices.index[t]`.
2. Generate scenarios from the most recent observed price; solve once.
3. Execute only the first-period trades from the resulting plan (receding-horizon principle).
4. Mark positions to market at $t+1$ actual prices and advance.

**Interface:**

```python
class Backtester:
    base_inputs: dict
    actual_prices: pd.DataFrame           # rows = realized dates, cols = tickers

    def __init__(self, base_inputs: dict, actual_prices: pd.DataFrame) -> None: ...
    def run(self) -> list[dict]: ...
```

`run()` loops $t = 0, \dots, |\text{actual\_prices}| - 2$ and returns a list of per-step dicts:

| Key | Description |
|---|---|
| `t` | Time step index |
| `positions_before` | `pd.DataFrame` snapshot entering this step |
| `sol` | Full optimizer solution from `RMPController.solve()` at this step |

**Position update** (`Backtester._update_positions`). Given `f0_sol = sol["filtration"][(0, 0)]`:
- For each existing lot $i$: `sell_shr_l[(i, 0)]` shares are removed (j=0 because all current holdings are starting lots in each fresh build). Remaining shares are revalued at $t+1$ prices; cost basis is scaled by the surviving fraction. Lots reduced to (near) zero are dropped.
- For each ticker with `buy_shr_h[tkr] > 0`: a new lot is created with cost basis equal to the purchase price (period $t$ actual) and `amt` marked at $t+1$ actual.
- The DataFrame is `reset_index()`'d so its row indices align with the lot indices the next `StoxOptimizer.build()` expects.

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
| $f \in \{0, \dots, T-1\}$ | Filtration index (month) |
| $k$ | Ticker (drawn from `positions["tkr"] ∪ model["tkr"]`) |
| $(i,\, j)$ | Lot identifier: row index $i$, column $j$ = period the lot was acquired |
| $\mathcal{L}_f$ | Set of lots available at filtration $f$ |
| $\mathcal{L}_f(k)$ | Subset of $\mathcal{L}_f$ belonging to ticker $k$ |

#### Parameters

| Symbol | Definition |
|---|---|
| $T$ | Number of filtration periods (`monthly_prices.shape[0]`) |
| $N$ | Number of assets in the model universe |
| $L$ | Number of starting lots (rows of `positions`) |
| $p_{k,f}$ | Market price of ticker $k$ at filtration $f$ |
| $w^*_k$ | Target portfolio weight for ticker $k$ |
| $\tau$ | Flat capital gains tax rate (`tax_rate`) |
| $\delta$ | Per-ticker weight tolerance band (`tkr_dev`); allows weight to deviate $\pm\delta$ from target at intermediate filtrations |
| $c_{i,j}$ | Per-share cost basis of lot $(i, j)$ |

For starting lots ($j = 0$), $c_{i,0} = \text{cb}_i \,/\, (\text{amt}_i / p_{\text{ticker}(i),\,0})$, where $\text{cb}_i$ and $\text{amt}_i$ are the `cost_basis_amt` and `amt` fields of position $i$. For purchased lots ($j \ge 1$), $c_{i,j} = p_{\text{ticker}(i),\,j-1}$ — the market price when the lot was opened.

---

### Lot Structure

Each lot is identified by $(i, j)$ where $j$ is the period the lot was acquired and $i$ is its row index *within that column*:

- **Column $j = 0$** (starting lots): $i \in \{0, \dots, L - 1\}$, ticker = $\text{ticker}(i, 0)$ (the $i$-th starting position).
- **Column $j \ge 1$** (purchased lots): $i \in \{0, \dots, N - 1\}$, ticker = $\text{ticker}(i, j)$ (the $i$-th model asset).

So at filtration $f$ the available lot set is

$$\mathcal{L}_f = \bigl\{(i, 0) : i \in \{0,\dots,L-1\}\bigr\} \;\cup\; \bigcup_{j=1}^{f}\bigl\{(i, j) : i \in \{0,\dots,N-1\}\bigr\},$$

with $|\mathcal{L}_f| = L + fN$. The total number of lot variables created over all filtrations is

$$\sum_{f=0}^{T-1}|\mathcal{L}_f| = LT + N\,\frac{T(T-1)}{2}.$$

Note the index range depends on the column: $(0, 0)$ refers to the first starting position, while $(0, 1)$ refers to the first model asset purchased at filtration 0.

---

### Decision Variables

All variables are denominated in **shares**, not portfolio weight. Upper bounds are derived from a heuristic portfolio-value cap $\bar{V}_f$ (starting AUM compounded by the maximum monthly growth observed across tickers up to filtration $f$):

$$\bar{V}_f = V_0 \cdot \prod_{t=1}^{f} \max_k \frac{p_{k,t}}{p_{k,t-1}}, \quad \text{share UB} = \lceil \bar{V}_f / p_{k,f} \rceil.$$

#### Lot-level variables — indexed over $(i, j) \in \mathcal{L}_f$

| Variable | Bounds | Description |
|---|---|---|
| $x_{i,j,f} \in \mathbb{R}_+$ | $[0,\, \bar{V}_f / p_{\text{ticker}(i),f}]$ | Shares of lot $(i,j)$ held at filtration $f$ (`lot_shr`) |
| $s^l_{i,j,f} \in \mathbb{R}_+$ | $[0,\, \bar{V}_f / p_{\text{ticker}(i),f}]$ | Shares sold from lot $(i,j)$ at filtration $f$ (`sell_shr_l`) |

#### Holding-level variables — indexed over ticker $k$ at filtration $f$

| Variable | Bounds | Description |
|---|---|---|
| $h_{k,f} \in \mathbb{R}_+$ | $[0,\, \bar{V}_f / p_{k,f}]$ | Total shares of ticker $k$ held (`shr_h`) |
| $s^h_{k,f} \in \mathbb{R}_+$ | $[0,\, \bar{V}_f / p_{k,f}]$ | Total shares of ticker $k$ sold (`sell_shr_h`) |
| $b^h_{k,f} \in \mathbb{R}_+$ | $[0,\, \bar{V}_f / p_{k,f}]$ | Total shares of ticker $k$ bought (`buy_shr_h`) |
| $\delta^s_{k,f} \in \{0,1\}$ | binary | 1 iff ticker $k$ has any sell at filtration $f$ (`sell_h`) |
| $\delta^b_{k,f} \in \{0,1\}$ | binary | 1 iff ticker $k$ has any buy at filtration $f$ (`buy_h`) |

---

### Constraints

#### Starting lot anchoring (`build_starting_lot_constraints`)

At $f = 0$, lot shares are pinned to the input portfolio:

$$x_{i,0,0} = \frac{\text{amt}_i}{p_{\text{ticker}(i),\,0}}, \qquad i \in \{0,\dots,L-1\}.$$

#### Lot ↔ holding linking (`build_lot_holding_linking_constraints`)

Per-ticker holding equals the sum of its lot shares:

$$h_{k,f} = \sum_{(i,j)\,\in\,\mathcal{L}_f(k)} x_{i,j,f} \qquad \forall\, k,\, f.$$

#### Sell aggregation (`build_wash_sales_constraints`)

Holding-level sells aggregate lot-level sells:

$$s^h_{k,f} = \sum_{(i,j)\,\in\,\mathcal{L}_f(k)} s^l_{i,j,f} \qquad \forall\, k,\, f.$$

#### Big-M indicator linking

Binary indicators activate only when the corresponding share quantity is nonzero. The big-M used is the variable's own upper bound $M_{k,f} = \lceil \bar{V}_f / p_{k,f} \rceil$:

$$s^h_{k,f} \le M_{k,f}\, \delta^s_{k,f}, \qquad b^h_{k,f} \le M_{k,f}\, \delta^b_{k,f}.$$

#### No simultaneous buy and sell

For tickers that appear in both the sell and buy universes:

$$\delta^b_{k,f} + \delta^s_{k,f} \le 1.$$

#### Self-financing (per period)

Dollars sold equal dollars bought within each filtration — the portfolio holds no cash:

$$\sum_k p_{k,f}\, s^h_{k,f} \;=\; \sum_k p_{k,f}\, b^h_{k,f} \qquad \forall\, f.$$

#### Lot dynamics (`build_lot_dynamics_constraints`)

Existing lots at $f$ propagate to $f+1$ by subtracting sells:

$$x_{i,j,f+1} = x_{i,j,f} - s^l_{i,j,f}, \qquad (i,j) \in \mathcal{L}_f, \; f < T-1.$$

New lots at $f+1$ (column $j = f + 1$) are initialized from buys at $f$:

$$x_{i,\,f+1,\,f+1} = b^h_{\text{ticker}(i,\,f+1),\,f}, \qquad i \in \{0,\dots,N-1\}, \; f < T-1.$$

The inequality $s^l_{i,j,f} \le x_{i,j,f}$ is enforced *implicitly* by combining the dynamics equation with $x_{i,j,f+1} \ge 0$. (See "Modeling Notes" for the corresponding caveat at $f = T-1$.)

---

### Objectives

The optimizer registers three objectives via Gurobi's hierarchical multi-objective interface (`setObjectiveN`) and minimizes them lexicographically. Higher priority is optimized first; lower priority is minimized subject to the higher priority being optimal. Objectives at the **same** priority are combined with equal weight (Gurobi default).

#### Priority 1 — Terminal weight deviation (`build_terminal_deviation_objective`)

Let $V_T = \sum_k p_{k,T-1}\, h_{k,T-1}$ be the final portfolio value. For each model ticker $k$, an auxiliary $\xi_k \ge 0$ bounds the absolute deviation:

$$\xi_k \ge \big| p_{k,T-1}\,h_{k,T-1} - w^*_k\, V_T \big|.$$

Encoded as two linear inequalities. The objective is

$$\min \sum_{k \in \text{model}} \xi_k.$$

#### Priority 1 — Transitory weight deviation (`build_transitory_deviation_objective`)

At each intermediate filtration $f \in \{1, \dots, T-2\}$ the portfolio is allowed to deviate up to $\pm\delta$ (in weight) from the target. Only the overshoot beyond the band is penalized. Let $V_f = \sum_k p_{k,f}\,h_{k,f}$. For each model ticker $k$, an auxiliary $\zeta_{k,f} \ge 0$ captures the out-of-band dollar deviation:

$$\zeta_{k,f} \ge p_{k,f}\,h_{k,f} - (w^*_k + \delta)\,V_f,$$
$$\zeta_{k,f} \ge (w^*_k - \delta)\,V_f - p_{k,f}\,h_{k,f}.$$

Within the band both right-hand sides are $\le 0$, so $\zeta_{k,f}$ can stay at 0. The objective is

$$\min \sum_{f=1}^{T-2} \sum_{k \in \text{model}} \zeta_{k,f}.$$

This objective shares priority 1 with terminal deviation, so both are minimized jointly before tax cost.

#### Priority 0 — Realized tax cost (`build_tax_cost_objective`)

Total realized gain/loss across all sells, decomposed per filtration and then summed. For each filtration $f \in \{0,\dots,T-2\}$ a scalar variable $\text{tc}_f$ is defined by

$$\text{tc}_f = \tau \sum_{(i,j)\in\mathcal{L}_f} s^l_{i,j,f} \cdot \bigl(p_{\text{ticker}(i),f} - c_{i,j}\bigr),$$

and stored in `self.filtration[f]["tax_cost"]` for post-solve inspection. The aggregate objective is

$$\min \sum_f \text{tc}_f.$$

Losses contribute negatively, so the optimizer is incentivized to harvest them. The objective variable has $\text{lb} = -\infty$.

---

## Modeling Notes

- **Multi-scenario, equally-likely.** All three objectives are averaged across the supplied scenarios with uniform weight $1/S$. Probability-weighted scenarios are not yet supported.
- **Non-anticipativity at the root only.** The first-period decisions (`sell_shr_l[(s, 0)]`, `buy_shr_h[(s, 0)]`) are forced to be identical across scenarios via `build_information_pattern_constraints`, which pins `lot_shr[(s, 1)]` equal across $s$. From $f \ge 1$ onward, decisions (sells, buys, holdings) are scenario-specific — this is the full-recourse stage.
- **Per-(s, f) variable layout.** Each filtration entry carries `lot_shr` and `shr_h` (always); `sell_shr_l`, `sell_shr_h`, `buy_shr_h`, `sell_h`, `buy_h` and the per-period `tax_cost` scalar (only at $f < T-1$, since selling at the terminal period would not feed into any downstream state); `trans_dev` (only at $1 \le f \le T-2$, the band-penalty range); and `terminal_dev` (only at $f = T-1$).
- **Tax cost is dollar-denominated.** The objective is $\tau \cdot \text{realized gains}$ summed over $(s, f)$ and averaged over $s$, so its solved value is the actual expected tax bill in dollars. Multiplying by $\tau$ leaves the optimal trade-off unchanged but makes the objective magnitude interpretable. No short-term vs long-term distinction; loss harvesting is treated 1:1.
- **Self-financing forces full investment.** Cash positions are not modeled; every sell dollar must be matched by a buy dollar in the same period (within each scenario). A zero-action period is allowed (both sums = 0).
- **Wash-sale rule is local.** Only same-period buys and sells of the same ticker are blocked. Cross-period wash sales (real rule: 30 days) are not enforced.
- **MIP gap is per-stage.** When `MIPGap` is supplied, it loosens optimality only on the tax-cost stage of the lex hierarchy via `model.getMultiobjEnv(idx).setParam("MIPGap", value)` — the deviation stages keep Gurobi's default tight gap.

---

## Running the Project

### Prescient (perfect-foresight) benchmark

```bash
poetry run python run_prescient_case.py
```

Builds a single-scenario `StoxOptimizer` over realized 2024 prices (wrapped as a one-element list). Saves four PNGs:
`cumulative_tax_cost.png`, `portfolio_value.png`, `transition_pct.png`, `AAPL_weight_and_price.png`.

### MPC t=0 plan

```bash
poetry run python run_mpc.py
```

Runs `RMPController`: bootstrap `n_scenario` price paths from a historical window, build the multi-scenario `StoxOptimizer`, and run a single solve. The solution, scenarios, model, positions, realized prices, and config are pickled to `mpc_output.pkl` for downstream analysis.

### MPC analysis with prescient overlay

```bash
poetry run python analyze_mpc.py
```

Loads `mpc_output.pkl`, runs the prescient benchmark on the realized prices for comparison, and produces four overlay plots: cumulative tax cost, total portfolio value, % transition, and AAPL weight + price (dual axis). MPC-t=0 paths are translucent steelblue with a thick mean line; the prescient trajectory overlays as dashed crimson. Output PNGs: `mpc_t0_cumulative_tax_cost.png`, `mpc_t0_portfolio_value.png`, `mpc_t0_transition_pct.png`, `mpc_t0_AAPL_weight_and_price.png`.

> The plots show the **t=0 plan** (one stochastic solve at the start), not a rolling MPC trajectory. The rolling case is `Backtester` and is not yet wrapped by an analysis script.

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

# Add a dependency to stochastic-optimizer
cd packages/stochastic-optimizer && poetry add <package>

# Format all files manually
poetry run black .
```

### Code Style

[black](https://github.com/psf/black) and [isort](https://pycqa.github.io/isort/) are enforced via pre-commit hooks. They run automatically on every `git commit` and reformat staged `.py` files (isort uses the `--profile black` setting for compatibility).

---

## Dependencies

| Package | Purpose |
|---|---|
| `gurobipy` | Gurobi solver interface (used by `stochastic-optimizer`) |
| `numpy` / `scipy` | Numerical computation |
| `pandas` | Data handling (prices, tax lots, positions) |
| `yfinance` | Download historical price data from Yahoo Finance |
| `matplotlib` | Plotting results |
| `jupyter` | Notebooks for exploration and writeup |
| `lxml` / `requests` | HTML parsing and HTTP (yfinance dependencies) |
| `black` *(dev)* | Code formatter |
| `isort` *(dev)* | Import sorter |
| `pre-commit` *(dev)* | Git hook runner (enforces black + isort on commit) |