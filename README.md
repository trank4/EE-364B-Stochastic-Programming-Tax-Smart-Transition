# EE-364B Stochastic Programming — Tax-Smart Transition

Course project for EE364B (Convex Optimization II) at Stanford. Implements stochastic programming methods to optimize tax-smart portfolio transitions — minimizing tax impact while reallocating assets under uncertainty.

> **Status:** the optimizer currently solves the **prescient (single-scenario) case** — perfect foresight over a fixed monthly price path. Multi-scenario extensions are planned but not yet implemented.

---

## Repository Structure

```
.
├── packages/
│   └── stochastic-optimizer/       # Installable optimization library (Gurobi-backed)
│       ├── pyproject.toml
│       └── stochastic_optimizer/
│           ├── __init__.py
│           └── optimizer.py        # StoxOptimizer class
├── run_prescient_case.py           # Prescient (perfect-foresight) benchmark runner
├── pyproject.toml                  # Root project (application, not a library)
├── poetry.toml                     # Poetry config (in-project virtualenv)
├── poetry.lock
└── .pre-commit-config.yaml         # black + isort pre-commit hooks
```

---

## Packages

### `stochastic-optimizer` (`packages/stochastic-optimizer/`)

Reusable optimization library that wraps Gurobi. Declared as an editable path dependency of the root project.

**Public API** (`from stochastic_optimizer import StoxOptimizer`):

| Class | File | Description |
|---|---|---|
| `StoxOptimizer` | `optimizer.py` | Main optimizer object |

**`StoxOptimizer` interface:**

```python
class StoxOptimizer:
    model: gp.Model        # Gurobi model, initialized in __init__
    inputs: dict           # see "inputs dict keys" below
    T: int                 # number of monthly filtration periods
    n_asset: int           # size of model universe N
    n_start_pos: int       # number of starting positions L
    filtration: list[dict] # per-period state: variables, prices, lot info
    objectives: dict       # name -> [Var, priority] for hierarchical objectives

    def __init__(self, inputs: dict) -> None: ...
    def build(self) -> None: ...     # Construct variables, constraints, objectives
    def solve(self) -> None: ...     # Invoke Gurobi solver
```

`build()` runs the following pipeline in order:

1. `build_filtration` — create all decision variables and cache prices/cost basis per filtration.
2. `build_lot_holding_linking_constraints` — link lot-level shares to ticker-level holdings.
3. `build_starting_lot_constraints` — anchor lot shares at filtration 0 to the input portfolio.
4. `build_wash_sales_constraints` — sell aggregation, big-M indicator linking, no-simultaneous-buy-and-sell, dollar self-financing.
5. `build_lot_dynamics_constraints` — link consecutive periods via sells (existing lots) and buys (new lots).
6. `build_terminal_deviation_objective` — total absolute deviation from target weights at the final period.
7. `build_tax_cost_objective` — total realized gain/loss across all sells (proxy for tax cost).
8. `set_objective_hierarchy` — register both objectives with `setObjectiveN` for lexicographic minimization.

**`inputs` dict keys:**

| Key | Type | Description |
|---|---|---|
| `positions` | `pd.DataFrame` | Starting portfolio (`tkr`, `amt`, `cost_basis_amt`; `pnl`/`wt` optional) |
| `tax_rate` | `float` | Flat capital gains tax rate (currently unused — see "Modeling Notes") |
| `model` | `pd.DataFrame` | Target model portfolio (`tkr`, `tgt_wt`) |
| `monthly_prices` | `pd.DataFrame` | Month-start prices, rows = periods, columns = tickers |
| `scenario_prob` | `list \| None` | Reserved for multi-scenario extension; currently unused |

**Dependencies:** `gurobipy >= 11.0.0`, `numpy`, `pandas`

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
| $\tau$ | Flat capital gains tax rate (not currently used in the objective) |
| $c_{i,j}$ | Per-share cost basis of lot $(i, j)$ |

For starting lots ($j = 0$), $c_{i,0} = \text{cost\_basis\_amt}_i \,/\, (\text{amt}_i / p_{\text{ticker}(i),\,0})$ — i.e. cost basis dollars divided by shares acquired. For purchased lots ($j \ge 1$), $c_{i,j} = p_{\text{ticker}(i),\,j-1}$ since the lot was bought at filtration $j - 1$.

---

### Lot Structure

Each lot is identified by $(i, j)$ where $j$ is the period the lot was acquired and $i$ is its row index *within that column*:

- **Column $j = 0$** (starting lots): $i \in \{0, \dots, L - 1\}$, ticker = $\text{pos\_tkrs}[i]$.
- **Column $j \ge 1$** (purchased lots): $i \in \{0, \dots, N - 1\}$, ticker = $\text{model\_tkrs}[i]$.

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

$$x_{i,\,f+1,\,f+1} = b^h_{\text{model\_tkrs}[i],\,f}, \qquad i \in \{0,\dots,N-1\}, \; f < T-1.$$

The inequality $s^l_{i,j,f} \le x_{i,j,f}$ is enforced *implicitly* by combining the dynamics equation with $x_{i,j,f+1} \ge 0$. (See "Modeling Notes" for the corresponding caveat at $f = T-1$.)

---

### Objectives

The optimizer registers two objectives via Gurobi's hierarchical multi-objective interface (`setObjectiveN`) and minimizes them lexicographically. Higher priority is optimized first; lower priority is minimized subject to the higher priority being optimal.

#### Priority 1 — Terminal weight deviation (`build_terminal_deviation_objective`)

Let $V_T = \sum_k p_{k,T-1}\, h_{k,T-1}$ be the final portfolio value. For each model ticker $k$, an auxiliary $\xi_k \ge 0$ bounds the absolute deviation:

$$\xi_k \ge \big| p_{k,T-1}\,h_{k,T-1} - w^*_k\, V_T \big|.$$

Encoded as two linear inequalities. The objective is

$$\min \sum_{k \in \text{model}} \xi_k.$$

#### Priority 0 — Realized tax cost (`build_tax_cost_objective`)

Total realized gain/loss across all sells, summed over filtrations:

$$\min \sum_f \sum_{(i,j)\in\mathcal{L}_f} s^l_{i,j,f} \cdot \bigl(p_{\text{ticker}(i),f} - c_{i,j}\bigr).$$

The flat tax rate $\tau$ is omitted because, with a single rate, scaling by $\tau$ doesn't change the argmin. Losses contribute negatively, so the optimizer is incentivized to harvest them. The objective variable has $\text{lb} = -\infty$.

---

## Modeling Notes

- **Single scenario only.** Despite the class name, only one price path is consumed (`monthly_prices`). `scenario_prob` is reserved for a future multi-scenario extension.
- **Tax cost ≈ realized gain/loss.** With a single flat rate, minimizing $\tau \cdot \text{gains}$ and $\text{gains}$ are equivalent. No short-term vs long-term distinction; loss harvesting is treated 1:1.
- **Self-financing forces full investment.** Cash positions are not modeled; every sell dollar must be matched by a buy dollar in the same period. A zero-action period is allowed (both sums = 0).
- **Wash-sale rule is local.** Only same-period buys and sells of the same ticker are blocked. Cross-period wash sales (real rule: 30 days) are not enforced.
- **Last-period sells are not bounded by holdings.** The dynamics constraint runs for $f \in \{0, \dots, T-2\}$, so $s^l_{i,j,T-1}$ is bounded only by the variable's UB, not by $x_{i,j,T-1}$. This is a known issue (see TODO); it can be patched by adding $s^l_{i,j,f} \le x_{i,j,f}$ at every $f$, or by treating $T-1$ as a no-trade evaluation period.

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