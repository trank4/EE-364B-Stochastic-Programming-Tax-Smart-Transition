# EE-364B Stochastic Programming — Tax-Smart Transition

Course project for EE364B (Convex Optimization II) at Stanford. Implements stochastic programming methods to optimize tax-smart portfolio transitions — minimizing tax impact while reallocating assets under uncertainty.

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
    model: gp.Model   # Gurobi model, initialized in __init__

    def __init__(self, inputs: dict) -> None: ...  # Initialize optimizer state with inputs dict
    def build(self) -> None: ...                   # Construct variables, constraints, objective
    def solve(self) -> None: ...                   # Invoke Gurobi solver, extract results
```

**`inputs` dict keys:**

| Key | Type | Description |
|---|---|---|
| `positions` | `pd.DataFrame` | Starting portfolio (`Tkr`, `Amt`, `CostBasisAmt` columns) |
| `tax_rate` | `float` | Flat capital gains tax rate (e.g. `0.3`) |
| `model` | `pd.DataFrame` | Target model portfolio (`Tkr`, `TgtWt` columns) |
| `monthly_prices` | `pd.DataFrame` | Month-start prices, rows = periods, columns = tickers |

**Dependencies:** `gurobipy >= 11.0.0`

---

## Optimization Problem

### Overview

We manage a portfolio over $T$ monthly periods. The investor starts with $L$ tax lots — existing stock positions each carrying a cost basis — and wants to transition towards a target model portfolio of $N$ assets while minimizing capital gains taxes incurred along the way.

At each filtration $f$, the optimizer decides how much of each lot to sell and how much of each asset to buy, subject to budget balance. Selling a lot at a profit triggers a capital gains tax at flat rate $\tau$. The goal is to reach the target weights as efficiently as possible while keeping the tax drag small.

---

### Notation

#### Indices and Sets

| Symbol | Definition |
|---|---|
| $f \in \{0, \dots, T-1\}$ | Filtration index (month) |
| $k \in \{0, \dots, K_f-1\}$ | Ticker index within filtration $f$, where $K_f$ is the number of distinct tickers present at filtration $f$ |
| $(i,\, j)$ | Lot identifier: row index $i$, acquired at period $j \leq f$ |
| $\mathcal{L}_f$ | Set of all lots available at filtration $f$ (see Lot Structure below) |
| $\mathcal{L}_f(k)$ | Subset of $\mathcal{L}_f$ belonging to ticker $k$ |

#### Parameters

| Symbol | Definition |
|---|---|
| $T$ | Number of filtration periods |
| $N$ | Number of assets in the model universe |
| $L$ | Number of starting lots (existing positions) |
| $p_{k,f}$ | Market price of ticker $k$ at filtration $f$ |
| $w^*_k$ | Target portfolio weight for ticker $k$ |
| $\tau$ | Flat capital gains tax rate |
| $c_{i,j}$ | Cost basis of lot $(i, j)$: input value for starting lots ($j=0$), or price at period $j-1$ for purchased lots |

---

### Lot Structure

Each lot is identified by a pair $(i,\, j)$: $j$ is the period the lot was purchased and $i$ is the lot's row index. At filtration $f$, the optimizer has visibility into all lots acquired from period $0$ through $f$. The available lot set grows **triangularly** with $f$:

$$\mathcal{L}_f = \bigl\{(i,\, j) : j \in \{0, \dots, f\},\quad i \in \{0, \dots, L + j N - 1\}\bigr\}$$

The column $j$ of the triangle contains $L + jN$ lots — the $L$ starting lots plus $N$ new lots per purchase period elapsed. At each new filtration $f$, a new column $j = f$ is added with $N$ freshly purchased lots (one per model asset).

The ticker of lot $(i, j)$ depends only on the row index $i$:

$$\text{ticker}(i) = \begin{cases} \text{pos\_tkrs}[i] & \text{if } i < L \quad \text{(starting lot)} \\ \text{model\_tkrs}[(i - L) \bmod N] & \text{if } i \geq L \quad \text{(purchased lot)} \end{cases}$$

The total number of lot variables created across all filtrations is:

$$\sum_{f=0}^{T-1} |\mathcal{L}_f| = \sum_{f=0}^{T-1}\sum_{j=0}^{f}(L + jN) = LT^2/2 + N\,T(T-1)(T+1)/6 \quad \text{(triangular sum)}$$

---

### Decision Variables

All variables are replicated at every filtration $f \in \{0, \dots, T-1\}$.

#### Lot-level variables — indexed over $(i,\, j) \in \mathcal{L}_f$

| Variable | Bounds | Description |
|---|---|---|
| $x_{i,j,f} \in \mathbb{R}$ | $[0,\, 100]$ | Portfolio weight (%) of lot $(i,j)$ held at filtration $f$ |
| $s^l_{i,j,f} \in \mathbb{R}$ | $[0,\, 100]$ | Portfolio weight (%) sold from lot $(i,j)$ at filtration $f$ |

#### Holding-level variables — indexed over ticker $k$ at filtration $f$

| Variable | Bounds | Description |
|---|---|---|
| $s^h_{k,f} \in \mathbb{R}$ | $[0,\, 100]$ | Total portfolio weight (%) of ticker $k$ sold at filtration $f$ |
| $b^h_{k,f} \in \mathbb{R}$ | $[0,\, 100]$ | Portfolio weight (%) of ticker $k$ bought at filtration $f$ |
| $\delta^s_{k,f} \in \{0,1\}$ | binary | 1 if any of ticker $k$ is sold at filtration $f$ |
| $\delta^b_{k,f} \in \{0,1\}$ | binary | 1 if any of ticker $k$ is bought at filtration $f$ |

---

### Constraints

#### Sell aggregation (`build_wash_sales_constraints`)

The holding-level sell weight is the sum of all lot-level sells for that ticker:

$$s^h_{k,f} = \sum_{(i,j)\,\in\,\mathcal{L}_f(k)} s^l_{i,j,f} \qquad \forall\, k,\, f$$

#### Big-M indicator linking

The binary indicators activate only when the corresponding weight is nonzero. With portfolio weights bounded in $[0, 100]$:

$$s^h_{k,f} \leq 100\, \delta^s_{k,f} \qquad \forall\, k,\, f$$

$$b^h_{k,f} \leq 100\, \delta^b_{k,f} \qquad \forall\, k,\, f$$

#### No simultaneous buy and sell (wash-sale prevention)

A ticker cannot be both bought and sold in the same filtration period:

$$\delta^b_{k,f} + \delta^s_{k,f} \leq 1 \qquad \forall\, k,\, f$$

#### Lot dynamics (`build_lot_dynamics_constraints`)

> **In progress.**

#### Objective

> **In progress.** The intended objective minimizes a weighted combination of:
> 1. **Tax cost** — capital gains taxes on appreciated lots: $\tau \displaystyle\sum_f \sum_{(i,j)\in\mathcal{L}_f} s^l_{i,j,f} \cdot \max\!\bigl(p_{\text{ticker}(i),\,f} - c_{i,j},\; 0\bigr)$
> 2. **Tracking error** — deviation of end-of-horizon weights from target $w^*$

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
