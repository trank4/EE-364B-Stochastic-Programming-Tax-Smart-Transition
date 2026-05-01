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

At each period $t$, the optimizer decides how much of each lot to sell and how much of each asset to buy, subject to budget balance. Selling a lot at a profit triggers a capital gains tax at flat rate $\tau$. The goal is to reach the target weights as efficiently as possible while keeping the tax drag small.

---

### Notation

#### Indices and Sets

| Symbol | Definition |
|---|---|
| $t \in \{0, \dots, T-1\}$ | Time period (month) |
| $k \in \{0, \dots, N-1\}$ | Asset (ticker) index in the model universe |
| $(i,\, j)$ | Lot identifier: row index $i$, acquired at period $j$ |
| $\mathcal{L}_t$ | Set of all lots available at period $t$ (see Lot Structure below) |
| $\mathcal{L}_t(k)$ | Subset of $\mathcal{L}_t$ belonging to ticker $k$ |

#### Parameters

| Symbol | Definition |
|---|---|
| $T$ | Number of periods |
| $N$ | Number of assets in the model universe |
| $L$ | Number of starting lots (existing positions) |
| $p_{k,t}$ | Price of asset $k$ at period $t$ |
| $w^*_k$ | Target portfolio weight for asset $k$ |
| $\tau$ | Flat capital gains tax rate |
| $c_{i,j}$ | Cost basis per unit of lot $(i, j)$ |

---

### Lot Structure

Lots are indexed by a pair $(i,\, j)$ where $j$ is the period the lot was acquired and $i$ identifies the specific lot within that period.

The set of lots available at period $t$ is **triangular** — it grows as new purchases create new lots:

$$\mathcal{L}_t = \bigl\{(i,\, j) : j \in \{0, \dots, t\},\quad i \in \{0, \dots, L + j N - 1\}\bigr\}$$

- **$j = 0$ (starting lots):** $i \in \{0, \dots, L-1\}$. Each corresponds to one existing position from the input portfolio.
- **$j > 0$ (purchased lots):** $i \in \{L + (j-1)N, \dots, L + jN - 1\}$ are the $N$ new lots created by buying each model asset at period $j$.

The ticker of lot $(i, j)$ is:

$$\text{ticker}(i) = \begin{cases} \text{pos\_tkrs}[i] & \text{if } i < L \quad \text{(starting lot)} \\ \text{model\_tkrs}[(i - L) \bmod N] & \text{if } i \geq L \quad \text{(purchased lot)} \end{cases}$$

At period $T-1$, the total number of lots across all periods is:

$$|\mathcal{L}_{T-1}| = \sum_{j=0}^{T-1} (L + j N) = LT + N\frac{T(T-1)}{2}$$

---

### Decision Variables

All variables are defined at every period $t \in \{0, \dots, T-1\}$.

#### Lot-level variables — indexed over $(i,\, j) \in \mathcal{L}_t$

| Variable | Bounds | Description |
|---|---|---|
| $x_{i,j,t} \in \mathbb{R}$ | $[0,\, 100]$ | Portfolio weight (%) of lot $(i,j)$ held at period $t$ |
| $s^l_{i,j,t} \in \mathbb{R}$ | $[0,\, 100]$ | Portfolio weight (%) sold from lot $(i,j)$ at period $t$ |

#### Holding-level variables — indexed over ticker $k$

| Variable | Bounds | Description |
|---|---|---|
| $s^h_{k,t} \in \mathbb{R}$ | $[0,\, 100]$ | Total portfolio weight (%) of ticker $k$ sold at period $t$ |
| $b^h_{k,t} \in \mathbb{R}$ | $[0,\, 100]$ | Portfolio weight (%) of ticker $k$ bought at period $t$ |
| $\delta^s_{k,t} \in \{0,1\}$ | binary | 1 if any of ticker $k$ is sold at period $t$ |
| $\delta^b_{k,t} \in \{0,1\}$ | binary | 1 if any of ticker $k$ is bought at period $t$ |

The holding-level sell weight aggregates lot-level sells:

$$s^h_{k,t} = \sum_{(i,j)\,\in\,\mathcal{L}_t(k)} s^l_{i,j,t}$$

The binary indicators $\delta^s_{k,t}$ and $\delta^b_{k,t}$ are used to enforce that no ticker is simultaneously bought and sold in the same period (a wash-sale / round-trip constraint), and to model per-transaction costs if needed.

---

### Objective and Constraints

> **In progress.** The objective function and constraints are under active development.

The intended objective minimizes a weighted combination of:
1. **Tax cost** — capital gains taxes triggered by selling appreciated lots: $\tau \sum_t \sum_{(i,j)\in\mathcal{L}_t} s^l_{i,j,t} \cdot \max(p_{\text{ticker}(i),\,t} - c_{i,j},\; 0)$
2. **Tracking error** — deviation of the end-of-horizon portfolio weights from the target $w^*$

Subject to constraints including budget balance (proceeds from sales fund purchases), lot accounting (holdings evolve consistently across periods), and no simultaneous buy/sell of the same ticker.

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
