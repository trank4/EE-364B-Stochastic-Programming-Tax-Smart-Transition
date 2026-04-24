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
├── pyproject.toml                  # Root project (application, not a library)
├── poetry.toml                     # Poetry config (in-project virtualenv)
├── poetry.lock
└── .pre-commit-config.yaml         # black formatter hook
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
    model: gp.Model | None   # Gurobi model, None until build() is called

    def __init__(self) -> None: ...   # Initialize optimizer state
    def build(self) -> None: ...      # Construct variables, constraints, objective
    def solve(self) -> None: ...      # Invoke Gurobi solver, extract results
```

**Dependencies:** `gurobipy >= 11.0.0`

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

[black](https://github.com/psf/black) is enforced via a pre-commit hook. It runs automatically on every `git commit` and reformats staged `.py` files.

---

## Dependencies

| Package | Purpose |
|---|---|
| `cvxpy` | Convex optimization modelling (Clarabel, OSQP, SCS, HiGHS solvers) |
| `gurobipy` | Gurobi solver interface (used by `stochastic-optimizer`) |
| `numpy` / `scipy` | Numerical computation |
| `pandas` | Data handling (returns, prices, tax lots) |
| `matplotlib` | Plotting results |
| `jupyter` | Notebooks for exploration and writeup |
| `black` *(dev)* | Code formatter |
| `pre-commit` *(dev)* | Git hook runner |
