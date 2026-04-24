# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

EE364B (Convex Optimization II) course project implementing stochastic programming methods for tax-smart portfolio transition — optimizing asset liquidation/reallocation while minimizing tax impact under uncertainty.

## Environment

This project uses **Poetry** with the virtualenv stored at `.venv/` inside the repo (`poetry.toml` sets `virtualenvs.in-project = true`).

```bash
poetry install          # install all dependencies
poetry add <pkg>        # add a new package
poetry shell            # activate the venv in a new shell
poetry run python <script>.py
poetry run jupyter notebook
```

## Key Dependencies

- **cvxpy** — convex optimization problem formulation and solving (Clarabel, OSQP, SCS, HiGHS solvers bundled)
- **numpy / scipy** — numerical computation
- **pandas** — data handling (returns, prices, tax lots)
- **matplotlib** — plotting results
- **jupyter** — notebooks for exploration and writeup
