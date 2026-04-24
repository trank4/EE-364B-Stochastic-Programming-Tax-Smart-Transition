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

## Git Practices

### Commit messages
Write clear, descriptive commit messages that enumerate the specific changes included — not just the intent. The subject line names the primary change; the body bullet-points each distinct file or logical change (e.g. `- Add optimizer.py with StoxOptimizer stub`). Use imperative mood. Never write vague labels like "update files" or "misc changes".

### Staging new files
After creating any new file, consider whether it belongs in the remote repo. If yes, `git add` it immediately. Skip generated artifacts and anything covered by `.gitignore` (`.venv/`, `__pycache__/`, `.env`, etc.).

### Keeping documentation in sync
Before every commit, update `README.md` to reflect what changed — repo structure, package APIs, dependencies, setup instructions. Stage the updated README alongside the other changed files.
