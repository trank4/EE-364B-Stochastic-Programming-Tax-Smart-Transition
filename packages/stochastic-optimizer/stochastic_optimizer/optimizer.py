from __future__ import annotations

import gurobipy as gp


class StoxOptimizer:
    """Stochastic portfolio transition optimizer backed by Gurobi."""

    def __init__(self) -> None:
        self.model: gp.Model | None = None

    def build(self) -> None:
        """Construct variables, constraints, and objective in self.model."""
        raise NotImplementedError

    def solve(self) -> None:
        """Invoke the Gurobi solver and extract results."""
        raise NotImplementedError
