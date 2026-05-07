import math

import numpy as np
import pandas as pd
from stochastic_optimizer import StoxOptimizer
from stochastic_optimizer.analysis_utils import fetch_monthly_price


class RMPController:
    """
    Robust Model Predictive Controller to generate historical simulation and run stochastic optimizer for transition
    """

    def __init__(self, inputs: dict):
        self.inputs = inputs
        self.optimizers = []
        self.scenario_prices = []

        # all tickers
        self.all_tkrs = list(
            set(self.inputs["positions"]["tkr"]).union(set(self.inputs["model"]["tkr"]))
        )
        self.n_period = self.inputs["n_period"]
        self.n_scenario = self.inputs["n_scenario"]

    def build_price_scenarios(self):
        """
        Run simulation to set up price scenarios and build optimizer.

        Expected inputs keys beyond the standard StoxOptimizer inputs:
          start_date     — date of the actual portfolio (ISO string, e.g. "2024-01-01")
          sim_start_date — start of historical window used for block bootstrap
          sim_end_date   — end of historical window used for block bootstrap
          n_scenario     — number of price scenarios to generate
          n_period       — prediction horizon (number of monthly periods per scenario)
          seed           — (optional) RNG seed for reproducibility, default 42
        """
        # --- get the price on start date ---
        start_date = self.inputs["start_date"]
        # use a 2-month window so resample("MS") always captures the start_date row
        end_date_excl = pd.date_range(start=start_date, periods=3, freq="MS")[
            2
        ].strftime("%Y-%m-%d")
        start_price = (
            fetch_monthly_price(
                self.all_tkrs, start_date=start_date, end_date=end_date_excl
            )
            .iloc[0]
            .reindex(self.all_tkrs)
        )

        # --- set up price scenarios ---
        historical_prices = fetch_monthly_price(
            self.all_tkrs,
            start_date=self.inputs["sim_start_date"],
            end_date=self.inputs["sim_end_date"],
        ).reindex(columns=self.all_tkrs)

        # get monthly returns from prices
        monthly_returns = (
            historical_prices.pct_change().dropna().values
        )  # (n_hist, n_tkr)

        # build overlapping blocks of length _BLOCK_LEN from historical returns
        _BLOCK_LEN = self.inputs["block_length"]
        # blocks_arr shape: (n_blocks, _BLOCK_LEN, n_tkr)
        n_hist = len(monthly_returns)
        blocks_arr = np.stack(
            [
                monthly_returns[i : i + _BLOCK_LEN]
                for i in range(n_hist - _BLOCK_LEN + 1)
            ]
        )

        # run block bootstrapping for self.n_scenario scenarios
        # each scenario needs n_period-1 return periods (period 0 is the known starting price)
        n_blocks_needed = math.ceil((self.n_period - 1) / _BLOCK_LEN)
        rng = np.random.default_rng(seed=self.inputs.get("seed", 42))

        self.scenario_prices = []
        for _ in range(self.n_scenario):
            sampled_idx = rng.integers(0, len(blocks_arr), size=n_blocks_needed)
            # fancy-index blocks, flatten to (n_blocks_needed * _BLOCK_LEN, n_tkr), trim
            sampled_returns = blocks_arr[sampled_idx].reshape(
                -1, monthly_returns.shape[1]
            )[
                : self.n_period - 1
            ]  # (n_period-1, n_tkr)

            # stitch starting price with simulated returns to get price path
            price_matrix = np.empty((self.n_period, len(self.all_tkrs)))
            price_matrix[0] = start_price.values
            for t in range(1, self.n_period):
                price_matrix[t] = price_matrix[t - 1] * (1 + sampled_returns[t - 1])
            assert price_matrix.shape == (self.n_period, len(self.all_tkrs))
            self.scenario_prices.append(
                pd.DataFrame(price_matrix, columns=self.all_tkrs)
            )

    def execute(self) -> dict:
        """
        solve each step forward by running the stochastic optimizer
        """
        for t in self.n_period:
            pass
        sol = {}
        return sol
