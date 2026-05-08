import math

import numpy as np
import pandas as pd
from stochastic_optimizer.analysis_utils import fetch_monthly_price
from stochastic_optimizer.optimizer import run_optimizer


def run_RMPController(inputs: dict) -> dict:
    """Run a single stochastic-optimizer solve over bootstrapped scenarios."""
    controller = RMPController(inputs)
    controller.build_price_scenarios()
    return controller.solve()


class RMPController:
    """
    Robust MPC controller. Single responsibility:
      1. Generate price scenarios via block bootstrap from a historical window.
      2. Construct a StoxOptimizer with those scenarios + the input positions.
      3. Run a single solve and return the solution dict.

    Rolling forward in time with realized prices is delegated to Backtester.
    """

    def __init__(self, inputs: dict):
        self.inputs = inputs
        self.scenario_prices: list[pd.DataFrame] = []

        # all tickers
        self.all_tkrs = list(
            set(self.inputs["positions"]["tkr"]).union(set(self.inputs["model"]["tkr"]))
        )
        self.n_period = self.inputs["n_period"]
        self.n_scenario = self.inputs["n_scenario"]
        self.seed = self.inputs.get("seed", 42)
        # relative MIP gap applied to the tax-cost stage of the optimizer
        self.MIPGap = self.inputs.get("MIPGap", 0.05)

    def build_price_scenarios(self):
        """
        Block-bootstrap n_scenario price paths of length n_period from a
        historical return window. The first period of each scenario is anchored
        at the actual price observed on inputs["start_date"].

        Expected inputs keys beyond the standard StoxOptimizer inputs:
          start_date     — date of the actual portfolio (ISO string or Timestamp)
          sim_start_date — start of historical window used for block bootstrap
          sim_end_date   — end of historical window
          n_scenario     — number of price scenarios to generate
          n_period       — prediction horizon (monthly periods per scenario)
          block_length   — bootstrap block length in months
          seed           — (optional) RNG seed, default 42
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

        # build overlapping blocks of length block_length from historical returns
        block_length = self.inputs["block_length"]
        # blocks_arr shape: (n_blocks, block_length, n_tkr)
        n_hist = len(monthly_returns)
        blocks_arr = np.stack(
            [
                monthly_returns[i : i + block_length]
                for i in range(n_hist - block_length + 1)
            ]
        )

        # run block bootstrapping for self.n_scenario scenarios
        # each scenario needs n_period-1 return periods (period 0 is the known starting price)
        n_blocks_needed = math.ceil((self.n_period - 1) / block_length)
        rng = np.random.default_rng(seed=self.seed)

        self.scenario_prices = []
        for _ in range(self.n_scenario):
            sampled_idx = rng.integers(0, len(blocks_arr), size=n_blocks_needed)
            # fancy-index blocks, flatten to (n_blocks_needed * block_length, n_tkr), trim
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

    def solve(self) -> dict:
        """
        Build a single StoxOptimizer with the bootstrapped scenarios and run it.
        Returns the optimizer's solution dict. Must be called after
        build_price_scenarios().
        """
        opt_inputs = {
            "positions": self.inputs["positions"],
            "tax_rate": self.inputs["tax_rate"],
            "model": self.inputs["model"],
            "tkr_adev": self.inputs["tkr_adev"],
            "monthly_prices": self.scenario_prices,
            "MIPGap": self.MIPGap,
        }
        return run_optimizer(opt_inputs)
