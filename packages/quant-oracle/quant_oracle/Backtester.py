import pandas as pd
from quant_oracle.optimizer import ForwardOptimizer
from quant_oracle.RMPController import RMPController


class Backtester:
    """
    Rolling backtester anchored on a fixed transition target date. At each
    period t along an actual realized price path, the controller is invoked
    with a *receding* horizon (n_period = n_steps - t) and a *rolling* 1-year
    historical window for the bootstrap. The controller solves once, only the
    first-period trades are executed, positions are marked to market at the
    next period's actual price, and the loop advances.

    Mechanics per step:
      - horizon shrinks by one each step so the target end date stays fixed
        (n_period = 12, 11, ..., 1 across a 12-step backtest).
      - sim_end_date = current step start_date; sim_start_date = start_date
        minus `lookback_months` months. The bootstrap reflects the most
        recently observed year of returns.
      - at the final step (n_period == 1) the current-period price is already
        realized, so bootstrapping is skipped and ForwardOptimizer is called
        directly on a single deterministic price scenario.
    """

    def __init__(
        self,
        base_inputs: dict,
        actual_prices: pd.DataFrame,
        lookback_months: int = 12,
    ):
        """
        Args:
            base_inputs: template inputs for RMPController. The "positions",
                "start_date", "n_period", "sim_start_date", and "sim_end_date"
                entries are overridden each step.
            actual_prices: realized monthly prices for the backtest window with
                rows indexed by date and columns indexed by ticker. The loop
                advances over `len(actual_prices) - 1` steps, so the last row
                is only used for marking-to-market the final positions.
            lookback_months: length of the rolling historical window used to
                build the block-bootstrap return blocks. Defaults to 12 months.
        """
        self.base_inputs = base_inputs
        self.actual_prices = actual_prices
        self.lookback_months = lookback_months

    def run(self) -> list[dict]:
        """
        Returns a list of per-step dicts:
          - "t"                : time step index
          - "n_period"         : remaining horizon used at this step
          - "positions_before" : positions DataFrame entering this step
          - "sol"              : full optimizer solution dict from solve()
        """
        current_positions = self.base_inputs["positions"].copy()
        n_steps = len(self.actual_prices) - 1
        results = []

        for t in range(n_steps):
            step_start = self.actual_prices.index[t]
            n_period = n_steps - t

            if n_period == 1:
                sol = self._solve_deterministic_step(t, current_positions)
            else:
                step_inputs = {
                    **self.base_inputs,
                    "positions": current_positions,
                    "start_date": step_start,
                    "n_period": n_period,
                    "sim_end_date": step_start,
                    "sim_start_date": step_start
                    - pd.DateOffset(months=self.lookback_months),
                    # offset the seed by step so each step's block-bootstrap
                    # draws an independent sample (still reproducible
                    # end-to-end via the base seed in base_inputs).
                    "seed": self.base_inputs.get("seed", 0) + t,
                }
                controller = RMPController(step_inputs)
                controller.build_price_scenarios()
                sol = controller.solve()

            results.append(
                {
                    "t": t,
                    "n_period": n_period,
                    "positions_before": current_positions.copy(),
                    "sol": sol,
                }
            )

            current_positions = self._update_positions(
                current_positions,
                sol["filtration"][(0, 1)],
                self.actual_prices.iloc[t],
                self.actual_prices.iloc[t + 1],
            )

        return results

    def _solve_deterministic_step(
        self, t: int, current_positions: pd.DataFrame
    ) -> dict:
        """
        Final-step solve: horizon is 1 and the current-period price is
        realized, so the bootstrap collapses to a degenerate single-scenario
        problem on the known price. Build a ForwardOptimizer directly with the
        realized price row as the only scenario.
        """
        all_tkrs = list(
            set(current_positions["tkr"]).union(set(self.base_inputs["model"]["tkr"]))
        )
        single_price = (
            self.actual_prices.iloc[[t]]
            .reindex(columns=all_tkrs)
            .reset_index(drop=True)
        )
        opt_inputs = {
            "positions": current_positions,
            "tax_rate": self.base_inputs["tax_rate"],
            "model": self.base_inputs["model"],
            "tkr_adev": self.base_inputs["tkr_adev"],
            "monthly_prices": [single_price],
        }
        optimizer = ForwardOptimizer(opt_inputs)
        optimizer.build()
        return optimizer.solve()

    @staticmethod
    def _update_positions(
        positions: pd.DataFrame,
        f0_sol: dict,
        current_prices: pd.Series,
        next_prices: pd.Series,
    ) -> pd.DataFrame:
        """
        Apply the first-period trades to positions and mark to market at
        next_prices.

        For each existing lot i, sell_shr_l[(i, 0)] shares are removed (j=0
        because all current holdings are starting lots in each fresh build).
        Remaining shares are revalued at next_prices; cost basis is scaled by
        the surviving fraction. Lots reduced to (near) zero are dropped.

        Each buy_shr_h[tkr] entry creates a new lot whose cost basis equals the
        purchase price (current_prices[tkr]), with amt marked at next_prices.

        The returned DataFrame is reset-indexed so its row indices match the
        lot indices ForwardOptimizer expects on the next build() call. It carries
        the columns the optimizer reads: tkr, amt, cost_basis_amt, shr,
        price, cost_basis_price.
        """
        sell_shr_l = f0_sol.get("sell_shr_l", {})
        buy_shr_h = f0_sol.get("buy_shr_h", {})

        rows = []

        # update existing lots after sells
        for i in range(len(positions)):
            row = positions.iloc[i]
            tkr = row["tkr"]
            old_shares = row["shr"]
            sold_shares = sell_shr_l.get((i, 0), 0.0)
            new_shares = max(0.0, old_shares - sold_shares)

            if new_shares <= 0:
                continue  # lot fully liquidated

            frac = new_shares / old_shares
            new_cost_basis_amt = row["cost_basis_amt"] * frac
            rows.append(
                {
                    "tkr": tkr,
                    "amt": new_shares * next_prices[tkr],
                    "cost_basis_amt": new_cost_basis_amt,
                    "shr": new_shares,
                    "price": next_prices[tkr],
                    "cost_basis_price": new_cost_basis_amt / new_shares,
                }
            )

        # add new lots from buys; cost basis = purchase price at current period
        for tkr, bought_shares in buy_shr_h.items():
            if bought_shares <= 0:
                continue
            cost_basis_amt = bought_shares * current_prices[tkr]
            rows.append(
                {
                    "tkr": tkr,
                    "amt": bought_shares * next_prices[tkr],
                    "cost_basis_amt": cost_basis_amt,
                    "shr": bought_shares,
                    "price": next_prices[tkr],
                    "cost_basis_price": current_prices[tkr],
                }
            )

        return pd.DataFrame(rows).reset_index(drop=True)
