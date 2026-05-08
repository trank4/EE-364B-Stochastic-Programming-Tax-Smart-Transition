import pandas as pd
from stochastic_optimizer.RMPController import RMPController


class Backtester:
    """
    Rolling backtester. At each period t along an actual realized price path,
    invokes an RMPController to compute trades for the upcoming horizon,
    executes only the first-period trades (receding-horizon principle), marks
    positions to market at the next period's actual price, and advances.

    The controller's scenarios are regenerated from the historical window at
    every step so the bootstrap reflects the most recent observed price.
    """

    def __init__(self, base_inputs: dict, actual_prices: pd.DataFrame):
        """
        Args:
            base_inputs: template inputs for RMPController. The "positions" and
                "start_date" entries are overridden each step.
            actual_prices: realized monthly prices for the backtest window with
                rows indexed by date and columns indexed by ticker. The loop
                advances over `len(actual_prices) - 1` steps, so the last row
                is only used for marking-to-market the final positions.
        """
        self.base_inputs = base_inputs
        self.actual_prices = actual_prices

    def run(self) -> list[dict]:
        """
        Loop over t = 0 .. len(actual_prices) - 2:
          1. Build a fresh RMPController with current positions and start_date
             advanced to actual_prices.index[t].
          2. Generate scenarios, solve once, get the multi-period plan.
          3. Apply only the first-period trades (sell_shr_l, buy_shr_h from
             filtration[0]) using actual prices at t and t+1.
          4. Advance positions.

        Returns a list of per-step dicts:
          - "t"                : time step index
          - "positions_before" : positions DataFrame entering this step
          - "sol"              : full optimizer solution dict from solve()
        """
        current_positions = self.base_inputs["positions"].copy()
        n_steps = len(self.actual_prices) - 1
        results = []

        for t in range(n_steps):
            step_inputs = {
                **self.base_inputs,
                "positions": current_positions,
                "start_date": self.actual_prices.index[t],
            }
            controller = RMPController(step_inputs)
            controller.build_price_scenarios()
            sol = controller.solve()

            results.append(
                {
                    "t": t,
                    "positions_before": current_positions.copy(),
                    "sol": sol,
                }
            )

            current_positions = self._update_positions(
                current_positions,
                sol["filtration"][0],
                self.actual_prices.iloc[t],
                self.actual_prices.iloc[t + 1],
            )

        return results

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
        lot indices StoxOptimizer expects on the next build() call. It carries
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

            if new_shares < 1e-8:
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
            if bought_shares < 1e-8:
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
