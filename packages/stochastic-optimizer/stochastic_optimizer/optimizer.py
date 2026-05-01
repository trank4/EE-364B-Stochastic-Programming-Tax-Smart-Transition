from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB


class StoxOptimizer:
    """Stochastic portfolio transition optimizer backed by Gurobi."""

    def __init__(self, inputs) -> None:
        self.model = gp.Model("stochastic_optimizer")
        self.inputs = inputs

        # infer number of period T from inputs
        self.T = self.inputs["monthly_prices"].shape[0]

        # infer number of assets in the universe
        self.n_asset = self.inputs["model"].shape[0]

        # infer number of starting positions
        self.n_start_pos = self.inputs["positions"].shape[0]
        # create filtration
        self.filtration = [{} for _ in range(self.T)]

        self.n_lot = None  # total lots in optimization, populated later

    def build(self) -> None:
        """Construct variables, constraints, and objective in self.model."""
        self.build_filtration()
        self.build_wash_sales_constraints()
        self.build_lot_dynamics_constraints()

    def solve(self) -> None:
        """Invoke the Gurobi solver and extract results."""
        self.model.optimize()
        # get solutions

    def lot_ticker(self, i):
        """helper function to get ticker name from lot indices"""
        pos_tkrs = list(self.inputs["positions"]["tkr"])
        model_tkrs = list(self.inputs["model"]["tkr"])
        if i < self.n_start_pos:
            return pos_tkrs[i]
        else:
            block_idx = (i - self.n_start_pos) % self.n_asset
            return model_tkrs[block_idx]

    def build_filtration(self) -> None:
        """Build filtration variables."""
        total_lot = 0
        for f in range(self.T):
            # set up lot variables
            # the lot structure is triangular with ragged columns
            # we start with lots existing in the starting positions
            # after each filtration, there is a new column with n_asset new lots that we can sell from
            lot_indices = [
                (i, j)
                for j in range(f + 1)
                for i in range(self.n_start_pos + j * self.n_asset)
            ]
            total_lot += len(lot_indices)
            lot_info = {
                (i, j): {} for i, j in lot_indices
            }  # truth source for all info lot-level
            # set up the ticker names associated with each lot
            lot_info = {(i, j): {"tkr": self.lot_ticker(i)} for (i, j) in lot_info}
            lot_names = {
                (i, j): f"lot({info['tkr']},l={i},t={j},f={f})"
                for (i, j), info in lot_info.items()
            }
            self.filtration[f]["lot"] = self.model.addVars(
                lot_indices, lb=0.0, ub=100, name=lot_names
            )

            # set up sell_wt_l
            # sell_wt has the same indices as lots, because we can decide selling these lots to reach next filtration
            sell_wt_l_names = {
                (i, j): f"sell_wt({info['tkr']},l={i},t={j},f={f})"
                for (i, j), info in lot_info.items()
            }
            self.filtration[f]["sell_wt_l"] = self.model.addVars(
                lot_indices, lb=0.0, ub=100, name=sell_wt_l_names
            )

            # --- ticker holdings -> lot indices mapping ---
            tkr_to_lot_indices = defaultdict(list)
            for i, j in lot_indices:
                tkr_to_lot_indices[self.lot_ticker(i)].append((i, j))
            self.filtration[f]["tkr_to_lot_indices"] = dict(tkr_to_lot_indices)

            # set up sell_wt_h
            all_tkrs = list(tkr_to_lot_indices.keys())
            sell_wt_h_names = {
                i: f"sell_wt_h({tkr}, f={f})" for i, tkr in enumerate(all_tkrs)
            }
            self.filtration[f]["sell_wt_h"] = self.model.addVars(
                len(all_tkrs), lb=0.0, ub=100, name=sell_wt_h_names
            )

            # set up buy_wt_h
            buy_wt_h_names = {
                i: f"buy_wt_h({tkr}, f={f})" for i, tkr in enumerate(all_tkrs)
            }
            self.filtration[f]["buy_wt_h"] = self.model.addVars(
                len(all_tkrs), lb=0.0, ub=100, name=buy_wt_h_names
            )

            # set up buy_h
            buy_h_names = {i: f"buy_h({tkr}, f={f})" for i, tkr in enumerate(all_tkrs)}
            self.filtration[f]["buy_h"] = self.model.addVars(
                len(all_tkrs), vtype=GRB.BINARY, name=buy_h_names
            )
            # set up sell_h
            sell_h_names = {
                i: f"sell_h({tkr}, f={f})" for i, tkr in enumerate(all_tkrs)
            }
            self.filtration[f]["sell_h"] = self.model.addVars(
                len(all_tkrs), vtype=GRB.BINARY, name=sell_h_names
            )

            # set prices & cost basis
            for (i, j), info in lot_info.items():
                # prices depends on the current filtration period
                info["price"] = self.inputs["monthly_prices"].iloc[f][info["tkr"]]
                if i < self.n_start_pos:
                    # For starting lot, the cost basis is input
                    info["cost_basis"] = self.inputs["positions"].iloc[i][
                        "cost_basis_amt"
                    ]
                else:
                    # this block of code only executed for j >= 1 as at j = 0, there are only starting lots
                    # For other lots, cost basis depends on when the lots was added, so for lot (i,j) it's the price of previous period j - 1
                    info["cost_basis"] = self.inputs["monthly_prices"].iloc[j - 1][
                        info["tkr"]
                    ]
            self.filtration[f]["lot_info"] = lot_info
        self.model.update()
        self.n_lot = total_lot

    def build_wash_sales_constraints(self):
        for f, filtration in enumerate(self.filtration):
            tkr_to_lot_indices = filtration["tkr_to_lot_indices"]
            all_tkrs = list(tkr_to_lot_indices.keys())

            for k, tkr in enumerate(all_tkrs):
                lot_indices_for_tkr = tkr_to_lot_indices[tkr]
                # constraint to map sell_wt_l with sell_wt_h
                self.model.addConstr(
                    filtration["sell_wt_h"][k]
                    == gp.quicksum(
                        filtration["sell_wt_l"][i, j] for i, j in lot_indices_for_tkr
                    ),
                    name=f"sell_l_to_h_mapping({tkr},f={f})",
                )
                # constraint that sell_wt_h is upperbound by sell_h * 100
                self.model.addConstr(
                    filtration["sell_wt_h"][k] <= filtration["sell_h"][k] * 100,
                    name=f"upper_sell_binary({tkr}, f={f})",
                )
                # constraint that buy_wt_h is upperbound by buy_h * 100
                self.model.addConstr(
                    filtration["buy_wt_h"][k] <= filtration["buy_h"][k] * 100,
                    name=f"upper_buy_binary({tkr}, f={f})",
                )
                # constraint that only one of buy_h and sell_h can be one
                self.model.addConstr(
                    filtration["buy_h"][k] + filtration["sell_h"][k] <= 1.0,
                    name=f"upper_buy_binary({tkr}, f={f})",
                )
        self.model.update()

    def build_lot_dynamics_constraints(self):
        pass
