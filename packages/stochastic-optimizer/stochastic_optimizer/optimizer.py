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
        self.build_starting_lot_constraints()
        self.build_wash_sales_constraints()
        self.build_lot_dynamics_constraints()

    def solve(self) -> None:
        """Invoke the Gurobi solver and extract results."""
        self.model.optimize()
        # get solutions

    def lot_ticker(self, i, j):
        """helper function to get ticker name from lot indices"""
        pos_tkrs = list(self.inputs["positions"]["tkr"])
        model_tkrs = list(self.inputs["model"]["tkr"])
        if j == 0:
            return pos_tkrs[i]
        else:
            return model_tkrs[i]

    def build_filtration(self) -> None:
        """Build filtration variables."""
        total_lot = 0
        # set up lot variables
        # first period only include starting lots
        # after each filtration, there is a new column with n_asset new lots that we can sell from
        # So the total number of lots grows linearly with filtration = start lots + f * n_asset
        lot_indices = [(i, 0) for i in range(self.n_start_pos)]
        for f in range(self.T):
            assert len(lot_indices) == self.n_start_pos + f * self.n_asset
            total_lot += len(lot_indices)
            lot_info = {
                (i, j): {} for i, j in lot_indices
            }  # truth source for all info lot-level
            # set up the ticker names associated with each lot
            lot_info = {(i, j): {"tkr": self.lot_ticker(i, j)} for (i, j) in lot_info}
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
                tkr_to_lot_indices[self.lot_ticker(i, j)].append((i, j))
            self.filtration[f]["tkr_to_lot_indices"] = dict(tkr_to_lot_indices)

            # set up sell_wt_h
            all_tkrs_to_sell = list(tkr_to_lot_indices.keys())
            self.filtration[f]["all_tkrs_to_sell"] = all_tkrs_to_sell
            sell_wt_h_names = {
                tkr: f"sell_wt_h({tkr}, f={f})" for tkr in all_tkrs_to_sell
            }
            self.filtration[f]["sell_wt_h"] = self.model.addVars(
                all_tkrs_to_sell, lb=0.0, ub=100, name=sell_wt_h_names
            )

            # set up buy_wt_h
            all_tkrs_to_buy = list(self.inputs["model"]["tkr"])
            self.filtration[f]["all_tkrs_to_buy"] = all_tkrs_to_buy
            buy_wt_h_names = {tkr: f"buy_wt_h({tkr}, f={f})" for tkr in all_tkrs_to_buy}
            self.filtration[f]["buy_wt_h"] = self.model.addVars(
                all_tkrs_to_buy, lb=0.0, ub=100, name=buy_wt_h_names
            )

            # set up buy_h
            buy_h_names = {tkr: f"buy_h({tkr}, f={f})" for tkr in all_tkrs_to_buy}
            self.filtration[f]["buy_h"] = self.model.addVars(
                all_tkrs_to_buy, vtype=GRB.BINARY, name=buy_h_names
            )
            # set up sell_h
            sell_h_names = {tkr: f"sell_h({tkr}, f={f})" for tkr in all_tkrs_to_sell}
            self.filtration[f]["sell_h"] = self.model.addVars(
                all_tkrs_to_sell, vtype=GRB.BINARY, name=sell_h_names
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

            # update lot indices for next filtration
            if f < self.T - 1:  # dont update in last filtration
                lot_indices += [(i, f + 1) for i in range(self.n_asset)]
        self.model.update()
        self.n_lot = total_lot

    def build_wash_sales_constraints(self):
        for f, filtration in enumerate(self.filtration):
            tkr_to_lot_indices = filtration["tkr_to_lot_indices"]

            for tkr in self.filtration[f]["all_tkrs_to_sell"]:
                lot_indices_for_tkr = tkr_to_lot_indices[tkr]
                # constraint to map sell_wt_l with sell_wt_h
                self.model.addConstr(
                    filtration["sell_wt_h"][tkr]
                    == gp.quicksum(
                        filtration["sell_wt_l"][i, j] for i, j in lot_indices_for_tkr
                    ),
                    name=f"sell_l_to_h_mapping({tkr},f={f})",
                )
                # constraint that sell_wt_h is upperbound by sell_h * 100
                self.model.addConstr(
                    filtration["sell_wt_h"][tkr] <= filtration["sell_h"][tkr] * 100,
                    name=f"upper_sell_binary({tkr}, f={f})",
                )

            for tkr in self.filtration[f]["all_tkrs_to_buy"]:
                # constraint that buy_wt_h is upperbound by buy_h * 100
                self.model.addConstr(
                    filtration["buy_wt_h"][tkr] <= filtration["buy_h"][tkr] * 100,
                    name=f"upper_buy_binary({tkr}, f={f})",
                )

            tkrs_both_buy_and_sell = [
                tkr
                for tkr in self.filtration[f]["all_tkrs_to_sell"]
                if tkr in self.filtration[f]["all_tkrs_to_buy"]
            ]
            for tkr in tkrs_both_buy_and_sell:
                # constraint that only one of buy_h and sell_h can be one
                self.model.addConstr(
                    filtration["buy_h"][tkr] + filtration["sell_h"][tkr] <= 1.0,
                    name=f"upper_buy_binary({tkr}, f={f})",
                )
        self.model.update()

    def build_lot_dynamics_constraints(self):
        for f in range(len(self.filtration) - 1):
            current_filtration = self.filtration[f]
            next_filtration = self.filtration[f + 1]
            # for existing lots in current_filtration, the dynamics to next filtration depends on sells
            for i, j in current_filtration["lot_info"].keys():
                lot_name = current_filtration["lot_info"][i, j]["tkr"]
                # constraint to link the current lot, sell_wt_l and the associated lot in next filtration
                self.model.addConstr(
                    current_filtration["lot"][i, j]
                    - current_filtration["sell_wt_l"][i, j]
                    == next_filtration["lot"][i, j],
                    name=f"sell_lot_dynamics({lot_name},l={i},t={j},from f={f} to f={f+1})",
                )

            # for new lots that exist only in next filtration, the dynamics depends on buys
            new_lot_indices = (
                next_filtration["lot_info"].keys()
                - current_filtration["lot_info"].keys()
            )
            for i, j in new_lot_indices:
                lot_name = next_filtration["lot_info"][i, j]["tkr"]
                # make sure we get the correct buy variables
                assert current_filtration["buy_wt_h"][lot_name] is not None
                # constraint to link the buy_wt_h of the current filtration to new lots in next filtration
                self.model.addConstr(
                    next_filtration["lot"][i, j]
                    == current_filtration["buy_wt_h"][lot_name],
                    name=f"buy_lot_dynamics({lot_name},l={i},t={j}, from f={f} to f={f+1})",
                )

        self.model.update()

    def build_starting_lot_constraints(self):
        starting_filtration = self.filtration[0]
        for i, j in starting_filtration["lot_info"].keys():
            assert (
                self.inputs["positions"].iloc[i]["tkr"]
                == starting_filtration["lot_info"][i, j]["tkr"]
            )
            self.model.addConstr(
                starting_filtration["lot"][i, j]
                == self.inputs["positions"].iloc[i]["wt"],
                name=f"start_lot_wt({i},t={j})",
            )
