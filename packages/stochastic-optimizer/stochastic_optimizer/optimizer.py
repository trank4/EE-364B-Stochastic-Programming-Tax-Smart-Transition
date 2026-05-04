from collections import defaultdict

import gurobipy as gp
import numpy as np
from gurobipy import GRB


class StoxOptimizer:
    """
    Stochastic portfolio transition optimizer backed by Gurobi.

    Models a multi-period tax-smart transition from a starting portfolio to a
    target model portfolio. At each filtration period the optimizer can sell
    existing lots and buy new ones. The full problem is assembled by calling build() then solved by
    calling solve().
    """

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

        self.objectives = {}  # dict to contain objective variables and their priorities

        # deduce the upperbound on portfolio value over periods to deduce upper bounds for variables later
        # the heuristics is to assume the portfolio is growing at highest returns from prices
        monthly_growth = (
            (self.inputs["monthly_prices"] / self.inputs["monthly_prices"].shift(1))
            .dropna()
            .values
        )

        max_growth = np.max(monthly_growth, axis=1)

        starting_value = self.inputs["positions"]["amt"].sum()

        self.portfolio_ub = np.concatenate(
            ([starting_value], starting_value * np.cumprod(max_growth))
        )

        assert self.portfolio_ub.shape[0] == self.T

    def build(self) -> None:
        """
        Assemble the full Gurobi model by running each build stage in order:
        filtration variables, starting-lot anchors, wash-sale constraints, and
        lot dynamics constraints. Must be called before solve().
        """
        self.build_filtration()
        self.build_lot_holding_linking_constraints()
        self.build_starting_lot_constraints()
        self.build_wash_sales_constraints()
        self.build_lot_dynamics_constraints()

        # build hierarchical objectives for lexicographic optimization
        self.build_terminal_deviation_objective()
        self.build_tax_cost_objective()
        self.set_objective_hierarchy()

    def solve(self) -> None:
        """
        Hand the assembled model to the Gurobi solver and extract the solution.
        build() must be called first.
        """
        self.model.optimize()
        # get solutions

    def lot_ticker(self, i, j):
        """
        Returns the ticker string for lot (i, j). Starting lots (j == 0) map to
        the i-th position in the input portfolio; purchased lots (j > 0) map to
        the i-th asset in the model universe.
        """
        pos_tkrs = list(self.inputs["positions"]["tkr"])
        model_tkrs = list(self.inputs["model"]["tkr"])
        if j == 0:
            return pos_tkrs[i]
        else:
            return model_tkrs[i]

    def build_filtration(self) -> None:
        """
        Creates all Gurobi decision variables for every filtration period and
        stores them in self.filtration[f]. The lot set grows by N new lots each
        period (one per model asset), producing a triangular structure over time.
        Also computes and caches each lot's current market price and cost basis
        so constraint-building methods can read them directly from lot_info.
        """
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

            # set prices & cost basis
            all_tkrs = list(
                set(self.inputs["positions"]["tkr"]).union(
                    set(self.inputs["model"]["tkr"])
                )
            )
            self.filtration[f]["tkr_prices"] = {}
            for tkr in all_tkrs:
                # prices depends on the current filtration period
                self.filtration[f]["tkr_prices"][tkr] = self.inputs[
                    "monthly_prices"
                ].iloc[f][tkr]

            for (i, j), info in lot_info.items():
                if j == 0:
                    # For starting lot, the cost basis is input
                    cost_basis_amt = self.inputs["positions"].iloc[i]["cost_basis_amt"]
                    tkr_shares = (
                        self.inputs["positions"].iloc[i]["amt"]
                        / self.inputs["monthly_prices"].iloc[0][info["tkr"]]
                    )
                    cost_basis_price = cost_basis_amt / tkr_shares
                    info["cost_basis_price"] = cost_basis_price
                else:
                    # this block of code only executed for j >= 1 as at j = 0, there are only starting lots
                    # For other lots, cost basis depends on when the lots was added, so for lot (i,j) it's the price of previous period j - 1
                    info["cost_basis_price"] = self.inputs["monthly_prices"].iloc[
                        j - 1
                    ][info["tkr"]]

            portfolio_ub = self.portfolio_ub[f]
            # upper bound is based on heuristics putting the whole portfolio in 1 lot
            lot_shr_ub = {
                (i, j): np.ceil(
                    portfolio_ub
                    / self.filtration[f]["tkr_prices"][lot_info[i, j]["tkr"]]
                )
                for i, j in lot_indices
            }
            self.filtration[f]["lot_shr"] = self.model.addVars(
                lot_indices, lb=0.0, ub=lot_shr_ub, name=lot_names
            )

            # --- ticker holdings -> lot indices mapping ---
            tkr_to_lot_indices = defaultdict(list)
            for i, j in lot_indices:
                tkr_to_lot_indices[self.lot_ticker(i, j)].append((i, j))
            self.filtration[f]["tkr_to_lot_indices"] = dict(tkr_to_lot_indices)

            # set up holding shares variables
            shr_h_names = {
                tkr: f"shr_h({tkr}, f={f})" for tkr in tkr_to_lot_indices.keys()
            }
            shr_h_ub = {
                tkr: np.ceil(portfolio_ub / self.filtration[f]["tkr_prices"][tkr])
                for tkr in tkr_to_lot_indices.keys()
            }
            self.filtration[f]["shr_h"] = self.model.addVars(
                tkr_to_lot_indices.keys(), lb=0.0, ub=shr_h_ub, name=shr_h_names
            )
            # set up sell_shr_l
            # sell_shr_l has the same indices as lots, because we can decide selling these lots to reach next filtration
            sell_shr_l_names = {
                (i, j): f"sell_shr_l({info['tkr']},l={i},t={j},f={f})"
                for (i, j), info in lot_info.items()
            }
            sell_shr_l_ub = {
                (i, j): np.ceil(
                    portfolio_ub
                    / self.filtration[f]["tkr_prices"][lot_info[i, j]["tkr"]]
                )
                for i, j in lot_indices
            }
            self.filtration[f]["sell_shr_l"] = self.model.addVars(
                lot_indices, lb=0.0, ub=sell_shr_l_ub, name=sell_shr_l_names
            )

            # set up sell_shr_h
            all_tkrs_to_sell = list(tkr_to_lot_indices.keys())
            self.filtration[f]["all_tkrs_to_sell"] = all_tkrs_to_sell
            sell_shr_h_names = {
                tkr: f"sell_shr_h({tkr}, f={f})" for tkr in all_tkrs_to_sell
            }
            sell_shr_h_ub = {
                tkr: np.ceil(portfolio_ub / self.filtration[f]["tkr_prices"][tkr])
                for tkr in all_tkrs_to_sell
            }
            self.filtration[f]["sell_shr_h"] = self.model.addVars(
                all_tkrs_to_sell, lb=0.0, ub=sell_shr_h_ub, name=sell_shr_h_names
            )

            # set up buy_shr_h
            all_tkrs_to_buy = list(self.inputs["model"]["tkr"])
            self.filtration[f]["all_tkrs_to_buy"] = all_tkrs_to_buy
            buy_shr_h_names = {
                tkr: f"buy_shr_h({tkr}, f={f})" for tkr in all_tkrs_to_buy
            }
            buy_shr_h_ub = {
                tkr: np.ceil(portfolio_ub / self.filtration[f]["tkr_prices"][tkr])
                for tkr in all_tkrs_to_buy
            }
            self.filtration[f]["buy_shr_h"] = self.model.addVars(
                all_tkrs_to_buy, lb=0.0, ub=buy_shr_h_ub, name=buy_shr_h_names
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

            self.filtration[f]["lot_info"] = lot_info

            # update lot indices for next filtration
            if f < self.T - 1:  # dont update in last filtration
                lot_indices += [(i, f + 1) for i in range(self.n_asset)]
        self.model.update()
        self.n_lot = total_lot

    def build_lot_holding_linking_constraints(self):
        """
        Defines shr_h[tkr] as the sum of all lot shares belonging to that ticker.
        This makes shr_h the canonical per-ticker holding shares that downstream
        constraints (terminal deviation objective) can reference
        without having to sum over lots themselves.
        """
        for f, filtration in enumerate(self.filtration):
            tkr_to_lot_indices = filtration["tkr_to_lot_indices"]
            for tkr, lot_indices in tkr_to_lot_indices.items():
                self.model.addConstr(
                    filtration["shr_h"][tkr]
                    == gp.quicksum(filtration["lot_shr"][i, j] for i, j in lot_indices),
                    name=f"lot_shr_to_shr_h({tkr},f={f})",
                )

    def build_wash_sales_constraints(self):
        """
        Adds three groups of constraints for every filtration period:
        1. Aggregation — sell_shr_h[tkr] equals the sum of sell_shr_l across all
           lots of that ticker, linking lot-level and holding-level sell weights.
        2. Big-M linking — sell_shr_h and buy_shr_h are each upper-bounded by 100
           times their binary indicator, so a shares can only be nonzero when the
           corresponding indicator is 1.
        3. Wash-sale prevention — for tickers that appear in both the sell and buy
           universes, buy_h + sell_h <= 1 prevents simultaneous buy and sell.
        """
        for f, filtration in enumerate(self.filtration):
            tkr_to_lot_indices = filtration["tkr_to_lot_indices"]

            for tkr in self.filtration[f]["all_tkrs_to_sell"]:
                lot_indices_for_tkr = tkr_to_lot_indices[tkr]
                # constraint to map sell_shr_l with sell_shr_h
                self.model.addConstr(
                    filtration["sell_shr_h"][tkr]
                    == gp.quicksum(
                        filtration["sell_shr_l"][i, j] for i, j in lot_indices_for_tkr
                    ),
                    name=f"sell_l_to_h_mapping({tkr},f={f})",
                )

                self.model.addConstr(
                    filtration["sell_shr_h"][tkr]
                    <= filtration["sell_h"][tkr] * filtration["sell_shr_h"][tkr].UB,
                    name=f"upper_sell_binary({tkr}, f={f})",
                )

            for tkr in self.filtration[f]["all_tkrs_to_buy"]:
                self.model.addConstr(
                    filtration["buy_shr_h"][tkr]
                    <= filtration["buy_h"][tkr] * filtration["buy_shr_h"][tkr].UB,
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
                    name=f"buy_sell_exclusivity({tkr}, f={f})",
                )

            # buy and sell in same filtration has to equal $ amount
            sell_amount = gp.quicksum(
                filtration["sell_shr_h"][tkr] * filtration["tkr_prices"][tkr]
                for tkr in filtration["sell_shr_h"]
            )
            buy_amount = gp.quicksum(
                filtration["buy_shr_h"][tkr] * filtration["tkr_prices"][tkr]
                for tkr in filtration["buy_shr_h"]
            )
            self.model.addConstr(
                sell_amount == buy_amount,
                name=f"self-financing-constr(f={f})",
            )
        self.model.update()

    def build_lot_dynamics_constraints(self):
        """
        Links the portfolio state across consecutive filtration periods.
        For lots that already exist in the current filtration, the holding in
        the next period equals the current holding minus whatever was sold.
        For lots that are new in the next filtration (just purchased), their
        initial holding is set equal to the buy shares from the current period.
        """
        for f in range(len(self.filtration) - 1):
            current_filtration = self.filtration[f]
            next_filtration = self.filtration[f + 1]
            # for existing lots in current_filtration, the dynamics to next filtration depends on sells
            for i, j in current_filtration["lot_info"].keys():
                lot_name = current_filtration["lot_info"][i, j]["tkr"]
                # constraint to link the current lot, sell_shr_l and the associated lot in next filtration
                self.model.addConstr(
                    current_filtration["lot_shr"][i, j]
                    - current_filtration["sell_shr_l"][i, j]
                    == next_filtration["lot_shr"][i, j],
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
                assert current_filtration["buy_shr_h"][lot_name] is not None
                # constraint to link the buy_wt_h of the current filtration to new lots in next filtration
                self.model.addConstr(
                    next_filtration["lot_shr"][i, j]
                    == current_filtration["buy_shr_h"][lot_name],
                    name=f"buy_lot_dynamics({lot_name},l={i},t={j}, from f={f} to f={f+1})",
                )

        self.model.update()

    def build_starting_lot_constraints(self):
        """
        Anchors the lot shares at filtration 0 to the actual portfolio shares
        from the input positions. Without this, the solver would be free to set
        the initial holdings to any value.
        """
        starting_filtration = self.filtration[0]
        for i, j in starting_filtration["lot_info"].keys():
            assert (
                self.inputs["positions"].iloc[i]["tkr"]
                == starting_filtration["lot_info"][i, j]["tkr"]
            )
            lot_tkr = starting_filtration["lot_info"][i, j]["tkr"]
            lot_price = starting_filtration["tkr_prices"][lot_tkr]
            self.model.addConstr(
                starting_filtration["lot_shr"][i, j]
                == (self.inputs["positions"].iloc[i]["amt"] / lot_price),
                name=f"start_lot_shr({i},t={j})",
            )

    def build_terminal_deviation_objective(self):
        last_filtration = self.filtration[-1]
        final_portfolio_amount = gp.quicksum(
            last_filtration["shr_h"][tkr] * last_filtration["tkr_prices"][tkr]
            for tkr in last_filtration["shr_h"]
        )
        # create variables for terminal deviation
        model_tkrs = list(self.inputs["model"]["tkr"])
        terminal_dev_names = {tkr: f"terminal_dev({tkr})" for tkr in model_tkrs}
        terminal_dev = self.model.addVars(model_tkrs, lb=0, name=terminal_dev_names)

        # set up terminal deviation objective variable
        total_terminal_dev = self.model.addVar(name="total_terminal_dev_objective")
        self.objectives["total_terminal_dev"] = [total_terminal_dev, 1]
        self.model.addConstr(
            total_terminal_dev == gp.quicksum(terminal_dev[tkr] for tkr in model_tkrs),
            name="total_terminal_dev_objective",
        )

        for tkr in terminal_dev_names.keys():
            tkr_tgt_wt = (
                self.inputs["model"]
                .loc[self.inputs["model"]["tkr"] == tkr, "tgt_wt"]
                .item()
            )
            tkr_amount = (
                last_filtration["shr_h"][tkr] * last_filtration["tkr_prices"][tkr]
            )
            self.model.addConstr(
                tkr_amount - tkr_tgt_wt * final_portfolio_amount <= terminal_dev[tkr],
                name=f"upper_bound_terminal_dev({tkr})",
            )
            self.model.addConstr(
                tkr_tgt_wt * final_portfolio_amount - tkr_amount <= terminal_dev[tkr],
                name=f"lower_bound_terminal_dev({tkr})",
            )

    def build_tax_cost_objective(self):
        """we want to minimize the expected tax cost realized from all sells across all filtrations and scenarios.
        We assume that, the optimizer will take the same action across all scenarios.
        This assumption simplifies and make the optimization problem more tractable as it is equivalent to using the same variables across all scenarios.
        Different prices in scenarios will only affect the tax cost objective, and no where else.
        And if we assume same likelihood for every scenarios, minimizing expected tax cost across scenario is equivalent to minimizing sum of tax cost across scenarios.
        Assume tax cost is calculated as follow:
        tax of selling from a lot = $_lot_sell_amount - ($_lot_sell_amount / price) * cost_basis_price
                                  = $_lot_sell_amount * (1 - cost_basis_price/price)
        -> (tax / total_aum) = ($_lot_sell_amount / total_aum) * (1 - cost_basis_price/price)
                             = sell_wt_l * (1 - cost_basis_price/price)
        """
        # create total tax cost objective variable (across all filtrations and all scenarios)
        # the lower bound is -inf to minimize tax cost as much as possible
        total_tax_cost = self.model.addVar(lb=-GRB.INFINITY, name="total_tax_cost")
        self.objectives["total_tax_cost"] = [total_tax_cost, 0]
        # create list to hold all lot tax cost
        lot_tax_cost_list = []
        for f in range(len(self.filtration) - 1):
            filtration = self.filtration[f]
            for i, j in filtration["lot_info"].keys():
                lot_tkr = filtration["lot_info"][i, j]["tkr"]
                lot_price = filtration["tkr_prices"][lot_tkr]
                lot_cost_basis_price = filtration["lot_info"][i, j]["cost_basis_price"]
                # no multiplying for tax rate here because minimization tax cost over 1 tax rate is equivalent to minimization of realized gain/loss
                lot_tax_cost = (
                    self.inputs["tax_rate"]
                    * filtration["sell_shr_l"][i, j]
                    * (lot_price - lot_cost_basis_price)
                )
                lot_tax_cost_list.append(lot_tax_cost)
        self.model.addConstr(
            gp.quicksum(lot_tax_cost_list) <= total_tax_cost,
            name="total_tax_cost_objective",
        )

    def set_objective_hierarchy(self):
        for index, (name, (var, priority)) in enumerate(self.objectives.items()):
            self.model.setObjectiveN(var, index=index, priority=priority, name=name)
        self.model.ModelSense = GRB.MINIMIZE
