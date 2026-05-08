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
        # monthly_prices can be list of prices for multiple scenarios, but they should have same number of periods
        self.T = self.inputs["monthly_prices"][0].shape[0]
        self.n_scenario = len(self.inputs["monthly_prices"])
        # infer number of assets in the universe
        self.n_asset = self.inputs["model"].shape[0]

        # infer number of starting positions
        self.n_start_pos = self.inputs["positions"].shape[0]
        # create filtration
        self.filtration = {
            (s, f): {} for f in range(self.T) for s in range(self.n_scenario)
        }

        self.n_lot = None  # total lots in optimization, populated later

        self.objectives = {}  # dict to contain objective variables and their priorities

        # deduce the upperbound on portfolio value over periods to deduce upper bounds for variables later
        # the heuristics is to assume the portfolio is growing at highest returns from prices
        # portfolio_ub shape: (n_scenario, T) — row s is the per-period UB for scenario s
        starting_value = self.inputs["positions"]["amt"].sum()

        ub_rows = []
        for prices in self.inputs["monthly_prices"]:
            monthly_growth = (prices / prices.shift(1)).dropna().values
            max_growth = np.max(monthly_growth, axis=1)
            ub_rows.append(
                np.concatenate(
                    ([starting_value], starting_value * np.cumprod(max_growth))
                )
            )

        self.portfolio_ub = np.stack(ub_rows)  # (n_scenario, T)

        assert self.portfolio_ub.shape == (self.n_scenario, self.T)

    def build(self) -> None:
        """
        Assemble the full Gurobi model by running each build stage in order:
        filtration variables, lot/holding linking, starting-lot anchors,
        wash-sale + self-financing constraints, and lot dynamics constraints,
        followed by the two hierarchical objectives (terminal deviation, tax
        cost) registered for lexicographic minimization. Must be called before
        solve().
        """
        self.build_filtration()
        self.build_lot_holding_linking_constraints()
        self.build_starting_lot_constraints()
        self.build_wash_sales_constraints()
        self.build_lot_dynamics_constraints()

        # build hierarchical objectives for lexicographic optimization
        self.build_terminal_deviation_objective()
        self.build_transitory_deviation_objective()
        self.build_tax_cost_objective()
        self.set_objective_hierarchy()

    def solve(self) -> dict:
        """
        Hand the assembled model to the Gurobi solver and extract the solution.
        build() must be called first.

        Returns a dict with:
        - "filtration": list of per-filtration dicts, one per period, each containing
          the optimal values (.X) of all decision variables at that filtration:
            lot_shr      — {(i,j): float}  shares held per lot
            sell_shr_l   — {(i,j): float}  shares sold per lot
            shr_h        — {tkr: float}    total shares held per ticker
            sell_shr_h   — {tkr: float}    total shares sold per ticker
            buy_shr_h    — {tkr: float}    total shares bought per ticker
            sell_h       — {tkr: float}    sell binary indicator per ticker
            buy_h        — {tkr: float}    buy binary indicator per ticker
            tax_cost     — float           realized tax cost (filtrations 0..T-2 only)
            trans_dev    — {tkr: float}    out-of-band deviation (filtrations 1..T-2 only)
        - one key per objective name (e.g. "total_terminal_dev", "total_trans_dev",
          "total_tax_cost") mapping to its optimal scalar value.
        """
        self.model.optimize()
        sol = {}

        sol["filtration"] = []
        for f, filtration in enumerate(self.filtration):
            f_sol = {
                "lot_shr": {k: v.X for k, v in filtration["lot_shr"].items()},
                "shr_h": {k: v.X for k, v in filtration["shr_h"].items()},
            }
            # sell/buy variables are absent at the terminal filtration
            if f < self.T - 1:
                f_sol["sell_shr_l"] = {
                    k: v.X for k, v in filtration["sell_shr_l"].items()
                }
                f_sol["sell_shr_h"] = {
                    k: v.X for k, v in filtration["sell_shr_h"].items()
                }
                f_sol["buy_shr_h"] = {
                    k: v.X for k, v in filtration["buy_shr_h"].items()
                }
                f_sol["sell_h"] = {k: v.X for k, v in filtration["sell_h"].items()}
                f_sol["buy_h"] = {k: v.X for k, v in filtration["buy_h"].items()}
            if "tax_cost" in filtration:
                f_sol["tax_cost"] = filtration["tax_cost"].X
            if "trans_dev" in filtration:
                f_sol["trans_dev"] = {
                    k: v.X for k, v in filtration["trans_dev"].items()
                }
            sol["filtration"].append(f_sol)

        for name, (var, _priority) in self.objectives.items():
            sol[name] = var.X

        return sol

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
        Creates all Gurobi decision variables for every (filtration, scenario)
        pair and stores them in self.filtration[(s, f)]. The lot index set grows
        by N new lots each period (one per model asset) and is identical across
        scenarios — only the variable values, prices, and bounds differ per
        scenario. Also computes and caches each lot's market price and cost
        basis (per scenario) so downstream constraint-building methods can read
        them directly from lot_info.
        """
        all_tkrs = list(
            set(self.inputs["positions"]["tkr"]).union(set(self.inputs["model"]["tkr"]))
        )
        all_tkrs_to_buy_const = list(self.inputs["model"]["tkr"])

        total_lot = 0
        for s in range(self.n_scenario):
            scenario_prices = self.inputs["monthly_prices"][s]
            # first period only include starting lots; after each filtration there
            # is a new column with n_asset new lots that we can sell from. So the
            # total number of lots grows linearly: start lots + f * n_asset
            lot_indices = [(i, 0) for i in range(self.n_start_pos)]
            for f in range(self.T):
                assert len(lot_indices) == self.n_start_pos + f * self.n_asset
                total_lot += len(lot_indices)

                # ticker name + cost-basis info per lot (scenario-specific)
                lot_info = {
                    (i, j): {"tkr": self.lot_ticker(i, j)} for (i, j) in lot_indices
                }
                lot_names = {
                    (i, j): f"lot({info['tkr']},l={i},t={j},f={f},s={s})"
                    for (i, j), info in lot_info.items()
                }

                # cache per-ticker prices for this (s, f)
                self.filtration[(s, f)]["tkr_prices"] = {
                    tkr: scenario_prices.iloc[f][tkr] for tkr in all_tkrs
                }

                for (i, j), info in lot_info.items():
                    if j == 0:
                        assert self.inputs["positions"].iloc[i]["tkr"] == info["tkr"]
                        # starting lot: cost basis from input
                        info["cost_basis_price"] = self.inputs["positions"].iloc[i][
                            "cost_basis_price"
                        ]
                    else:
                        # purchased lot: cost basis = price at the period it was bought
                        info["cost_basis_price"] = scenario_prices.iloc[j - 1][
                            info["tkr"]
                        ]

                # heuristic UB: putting the whole portfolio (at scenario s, period f) in 1 lot
                portfolio_ub = self.portfolio_ub[s, f]
                lot_shr_ub = {
                    (i, j): float(
                        np.ceil(
                            portfolio_ub
                            / self.filtration[(s, f)]["tkr_prices"][
                                lot_info[i, j]["tkr"]
                            ]
                        )
                    )
                    for i, j in lot_indices
                }
                self.filtration[s, f]["lot_shr"] = self.model.addVars(
                    lot_indices, lb=0.0, ub=lot_shr_ub, name=lot_names
                )

                # ticker -> lot indices mapping (same across scenarios but stored per-key for ergonomics)
                tkr_to_lot_indices = defaultdict(list)
                for i, j in lot_indices:
                    tkr_to_lot_indices[self.lot_ticker(i, j)].append((i, j))
                self.filtration[s, f]["tkr_to_lot_indices"] = dict(tkr_to_lot_indices)

                shr_h_names = {
                    tkr: f"shr_h({tkr},f={f},s={s})" for tkr in tkr_to_lot_indices
                }
                shr_h_ub = {
                    tkr: np.ceil(
                        portfolio_ub / self.filtration[(s, f)]["tkr_prices"][tkr]
                    )
                    for tkr in tkr_to_lot_indices
                }
                self.filtration[(s, f)]["shr_h"] = self.model.addVars(
                    tkr_to_lot_indices.keys(),
                    lb=0.0,
                    ub=shr_h_ub,
                    name=shr_h_names,
                )

                # sell/buy variables only exist for non-terminal filtrations: selling at T-1
                # has no effect on shr_h[T-1] (used by terminal deviation) and is never
                # linked forward by lot dynamics, so these variables would be unconstrained.
                if f < self.T - 1:
                    sell_shr_l_names = {
                        (i, j): f"sell_shr_l({info['tkr']},l={i},t={j},f={f},s={s})"
                        for (i, j), info in lot_info.items()
                    }
                    sell_shr_l_ub = {
                        (i, j): float(
                            np.ceil(
                                portfolio_ub
                                / self.filtration[(s, f)]["tkr_prices"][
                                    lot_info[i, j]["tkr"]
                                ]
                            )
                        )
                        for i, j in lot_indices
                    }
                    self.filtration[(s, f)]["sell_shr_l"] = self.model.addVars(
                        lot_indices,
                        lb=0.0,
                        ub=sell_shr_l_ub,
                        name=sell_shr_l_names,
                    )

                    all_tkrs_to_sell = list(tkr_to_lot_indices.keys())
                    self.filtration[(s, f)]["all_tkrs_to_sell"] = all_tkrs_to_sell
                    sell_shr_h_names = {
                        tkr: f"sell_shr_h({tkr},f={f},s={s})"
                        for tkr in all_tkrs_to_sell
                    }
                    sell_shr_h_ub = {
                        tkr: np.ceil(
                            portfolio_ub / self.filtration[(s, f)]["tkr_prices"][tkr]
                        )
                        for tkr in all_tkrs_to_sell
                    }
                    self.filtration[(s, f)]["sell_shr_h"] = self.model.addVars(
                        all_tkrs_to_sell,
                        lb=0.0,
                        ub=sell_shr_h_ub,
                        name=sell_shr_h_names,
                    )

                    self.filtration[(s, f)]["all_tkrs_to_buy"] = all_tkrs_to_buy_const
                    buy_shr_h_names = {
                        tkr: f"buy_shr_h({tkr},f={f},s={s})"
                        for tkr in all_tkrs_to_buy_const
                    }
                    buy_shr_h_ub = {
                        tkr: np.ceil(
                            portfolio_ub / self.filtration[(s, f)]["tkr_prices"][tkr]
                        )
                        for tkr in all_tkrs_to_buy_const
                    }
                    self.filtration[(s, f)]["buy_shr_h"] = self.model.addVars(
                        all_tkrs_to_buy_const,
                        lb=0.0,
                        ub=buy_shr_h_ub,
                        name=buy_shr_h_names,
                    )

                    buy_h_names = {
                        tkr: f"buy_h({tkr},f={f},s={s})"
                        for tkr in all_tkrs_to_buy_const
                    }
                    self.filtration[(s, f)]["buy_h"] = self.model.addVars(
                        all_tkrs_to_buy_const, vtype=GRB.BINARY, name=buy_h_names
                    )

                    sell_h_names = {
                        tkr: f"sell_h({tkr},f={f},s={s})" for tkr in all_tkrs_to_sell
                    }
                    self.filtration[(s, f)]["sell_h"] = self.model.addVars(
                        all_tkrs_to_sell, vtype=GRB.BINARY, name=sell_h_names
                    )

                self.filtration[(s, f)]["lot_info"] = lot_info

                # update lot indices for next filtration (same growth across scenarios)
                if f < self.T - 1:
                    lot_indices = lot_indices + [
                        (i, f + 1) for i in range(self.n_asset)
                    ]

        self.model.update()
        self.n_lot = total_lot

    def build_lot_holding_linking_constraints(self):
        """
        Defines shr_h[tkr] as the sum of all lot shares belonging to that ticker,
        per (scenario, filtration). This makes shr_h the canonical per-ticker
        holding shares that downstream constraints (terminal deviation objective)
        can reference without having to sum over lots themselves.
        """
        for (s, f), filtration in self.filtration.items():
            tkr_to_lot_indices = filtration["tkr_to_lot_indices"]
            for tkr, lot_indices in tkr_to_lot_indices.items():
                self.model.addConstr(
                    filtration["shr_h"][tkr]
                    == gp.quicksum(filtration["lot_shr"][i, j] for i, j in lot_indices),
                    name=f"lot_shr_to_shr_h({tkr},f={f},s={s})",
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
        for f, filtration in enumerate(self.filtration[:-1]):
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
            self.model.addConstr(
                starting_filtration["lot_shr"][i, j]
                == (self.inputs["positions"].iloc[i]["shr"]),
                name=f"start_lot_shr({i},t={j})",
            )

    def build_terminal_deviation_objective(self):
        """
        Minimize the total absolute dollar deviation from target weights at
        the final filtration. For each model ticker k, an auxiliary
        terminal_dev[k] >= 0 bounds |amount_k - tgt_wt_k * V_T|, encoded by
        two linear inequalities. The aggregate sum is registered as the
        priority-1 objective in the lex hierarchy.
        """
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
        self.objectives["total_terminal_dev"] = [total_terminal_dev, 2]
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
        """
        Minimize total realized gain/loss across all sells and filtrations.

        For each lot (i, j) at filtration f, the per-share gain is
            price[f] - cost_basis_price[i, j]
        and the realized P&L from selling sell_shr_l[i, j] shares is
            sell_shr_l[i, j] * (price[f] - cost_basis_price[i, j]).

        The flat tax rate tau is omitted: with a single rate, scaling the
        objective by tau doesn't change the argmin. Negative contributions
        (losses) are kept, so the optimizer can harvest them — the objective
        variable has lb = -inf accordingly.

        When multi-scenario support is added, prices and the inner sum will
        gain a scenario index s; the outer expectation E_s[tax(s)] reduces to
        a sum if scenarios are equally likely, and to a probability-weighted
        sum otherwise (via inputs["scenario_prob"]).
        """
        # create total tax cost objective variable (across all filtrations and all scenarios)
        # the lower bound is -inf to minimize tax cost as much as possible
        total_tax_cost = self.model.addVar(lb=-GRB.INFINITY, name="total_tax_cost")
        self.objectives["total_tax_cost"] = [total_tax_cost, 0]
        filtration_tax_cost_vars = []
        for f in range(len(self.filtration) - 1):
            filtration = self.filtration[f]
            lot_tax_cost_list = []
            for i, j in filtration["lot_info"].keys():
                lot_tkr = filtration["lot_info"][i, j]["tkr"]
                lot_price = filtration["tkr_prices"][lot_tkr]
                lot_cost_basis_price = filtration["lot_info"][i, j]["cost_basis_price"]
                lot_tax_cost = (
                    self.inputs["tax_rate"]
                    * filtration["sell_shr_l"][i, j]
                    * (lot_price - lot_cost_basis_price)
                )
                lot_tax_cost_list.append(lot_tax_cost)
            tax_cost_f = self.model.addVar(lb=-GRB.INFINITY, name=f"tax_cost(f={f})")
            self.model.addConstr(
                tax_cost_f == gp.quicksum(lot_tax_cost_list),
                name=f"tax_cost_def(f={f})",
            )
            self.filtration[f]["tax_cost"] = tax_cost_f
            filtration_tax_cost_vars.append(tax_cost_f)
        self.model.addConstr(
            gp.quicksum(filtration_tax_cost_vars) <= total_tax_cost,
            name="total_tax_cost_objective",
        )

    def build_transitory_deviation_objective(self):
        """
        Minimize the total dollar deviation outside the ±tkr_dev tolerance band
        around target weights across intermediate filtrations (f=1 to T-2).

        For each model ticker k at filtration f, trans_dev[f][k] >= 0 captures
        how far the holding strays outside the band [tgt_wt_k - tkr_dev, tgt_wt_k + tkr_dev],
        encoded by two linear inequalities in dollar space:
            tkr_amount_k - (tgt_wt_k + tkr_dev) * V_f <= trans_dev[f][k]
            (tgt_wt_k - tkr_dev) * V_f - tkr_amount_k <= trans_dev[f][k]
        Within the band both RHS's are <= 0 so trans_dev[f][k] can stay at 0.

        The aggregate sum is registered as priority-1 (same lex level as terminal
        deviation) so intermediate alignment is optimized alongside terminal alignment
        before tax cost minimization.
        """
        model_tkrs = list(self.inputs["model"]["tkr"])
        tkr_dev = self.inputs["tkr_adev"]
        all_trans_dev_vars = []

        for f in range(1, len(self.filtration) - 1):
            filtration = self.filtration[f]
            portfolio_amount_f = gp.quicksum(
                filtration["shr_h"][tkr] * filtration["tkr_prices"][tkr]
                for tkr in filtration["shr_h"]
            )
            trans_dev_names = {tkr: f"trans_dev({tkr},f={f})" for tkr in model_tkrs}
            trans_dev = self.model.addVars(model_tkrs, lb=0, name=trans_dev_names)
            self.filtration[f]["trans_dev"] = trans_dev

            for tkr in model_tkrs:
                tkr_tgt_wt = (
                    self.inputs["model"]
                    .loc[self.inputs["model"]["tkr"] == tkr, "tgt_wt"]
                    .item()
                )
                tkr_amount = filtration["shr_h"][tkr] * filtration["tkr_prices"][tkr]
                self.model.addConstr(
                    tkr_amount - (tkr_tgt_wt + tkr_dev) * portfolio_amount_f
                    <= trans_dev[tkr],
                    name=f"upper_bound_trans_dev({tkr},f={f})",
                )
                self.model.addConstr(
                    (tkr_tgt_wt - tkr_dev) * portfolio_amount_f - tkr_amount
                    <= trans_dev[tkr],
                    name=f"lower_bound_trans_dev({tkr},f={f})",
                )
                all_trans_dev_vars.append(trans_dev[tkr])

        total_trans_dev = self.model.addVar(lb=0, name="total_trans_dev_objective")
        self.objectives["total_trans_dev"] = [total_trans_dev, 1]
        self.model.addConstr(
            total_trans_dev == gp.quicksum(all_trans_dev_vars),
            name="total_trans_dev_objective",
        )

    def set_objective_hierarchy(self):
        """
        Register all objective variables in self.objectives with Gurobi's
        multi-objective interface. setObjectiveN treats higher `priority`
        values as more important: priority-1 is minimized first, then
        priority-0 is minimized subject to priority-1 staying optimal
        (lexicographic minimization).
        """
        for index, (name, (var, priority)) in enumerate(self.objectives.items()):
            self.model.setObjectiveN(var, index=index, priority=priority, name=name)
        self.model.ModelSense = GRB.MINIMIZE
