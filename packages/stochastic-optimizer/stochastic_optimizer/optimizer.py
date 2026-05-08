from collections import defaultdict

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def run_optimizer(inputs: dict) -> dict:
    optimizer = StoxOptimizer(inputs)
    optimizer.build()
    sol = optimizer.solve()
    return sol


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
        self.build_information_pattern_constraints()

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
        - "filtration": dict keyed by (s, f) with per-(scenario, filtration)
          dicts containing the optimal values (.X) of all decision variables
          at that (scenario, filtration):
            lot_shr      — {(i,j): float}  shares held per lot
            shr_h        — {tkr: float}    total shares held per ticker
            sell_shr_l   — {(i,j): float}  shares sold per lot       (f < T-1 only)
            sell_shr_h   — {tkr: float}    total shares sold per tkr (f < T-1 only)
            buy_shr_h    — {tkr: float}    total shares bought       (f < T-1 only)
            sell_h       — {tkr: float}    sell binary indicator     (f < T-1 only)
            buy_h        — {tkr: float}    buy binary indicator      (f < T-1 only)
            tax_cost     — float           realized tax cost         (f < T-1 only)
            trans_dev    — {tkr: float}    out-of-band deviation     (1 <= f <= T-2)
            terminal_dev — {tkr: float}    terminal target deviation (f == T-1)
        - one key per objective name (e.g. "total_terminal_dev",
          "total_trans_dev", "total_tax_cost") mapping to its optimal scalar
          value (averaged across equally-likely scenarios).
        """
        self.model.optimize()
        sol = {"filtration": {}}

        for (s, f), filtration in self.filtration.items():
            f_sol = {
                "lot_shr": {k: v.X for k, v in filtration["lot_shr"].items()},
                "shr_h": {k: v.X for k, v in filtration["shr_h"].items()},
            }
            # sell/buy variables exist only for non-terminal filtrations
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
            if "terminal_dev" in filtration:
                f_sol["terminal_dev"] = {
                    k: v.X for k, v in filtration["terminal_dev"].items()
                }
            sol["filtration"][(s, f)] = f_sol

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
        Adds four groups of constraints for every (scenario, non-terminal
        filtration) pair:
        1. Aggregation — sell_shr_h[tkr] equals the sum of sell_shr_l across all
           lots of that ticker, linking lot-level and holding-level sell weights.
        2. Big-M linking — sell_shr_h and buy_shr_h are each upper-bounded by
           their UB times their binary indicator, so shares can only be nonzero
           when the corresponding indicator is 1.
        3. Wash-sale prevention — for tickers that appear in both the sell and buy
           universes, buy_h + sell_h <= 1 prevents simultaneous buy and sell.
        4. Self-financing — dollar value sold equals dollar value bought within
           the same period (and same scenario), at the scenario's prices.

        Skips terminal filtrations (f == T-1) since sell/buy variables don't
        exist there.
        """
        for (s, f), filtration in self.filtration.items():
            if f == self.T - 1:
                continue

            tkr_to_lot_indices = filtration["tkr_to_lot_indices"]

            for tkr in filtration["all_tkrs_to_sell"]:
                lot_indices_for_tkr = tkr_to_lot_indices[tkr]
                # sell_shr_l aggregates to sell_shr_h
                self.model.addConstr(
                    filtration["sell_shr_h"][tkr]
                    == gp.quicksum(
                        filtration["sell_shr_l"][i, j] for i, j in lot_indices_for_tkr
                    ),
                    name=f"sell_l_to_h_mapping({tkr},f={f},s={s})",
                )
                # big-M linking sell shares to sell binary
                self.model.addConstr(
                    filtration["sell_shr_h"][tkr]
                    <= filtration["sell_h"][tkr] * filtration["sell_shr_h"][tkr].UB,
                    name=f"upper_sell_binary({tkr},f={f},s={s})",
                )

            for tkr in filtration["all_tkrs_to_buy"]:
                # big-M linking buy shares to buy binary
                self.model.addConstr(
                    filtration["buy_shr_h"][tkr]
                    <= filtration["buy_h"][tkr] * filtration["buy_shr_h"][tkr].UB,
                    name=f"upper_buy_binary({tkr},f={f},s={s})",
                )

            tkrs_both_buy_and_sell = [
                tkr
                for tkr in filtration["all_tkrs_to_sell"]
                if tkr in filtration["all_tkrs_to_buy"]
            ]
            for tkr in tkrs_both_buy_and_sell:
                # only one of buy_h and sell_h can be 1 (wash-sale prevention)
                self.model.addConstr(
                    filtration["buy_h"][tkr] + filtration["sell_h"][tkr] <= 1.0,
                    name=f"buy_sell_exclusivity({tkr},f={f},s={s})",
                )

            # self-financing: dollar value sold = dollar value bought
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
                name=f"self-financing-constr(f={f},s={s})",
            )
        self.model.update()

    def build_lot_dynamics_constraints(self):
        """
        Links the portfolio state across consecutive filtration periods within
        each scenario. Dynamics are scenario-local: scenario s at period f only
        connects to scenario s at period f+1.

        For lots that already exist in the current filtration, the holding in
        the next period equals the current holding minus whatever was sold.
        For lots that are new in the next filtration (just purchased), their
        initial holding is set equal to the buy shares from the current period.
        """
        for s in range(self.n_scenario):
            for f in range(self.T - 1):
                current_filtration = self.filtration[(s, f)]
                next_filtration = self.filtration[(s, f + 1)]

                # existing lots: x[f+1] == x[f] - sell_shr_l[f]
                for i, j in current_filtration["lot_info"].keys():
                    lot_name = current_filtration["lot_info"][i, j]["tkr"]
                    self.model.addConstr(
                        current_filtration["lot_shr"][i, j]
                        - current_filtration["sell_shr_l"][i, j]
                        == next_filtration["lot_shr"][i, j],
                        name=f"sell_lot_dynamics({lot_name},l={i},t={j},f={f}->f={f+1},s={s})",
                    )

                # new lots in next filtration are seeded by the current buy
                new_lot_indices = (
                    next_filtration["lot_info"].keys()
                    - current_filtration["lot_info"].keys()
                )
                for i, j in new_lot_indices:
                    lot_name = next_filtration["lot_info"][i, j]["tkr"]
                    assert current_filtration["buy_shr_h"][lot_name] is not None
                    self.model.addConstr(
                        next_filtration["lot_shr"][i, j]
                        == current_filtration["buy_shr_h"][lot_name],
                        name=f"buy_lot_dynamics({lot_name},l={i},t={j},f={f}->f={f+1},s={s})",
                    )

        self.model.update()

    def build_information_pattern_constraints(self):
        """
        Non-anticipativity at the first decision point: the post-decision state
        at filtration f=1 (lot shares) must be identical across all scenarios.
        The optimizer does not know which scenario will materialize when it
        chooses the f=0 sells and buys, so the resulting state at f=1 cannot
        depend on s.

        Anchoring lot_shr[(s, 1)] == lot_shr[(0, 1)] for all s >= 1 is
        equivalent to forcing the first-period sell_shr_l and buy_shr_h
        decisions to be identical across scenarios:
          - existing lots at f=0: lot_shr[(s,1)] = lot_shr[(s,0)] - sell_shr_l[(s,0)],
            and lot_shr[(s,0)] is already pinned by build_starting_lot_constraints
            to the same value across scenarios, so equal lot_shr[(s,1)] forces
            equal sell_shr_l[(s,0)].
          - new lots at f=1: lot_shr[(s,1)][i, 1] = buy_shr_h[(s,0)][tkr], so
            equal lot_shr[(s,1)] forces equal buy_shr_h[(s,0)].
        """

        base_lot_shr = self.filtration[(0, 1)]["lot_shr"]
        for s in range(1, self.n_scenario):
            scenario_lot_shr = self.filtration[(s, 1)]["lot_shr"]
            for (i, j), var in scenario_lot_shr.items():
                self.model.addConstr(
                    var == base_lot_shr[i, j],
                    name=f"non_anticip_lot_shr(l={i},t={j},s={s})",
                )
        self.model.update()

    def build_starting_lot_constraints(self):
        """
        Anchors the lot shares at filtration 0 to the actual portfolio shares
        from the input positions, in every scenario. Without this, the solver
        would be free to set the initial holdings to any value. Anchoring all
        scenarios to the same `shr` also enforces non-anticipativity at the
        root: every scenario must start from the observed portfolio.
        """
        for (s, f), filtration in self.filtration.items():
            if f != 0:
                continue
            for i, j in filtration["lot_info"].keys():
                assert (
                    self.inputs["positions"].iloc[i]["tkr"]
                    == filtration["lot_info"][i, j]["tkr"]
                )
                self.model.addConstr(
                    filtration["lot_shr"][i, j]
                    == self.inputs["positions"].iloc[i]["shr"],
                    name=f"start_lot_shr({i},t={j},s={s})",
                )

    def build_terminal_deviation_objective(self):
        """
        Minimize the average (across equally-likely scenarios) total absolute
        dollar deviation from target weights at the final filtration. For each
        scenario s and model ticker k, an auxiliary terminal_dev[s][k] >= 0
        bounds |amount_k - tgt_wt_k * V_T(s)| in scenario s, encoded by two
        linear inequalities. The total objective is

            sum_s sum_k terminal_dev[s][k]

        and is registered as the highest-priority objective in the lex
        hierarchy.
        """
        model_tkrs = list(self.inputs["model"]["tkr"])
        scenario_total_devs = []

        for s in range(self.n_scenario):
            last_filtration = self.filtration[(s, self.T - 1)]
            final_portfolio_amount = gp.quicksum(
                last_filtration["shr_h"][tkr] * last_filtration["tkr_prices"][tkr]
                for tkr in last_filtration["shr_h"]
            )
            terminal_dev_names = {
                tkr: f"terminal_dev({tkr},s={s})" for tkr in model_tkrs
            }
            terminal_dev = self.model.addVars(model_tkrs, lb=0, name=terminal_dev_names)
            last_filtration["terminal_dev"] = terminal_dev

            for tkr in model_tkrs:
                tkr_tgt_wt = (
                    self.inputs["model"]
                    .loc[self.inputs["model"]["tkr"] == tkr, "tgt_wt"]
                    .item()
                )
                tkr_amount = (
                    last_filtration["shr_h"][tkr] * last_filtration["tkr_prices"][tkr]
                )
                self.model.addConstr(
                    tkr_amount - tkr_tgt_wt * final_portfolio_amount
                    <= terminal_dev[tkr],
                    name=f"upper_bound_terminal_dev({tkr},s={s})",
                )
                self.model.addConstr(
                    tkr_tgt_wt * final_portfolio_amount - tkr_amount
                    <= terminal_dev[tkr],
                    name=f"lower_bound_terminal_dev({tkr},s={s})",
                )

            scenario_total_devs.append(
                gp.quicksum(terminal_dev[tkr] for tkr in model_tkrs)
            )

        total_terminal_dev = self.model.addVar(name="total_terminal_dev_objective")
        self.objectives["total_terminal_dev"] = [total_terminal_dev, 2]
        self.model.addConstr(
            total_terminal_dev == gp.quicksum(scenario_total_devs),
            name="total_terminal_dev_objective",
        )

    def build_transitory_deviation_objective(self):
        """
        Minimize the average (across equally-likely scenarios) total dollar
        deviation outside the ±tkr_adev tolerance band around target weights at
        intermediate filtrations f=1..T-2.

        For each (s, f, k), trans_dev[s][f][k] >= 0 captures how far the
        holding strays outside the band [tgt_wt_k - tkr_adev, tgt_wt_k + tkr_adev]
        in scenario s at period f, encoded as two linear inequalities in dollar
        space. Within the band both RHSs are <= 0 so trans_dev can stay at 0.

        The aggregate is

            sum_s[ sum_{f=1..T-2} sum_k trans_dev[s][f][k] ]

        registered at priority 1.
        """
        model_tkrs = list(self.inputs["model"]["tkr"])
        tkr_dev = self.inputs["tkr_adev"]
        scenario_trans_dev_sums = []

        for s in range(self.n_scenario):
            scenario_dev_vars = []
            for f in range(1, self.T - 1):
                filtration = self.filtration[(s, f)]
                portfolio_amount_f = gp.quicksum(
                    filtration["shr_h"][tkr] * filtration["tkr_prices"][tkr]
                    for tkr in filtration["shr_h"]
                )
                trans_dev_names = {
                    tkr: f"trans_dev({tkr},f={f},s={s})" for tkr in model_tkrs
                }
                trans_dev = self.model.addVars(model_tkrs, lb=0, name=trans_dev_names)
                filtration["trans_dev"] = trans_dev

                for tkr in model_tkrs:
                    tkr_tgt_wt = (
                        self.inputs["model"]
                        .loc[self.inputs["model"]["tkr"] == tkr, "tgt_wt"]
                        .item()
                    )
                    tkr_amount = (
                        filtration["shr_h"][tkr] * filtration["tkr_prices"][tkr]
                    )
                    self.model.addConstr(
                        tkr_amount - (tkr_tgt_wt + tkr_dev) * portfolio_amount_f
                        <= trans_dev[tkr],
                        name=f"upper_bound_trans_dev({tkr},f={f},s={s})",
                    )
                    self.model.addConstr(
                        (tkr_tgt_wt - tkr_dev) * portfolio_amount_f - tkr_amount
                        <= trans_dev[tkr],
                        name=f"lower_bound_trans_dev({tkr},f={f},s={s})",
                    )
                    scenario_dev_vars.append(trans_dev[tkr])

            scenario_trans_dev_sums.append(gp.quicksum(scenario_dev_vars))

        total_trans_dev = self.model.addVar(lb=0, name="total_trans_dev_objective")
        self.objectives["total_trans_dev"] = [total_trans_dev, 1]
        self.model.addConstr(
            total_trans_dev == gp.quicksum(scenario_trans_dev_sums),
            name="total_trans_dev_objective",
        )

    def build_tax_cost_objective(self):
        """
        Minimize the average (across equally-likely scenarios) total realized
        tax cost across all sells and filtrations.

        For each (s, f), a per-period tax cost variable

            tax_cost[s, f] = tau * sum_{(i,j) in L_f} sell_shr_l[s,f][i,j] *
                             (price[s,f][tkr] - cost_basis_price[s][i,j])

        is created and stored in self.filtration[(s, f)]["tax_cost"] for easy
        downstream access (e.g. plotting per-period tax cost). The objective is

            E_s[ sum_{f=0..T-2} tax_cost[s, f] ] = (1/S) sum_s sum_f tax_cost[s, f]

        Negative contributions (losses) are kept so the optimizer can harvest
        them — the objective variable has lb = -inf accordingly. Registered at
        priority 0 (lowest in the lex hierarchy).
        """
        total_tax_cost = self.model.addVar(lb=-GRB.INFINITY, name="total_tax_cost")
        self.objectives["total_tax_cost"] = [total_tax_cost, 0]

        scenario_tax_cost_sums = []
        for s in range(self.n_scenario):
            scenario_tax_cost_vars = []
            for f in range(self.T - 1):
                filtration = self.filtration[(s, f)]
                lot_tax_cost_list = []
                for i, j in filtration["lot_info"].keys():
                    lot_tkr = filtration["lot_info"][i, j]["tkr"]
                    lot_price = filtration["tkr_prices"][lot_tkr]
                    lot_cost_basis_price = filtration["lot_info"][i, j][
                        "cost_basis_price"
                    ]
                    lot_tax_cost = (
                        self.inputs["tax_rate"]
                        * filtration["sell_shr_l"][i, j]
                        * (lot_price - lot_cost_basis_price)
                    )
                    lot_tax_cost_list.append(lot_tax_cost)

                tax_cost_f = self.model.addVar(
                    lb=-GRB.INFINITY, name=f"tax_cost(f={f},s={s})"
                )
                self.model.addConstr(
                    tax_cost_f == gp.quicksum(lot_tax_cost_list),
                    name=f"tax_cost_def(f={f},s={s})",
                )
                # reserved per (s, f) for downstream plotting / inspection
                filtration["tax_cost"] = tax_cost_f
                scenario_tax_cost_vars.append(tax_cost_f)

            scenario_tax_cost_sums.append(gp.quicksum(scenario_tax_cost_vars))

        self.model.addConstr(
            total_tax_cost == gp.quicksum(scenario_tax_cost_sums) / self.n_scenario,
            name="total_tax_cost_objective",
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
