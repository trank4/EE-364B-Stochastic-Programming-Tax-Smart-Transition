"""
Analyze a saved rolling backtest produced by run_backtest.py without
re-solving.

Loads the pickle at BACKTEST_INPUT_PKL, the prescient (perfect-foresight)
benchmark sol from the pickle written by run_prescient_case.py, and the
MPC t=0 plan sol from the pickle written by run_mpc.py. Produces four
overlay plots:
  - cumulative realized tax cost   (backtest solid, prescient dashed,
                                    MPC t=0 plan mean dashed)
  - total portfolio value          (backtest solid, prescient dashed,
                                    MPC t=0 plan mean dashed)
  - % transition                   (backtest solid, prescient dashed,
                                    MPC t=0 plan mean dashed)
  - AAPL weight & price            (dual axis, backtest weight solid,
                                    prescient weight navy dashed,
                                    MPC t=0 plan mean weight seagreen dashed,
                                    actual price orange dashed)
"""

import pickle

from quant_oracle.analysis_utils import (
    plot_backtest_cumulative_tax_cost,
    plot_backtest_portfolio_value,
    plot_backtest_ticker_weight_and_price,
    plot_backtest_transition_pct,
)

BACKTEST_INPUT_PKL = "backtest_output.pkl"
PRESCIENT_INPUT_PKL = "prescient_output.pkl"
MPC_INPUT_PKL = "mpc_output.pkl"


if __name__ == "__main__":
    with open(BACKTEST_INPUT_PKL, "rb") as f:
        bt = pickle.load(f)

    results = bt["results"]
    actual_prices = bt["actual_prices"]
    model = bt["model"]

    print(f"Loading prescient benchmark from {PRESCIENT_INPUT_PKL}...")
    with open(PRESCIENT_INPUT_PKL, "rb") as f:
        prescient_out = pickle.load(f)
    prescient_sol = prescient_out["sol"]

    print(f"Loading MPC t=0 plan from {MPC_INPUT_PKL}...")
    with open(MPC_INPUT_PKL, "rb") as f:
        mpc_out = pickle.load(f)
    mpc_sol = mpc_out["sol"]
    mpc_scenario_prices = mpc_out["scenario_prices"]
    mpc_n_scenario = mpc_out["n_scenario"]

    plot_backtest_cumulative_tax_cost(
        results,
        actual_prices,
        prescient_sol=prescient_sol,
        mpc_sol=mpc_sol,
        mpc_n_scenario=mpc_n_scenario,
    )
    plot_backtest_portfolio_value(
        results,
        actual_prices,
        prescient_sol=prescient_sol,
        mpc_sol=mpc_sol,
        mpc_scenario_prices=mpc_scenario_prices,
    )
    plot_backtest_transition_pct(
        results,
        actual_prices,
        model,
        prescient_sol=prescient_sol,
        mpc_sol=mpc_sol,
        mpc_scenario_prices=mpc_scenario_prices,
    )
    plot_backtest_ticker_weight_and_price(
        results,
        actual_prices,
        "AAPL",
        prescient_sol=prescient_sol,
        mpc_sol=mpc_sol,
        mpc_scenario_prices=mpc_scenario_prices,
    )

    print("Done.")
