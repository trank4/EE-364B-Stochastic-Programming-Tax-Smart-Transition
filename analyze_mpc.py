"""
Analyze a saved MPC run from run_mpc.py without re-solving.

Loads the pickle written by run_mpc.py and produces multi-scenario plots:
  - cumulative tax cost per scenario (translucent) + mean line
  - % transition per scenario (translucent) + mean line
  - AAPL portfolio weight & price per scenario (translucent) + mean lines
"""

import pickle

import pandas as pd
from stochastic_optimizer.analysis_utils import (
    plot_mpc_cumulative_tax_cost,
    plot_mpc_ticker_weight_and_price,
    plot_mpc_transition_pct,
)

INPUT_PKL = "mpc_output.pkl"

if __name__ == "__main__":
    with open(INPUT_PKL, "rb") as f:
        out = pickle.load(f)

    sol = out["sol"]
    scenario_prices = out["scenario_prices"]
    model = out["model"]
    start_date = out["start_date"]
    n_period = out["n_period"]
    n_scenario = out["n_scenario"]

    # derive month-start dates (scenario_prices DataFrames have integer index)
    dates = pd.date_range(start=start_date, periods=n_period, freq="MS")

    plot_mpc_cumulative_tax_cost(sol, dates, n_scenario, n_period)
    plot_mpc_transition_pct(sol, scenario_prices, model, dates)
    plot_mpc_ticker_weight_and_price(sol, scenario_prices, "AAPL", dates)

    print("Done.")
