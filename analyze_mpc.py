"""
Analyze a saved MPC t=0 plan run from run_mpc.py without re-solving.

The plots show the stochastic optimizer's t=0 plan: per-scenario trajectories
the solver computed under each bootstrapped price path. They are not rolling
MPC trajectories — Backtester is responsible for the rolling case.

Loads the pickle written by run_mpc.py, loads the prescient (perfect-foresight)
benchmark sol from the pickle written by run_prescient_case.py, and produces
four overlay plots:
  - cumulative tax cost            (MPC t=0 plan paths + mean, prescient dashed)
  - total portfolio value          (MPC t=0 plan paths + mean, prescient dashed)
  - % transition                   (MPC t=0 plan paths + mean, prescient dashed)
  - AAPL weight & price            (dual axis, MPC t=0 plan paths + mean,
                                    prescient weight + actual price dashed)
"""

import pickle

import pandas as pd
from quant_oracle.analysis_utils import (
    fetch_monthly_price,
    plot_mpc_t0_cumulative_tax_cost,
    plot_mpc_t0_portfolio_value,
    plot_mpc_t0_ticker_weight_and_price,
    plot_mpc_t0_transition_pct,
)

MPC_INPUT_PKL = "mpc_output.pkl"
PRESCIENT_INPUT_PKL = "prescient_output.pkl"


if __name__ == "__main__":
    with open(MPC_INPUT_PKL, "rb") as f:
        out = pickle.load(f)

    sol = out["sol"]
    scenario_prices = out["scenario_prices"]
    model = out["model"]
    start_date = out["start_date"]
    n_period = out["n_period"]
    n_scenario = out["n_scenario"]

    # backfill actual_prices via yfinance if the pickle predates the field
    actual_prices = out.get("actual_prices")
    if actual_prices is None:
        end_date_excl = (
            pd.Timestamp(start_date) + pd.DateOffset(months=n_period)
        ).strftime("%Y-%m-%d")
        print(
            f"actual_prices missing from pickle; fetching realized prices from "
            f"yfinance ({pd.Timestamp(start_date).date()} → {end_date_excl})..."
        )
        actual_prices = fetch_monthly_price(
            list(scenario_prices[0].columns),
            start_date=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
            end_date=end_date_excl,
        )

    # derive month-start dates (scenario_prices DataFrames have integer index)
    dates = pd.date_range(start=start_date, periods=n_period, freq="MS")

    # load prescient sol from the prescient pickle instead of re-solving
    print(f"Loading prescient benchmark from {PRESCIENT_INPUT_PKL}...")
    with open(PRESCIENT_INPUT_PKL, "rb") as f:
        prescient_out = pickle.load(f)
    prescient_sol = prescient_out["sol"]

    plot_mpc_t0_cumulative_tax_cost(
        sol, dates, n_scenario, n_period, prescient_sol=prescient_sol
    )
    plot_mpc_t0_portfolio_value(
        sol,
        scenario_prices,
        dates,
        prescient_sol=prescient_sol,
        actual_prices=actual_prices,
    )
    plot_mpc_t0_transition_pct(
        sol,
        scenario_prices,
        model,
        dates,
        prescient_sol=prescient_sol,
        actual_prices=actual_prices,
    )
    plot_mpc_t0_ticker_weight_and_price(
        sol,
        scenario_prices,
        "AAPL",
        dates,
        prescient_sol=prescient_sol,
        actual_prices=actual_prices,
    )

    print("Done.")
