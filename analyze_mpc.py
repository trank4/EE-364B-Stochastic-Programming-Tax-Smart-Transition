"""
Analyze a saved MPC t=0 plan run from run_mpc.py without re-solving.

The plots show the stochastic optimizer's t=0 plan: per-scenario trajectories
the solver computed under each bootstrapped price path. They are not rolling
MPC trajectories — Backtester is responsible for the rolling case.

Loads the pickle written by run_mpc.py, runs the prescient (perfect-foresight)
benchmark on the realized prices, and produces four overlay plots:
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
from quant_oracle.optimizer import run_optimizer

INPUT_PKL = "mpc_output.pkl"

# fallback defaults used only if the pickle was generated before run_mpc.py
# started saving these fields (Option 2: backfill instead of forcing a re-solve)
DEFAULT_TAX_RATE = 0.3
DEFAULT_TKR_ADEV = 0.05

if __name__ == "__main__":
    with open(INPUT_PKL, "rb") as f:
        out = pickle.load(f)

    sol = out["sol"]
    scenario_prices = out["scenario_prices"]
    model = out["model"]
    positions = out["positions"]
    start_date = out["start_date"]
    n_period = out["n_period"]
    n_scenario = out["n_scenario"]
    tax_rate = out.get("tax_rate", DEFAULT_TAX_RATE)
    tkr_adev = out.get("tkr_adev", DEFAULT_TKR_ADEV)

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

    # solve the prescient benchmark on the realized 2024 prices (single scenario)
    print("Solving prescient benchmark on realized prices...")
    prescient_sol = run_optimizer(
        {
            "positions": positions,
            "tax_rate": tax_rate,
            "model": model,
            "tkr_adev": tkr_adev,
            "monthly_prices": [actual_prices],
        }
    )

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
