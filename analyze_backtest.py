"""
Analyze a saved rolling backtest produced by run_backtest.py without
re-solving.

Loads the pickle at BACKTEST_INPUT_PKL, optionally loads the prescient
(perfect-foresight) benchmark sol from the pickle written by
run_prescient_case.py, and produces four overlay plots:
  - cumulative realized tax cost   (backtest solid, prescient dashed)
  - total portfolio value          (backtest solid, prescient dashed)
  - % transition                   (backtest solid, prescient dashed)
  - AAPL weight & price            (dual axis, backtest weight solid,
                                    prescient weight navy dashed,
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

    plot_backtest_cumulative_tax_cost(
        results, actual_prices, prescient_sol=prescient_sol
    )
    plot_backtest_portfolio_value(results, actual_prices, prescient_sol=prescient_sol)
    plot_backtest_transition_pct(
        results, actual_prices, model, prescient_sol=prescient_sol
    )
    plot_backtest_ticker_weight_and_price(
        results, actual_prices, "AAPL", prescient_sol=prescient_sol
    )

    print("Done.")
