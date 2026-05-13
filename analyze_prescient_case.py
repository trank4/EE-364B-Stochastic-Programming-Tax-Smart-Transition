"""
Analyze a saved prescient (perfect-foresight) optimizer run produced by
run_prescient_case.py, without re-solving.

Loads the pickle at PRESCIENT_OUTPUT_PKL and produces:
  - cumulative tax cost            (cumulative_tax_cost.png)
  - total portfolio value          (portfolio_value.png)
  - % transition over time         (transition_pct.png)
  - AAPL weight & price            (AAPL_weight_and_price.png)
"""

import pickle

from quant_oracle.analysis_utils import (
    calculate_portfolio_weights,
    calculate_transition_pct,
    plot_cumulative_tax_cost,
    plot_portfolio_value,
    plot_ticker_weight_and_price,
    plot_transition_pct,
)

INPUT_PKL = "prescient_output.pkl"


if __name__ == "__main__":
    with open(INPUT_PKL, "rb") as f:
        out = pickle.load(f)

    sol = out["sol"]
    monthly_prices = out["monthly_prices"]
    model = out["model"]

    plot_cumulative_tax_cost(sol, monthly_prices)
    plot_portfolio_value(sol, monthly_prices)

    weights_df = calculate_portfolio_weights(sol, monthly_prices)
    transition_pct = calculate_transition_pct(weights_df, model)
    plot_transition_pct(transition_pct)
    plot_ticker_weight_and_price(weights_df, monthly_prices, "AAPL")

    print("\nPortfolio weights over time:")
    print(weights_df.to_string())
    print("\n% Transition over time:")
    print(transition_pct.to_string())

    print("\nDone.")
