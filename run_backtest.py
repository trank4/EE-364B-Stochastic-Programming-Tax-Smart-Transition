"""
Rolling monthly backtest of the tax-smart transition optimizer from 01/2024
through 12/2024.

At each month t the Backtester solves a fresh stochastic problem with:
  - receding horizon         (n_period = 12 - t, so the target end of
                              transition stays anchored at 12/2024)
  - rolling 1-year history   (sim window = [t - 12 months, t]) used by the
                              block-bootstrap scenario generator
  - deterministic final step (at t = 12/2024 the realized price is already
                              known; ForwardOptimizer is called directly on
                              the single-row price)

Only the first-period (f=1) trades from each step are executed; positions
are marked to market at the next month's realized price and the loop
advances. The resulting list of per-step solves is pickled to
BACKTEST_OUTPUT_PKL for downstream analysis by analyze_backtest.py.
"""

import pickle

import pandas as pd
from quant_oracle import Backtester
from quant_oracle.analysis_utils import fetch_monthly_price

top20_spy_tickers = [
    "AAPL",
    "MSFT",
    "AMZN",
    "NVDA",
    "GOOGL",
    "GOOG",
    "META",
    "TSLA",
    "LLY",
    "AVGO",
    "JPM",
    "V",
    "XOM",
    "UNH",
    "MA",
    "JNJ",
    "PG",
    "COST",
    "HD",
    "KO",
]

BACKTEST_OUTPUT_PKL = "backtest_output.pkl"


if __name__ == "__main__":
    # Realized monthly prices: Jan 2024 through Jan 2025 (13 month-starts).
    # The extra Jan-2025 row is consumed only for marking-to-market the final
    # Dec-2024 trade, so the loop runs exactly 12 steps.
    actual_prices = fetch_monthly_price(
        top20_spy_tickers, start_date="2024-01-01", end_date="2025-02-01"
    )

    # Same starting portfolio as run_mpc.py / run_prescient_case.py:
    # a single AAPL lot with high unrealized gains (cost basis = $500K on $1M).
    positions = pd.DataFrame(
        {"tkr": ["AAPL"], "amt": [1000000], "cost_basis_amt": [500000]}
    )
    start_prices = actual_prices.iloc[0]
    positions["price"] = positions["tkr"].map(start_prices)
    positions["shr"] = positions["amt"] / positions["price"]
    positions["cost_basis_price"] = positions["cost_basis_amt"] / positions["shr"]

    tax_rate = 0.3
    tkr_adev = 0.05

    # Equally-weighted top-20 SPY target.
    model = pd.DataFrame(
        {"tkr": top20_spy_tickers, "tgt_wt": 1 / len(top20_spy_tickers)}
    )

    n_steps = len(actual_prices) - 1  # 12
    base_inputs = {
        "seed": 0,
        "n_period": n_steps,  # initial horizon; receding each step inside Backtester
        "n_scenario": 10,
        "block_length": 3,
        "tax_rate": tax_rate,
        "model": model,
        "positions": positions,
        "tkr_adev": tkr_adev,
    }

    backtester = Backtester(base_inputs, actual_prices, lookback_months=12)
    results = backtester.run()

    output = {
        "results": results,
        "actual_prices": actual_prices,
        "model": model,
        "positions": positions,
        "tax_rate": tax_rate,
        "tkr_adev": tkr_adev,
        "n_steps": n_steps,
        "all_tkrs": sorted(set(top20_spy_tickers) | set(positions["tkr"])),
    }
    with open(BACKTEST_OUTPUT_PKL, "wb") as f:
        pickle.dump(output, f)
    print(f"\nSaved backtest output to {BACKTEST_OUTPUT_PKL}")
