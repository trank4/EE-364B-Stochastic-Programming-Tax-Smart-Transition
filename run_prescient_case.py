"""
Prescient-case benchmark for the stochastic portfolio transition optimizer.

Steps:
  1. Select the top-20 S&P 500 constituents by market cap as of 2024-01-01.
  2. Download monthly price data for 2024 and compute realized monthly prices.
  3. Pass the price matrix to ForwardOptimizer and run build() / solve().
  4. Pickle the optimizer output and inputs to PRESCIENT_OUTPUT_PKL for
     downstream analysis by analyze_prescient_case.py and analyze_mpc.py
     (which uses the prescient sol as a perfect-foresight benchmark) so that
     the optimizer does not need to be re-run.

"Prescient" means perfect foresight: the optimizer sees the full realized
price path for 2024, serving as an upper-bound benchmark against which
stochastic (multi-scenario) solutions are compared.
"""

import pickle

import pandas as pd
from quant_oracle.analysis_utils import fetch_monthly_price
from quant_oracle.optimizer import run_optimizer

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

PRESCIENT_OUTPUT_PKL = "prescient_output.pkl"

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Month-start prices
    monthly_prices = fetch_monthly_price(
        top20_spy_tickers, start_date="2024-01-01", end_date="2025-01-01"
    )

    # construct starting position dataframe
    # assuming the portfolio has 1 stock "AAPL" with low cost basis high unrealized gains
    positions = pd.DataFrame(
        {"tkr": ["AAPL"], "amt": [1000000], "cost_basis_amt": [500000]}
    )
    start_prices = monthly_prices.iloc[0]  # Series indexed by ticker
    positions["price"] = positions["tkr"].map(start_prices)
    positions["shr"] = positions["amt"] / positions["price"]
    positions["cost_basis_price"] = positions["cost_basis_amt"] / positions["shr"]

    # assuming for simplicity 1 tax rate of 30%
    tax_rate = 0.3

    # assume equally-weighted model of the top 20 tickers
    model = pd.DataFrame(
        {"tkr": top20_spy_tickers, "tgt_wt": 1 / len(top20_spy_tickers)}
    )

    # combine everything into inputs dict
    # monthly_prices is wrapped in a list because the optimizer accepts a
    # per-scenario list of price DataFrames; the prescient case is a single
    # scenario with the realized 2024 prices.
    tkr_adev = 0.05
    inputs = {
        "positions": positions,
        "tax_rate": tax_rate,
        "model": model,
        "tkr_adev": tkr_adev,
        "monthly_prices": [monthly_prices],
    }
    # build and run optimizer
    sol = run_optimizer(inputs)

    # persist everything analyze_prescient_case.py and analyze_mpc.py need so
    # the optimizer does not have to re-run
    output = {
        "sol": sol,
        "monthly_prices": monthly_prices,
        "positions": positions,
        "model": model,
        "tax_rate": tax_rate,
        "tkr_adev": tkr_adev,
    }
    with open(PRESCIENT_OUTPUT_PKL, "wb") as f:
        pickle.dump(output, f)
    print(f"\nSaved prescient output to {PRESCIENT_OUTPUT_PKL}")
