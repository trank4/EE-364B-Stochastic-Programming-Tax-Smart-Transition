import pandas as pd
from stochastic_optimizer import RMPController
from stochastic_optimizer.analysis_utils import *

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


def run_RMPController(inputs: dict) -> dict:
    controller = RMPController(inputs)
    controller.build_price_scenarios()
    sol = controller.execute()
    return sol


if __name__ == "__main__":
    monthly_prices = fetch_monthly_price(
        top20_spy_tickers, start_date="2024-01-01", end_date="2025-01-01"
    )
    # construct starting position dataframe
    # assuming the portfolio has 1 stock "AAPL" with low cost basis high unrealized gains
    positions = pd.DataFrame(
        {"tkr": ["AAPL"], "amt": [1000000], "cost_basis_amt": [500000]}
    )
    positions["pnl"] = (positions["amt"] - positions["cost_basis_amt"]) / positions[
        "amt"
    ].sum()
    positions["wt"] = positions["amt"] / positions["amt"].sum()

    # assuming for simplicity 1 tax rate of 30%
    tax_rate = 0.3

    # assume equally-weighted model of the top 20 tickers
    model = pd.DataFrame(
        {"tkr": top20_spy_tickers, "tgt_wt": 1 / len(top20_spy_tickers)}
    )

    inputs = {
        "start_date": pd.to_datetime("2024-01-01"),
        "sim_start_date": pd.to_datetime("2023-01-01"),
        "sim_end_date": pd.to_datetime("2024-01-01"),
        "n_period": 12,
        "n_scenario": 10,
        "block_length": 3,
        "tax_rate": tax_rate,
        "model": model,
        "positions": positions,
    }

    sol = run_RMPController(inputs)
