"""
Prescient-case benchmark for the stochastic portfolio transition optimizer.

Steps:
  1. Select the top-20 S&P 500 constituents by market cap as of 2024-01-01.
  2. Download monthly price data for 2024 and compute realized monthly prices.
  3. Pass the price matrix to StoxOptimizer and run build() / solve().

"Prescient" means perfect foresight: the optimizer sees the full realized
price path for 2024, serving as an upper-bound benchmark against which
stochastic (multi-scenario) solutions are compared.
"""

import pandas as pd
import yfinance as yf
from stochastic_optimizer import StoxOptimizer

# ---------------------------------------------------------------------------
# Step 1: top S&P 500 constituents as of 2024-01-01
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Step 2: Monthly returns for 2025
# ---------------------------------------------------------------------------


def fetch_monthly_price(tickers: list[str]) -> pd.DataFrame:
    """
    Download daily adjusted closes, resample to month-start
    """
    raw = yf.download(
        tickers,
        start="2024-01-01",  # need one prior month for the first return
        end="2025-01-01",
        auto_adjust=True,
        progress=False,
    )["Close"]

    # Align columns to requested tickers (some may have been delisted)
    raw = raw.reindex(columns=tickers)

    # Month-end prices
    monthly_prices = raw.resample("MS").first()

    return monthly_prices


# ---------------------------------------------------------------------------
# Step 3: Run StoxOptimizer (prescient / perfect-foresight mode)
# ---------------------------------------------------------------------------


def run_optimizer(inputs: dict) -> None:
    optimizer = StoxOptimizer(inputs)
    optimizer.build()
    optimizer.solve()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Monthly returns
    monthly_prices = fetch_monthly_price(top20_spy_tickers)

    # construct starting position dataframe
    # assuming the portfolio has 1 stock "AAPL" with low cost basis high unrealized gains
    positions = pd.DataFrame(
        {"tkr": ["AAPL"], "amt": [10000000], "cost_basis_amt": [500000]}
    )
    positions["pnl"] = (positions["amt"] - positions["cost_basis_amt"]) / positions[
        "amt"
    ]
    positions["wt"] = positions["amt"] / positions["amt"].sum()

    # assuming for simplicity 1 tax rate of 30%
    tax_rate = 0.3

    # assume equally-weighted model of the top 20 tickers
    model = pd.DataFrame(
        {"tkr": top20_spy_tickers, "tgt_wt": 1 / len(top20_spy_tickers)}
    )

    # combine everything into inputs dict
    inputs = {
        "positions": positions,
        "tax_rate": tax_rate,
        "model": model,
        "monthly_prices": monthly_prices,
    }
    # build and run optimizer
    run_optimizer(inputs)

    print("\nDone.")
