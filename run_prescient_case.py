"""
Prescient-case benchmark for the stochastic portfolio transition optimizer.

Steps:
  1. Fetch the top-100 S&P 500 constituents by market cap as of 2025-01-01.
  2. Download monthly price data for 2025 and compute realized monthly returns.
  3. Pass the return matrix to StoxOptimizer and run build() / solve().

"Prescient" means perfect foresight: the optimizer sees the full realized
return path for 2025, serving as an upper-bound benchmark against which
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
    # assuming the portfolio has 1 stock "APPL" with low cost basis high unrealized gains
    positions = pd.DataFrame(
        {"Tkr": ["AAPL"], "Amt": [10000000], "CostBasisAmt": [500000]}
    )
    positions["PNL"] = (positions["Amt"] - positions["CostBasisAmt"]) / positions["Amt"]

    # assuming for simplicity 1 tax rate of 30%
    tax_rate = 0.3

    # assume equally-weighted model of the top 20 tickers
    model = pd.DataFrame(
        {"Tkr": top20_spy_tickers, "TgtWt": 1 / len(top20_spy_tickers)}
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
