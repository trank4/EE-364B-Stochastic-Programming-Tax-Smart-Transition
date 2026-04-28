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


def fetch_monthly_returns(tickers: list[str]) -> pd.DataFrame:
    """
    Download daily adjusted closes for 2025, resample to month-end, and
    compute simple monthly returns.

    Returns a DataFrame of shape (12, len(tickers)) indexed by month-end date.
    """
    print(f"\nDownloading 2024 daily prices for {len(tickers)} tickers …")
    raw = yf.download(
        tickers,
        start="2023-12-1",  # need one prior month for the first return
        end="2025-01-01",
        auto_adjust=True,
        progress=False,
    )["Close"]

    # Align columns to requested tickers (some may have been delisted)
    raw = raw.reindex(columns=tickers)

    # Month-end prices
    monthly_prices = raw.resample("ME").last()

    # Simple returns: (P_t - P_{t-1}) / P_{t-1}
    monthly_returns = monthly_prices.pct_change().dropna(how="all")

    # Keep only the 12 months of 2025
    monthly_returns = monthly_returns[
        (monthly_returns.index >= "2024-01-01")
        & (monthly_returns.index <= "2024-12-31")
    ]

    print(f"\nMonthly return matrix: {monthly_returns.shape}  (months × assets)")
    print(monthly_returns.to_string())

    return monthly_prices, monthly_returns


# ---------------------------------------------------------------------------
# Step 3: Run StoxOptimizer (prescient / perfect-foresight mode)
# ---------------------------------------------------------------------------


def run_optimizer(inputs: dict) -> None:
    optimizer = StoxOptimizer()
    optimizer.build()
    optimizer.solve()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Monthly returns
    monthly_price, monthly_returns = fetch_monthly_returns(top20_spy_tickers)

    # construct starting position dataframe
    # assuming the portfolio has 1 stock "XYZ" which is the start-up by the founder, thus cost basis of 1
    positions = pd.DataFrame({"Tkr": ["XYZ"], "Amt": [10000000], "CostBasisAmt": [1]})

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
        "monthly_returns": monthly_returns,
        "monthly_price": monthly_price,
    }
    # 4. Optimize
    run_optimizer(inputs)

    print("\nDone.")
