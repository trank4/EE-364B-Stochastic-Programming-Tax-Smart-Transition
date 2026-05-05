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

import matplotlib.pyplot as plt
import numpy as np
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
# Step 2: Month-start prices for 2024
# ---------------------------------------------------------------------------


def fetch_monthly_price(tickers: list[str]) -> pd.DataFrame:
    """
    Download daily adjusted closes and take the first observation of each month.
    """
    raw = yf.download(
        tickers,
        start="2024-01-01",
        end="2025-01-01",
        auto_adjust=True,
        progress=False,
    )["Close"]

    # Align columns to requested tickers (some may have been delisted)
    raw = raw.reindex(columns=tickers)

    # Month-start prices
    monthly_prices = raw.resample("MS").first()

    return monthly_prices


# ---------------------------------------------------------------------------
# Step 3: Run StoxOptimizer (prescient / perfect-foresight mode)
# ---------------------------------------------------------------------------


def run_optimizer(inputs: dict) -> dict:
    optimizer = StoxOptimizer(inputs)
    optimizer.build()
    sol = optimizer.solve()
    return sol


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def plot_cumulative_tax_cost(sol: dict, monthly_prices: pd.DataFrame) -> None:
    """
    Plot cumulative realized tax cost across filtration periods.
    Saves the figure to cumulative_tax_cost.png.
    """
    dates = monthly_prices.index
    per_period_tax = np.array(
        [f_sol["tax_cost"] for f_sol in sol["filtration"] if "tax_cost" in f_sol]
    )
    cumulative_tax = np.cumsum(per_period_tax)
    trade_dates = dates[: len(per_period_tax)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trade_dates, cumulative_tax, marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Tax Cost ($)")
    ax.set_title("Cumulative Realized Tax Cost Over Transition")
    ax.xaxis.set_tick_params(rotation=45)
    fig.tight_layout()
    plt.savefig("cumulative_tax_cost.png", dpi=150)
    plt.show()


def calculate_portfolio_weights(
    sol: dict, monthly_prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute portfolio weight of each ticker at every filtration period.

    For filtration f, weight_k = (shr_h[k] * price_k) / total_portfolio_value.
    Tickers not present in shr_h at a given period are assigned weight 0.

    Returns a DataFrame indexed by filtration date with one column per ticker.
    """
    all_tkrs = list(monthly_prices.columns)
    records = []
    for f, f_sol in enumerate(sol["filtration"]):
        prices = monthly_prices.iloc[f]
        dollar_vals = {
            tkr: f_sol["shr_h"].get(tkr, 0.0) * prices[tkr] for tkr in all_tkrs
        }
        total_val = sum(dollar_vals.values())
        weights = {
            tkr: (v / total_val if total_val > 0 else 0.0)
            for tkr, v in dollar_vals.items()
        }
        records.append(weights)
    return pd.DataFrame(records, index=monthly_prices.index)


def plot_transition_pct(transition_pct: pd.Series) -> None:
    """
    Plot % transition towards the target model portfolio over filtration periods.
    Saves the figure to transition_pct.png.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(transition_pct.index, transition_pct.values, marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("% Transition")
    ax.set_title("Portfolio Transition Progress Over Time")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_tick_params(rotation=45)
    fig.tight_layout()
    plt.savefig("transition_pct.png", dpi=150)
    plt.show()


def calculate_transition_pct(
    weights_df: pd.DataFrame, model: pd.DataFrame
) -> pd.Series:
    """
    Proxy for how far the portfolio has transitioned towards the target model.

        % transition = 1 - max(sum_overweights, sum_underweights)

    where overweight_k = max(0, w_k - tgt_wt_k) and
          underweight_k = max(0, tgt_wt_k - w_k).

    When fully transitioned both sums are 0 and the metric is 1.0; at the
    starting position the metric reflects the total active weight distance.

    Returns a Series indexed by filtration date with values in [0, 1].
    """
    tgt_wt = model.set_index("tkr")["tgt_wt"]
    result = {}
    for date, row in weights_df.iterrows():
        sum_over = sum(max(0.0, row.get(tkr, 0.0) - wt) for tkr, wt in tgt_wt.items())
        sum_under = sum(max(0.0, wt - row.get(tkr, 0.0)) for tkr, wt in tgt_wt.items())
        result[date] = 1.0 - max(sum_over, sum_under)
    return pd.Series(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Month-start prices
    monthly_prices = fetch_monthly_price(top20_spy_tickers)

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

    # combine everything into inputs dict
    inputs = {
        "positions": positions,
        "tax_rate": tax_rate,
        "model": model,
        "tkr_adev": 0.05,
        "monthly_prices": monthly_prices,
    }
    # build and run optimizer
    sol = run_optimizer(inputs)

    plot_cumulative_tax_cost(sol, monthly_prices)

    weights_df = calculate_portfolio_weights(sol, monthly_prices)
    transition_pct = calculate_transition_pct(weights_df, model)
    plot_transition_pct(transition_pct)
    print("\nPortfolio weights over time:")
    print(weights_df.to_string())
    print("\n% Transition over time:")
    print(transition_pct.to_string())

    print("\nDone.")
