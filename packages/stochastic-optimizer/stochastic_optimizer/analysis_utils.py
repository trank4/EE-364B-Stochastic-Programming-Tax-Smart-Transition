import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


def fetch_monthly_price(tickers: list[str], start_date, end_date) -> pd.DataFrame:
    """
    Download daily adjusted closes and take the first observation of each month.
    """
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )["Close"]

    # Align columns to requested tickers (some may have been delisted)
    raw = raw.reindex(columns=tickers)

    # Month-start prices
    monthly_prices = raw.resample("MS").first()

    return monthly_prices


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _scenario_filtrations(sol: dict, s: int) -> list[dict]:
    """
    Extract per-period filtration dicts for a single scenario s, in order
    f=0..T-1. sol["filtration"] is keyed by (s, f) tuples.
    """
    items = [(f, fs) for (sk, f), fs in sol["filtration"].items() if sk == s]
    items.sort(key=lambda kv: kv[0])
    return [fs for _, fs in items]


def plot_cumulative_tax_cost(
    sol: dict, monthly_prices: pd.DataFrame, s: int = 0
) -> None:
    """
    Plot cumulative realized tax cost across filtration periods for scenario s.
    Saves the figure to cumulative_tax_cost.png.
    """
    dates = monthly_prices.index
    per_period_tax = np.array(
        [
            f_sol["tax_cost"]
            for f_sol in _scenario_filtrations(sol, s)
            if "tax_cost" in f_sol
        ]
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
    sol: dict, monthly_prices: pd.DataFrame, s: int = 0
) -> pd.DataFrame:
    """
    Compute portfolio weight of each ticker at every filtration period for
    scenario s.

    For filtration f, weight_k = (shr_h[k] * price_k) / total_portfolio_value.
    Tickers not present in shr_h at a given period are assigned weight 0.

    Returns a DataFrame indexed by filtration date with one column per ticker.
    """
    all_tkrs = list(monthly_prices.columns)
    records = []
    for f, f_sol in enumerate(_scenario_filtrations(sol, s)):
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


def plot_portfolio_value(sol: dict, monthly_prices: pd.DataFrame, s: int = 0) -> None:
    """
    Plot total portfolio value in dollars at each filtration period for
    scenario s. Saves the figure to portfolio_value.png.
    """
    dates = monthly_prices.index
    total_values = []
    for f, f_sol in enumerate(_scenario_filtrations(sol, s)):
        prices = monthly_prices.iloc[f]
        total_val = sum(
            f_sol["shr_h"].get(tkr, 0.0) * prices[tkr] for tkr in monthly_prices.columns
        )
        total_values.append(total_val)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, total_values, marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Total Portfolio Value Over Transition")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax.xaxis.set_tick_params(rotation=45)
    fig.tight_layout()
    plt.savefig("portfolio_value.png", dpi=150)
    plt.show()


def plot_ticker_weight_and_price(
    weights_df: pd.DataFrame, monthly_prices: pd.DataFrame, tkr: str
) -> None:
    """
    Plot portfolio weight and market price of a single ticker over time on dual axes.
    Saves the figure to <tkr>_weight_and_price.png.
    """
    dates = weights_df.index
    weights = weights_df[tkr]
    prices = monthly_prices[tkr]

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(dates, weights, marker="o", color="steelblue", label=f"{tkr} weight")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Weight", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1.xaxis.set_tick_params(rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(
        dates,
        prices,
        marker="s",
        color="darkorange",
        linestyle="--",
        label=f"{tkr} price",
    )
    ax2.set_ylabel("Price ($)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.suptitle(f"{tkr} Weight and Price Over Transition")
    fig.tight_layout()
    plt.savefig(f"{tkr}_weight_and_price.png", dpi=150)
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
