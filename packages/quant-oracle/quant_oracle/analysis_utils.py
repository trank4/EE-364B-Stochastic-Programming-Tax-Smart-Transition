import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


def fetch_monthly_price(tickers: list[str], start_date, end_date) -> pd.DataFrame:
    """
    Download daily adjusted closes and take the first observation of each month.
    Forward-fills any gaps so month-start rows are never NaN for tickers that
    have at least one earlier observation in the window (e.g. when the
    nominal month-start is a non-trading day or has a halted print).
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

    # Month-start prices, forward-filled across any missing months
    monthly_prices = raw.resample("MS").first().ffill()

    return monthly_prices


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _scenario_filtrations(sol: dict, s: int) -> list[dict]:
    """
    Extract per-period filtration dicts for a single scenario s, in order
    f=1..T. sol["filtration"] is keyed by (s, f) tuples. The returned list
    is 0-indexed: index k corresponds to filtration f = k + 1.
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


# ---------------------------------------------------------------------------
# Multi-scenario MPC t=0 plan plotting helpers
# ---------------------------------------------------------------------------


def _per_scenario_cumulative_tax(
    sol: dict, n_scenario: int, n_period: int
) -> np.ndarray:
    """
    Returns shape (n_scenario, n_period) of cumulative tax cost over time
    for each scenario. Filtration index f runs from 1 to n_period.
    """
    out = np.zeros((n_scenario, n_period))
    for s in range(n_scenario):
        per_period = [
            sol["filtration"][(s, f)]["tax_cost"] for f in range(1, n_period + 1)
        ]
        out[s] = np.cumsum(per_period)
    return out


def _per_scenario_weights(
    sol: dict, scenario_prices: list[pd.DataFrame], all_tkrs: list[str]
) -> np.ndarray:
    """
    Returns shape (n_scenario, n_period, n_tkr) of portfolio weights per
    scenario, period, and ticker, computed from shr_h and the scenario's own
    price path.
    """
    n_scenario = len(scenario_prices)
    n_period = scenario_prices[0].shape[0]
    n_tkr = len(all_tkrs)
    out = np.zeros((n_scenario, n_period, n_tkr))
    for s in range(n_scenario):
        for f in range(1, n_period + 1):
            shr_h = sol["filtration"][(s, f)]["shr_h"]
            prices = scenario_prices[s].iloc[f - 1]
            dollars = np.array([shr_h.get(tkr, 0.0) * prices[tkr] for tkr in all_tkrs])
            total = dollars.sum()
            if total > 0:
                out[s, f - 1, :] = dollars / total
    return out


def _per_scenario_transition_pct(
    weights: np.ndarray, all_tkrs: list[str], model: pd.DataFrame
) -> np.ndarray:
    """
    Returns shape (n_scenario, n_period) of transition pct per scenario.
    """
    n_scenario, n_period, _ = weights.shape
    tgt_series = model.set_index("tkr")["tgt_wt"]
    tgt_arr = np.array([float(tgt_series.get(tkr, 0.0)) for tkr in all_tkrs])
    out = np.zeros((n_scenario, n_period))
    for s in range(n_scenario):
        for f in range(n_period):
            diff = weights[s, f] - tgt_arr
            sum_over = np.maximum(diff, 0.0).sum()
            sum_under = np.maximum(-diff, 0.0).sum()
            out[s, f] = 1.0 - max(sum_over, sum_under)
    return out


def _per_scenario_portfolio_value(
    sol: dict, scenario_prices: list[pd.DataFrame], all_tkrs: list[str]
) -> np.ndarray:
    """
    Returns shape (n_scenario, n_period) of total portfolio value per scenario,
    computed as sum_k shr_h[(s,f)][k] * scenario_prices[s].iloc[f][k].
    """
    n_scenario = len(scenario_prices)
    n_period = scenario_prices[0].shape[0]
    out = np.zeros((n_scenario, n_period))
    for s in range(n_scenario):
        for f in range(1, n_period + 1):
            shr_h = sol["filtration"][(s, f)]["shr_h"]
            prices = scenario_prices[s].iloc[f - 1]
            out[s, f - 1] = sum(shr_h.get(tkr, 0.0) * prices[tkr] for tkr in all_tkrs)
    return out


def _plot_paths_with_mean(
    ax, dates, paths: np.ndarray, color: str, label_prefix: str
) -> None:
    """
    Plot every row of `paths` (shape n_scenario x n_period) on `ax` with low
    alpha, plus the cross-scenario mean as a thick solid line.
    """
    for path in paths:
        ax.plot(dates, path, color=color, alpha=0.2, linewidth=1)
    mean_path = paths.mean(axis=0)
    label = f"{label_prefix} mean" if label_prefix else "mean"
    ax.plot(dates, mean_path, color=color, linewidth=2.5, label=label)


def plot_mpc_t0_cumulative_tax_cost(
    sol: dict,
    dates,
    n_scenario: int,
    n_period: int,
    prescient_sol: dict | None = None,
) -> None:
    """
    Plot per-scenario cumulative realized tax cost (translucent) plus the mean
    across scenarios. If `prescient_sol` is provided (a single-scenario sol
    from the prescient run), overlay it as a dashed crimson line for
    comparison. Saves to mpc_t0_cumulative_tax_cost.png.
    """
    paths = _per_scenario_cumulative_tax(sol, n_scenario, n_period)  # (S, T)
    trade_dates = dates[: paths.shape[1]]

    fig, ax = plt.subplots(figsize=(10, 4))
    _plot_paths_with_mean(
        ax, trade_dates, paths, color="steelblue", label_prefix="MPC t=0 plan"
    )

    if prescient_sol is not None:
        prescient_paths = _per_scenario_cumulative_tax(prescient_sol, 1, n_period)
        ax.plot(
            trade_dates,
            prescient_paths[0],
            color="crimson",
            linewidth=2,
            linestyle="--",
            label="prescient",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Tax Cost ($)")
    ax.set_title("Cumulative Realized Tax Cost: MPC t=0 plan vs Prescient")
    ax.xaxis.set_tick_params(rotation=45)
    ax.legend()
    fig.tight_layout()
    plt.savefig("mpc_t0_cumulative_tax_cost.png", dpi=150)
    plt.show()


def plot_mpc_t0_transition_pct(
    sol: dict,
    scenario_prices: list[pd.DataFrame],
    model: pd.DataFrame,
    dates,
    prescient_sol: dict | None = None,
    actual_prices: pd.DataFrame | None = None,
) -> None:
    """
    Plot per-scenario % transition (translucent) plus the mean across scenarios.
    If `prescient_sol` and `actual_prices` are provided, overlay the prescient
    transition trajectory as a dashed crimson line. Saves to
    mpc_t0_transition_pct.png.
    """
    all_tkrs = list(scenario_prices[0].columns)
    weights = _per_scenario_weights(sol, scenario_prices, all_tkrs)
    paths = _per_scenario_transition_pct(weights, all_tkrs, model)  # (S, T)

    fig, ax = plt.subplots(figsize=(10, 4))
    _plot_paths_with_mean(
        ax, dates, paths, color="steelblue", label_prefix="MPC t=0 plan"
    )

    if prescient_sol is not None and actual_prices is not None:
        prescient_tkrs = list(actual_prices.columns)
        prescient_weights = _per_scenario_weights(
            prescient_sol, [actual_prices], prescient_tkrs
        )
        prescient_paths = _per_scenario_transition_pct(
            prescient_weights, prescient_tkrs, model
        )
        ax.plot(
            dates,
            prescient_paths[0],
            color="crimson",
            linewidth=2,
            linestyle="--",
            label="prescient",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("% Transition")
    ax.set_title("Portfolio Transition Progress: MPC t=0 plan vs Prescient")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_tick_params(rotation=45)
    ax.legend()
    fig.tight_layout()
    plt.savefig("mpc_t0_transition_pct.png", dpi=150)
    plt.show()


def plot_mpc_t0_portfolio_value(
    sol: dict,
    scenario_prices: list[pd.DataFrame],
    dates,
    prescient_sol: dict | None = None,
    actual_prices: pd.DataFrame | None = None,
) -> None:
    """
    Plot per-scenario total portfolio value (translucent) plus the mean across
    scenarios. If `prescient_sol` and `actual_prices` are provided, overlay the
    prescient trajectory marked-to-market at realized prices as a dashed
    crimson line. Saves to mpc_t0_portfolio_value.png.
    """
    all_tkrs = list(scenario_prices[0].columns)
    paths = _per_scenario_portfolio_value(sol, scenario_prices, all_tkrs)

    fig, ax = plt.subplots(figsize=(10, 4))
    _plot_paths_with_mean(
        ax, dates, paths, color="steelblue", label_prefix="MPC t=0 plan"
    )

    if prescient_sol is not None and actual_prices is not None:
        prescient_tkrs = list(actual_prices.columns)
        prescient_paths = _per_scenario_portfolio_value(
            prescient_sol, [actual_prices], prescient_tkrs
        )
        ax.plot(
            dates,
            prescient_paths[0],
            color="crimson",
            linewidth=2,
            linestyle="--",
            label="prescient",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Total Portfolio Value: MPC t=0 plan vs Prescient")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax.xaxis.set_tick_params(rotation=45)
    ax.legend()
    fig.tight_layout()
    plt.savefig("mpc_t0_portfolio_value.png", dpi=150)
    plt.show()


def plot_mpc_t0_ticker_weight_and_price(
    sol: dict,
    scenario_prices: list[pd.DataFrame],
    tkr: str,
    dates,
    prescient_sol: dict | None = None,
    actual_prices: pd.DataFrame | None = None,
) -> None:
    """
    Dual-axis plot: per-scenario portfolio weight of `tkr` (translucent + mean)
    on the left axis, per-scenario price of `tkr` (translucent + mean) on the
    right axis. If `prescient_sol`/`actual_prices` are provided, overlay the
    prescient weight (left axis, navy dashed) and the realized actual price
    (right axis, crimson dashed). Saves to mpc_<tkr>_weight_and_price.png.
    """
    all_tkrs = list(scenario_prices[0].columns)
    tkr_idx = all_tkrs.index(tkr)
    weights = _per_scenario_weights(sol, scenario_prices, all_tkrs)  # (S, T, K)
    weight_paths = weights[:, :, tkr_idx]  # (S, T)
    price_paths = np.stack([sp[tkr].values for sp in scenario_prices])  # (S, T)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    _plot_paths_with_mean(
        ax1,
        dates,
        weight_paths,
        color="steelblue",
        label_prefix=f"MPC t=0 plan {tkr} weight",
    )
    if prescient_sol is not None and actual_prices is not None:
        prescient_tkrs = list(actual_prices.columns)
        prescient_weights = _per_scenario_weights(
            prescient_sol, [actual_prices], prescient_tkrs
        )
        ax1.plot(
            dates,
            prescient_weights[0, :, prescient_tkrs.index(tkr)],
            color="navy",
            linewidth=2,
            linestyle="--",
            label=f"prescient {tkr} weight",
        )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Weight", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1.xaxis.set_tick_params(rotation=45)

    ax2 = ax1.twinx()
    _plot_paths_with_mean(
        ax2,
        dates,
        price_paths,
        color="darkorange",
        label_prefix=f"MPC t=0 plan {tkr} price",
    )
    if actual_prices is not None:
        ax2.plot(
            dates,
            actual_prices[tkr].values,
            color="crimson",
            linewidth=2,
            linestyle="--",
            label=f"actual {tkr} price",
        )
    ax2.set_ylabel("Price ($)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.suptitle(f"{tkr} Weight and Price: MPC t=0 plan vs Prescient")
    fig.tight_layout()
    plt.savefig(f"mpc_t0_{tkr}_weight_and_price.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Rolling backtest plotting helpers
# ---------------------------------------------------------------------------


def _backtest_realized_trajectory(
    results: list[dict],
    actual_prices: pd.DataFrame,
) -> dict:
    """
    Extract realized backtest trajectories from a list of per-step Backtester
    results. Each step contributes its first-period decisions (filtration
    (0, 1)) and is marked to market at the actual price observed at the
    decision moment — matching the price convention of plot_portfolio_value
    (`monthly_prices.iloc[f]`).

    Returns a dict with arrays of shape (n_steps,) or (n_steps, n_tkr) on the
    full ticker universe of `actual_prices.columns`:
        - tkrs              : list of tickers (column order from actual_prices)
        - cum_tax           : (n_steps,) cumulative realized tax cost
        - portfolio_value   : (n_steps,) post-trade portfolio value
        - weights           : (n_steps, n_tkr) realized portfolio weights
    """
    tkrs = list(actual_prices.columns)
    n_steps = len(results)
    n_tkr = len(tkrs)

    per_period_tax = np.zeros(n_steps)
    portfolio_value = np.zeros(n_steps)
    weights = np.zeros((n_steps, n_tkr))

    for t, step in enumerate(results):
        f_sol = step["sol"]["filtration"][(0, 1)]
        per_period_tax[t] = f_sol.get("tax_cost", 0.0)
        prices = actual_prices.iloc[t]
        dollars = np.array([f_sol["shr_h"].get(tkr, 0.0) * prices[tkr] for tkr in tkrs])
        total = dollars.sum()
        portfolio_value[t] = total
        if total > 0:
            weights[t] = dollars / total

    return {
        "tkrs": tkrs,
        "cum_tax": np.cumsum(per_period_tax),
        "portfolio_value": portfolio_value,
        "weights": weights,
    }


def plot_backtest_cumulative_tax_cost(
    results: list[dict],
    actual_prices: pd.DataFrame,
    prescient_sol: dict | None = None,
) -> None:
    """
    Plot realized cumulative tax cost of the rolling backtest, optionally
    overlaying the prescient (perfect-foresight) trajectory as a dashed
    crimson line. Saves to backtest_cumulative_tax_cost.png.
    """
    traj = _backtest_realized_trajectory(results, actual_prices)
    n_steps = len(results)
    dates = actual_prices.index[:n_steps]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        dates,
        traj["cum_tax"],
        marker="o",
        color="steelblue",
        linewidth=2,
        label="backtest",
    )

    if prescient_sol is not None:
        prescient_paths = _per_scenario_cumulative_tax(prescient_sol, 1, n_steps)
        ax.plot(
            dates,
            prescient_paths[0],
            color="crimson",
            linewidth=2,
            linestyle="--",
            label="prescient",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Tax Cost ($)")
    ax.set_title("Cumulative Realized Tax Cost: Backtest vs Prescient")
    ax.xaxis.set_tick_params(rotation=45)
    ax.legend()
    fig.tight_layout()
    plt.savefig("backtest_cumulative_tax_cost.png", dpi=150)
    plt.show()


def plot_backtest_portfolio_value(
    results: list[dict],
    actual_prices: pd.DataFrame,
    prescient_sol: dict | None = None,
) -> None:
    """
    Plot realized portfolio value of the rolling backtest, optionally
    overlaying the prescient trajectory marked-to-market at realized prices
    as a dashed crimson line. Saves to backtest_portfolio_value.png.
    """
    traj = _backtest_realized_trajectory(results, actual_prices)
    n_steps = len(results)
    dates = actual_prices.index[:n_steps]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        dates,
        traj["portfolio_value"],
        marker="o",
        color="steelblue",
        linewidth=2,
        label="backtest",
    )

    if prescient_sol is not None:
        prescient_tkrs = list(actual_prices.columns)
        prescient_paths = _per_scenario_portfolio_value(
            prescient_sol,
            [actual_prices.iloc[:n_steps].reset_index(drop=True)],
            prescient_tkrs,
        )
        ax.plot(
            dates,
            prescient_paths[0],
            color="crimson",
            linewidth=2,
            linestyle="--",
            label="prescient",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Total Portfolio Value: Backtest vs Prescient")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax.xaxis.set_tick_params(rotation=45)
    ax.legend()
    fig.tight_layout()
    plt.savefig("backtest_portfolio_value.png", dpi=150)
    plt.show()


def plot_backtest_transition_pct(
    results: list[dict],
    actual_prices: pd.DataFrame,
    model: pd.DataFrame,
    prescient_sol: dict | None = None,
) -> None:
    """
    Plot realized % transition of the rolling backtest, optionally overlaying
    the prescient trajectory as a dashed crimson line. Saves to
    backtest_transition_pct.png.
    """
    traj = _backtest_realized_trajectory(results, actual_prices)
    n_steps = len(results)
    dates = actual_prices.index[:n_steps]
    tkrs = traj["tkrs"]

    tgt_series = model.set_index("tkr")["tgt_wt"]
    tgt_arr = np.array([float(tgt_series.get(tkr, 0.0)) for tkr in tkrs])
    diff = traj["weights"] - tgt_arr
    sum_over = np.maximum(diff, 0.0).sum(axis=1)
    sum_under = np.maximum(-diff, 0.0).sum(axis=1)
    transition_pct = 1.0 - np.maximum(sum_over, sum_under)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        dates,
        transition_pct,
        marker="o",
        color="steelblue",
        linewidth=2,
        label="backtest",
    )

    if prescient_sol is not None:
        prescient_tkrs = list(actual_prices.columns)
        prescient_weights = _per_scenario_weights(
            prescient_sol,
            [actual_prices.iloc[:n_steps].reset_index(drop=True)],
            prescient_tkrs,
        )
        prescient_paths = _per_scenario_transition_pct(
            prescient_weights, prescient_tkrs, model
        )
        ax.plot(
            dates,
            prescient_paths[0],
            color="crimson",
            linewidth=2,
            linestyle="--",
            label="prescient",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("% Transition")
    ax.set_title("Portfolio Transition Progress: Backtest vs Prescient")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_tick_params(rotation=45)
    ax.legend()
    fig.tight_layout()
    plt.savefig("backtest_transition_pct.png", dpi=150)
    plt.show()


def plot_backtest_ticker_weight_and_price(
    results: list[dict],
    actual_prices: pd.DataFrame,
    tkr: str,
    prescient_sol: dict | None = None,
) -> None:
    """
    Dual-axis plot: realized portfolio weight of `tkr` (steelblue, left axis)
    and realized market price (orange dashed, right axis) over the backtest
    window. If `prescient_sol` is provided, overlay the prescient weight on
    the left axis as a navy dashed line. Saves to
    backtest_<tkr>_weight_and_price.png.
    """
    traj = _backtest_realized_trajectory(results, actual_prices)
    n_steps = len(results)
    dates = actual_prices.index[:n_steps]
    tkr_idx = traj["tkrs"].index(tkr)
    weights = traj["weights"][:, tkr_idx]
    prices = actual_prices[tkr].iloc[:n_steps].values

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(
        dates,
        weights,
        marker="o",
        color="steelblue",
        linewidth=2,
        label=f"backtest {tkr} weight",
    )

    if prescient_sol is not None:
        prescient_tkrs = list(actual_prices.columns)
        prescient_weights = _per_scenario_weights(
            prescient_sol,
            [actual_prices.iloc[:n_steps].reset_index(drop=True)],
            prescient_tkrs,
        )
        ax1.plot(
            dates,
            prescient_weights[0, :, prescient_tkrs.index(tkr)],
            color="navy",
            linewidth=2,
            linestyle="--",
            label=f"prescient {tkr} weight",
        )

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
        linewidth=2,
        label=f"actual {tkr} price",
    )
    ax2.set_ylabel("Price ($)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.suptitle(f"{tkr} Weight and Price: Backtest vs Prescient")
    fig.tight_layout()
    plt.savefig(f"backtest_{tkr}_weight_and_price.png", dpi=150)
    plt.show()
