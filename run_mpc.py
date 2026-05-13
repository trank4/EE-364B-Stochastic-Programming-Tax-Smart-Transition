import pickle

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

OUTPUT_PKL = "mpc_output.pkl"

if __name__ == "__main__":
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

    inputs = {
        "seed": 0,
        "start_date": pd.to_datetime("2024-01-01"),
        "sim_start_date": pd.to_datetime("2023-01-01"),
        "sim_end_date": pd.to_datetime("2024-01-01"),
        "n_period": 12,
        "n_scenario": 10,
        "block_length": 3,
        "tax_rate": tax_rate,
        "model": model,
        "positions": positions,
        "tkr_adev": 0.05,
    }

    controller = RMPController(inputs)
    controller.build_price_scenarios()
    sol = controller.solve()

    # persist everything analyze_mpc.py needs so we don't have to re-solve
    # (also includes the realized 2024 prices and the optimizer config so the
    # prescient benchmark can be re-run inside analyze_mpc.py)
    output = {
        "sol": sol,
        "scenario_prices": controller.scenario_prices,
        "model": model,
        "positions": positions,
        "start_date": inputs["start_date"],
        "n_period": inputs["n_period"],
        "n_scenario": inputs["n_scenario"],
        "all_tkrs": controller.all_tkrs,
        "actual_prices": monthly_prices,
        "tax_rate": tax_rate,
        "tkr_adev": inputs["tkr_adev"],
    }
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(output, f)
    print(f"\nSaved MPC output to {OUTPUT_PKL}")
