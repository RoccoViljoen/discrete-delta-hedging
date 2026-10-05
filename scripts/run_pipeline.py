# Runs the full study:  python -m scripts.run_pipeline
# Everything is simulated, so there's no data to download.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.black_scholes import call_price
from src.experiments import HULL_TABLE_19_4, convergence, convergence_slope
from src.experiments import gamma_breakdown, hull_examples, hull_table_19_4, misspecification
from src.simulation import FREQUENCIES, GAMMA_FREQUENCY, K, N_PATHS, N_STEPS, R
from src.simulation import RESULTS_FOLDER, S0, SEED, SIGMA, SIGMA_HEDGE, SIGMA_REAL, T
from src.simulation import simulate_gbm

pd.options.display.max_columns = 20
pd.options.display.width = 120


def table_rows(experiment, table, key, columns):
    """Rows of (experiment, item, value) for summary.csv."""
    rows = []
    for i in range(len(table)):
        label = str(table[key].iloc[i])
        for column in columns:
            rows.append({"experiment": experiment, "item": column + "_" + label,
                         "value": table[column].iloc[i]})
    return rows


def run_hull_checks():
    examples = hull_examples()
    table = hull_table_19_4(SEED)
    print(examples.round(4))
    print(table.round(3))

    rows = table_rows("hull_examples", examples, "quantity", ["hull", "this_repo"])
    rows = rows + table_rows("hull_table_19_4", table, "weeks", ["hull", "this_repo"])
    return table, rows


def run_convergence(paths):
    table = convergence(paths, S0, K, R, SIGMA, T, FREQUENCIES)
    slope = convergence_slope(table)
    print(table.round(3))
    print("Slope of log(SD) on log(dt):", round(slope, 3))

    rows = table_rows("convergence", table, "rebalances",
                      ["mean", "sd", "rmse", "q05", "q95", "predicted_sd",
                       "turnover", "gamma_r2"])
    rows.append({"experiment": "convergence", "item": "slope", "value": slope})
    return table, slope, rows


def run_misspecification():
    table = misspecification(S0, K, R, T, N_STEPS, N_PATHS, SIGMA_REAL, SIGMA_HEDGE, SEED)
    print(table.round(3))
    return table_rows("misspecification", table, "sigma_real",
                      ["fair_price", "mean", "rmse", "price_gap", "mean_gamma_pnl",
                       "mean_if_sold_at_hedge_vol"])


def run_gamma(paths):
    quintiles, fit, df = gamma_breakdown(paths, S0, K, R, SIGMA, T, GAMMA_FREQUENCY)
    print(quintiles.round(3))
    print("Weekly hedge | R^2:", round(fit["r2"], 3), "| slope:", round(fit["slope"], 3),
          "| intercept:", round(fit["intercept"], 3), "| gross R^2:", round(fit["r2_gross"], 3))

    rows = []
    for key in list(fit.keys()):
        rows.append({"experiment": "gamma", "item": key, "value": fit[key]})
    return quintiles, df, fit, rows


def make_figures(table, slope, hull_table, gamma_df, fit, folder):
    figures = folder + "/figures"
    os.makedirs(figures, exist_ok=True)

    # log(SD) against log(dt) is the regression the slope comes from.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    log_dt = np.log(table["dt"])
    axes[0].plot(log_dt, np.log(table["sd"]), "o-", label="Simulated")
    axes[0].plot(log_dt, np.log(table["predicted_sd"]), "k--", label="Predicted from the gamma term")
    axes[0].set_xticks(log_dt)
    axes[0].set_xticklabels(["yearly", "quarterly", "monthly", "weekly", "daily"])
    axes[0].set_title("S0 = K = 100, 1 year (slope " + str(round(slope, 2)) + ")")
    axes[0].set_xlabel("Rebalancing frequency (log(dt) scale)")
    axes[0].set_ylabel("log(SD of P&L at expiry)")
    axes[0].legend()

    weeks = list(HULL_TABLE_19_4.keys())
    axes[1].plot(weeks, hull_table["hull"], "s-", color="grey", label="Hull Table 19.4")
    axes[1].plot(weeks, hull_table["this_repo"], "o-", label="This repo, 200,000 paths")
    axes[1].set_title("Hull's example: S0 = 49, K = 50, 20 weeks")
    axes[1].set_xlabel("Weeks between rebalances")
    axes[1].set_ylabel("SD of hedging cost / BSM price")
    axes[1].legend()
    fig.savefig(figures + "/convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 50,000 points is too many to see, so only the first 5,000 are plotted.
    sample = gamma_df.iloc[:5000]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sample["gamma_pnl"], sample["pnl"], alpha=0.2)
    low = min(sample["gamma_pnl"].min(), sample["pnl"].min())
    high = max(sample["gamma_pnl"].max(), sample["pnl"].max())
    ax.plot([low, high], [low, high], "k--", label="P&L = prediction")
    ax.set_title("Weekly hedge (R^2 = " + str(round(fit["r2"], 2)) + ", slope = "
                 + str(round(fit["slope"], 2)) + ")")
    ax.set_xlabel("Predicted from the gamma term")
    ax.set_ylabel("Simulated P&L at expiry")
    ax.legend()
    fig.savefig(figures + "/gamma_prediction.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(folder):
    os.makedirs(folder, exist_ok=True)
    print("BSM price:", round(call_price(S0, K, R, SIGMA, T), 4))

    hull_table, summary = run_hull_checks()

    # Experiments A and C use the same paths.
    paths = simulate_gbm(S0, R, SIGMA, T, N_STEPS, N_PATHS, SEED)
    table, slope, rows = run_convergence(paths)
    summary = summary + rows
    summary = summary + run_misspecification()
    quintiles, gamma_df, fit, rows = run_gamma(paths)
    summary = summary + rows

    pd.DataFrame(summary).to_csv(folder + "/summary.csv", index=False)
    quintiles.to_csv(folder + "/gamma_quintiles.csv", index=False)
    make_figures(table, slope, hull_table, gamma_df, fit, folder)
    print("Saved to", folder)


def main():
    run(RESULTS_FOLDER)


if __name__ == "__main__":
    main()
