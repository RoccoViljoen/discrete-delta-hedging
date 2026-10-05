# The checks against Hull and the three experiments: convergence (A),
# the wrong hedge volatility (B) and the gamma breakdown (C).

import numpy as np
import pandas as pd

from src.black_scholes import call_delta, call_gamma, call_price, call_theta
from src.hedging import delta_hedge, pnl_stats, rebalance_dates
from src.simulation import HULL, HULL_N_PATHS, HULL_N_STEPS
from src.simulation import gbm_from_normals, simulate_gbm

# Hull Table 19.4: weeks between rebalancing -> SD of hedging cost / BSM price.
HULL_TABLE_19_4 = {5: 0.42, 4: 0.38, 2: 0.28, 1: 0.21, 0.5: 0.16, 0.25: 0.13}


def r_squared(y, x):
    return pd.Series(y).corr(pd.Series(x)) ** 2


def ols_slope(y, x):
    """Slope and intercept of a straight-line fit of y on x: cov(x, y) / var(x)."""
    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    slope = x.cov(y) / x.var()
    intercept = y.mean() - slope * x.mean()
    return slope, intercept


def hull_examples():
    """Price and Greeks for the option in Hull's Examples 19.1-19.4."""
    S0, K, r, sigma, T = HULL["S0"], HULL["K"], HULL["r"], HULL["sigma"], HULL["T"]
    rows = [{"quantity": "price", "hull": 2.40, "this_repo": call_price(S0, K, r, sigma, T)},
            {"quantity": "delta", "hull": 0.522, "this_repo": call_delta(S0, K, r, sigma, T)},
            {"quantity": "theta", "hull": -4.31, "this_repo": call_theta(S0, K, r, sigma, T)},
            {"quantity": "gamma", "hull": 0.066, "this_repo": call_gamma(S0, K, r, sigma, T)}]
    return pd.DataFrame(rows)


def hull_table_19_4(seed):
    """Redo Hull's Table 19.4 with his parameters, including mu = 13%."""
    S0, K, r, sigma, T = HULL["S0"], HULL["K"], HULL["r"], HULL["sigma"], HULL["T"]
    C0 = call_price(S0, K, r, sigma, T)
    paths = simulate_gbm(S0, HULL["mu"], sigma, T, HULL_N_STEPS, HULL_N_PATHS, seed)

    rows = []
    for weeks in list(HULL_TABLE_19_4.keys()):
        dates = rebalance_dates(HULL_N_STEPS, round(20 / weeks))
        pnl = delta_hedge(paths, K, r, sigma, T, dates, C0)["pnl"]

        # Hull uses the PV of the hedging cost, which is C0 - pnl * exp(-rT).
        measure = pnl.std(ddof=1) * np.exp(-r * T) / C0
        rows.append({"weeks": weeks, "hull": HULL_TABLE_19_4[weeks], "this_repo": measure})
    return pd.DataFrame(rows)


def predicted_sd(paths, K, r, sigma, T):
    """SD of the P&L predicted from the gamma term, without the sqrt(dt).

    The derivation is in the README. The integral is done on the daily grid.
    """
    n_steps = paths.shape[1] - 1
    dt = T / n_steps
    t = np.arange(n_steps) * dt
    S = paths[:, :-1]
    gamma = call_gamma(S, K, r, sigma, T - t)
    integral = (gamma ** 2 * S ** 4 * np.exp(2 * r * (T - t))).sum(axis=1) * dt
    return sigma ** 2 * np.sqrt(0.5 * integral.mean())


def convergence(paths, S0, K, r, sigma, T, frequencies):
    """Experiment A: the same paths hedged at each rebalancing frequency."""
    n_steps = paths.shape[1] - 1
    C0 = call_price(S0, K, r, sigma, T)
    factor = predicted_sd(paths, K, r, sigma, T)

    rows = []
    for f in frequencies:
        res = delta_hedge(paths, K, r, sigma, T, rebalance_dates(n_steps, f), C0)
        row = {"rebalances": f, "dt": T / f}
        row.update(pnl_stats(res["pnl"]))
        row["predicted_sd"] = factor * np.sqrt(T / f)
        row["turnover"] = res["turnover"].mean()
        row["gamma_r2"] = r_squared(res["pnl"], res["gamma_pnl"])
        rows.append(row)
    return pd.DataFrame(rows)


def convergence_slope(table):
    # The once-a-year hedge is left out because it's really a static position.
    sub = table[table["rebalances"] >= 4]
    slope, _ = ols_slope(np.log(sub["sd"]), np.log(sub["dt"]))
    return slope


def misspecification(S0, K, r, T, n_steps, n_paths, sigma_real, sigma_hedge, seed):
    """Experiment B: the real vol is sigma_real but the deltas use sigma_hedge.

    The option is sold at its fair price, so only the hedge ratio is wrong.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    dates = rebalance_dates(n_steps, n_steps)

    rows = []
    for s in sigma_real:
        paths = gbm_from_normals(S0, r, s, T, z)
        fair_price = call_price(S0, K, r, s, T)
        res = delta_hedge(paths, K, r, sigma_hedge, T, dates, fair_price)
        stats = pnl_stats(res["pnl"])

        # What selling at the sigma_hedge price would have cost, grown to T.
        price_gap = (fair_price - call_price(S0, K, r, sigma_hedge, T)) * np.exp(r * T)
        rows.append({"sigma_real": s, "fair_price": fair_price,
                     "mean": stats["mean"], "rmse": stats["rmse"],
                     "price_gap": price_gap,
                     "mean_gamma_pnl": res["gamma_pnl"].mean(),
                     "mean_if_sold_at_hedge_vol": stats["mean"] - price_gap})
    return pd.DataFrame(rows)


def gamma_breakdown(paths, S0, K, r, sigma, T, frequency):
    """Experiment C: how much of the P&L the gamma term explains."""
    n_steps = paths.shape[1] - 1
    C0 = call_price(S0, K, r, sigma, T)
    res = delta_hedge(paths, K, r, sigma, T, rebalance_dates(n_steps, frequency), C0)
    df = pd.DataFrame({"pnl": res["pnl"], "gamma_pnl": res["gamma_pnl"],
                       "gamma_exposure": res["gamma_exposure"]})

    slope, intercept = ols_slope(df["pnl"], df["gamma_pnl"])
    fit = {"r2": r_squared(df["pnl"], df["gamma_pnl"]),
           "slope": slope,
           "intercept": intercept,
           "r2_gross": r_squared(df["pnl"].abs(), df["gamma_exposure"])}

    # The cruder version I tried first: 0.5 * Gamma * dS^2 with no theta part.
    df["quintile"] = pd.qcut(df["gamma_exposure"], 5, labels=False) + 1
    total = (df["pnl"] ** 2).sum()
    rows = []
    for q, group in df.groupby("quintile"):
        rows.append({"quintile": q,
                     "mean_pnl": group["pnl"].mean(),
                     "rmse": np.sqrt((group["pnl"] ** 2).mean()),
                     "share_of_squared_error": (group["pnl"] ** 2).sum() / total})
    return pd.DataFrame(rows), fit, df
