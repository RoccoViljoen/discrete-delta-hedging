# Discrete delta hedging of a short European call (Hull, 19.4). The hedge is
# self-financing: every share trade is paid for from a cash account earning r.

import numpy as np
import pandas as pd

from src.black_scholes import call_delta, call_gamma


def rebalance_dates(n_steps, n_rebalances):
    """Positions of the hedge dates on the simulation grid, starting at 0."""
    if n_rebalances < 1 or n_rebalances > n_steps:
        raise ValueError("n_rebalances has to be between 1 and n_steps")

    # 52 doesn't divide 252, so weekly hedges end up 4 or 5 days apart.
    dates = [round(i * n_steps / n_rebalances) for i in range(n_rebalances)]
    return np.unique(dates)


def delta_hedge(paths, K, r, sigma, T, dates, C0, keep_deltas=False):
    """Sell the call for C0 and delta hedge it on every path. No trade at expiry.

    gamma_pnl is what Hull's equations 19.3 and 19.4 predict the P&L to be:
    each interval adds about -0.5 * Gamma * (dS^2 - sigma^2 * S^2 * h).
    """
    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1
    dates = list(dates)
    if dates[0] != 0:
        raise ValueError("the first hedge has to be at t = 0")
    if dates[-1] >= n_steps:
        raise ValueError("hedge dates have to be before expiry")
    dt = T / n_steps
    dates = dates + [n_steps]  # expiry is the end of the last interval

    S = paths[:, 0]
    delta = call_delta(S, K, r, sigma, T)
    cash = C0 - delta * S
    turnover = np.abs(delta)
    deltas = [delta]
    gamma_pnl = np.zeros(n_paths)
    gamma_exposure = np.zeros(n_paths)

    for j in range(len(dates) - 1):
        t = dates[j] * dt
        t_next = dates[j + 1] * dt
        S = paths[:, dates[j]]
        S_next = paths[:, dates[j + 1]]

        if j > 0:
            new_delta = call_delta(S, K, r, sigma, T - t)
            cash = cash - (new_delta - delta) * S
            turnover = turnover + np.abs(new_delta - delta)
            delta = new_delta
            if keep_deltas:
                deltas.append(delta)

        # Grown to expiry so it's in the same units as the P&L.
        gamma = call_gamma(S, K, r, sigma, T - t)
        dS = S_next - S
        h = t_next - t
        term = -0.5 * gamma * (dS ** 2 - sigma ** 2 * S ** 2 * h)
        gamma_pnl = gamma_pnl + term * np.exp(r * (T - t_next))
        gamma_exposure = gamma_exposure + 0.5 * gamma * dS ** 2

        cash = cash * np.exp(r * h)

    S_T = paths[:, -1]
    pnl = cash + delta * S_T - np.maximum(S_T - K, 0)

    result = {"pnl": pnl, "gamma_pnl": gamma_pnl, "gamma_exposure": gamma_exposure,
              "turnover": turnover, "times": np.array(dates[:-1]) * dt}
    if keep_deltas:
        result["deltas"] = np.column_stack(deltas)
    return result


def pnl_stats(pnl):
    # RMSE is measured around 0, the P&L of a perfect hedge, so bias counts too.
    pnl = pd.Series(pnl)
    return {"mean": pnl.mean(),
            "sd": pnl.std(),
            "rmse": np.sqrt((pnl ** 2).mean()),
            "q05": pnl.quantile(0.05),
            "q95": pnl.quantile(0.95)}
