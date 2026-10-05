# Tests for the GBM simulation.

import numpy as np

from src.black_scholes import call_price
from src.simulation import simulate_gbm


def test_same_seed_same_paths():
    a = simulate_gbm(100.0, 0.05, 0.3, 1.0, 50, 1000, seed=7)
    b = simulate_gbm(100.0, 0.05, 0.3, 1.0, 50, 1000, seed=7)
    assert a.shape == (1000, 51)
    assert (a == b).all()
    assert (a > 0).all()
    assert (a[:, 0] == 100.0).all()


def test_log_returns_match_hull_14_7():
    # ln(S_T / S0) should be normal with mean (mu - sigma^2 / 2) T and variance sigma^2 T.
    mu, sigma, T, n = 0.10, 0.30, 2.0, 200_000
    log_return = np.log(simulate_gbm(100.0, mu, sigma, T, 10, n, seed=1)[:, -1] / 100.0)
    se = sigma * np.sqrt(T / n)
    assert abs(log_return.mean() - (mu - 0.5 * sigma ** 2) * T) < 4 * se
    assert abs(log_return.var() / (sigma ** 2 * T) - 1) < 0.01


def test_monte_carlo_price_matches_bsm():
    # With mu = r the discounted mean payoff should be the BSM price.
    S0, K, r, sigma, T, n = 100.0, 100.0, 0.02, 0.2, 1.0, 200_000
    S_T = simulate_gbm(S0, r, sigma, T, 1, n, seed=11)[:, -1]
    payoff = np.exp(-r * T) * np.maximum(S_T - K, 0)
    se = payoff.std(ddof=1) / np.sqrt(n)
    assert abs(payoff.mean() - call_price(S0, K, r, sigma, T)) < 3 * se
