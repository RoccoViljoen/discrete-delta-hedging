# Project settings, and simulation of the stock price paths.
# All fixed choices are defined here so they can be found in one place.

import numpy as np

RESULTS_FOLDER = "results"

S0 = 100.0
K = 100.0
T = 1.0
R = 0.02
SIGMA = 0.20

SEED = 42
N_PATHS = 50_000
N_STEPS = 252

FREQUENCIES = [1, 4, 12, 52, 252]  # rebalances per year
SIGMA_REAL = [0.10, 0.15, 0.20, 0.25, 0.30]
SIGMA_HEDGE = 0.20
GAMMA_FREQUENCY = 52

# Hull's example from chapter 19. His simulation uses the real-world drift mu = 13%.
HULL = {"S0": 49.0, "K": 50.0, "r": 0.05, "sigma": 0.20, "mu": 0.13, "T": 20 / 52}
HULL_N_PATHS = 200_000
HULL_N_STEPS = 80  # quarter weeks, so every interval in Table 19.4 fits exactly


def gbm_from_normals(S0, mu, sigma, T, z):
    """Turn an (n_paths, n_steps) array of N(0, 1) draws into GBM paths.

    Uses the exact lognormal step (Hull, 14.3 and 14.7), so there's no
    discretisation error in the prices.
    """
    if sigma <= 0:
        raise ValueError("sigma has to be positive")
    n_steps = z.shape[1]
    dt = T / n_steps
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(log_returns, axis=1)
    log_paths = np.hstack([np.zeros((z.shape[0], 1)), log_paths])
    return S0 * np.exp(log_paths)


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    return gbm_from_normals(S0, mu, sigma, T, z)
