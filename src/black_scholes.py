# Black-Scholes-Merton price and Greeks for a European call with no dividends
# (Hull 15.8, 19.4-19.6).

import numpy as np
from scipy.stats import norm


def d1_d2(S, K, r, sigma, tau):
    # Hull, equation 15.20
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return d1, d2


def check_inputs(S, sigma, tau):
    if (np.asarray(S) <= 0).any():
        raise ValueError("stock price of zero or below")
    if sigma <= 0:
        raise ValueError("sigma has to be positive")
    if (np.asarray(tau) < 0).any():
        raise ValueError("negative time to expiry")


def scalar_if_one(x):
    if np.ndim(x) == 0:
        return float(x)
    return x


def call_price(S, K, r, sigma, tau):
    """BSM call price (Hull, 15.8). At expiry this is exactly the payoff."""
    check_inputs(S, sigma, tau)
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)

    # tau is swapped for 1 at expiry only so the formula doesn't divide by
    # zero. Those values are thrown away and replaced by the payoff.
    safe_tau = np.where(tau > 0, tau, 1.0)
    d1, d2 = d1_d2(S, K, r, sigma, safe_tau)
    price = S * norm.cdf(d1) - K * np.exp(-r * safe_tau) * norm.cdf(d2)
    return scalar_if_one(np.where(tau > 0, price, np.maximum(S - K, 0)))


def call_delta(S, K, r, sigma, tau):
    """N(d1). At expiry it's 0 below the strike, 1 above, and 0.5 at it by convention."""
    check_inputs(S, sigma, tau)
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    safe_tau = np.where(tau > 0, tau, 1.0)
    d1, _ = d1_d2(S, K, r, sigma, safe_tau)
    at_expiry = 0.5 * (np.sign(S - K) + 1)
    return scalar_if_one(np.where(tau > 0, norm.cdf(d1), at_expiry))


def call_gamma(S, K, r, sigma, tau):
    # Gamma at the strike blows up as tau -> 0, so it is only defined for tau > 0.
    check_inputs(S, sigma, tau)
    if (np.asarray(tau) <= 0).any():
        raise ValueError("gamma needs tau > 0")
    d1, _ = d1_d2(S, K, r, sigma, tau)
    return norm.pdf(d1) / (S * sigma * np.sqrt(tau))


def call_theta(S, K, r, sigma, tau):
    """Theta per year (Hull, 19.5). Only defined for tau > 0, like gamma."""
    check_inputs(S, sigma, tau)
    if (np.asarray(tau) <= 0).any():
        raise ValueError("theta needs tau > 0")
    d1, d2 = d1_d2(S, K, r, sigma, tau)
    decay = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(tau))
    return decay - r * K * np.exp(-r * tau) * norm.cdf(d2)
