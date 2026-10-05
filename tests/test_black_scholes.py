# Tests for the price and Greeks: Hull's worked examples, finite differences,
# and Hull's equation 19.4.

import numpy as np
import pytest

from src.black_scholes import call_delta, call_gamma, call_price, call_theta

# Hull Examples 19.1-19.4: S0 = 49, K = 50, r = 5%, sigma = 20%, 20 weeks.
HULL = (49.0, 50.0, 0.05, 0.20, 20 / 52)

S = np.array([80.0, 95.0, 100.0, 105.0, 120.0])
K, R, SIGMA, TAU = 100.0, 0.02, 0.20, 0.5


# Hull's examples

def test_hull_price_example():
    assert np.isclose(call_price(*HULL), 2.40, atol=0.005)


def test_hull_delta_example():
    assert np.isclose(call_delta(*HULL), 0.522, atol=0.001)


def test_hull_theta_example():
    assert np.isclose(call_theta(*HULL), -4.31, atol=0.01)


def test_hull_gamma_example():
    assert np.isclose(call_gamma(*HULL), 0.066, atol=0.001)


# Finite differences

def test_delta_matches_finite_difference():
    h = 1e-4
    fd = (call_price(S + h, K, R, SIGMA, TAU) - call_price(S - h, K, R, SIGMA, TAU)) / (2 * h)
    assert np.isclose(call_delta(S, K, R, SIGMA, TAU), fd, atol=1e-6).all()


def test_gamma_matches_finite_difference():
    h = 1e-4
    fd = (call_delta(S + h, K, R, SIGMA, TAU) - call_delta(S - h, K, R, SIGMA, TAU)) / (2 * h)
    assert np.isclose(call_gamma(S, K, R, SIGMA, TAU), fd, atol=1e-6).all()


def test_theta_matches_finite_difference():
    # Theta is dC/dt. Time going forward means tau going down, hence the minus.
    h = 1e-4
    fd = -(call_price(S, K, R, SIGMA, TAU + h) - call_price(S, K, R, SIGMA, TAU - h)) / (2 * h)
    assert np.isclose(call_theta(S, K, R, SIGMA, TAU), fd, atol=1e-5).all()


# Identities and edge cases

def test_hull_equation_19_4():
    # Theta + r S Delta + 0.5 sigma^2 S^2 Gamma = r C
    lhs = (call_theta(S, K, R, SIGMA, TAU)
           + R * S * call_delta(S, K, R, SIGMA, TAU)
           + 0.5 * SIGMA ** 2 * S ** 2 * call_gamma(S, K, R, SIGMA, TAU))
    assert np.isclose(lhs, R * call_price(S, K, R, SIGMA, TAU), atol=1e-10).all()


def test_price_bounds():
    S_grid = np.linspace(50, 150, 101)
    price = call_price(S_grid, K, R, SIGMA, 1.0)
    delta = call_delta(S_grid, K, R, SIGMA, 1.0)
    assert (price >= np.maximum(S_grid - K * np.exp(-R), 0) - 1e-12).all()
    assert (price <= S_grid).all()
    assert ((delta > 0) & (delta < 1)).all()
    assert (call_gamma(S_grid, K, R, SIGMA, 1.0) > 0).all()


def test_expiry_gives_payoff():
    S_end = np.array([90.0, 100.0, 110.0])
    assert (call_price(S_end, K, R, SIGMA, 0.0) == [0.0, 0.0, 10.0]).all()
    assert (call_delta(S_end, K, R, SIGMA, 0.0) == [0.0, 0.5, 1.0]).all()


def test_gamma_and_theta_need_positive_tau():
    with pytest.raises(ValueError):
        call_gamma(100.0, K, R, SIGMA, 0.0)
    with pytest.raises(ValueError):
        call_theta(100.0, K, R, SIGMA, 0.0)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        call_price(100.0, K, R, 0.0, 1.0)
    with pytest.raises(ValueError):
        call_price(100.0, K, R, SIGMA, -0.1)
    with pytest.raises(ValueError):
        call_price(np.array([100.0, 0.0]), K, R, SIGMA, 1.0)
