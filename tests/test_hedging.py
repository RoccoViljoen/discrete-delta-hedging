# Tests for the hedge: the dates, the cash account by hand, and the size of the error.

import numpy as np
import pandas as pd
import pytest

from src.black_scholes import call_price
from src.experiments import ols_slope, predicted_sd
from src.hedging import delta_hedge, rebalance_dates
from src.simulation import simulate_gbm

K, R, SIGMA, T = 100.0, 0.02, 0.2, 1.0


# Hedge dates

def test_weekly_dates_on_daily_grid():
    dates = rebalance_dates(252, 52)
    assert len(dates) == 52
    assert dates[0] == 0
    assert dates[-1] < 252
    gaps = dates[1:] - dates[:-1]
    assert set(gaps) <= {4, 5}


def test_bad_dates_raise():
    paths = np.array([[100.0, 110.0, 105.0]])
    with pytest.raises(ValueError):
        delta_hedge(paths, K, R, SIGMA, 1.0, [1], 10.0)
    with pytest.raises(ValueError):
        delta_hedge(paths, K, R, SIGMA, 1.0, [0, 2], 10.0)
    with pytest.raises(ValueError):
        rebalance_dates(252, 0)


# Cash account by hand

def test_one_date_hedge_by_hand():
    # r = 0, one hedge, path 100 -> 120: 7.965567 - 53.9828 + 0.539828 * 120 - 20
    res = delta_hedge(np.array([[100.0, 120.0]]), K, 0.0, SIGMA, 1.0, [0], 7.965567455405804)
    assert np.isclose(res["pnl"][0], -1.2378757990536187)


def test_two_date_cash_account_by_hand():
    # r = 5%, hedges at t = 0 and 0.5, path 100 -> 110 -> 105. Worked by hand:
    # cash = (10.450584 - 63.6831) * e^0.025 - 0.184757 * 110 = -74.903329,
    # P&L = -74.903329 * e^0.025 + 0.821588 * 105 - 5 = 4.467179
    res = delta_hedge(np.array([[100.0, 110.0, 105.0]]), K, 0.05, SIGMA, 1.0, [0, 1],
                      10.450583572185565)
    assert np.isclose(res["pnl"][0], 4.467178764797765)
    assert np.isclose(res["turnover"][0], 0.6368306511756191 + 0.1847569154899166)


# Straight-line fit

def test_ols_slope_by_hand():
    x = np.array([0.0, 1.0, 2.0, 5.0])
    slope, intercept = ols_slope(2 + 3 * x, x)
    assert np.isclose(slope, 3)
    assert np.isclose(intercept, 2)


# Behaviour of the error

def test_more_rebalancing_less_error():
    paths = simulate_gbm(100.0, R, SIGMA, T, 252, 5000, seed=3)
    C0 = call_price(100.0, K, R, SIGMA, T)
    sd = {}
    for f in [12, 52, 252]:
        sd[f] = delta_hedge(paths, K, R, SIGMA, T, rebalance_dates(252, f), C0)["pnl"].std()
    assert sd[252] < sd[52] < sd[12]

    # sqrt(252 / 52) = 2.2
    assert 1.8 < sd[52] / sd[252] < 2.6


def test_gamma_term_predicts_pnl():
    paths = simulate_gbm(100.0, R, SIGMA, T, 252, 5000, seed=5)
    C0 = call_price(100.0, K, R, SIGMA, T)
    res = delta_hedge(paths, K, R, SIGMA, T, rebalance_dates(252, 52), C0)
    slope, _ = ols_slope(res["pnl"], res["gamma_pnl"])
    assert pd.Series(res["pnl"]).corr(pd.Series(res["gamma_pnl"])) > 0.95
    assert abs(slope - 1) < 0.05


def test_predicted_sd_matches_simulation():
    paths = simulate_gbm(100.0, R, SIGMA, T, 252, 5000, seed=9)
    C0 = call_price(100.0, K, R, SIGMA, T)
    sd = delta_hedge(paths, K, R, SIGMA, T, rebalance_dates(252, 52), C0)["pnl"].std()
    predicted = predicted_sd(paths, K, R, SIGMA, T) * np.sqrt(T / 52)
    assert abs(sd / predicted - 1) < 0.05
