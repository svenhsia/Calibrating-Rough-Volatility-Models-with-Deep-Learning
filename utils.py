import os
import datetime
import logging
import numpy as np
import QuantLib as ql
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_lets_be_rational.exceptions import BelowIntrinsicException

from rbergomi.rbergomi import rBergomi

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def heston_pricer(lambd, vbar, eta, rho, v0, r, q, tau, S0, K):
    """Computes European Call price under Heston dynamics with closedform solution.
    
    Parameters:
    -----------
        lambd: mean-reversion speed
        vbar: long-term average variance
        eta: volatility of variance
        rho: correlation between stock and vol
        v0: spot variance
        r: risk-free interest rate
        q: dividend rate
        tau: time to maturity in years (365 trading days per year)
        S0: initial spot price
        K: strike price
    """
    today = datetime.date.today()
    ql_date = ql.Date(today.day, today.month, today.year)
    day_count = ql.Actual365Fixed()
    ql.Settings.instance().evaluationDate = ql_date
    
    # option data
    option_type = ql.Option.Call
    payoff = ql.PlainVanillaPayoff(option_type, K)
    maturity_date = ql_date + int(round(tau * 365))
    exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, exercise)
    
    # Heston process
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, r, day_count))
    dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, q, day_count))
    heston_process = ql.HestonProcess(flat_ts, dividend_yield, spot_handle, v0, lambd, vbar, eta, rho)
    
    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process), 1e-15, int(1e6))
    european_option.setPricingEngine(engine)
    
    # check numerical stability
    try:
        price = european_option.NPV()
        if price <= 0 or price + K < S0:
            iv = np.nan
            logging.debug("NumStabProblem: Price {}. Intrinsic {}. Time {}. Strike {}.".format(price, S0-K, tau, K))
        else:
            logging.debug("Success: Price {} > intrinsic {}".format(price, S0-K))
            iv = implied_volatility(price, S0, K, tau, r, 'c')
    except RuntimeError:
        logging.info("RuntimeError: Intrinsic {}. Time {}. Strike {}.".format(S0-K, tau, K))
        price = np.nan
        iv = np.nan
    return price, iv


def rBergomi_pricer(H, eta, rho, v0, tau, K, S0, MC_samples=40000):
    """Computes European Call price under rBergomi dynamics with MC sampling.
    
    Parameters:
    -----------
        H: Hurst parameter
        eta: volatility of variance
        rho: correlation between stock and vol
        v0: spot variance
        tau: time to maturity in years (365 trading days per year)
        K: strike price
    """
    rB = rBergomi(n=365, N=MC_samples, T=tau, a=H-0.5)
    dW1, dW2 = rB.dW1(), rB.dW2()
    Y = rB.Y(dW1)
    dB = rB.dB(dW1, dW2, rho)
    xi = v0
    V = rB.V(Y, xi, eta)
    S = rB.S(V, dB)
    ST = S[:, -1]
    price = np.mean(np.maximum(ST-K, 0))
    
    # check numerical stability
    if price <= 0 or price + K < S0:
        iv = np.nan
        logging.debug("NumStabProblem: Price {}. Intrinsic {}. Time {}. Strike {}.".format(price, S0-K, tau, K))
    else:
        logging.debug("Success: Price {} > intrinsic {}".format(price, S0-K))
        iv = implied_volatility(price, S0, K, tau, 0, 'c')
    return price, iv
