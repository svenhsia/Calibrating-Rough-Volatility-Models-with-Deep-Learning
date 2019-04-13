import logging
import os
import datetime
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from tqdm import tqdm

from py_vollib.black_scholes.implied_volatility import implied_volatility
from rbergomi.rbergomi import rBergomi

tqdm.pandas()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

part_id = 18
part_size = 10000
# part_size = 100


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
    try:
        rB = rBergomi(n=365, N=MC_samples, T=tau, a=H-0.5)
        dW1, dW2 = rB.dW1(), rB.dW2()
        Y = rB.Y(dW1)
        dB = rB.dB(dW1, dW2, rho)
        xi = v0
        V = rB.V(Y, xi, eta)
        S = rB.S(V, dB)
        ST = S[:, -1]
        price = np.mean(np.maximum(ST-K, 0))
    except:
        return np.nan, np.nan

    # check numerical stability
    if price <= 0 or price + K < S0:
        iv = np.nan
        logging.debug("NumStabProblem: Price {}. Intrinsic {}. Time {}. Strike {}.".format(
            price, S0-K, tau, K))
    else:
        logging.debug("Success: Price {} > intrinsic {}".format(price, S0-K))
        iv = implied_volatility(price, S0, K, tau, 0, 'c')
    return price, iv


def generate_rBergomi_sample(K, T, param_generator, S0=1.0):
    counter = 0
    while counter < 10:
        params = param_generator()
        H, eta, rho, v0 = params['H'], params['eta'], params['rho'], params['v0']
        _, iv = rBergomi_pricer(H, eta, rho, v0, T, K, S0)
        if np.isnan(iv):
            counter += 1
        else:
            break
    else:
        logging.warning("Tried 10 times, none valid sample obtained.")
    sample = {
        'H': H,
        'eta': eta,
        'rho': rho,
        'v0': v0,
        'iv': iv
    }
    return sample


def param_generator(H_generator=truncnorm(-1.2, 8.6, 0.07, 0.05),
                    eta_generator=truncnorm(-3, 3, 2.5, 0.5),
                    rho_generator=truncnorm(-0.25, 2.25, -0.95, 0.2),
                    v0_generator=truncnorm(-2.5, 7, 0.3, 0.1)):
    rslt = {
        'H': H_generator.rvs(),
        'eta': eta_generator.rvs(),
        'rho': rho_generator.rvs(),
        'v0': v0_generator.rvs()
    }
    return rslt


logging.info("loading K_T data from file")
logging.info("rows from {} to {}".format(
    part_id*part_size, (part_id+1)*part_size))
K_T = pd.read_csv("./data/strike_maturity.csv",
                  index_col=0).iloc[part_id*part_size:(part_id+1)*part_size, :]

logging.info("K_T shape: {}".format(K_T.shape))

logging.info("K_T head: {}".format(K_T.head()))

# PARAMETERS
n_samples = K_T.shape[0]

# Market params
S0 = 1.

logging.info("start generating data")
data_nn = K_T.merge(K_T.progress_apply(
    lambda row: pd.Series(generate_rBergomi_sample(
        row['Moneyness'], row['Time to Maturity (years)'], param_generator, S0)),
    axis=1), left_index=True, right_index=True)

logging.info("data_nn shape: {}".format(data_nn.shape))

logging.info("data_nn head: {}".format(data_nn.head()))

logging.info("dropping nan")
data_nn.dropna(inplace=True)

logging.info("writing to local file")
data_nn.to_csv(
    "./data/rBergomi/labled_data_all_{}.csv".format(part_id), index=False)
