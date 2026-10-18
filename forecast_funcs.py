from SHARV_class import *
from arch import arch_model
import pandas as pd
import numpy as np
from LSTM_GARCH import *

def garch_forecast(data, model_res, step=1):
    """
    The arch package has built-in forecast function. Only need this because we do not want to update parameter estimate
    for every loop to save computation cost
    :param data:
    :param params:
    :return:
    """
    beta = model_res.params['beta[1]']
    omega = model_res.params['omega']
    alpha = model_res.params['alpha[1]']

    var = np.zeros(step)
    var[0] = omega + beta * model_res.conditional_volatility[-1] ** 2 + alpha * data[-1] ** 2
    if step > 1:
        for i in range(1, step):
            var[i] = omega + (beta + alpha) * var[i - 1]

    return np.sqrt(var)

def garch_forecast_var(data, model_res, q, parametric):
    """
    One-step ahead Value-at-Risk forecast under GARCH
    :param data:
    :param params:
    :oaram q: Quantile level for VaR
    :return:
    """
    beta = model_res.params['beta[1]']
    omega = model_res.params['omega']
    alpha = model_res.params['alpha[1]']

    var = omega + beta * model_res.conditional_volatility[-1] ** 2 + alpha * data[-1] ** 2
    std_res = model_res.std_resid
    if parametric:
        quant = sp.stats.norm.ppf(q)
    else:
        quant = np.quantile(std_res, q)

    VaR = np.sqrt(var) * quant
    return VaR

def out_of_sample(data, step=1, update=50, train_test_split=0.9, model='SHARV', **kwargs):
    """
    Out of sample volatility forecast for SHARV, ASHARV and GARCH
    :param data: pass Series for easy comparison with realized volatility with dates
    :param step: forecast step
    :param update: how many step do you want to re-estimate the model.
    :param train_test_split: percentage of training data
    :param model:
    :return:
    """
    if model == 'ASHARV':
        asymmetry = True
    else:
        asymmetry = False

    dates = data.index
    data = data.values
    train_len = int(len(data) * train_test_split)
    forecast = []
    for i in range(train_len, len(data) - update - step, update):
        # Only update the parameter estimation every update-th time
        train_data = data[:i]
        if model == 'GARCH':
            model_res = arch_model(train_data, mean="Zero", p=1, o=0, q=1).fit(disp=0)
        elif model == 'LSTM':
            hidden_dim_val = kwargs.get('hidden_dim', 16)
            lr_val = kwargs.get('lr', 0.005)
            epochs_val = kwargs.get('epochs', 100)
            model_res = VolatilityModel(hidden_dim=hidden_dim_val, lr=lr_val)
            model_res.fit(train_data, epochs=epochs_val)
        else:
            pars = Sharv(train_data, asymmetry=asymmetry).fit().params

        for j in range(i, min(i + update, len(data) - step)):
            # Use the same parameter to forecast when new data comes in until the next update round
            temp_date = dates[j:j + step]
            if model == 'GARCH':
                temp_model = arch_model(data[:j], mean="Zero", p=1, o=0, q=1).fix(model_res.params)
                temp_forecast = garch_forecast(data[:j], temp_model, step=step)
            elif model == 'LSTM':
                temp_forecast = model_res.forecast(data[:j], horizon=step).flatten()
            else:
                temp_forecast = Sharv(data[:j], asymmetry=asymmetry).vol_forecast(pars, step=step)

            temp = pd.DataFrame(temp_forecast, index=temp_date, columns=['Forecast'])
            forecast.append(temp)

    return forecast


import scipy as sp  # Required for sp.stats.norm.ppf


def out_of_sample_var(data, update=50, train_test_split=0.9, model='SHARV', q=0.05, parametric=True, **kwargs):
    """
    Out of sample Value-at-Risk forecast for SHARV, ASHARV, GARCH, and LSTM-GARCH
    """
    if model == 'ASHARV':
        asymmetry = True
    else:
        asymmetry = False

    dates = data.index
    data_values = data.values  # Convert pandas Series to numpy array safely
    train_len = int(len(data_values) * train_test_split)
    forecast = []
    indices = []

    # Extract keys if given by user for LSTM, otherwise fall back to explicit pipeline defaults
    hidden_dim_val = kwargs.get('hidden_dim', 16)
    lr_val = kwargs.get('lr', 0.005)
    epochs_val = kwargs.get('epochs', 100)

    for i in range(train_len, len(data_values) - update - 1, update):
        train_data = data_values[:i]

        if model == 'GARCH':
            model_res = arch_model(train_data, mean="Zero", p=1, o=0, q=1).fit(disp=0)
        elif model == 'LSTM':
            model_res = VolatilityModel(hidden_dim=hidden_dim_val, lr=lr_val)
            model_res.fit(train_data, epochs=epochs_val)
        else:
            temp_sharv = Sharv(train_data, asymmetry=asymmetry).fit()
            pars = temp_sharv.params

        for j in range(i, min(i + update, len(data_values) - 1)):
            temp_date = dates[j:j + 1]

            if model == 'GARCH':
                temp_model = arch_model(data_values[:j], mean="Zero", p=1, o=0, q=1).fix(model_res.params)
                temp_forecast = garch_forecast_var(data_values[:j], temp_model, q, parametric=parametric)
            elif model == 'LSTM':
                vol_forecast = model_res.forecast(data_values[:j], horizon=1).flatten()
                mu = model_res.mean
                if parametric:
                    z_q = sp.stats.norm.ppf(q)
                    temp_forecast = mu + z_q * vol_forecast
                else:
                    std_res = model_res.standardised_residuals(data_values[:j])
                    empirical_quantile = np.quantile(std_res, q)
                    temp_forecast = mu + empirical_quantile * vol_forecast
            else:
                # Returns a plain scalar number
                temp_forecast = Sharv(data_values[:j], asymmetry=asymmetry).VaR_forecast(pars, q, parametric=parametric)

            # Force whatever output came from the models above to be a clean, flat 1D sequence.
            # This handles numpy arrays, lists, and plain scalar floats perfectly.
            temp_forecast = np.atleast_1d(temp_forecast).flatten()

            # VaR violation checker now compares safely across consistent formats
            if data_values[j] < temp_forecast[0]:
                indices.append(1)
            else:
                indices.append(0)

            temp_df = pd.DataFrame(temp_forecast, index=temp_date, columns=['Forecast'])
            forecast.append(temp_df)

    n = len(indices)
    n00 = np.zeros(n)
    n01 = np.zeros(n)
    n10 = np.zeros(n)
    n11 = np.zeros(n)
    for j in range(1, n):
        if indices[j] == 1 and indices[j - 1] == 1:
            n11[j] = 1
        if indices[j] == 0 and indices[j - 1] == 1:
            n10[j] = 1
        if indices[j] == 1 and indices[j - 1] == 0:
            n01[j] = 1
        if indices[j] == 0 and indices[j - 1] == 0:
            n00[j] = 1

    # Unconditional coverage
    p1 = (sum(n01) + sum(n11)) / (sum(n11) + sum(n00) + sum(n01) + sum(n10))
    N1 = sum(n01) + sum(n11)
    N0 = sum(n00) + sum(n10)
    p0 = 1 - p1
    UC = sum(indices) / n
    LR_UC = 2 * (np.log(UC ** (sum(indices)) * (1 - UC) ** (n - sum(indices))) -
                 np.log(q ** sum(indices) * (1 - q) ** (n - sum(indices))))

    VR = sum(indices) / (q * n)

    # Conditional coverage and independence
    p00 = sum(n00) / (sum(n00) + sum(n01))
    p01 = sum(n01) / (sum(n00) + sum(n01))
    p10 = sum(n10) / (sum(n10) + sum(n11))
    p11 = sum(n11) / (sum(n10) + sum(n11))
    LR_CCI = 2 * (np.log(p00 ** (sum(n00)) * p01 ** (sum(n01)) * p10 ** (sum(n10)) * p11 ** (sum(n11))) -
                  np.log(p0 ** N0 * p1 ** N1))
    LR_CC = LR_UC + LR_CCI

    return {'VaR forecast': forecast, 'Conditional coverage': LR_CC, 'Unconditional coverage': LR_UC, 'Violation ratio':
        VR, 'p-value': sp.stats.chi2.sf(LR_CC, df=2)}

