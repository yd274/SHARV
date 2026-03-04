import numpy as np
import pandas as pd
import scipy as sp
from SHARV_class import *
from arch import arch_model

def pdf_sharv(par, data, model='SHARV'):
    """
    This function demonstrate the conditional density of SHARV, SHARV or GARCH. Since the conditional density depends
    on previous volatility value, let's assume that the model starts at steady-state, i.e., volatility starts at its
    unconditional mean.
    :param par:
    :param data:
    :param model:
    :return:
    """
    if model == 'SHARV':
        beta, omega1, gamma1 = par[0], par[1], par[2]
        mu, omega2, gamma2 = 0.0, 0.0, 0.0
    elif model == 'ASHARV':
        mu, beta, omega1, gamma1, omega2, gamma2 = par[0], par[1], par[2], par[3], par[4], par[5]
    elif model == 'GARCH':
        omega, beta, alpha = par[0], par[1], par[2]

    if model == 'SHARV' or model == 'ASHARV':
        sigma_0 = (omega1 + 0.5 * omega2 ) / (1 - beta - gamma1 - 0.5 * gamma2)
        data = data - mu * np.sqrt(sigma_0)
        indicator = 1 if data <= 0 else 0
        a_t = omega1 + gamma1 * sigma_0 + gamma2 * indicator * sigma_0 + omega2 * indicator
        b_t = beta * sigma_0
        d_t1 = np.sqrt(b_t * b_t + 4 * a_t * data * data)
        d_t2 = np.sign(data) * np.sqrt(d_t1 - b_t) / (np.sqrt(2 * a_t))
        if data == 0:
            result = 1 / np.sqrt(b_t) * sp.stats.norm.pdf(0, loc=0, scale=1)
        else:
            result = data / (d_t1 * d_t2) * sp.stats.norm.pdf(d_t2, loc=0, scale=1)

    elif model == 'GARCH':
        sigma_0 = omega / (1 - beta - alpha)
        b_t = omega + (beta + alpha) * sigma_0
        result = sp.stats.norm.pdf(data, loc=0, scale=np.sqrt(b_t))

    return result

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

def out_of_sample(data, step=1, update=50, train_test_split=0.9, model='SHARV'):
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
        else:
            pars = Sharv(train_data, asymmetry=asymmetry).fit().params

        for j in range(i, min(i + update, len(data) - step)):
            # Use the same parameter to forecast when new data comes in until the next update round
            temp_date = dates[j:j + step]
            if model == 'GARCH':
                temp_forecast = garch_forecast(data[:j], model_res, step=step)
            else:
                temp_forecast = Sharv(data[:j], asymmetry=asymmetry).vol_forecast(pars, step=step)

            temp = pd.DataFrame(temp_forecast, index=temp_date, columns=['Forecast'])
            forecast.append(temp)

    return forecast




