from SHARV_class import *
from arch import arch_model
import pandas as pd
import numpy as np

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