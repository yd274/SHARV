from SHARV_class import *
from loglike import *
import pandas as pd
import numpy as np
import scipy as sp

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
        mu, gamma2 = 0.0, 0.0
    elif model == 'ASHARV':
        mu, beta, omega1, gamma1, gamma2 = par[0], par[1], par[2], par[3], par[4]
    elif model == 'GARCH':
        omega, beta, alpha = par[0], par[1], par[2]

    if model == 'SHARV' or model == 'ASHARV':
        sigma_0 = (omega1) / (1 - beta - gamma1 - 0.5 * gamma2)
        data = data - mu * np.sqrt(sigma_0)
        indicator = 1 if data <= 0 else 0
        a_t = omega1 + gamma1 * sigma_0 + gamma2 * indicator * sigma_0
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


def finite_difference(par, data, dh=1e-4, asymmetry=False):
    """
    Using finite difference method to calculate the Hessian of the loglikelihood function of (A)SHARV
    :param par:
    :param data:
    :param dh: step size
    :param asymmetry: whether SHARV or ASHARV
    :return:
    """
    if isinstance(data, pd.DataFrame):
        data = data.values.reshape(len(data))

    n = len(par)
    par = np.array(par, dtype=np.float64).flatten()
    T = len(data)
    # MATLAB-style step size (eps^(1/4))
    h = np.array([max(dh, 1e-4 * np.abs(x)) for x in par], dtype=np.float64)

    hessian = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            par_ij_plus = par.copy()
            par_ij_minus = par.copy()
            par_ij_plus[i] += h[i]
            par_ij_plus[j] += h[j]
            par_ij_minus[i] += h[i]
            par_ij_minus[j] -= h[j]
            par_ij_minus2 = par.copy()
            par_ij_minus2[i] -= h[i]
            par_ij_minus2[j] -= h[j]
            par_ij_minus3 = par.copy()
            par_ij_minus3[i] -= h[i]
            par_ij_minus3[j] += h[j]

            f_plus_plus = compute_loglike(par_ij_plus, data, asymmetry)
            f_plus_minus = compute_loglike(par_ij_minus, data, asymmetry)
            f_minus_plus = compute_loglike(par_ij_minus3, data, asymmetry)
            f_minus_minus = compute_loglike(par_ij_minus2, data, asymmetry)

            hessian[i, j] = (f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus) / (4 * h[i] * h[j])
            if i != j:
                hessian[j, i] = hessian[i, j]

    return hessian

def score_vec(par, data, fun, dh=1e-7):
    """
    Use (central) finite difference method to calculate the average of the square score vector. This is the sandwich
    form for robust standard error where the middle term is E[S @ S.T] where S is the score of loglikelihood. The
    reason we use the score vector is so that we can use sample average to approximate the expectation
    :param par:
    :param data:
    :param fun:
    :param dh: step size
    :return:
    """
    if isinstance(data, pd.DataFrame):
        data = data.values.reshape(len(data))

    # Define step size for each parameter
    h = np.array([max(dh, 1e-4 * np.abs(x)) for x in par], dtype=np.float64)

    score = np.zeros((len(data), len(par)))
    for i in range(len(par)):
        par_up = par.copy()
        par_down = par.copy()
        par_up[i] = par[i] + h[i]
        par_down[i] = par[i] - h[i]
        # The function should be Sharv().filter()['Loglikelihood vector'] which returns a vector of the loglikelihood
        # function calculated at each data point
        fun_up = fun(par_up, data)
        fun_down = fun(par_down, data)
        temp = (fun_up - fun_down) / (2 * h[i])
        score[:, i] = np.array(temp).reshape(len(temp))

    score = np.delete(score, 0, axis=0)
    # At each data point, calculate the outer product S @ S.T
    temp = np.einsum('ij,ik->ijk', score, score)
    # Averaging over the sample to approximate the expectation
    score_vec = np.mean(temp, axis=0)
    score_vec = np.squeeze(score_vec)
    return score_vec


