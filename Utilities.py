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

