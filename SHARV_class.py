import numpy as np
from arch import arch_model
import scipy as sp
from statsmodels.base.model import GenericLikelihoodModel
from scipy.optimize import minimize, Bounds, LinearConstraint
from numba import njit
from loglike import compute_loglike
"""
@njit
def filter(par, data, asymmetry):
    if not asymmetry:
        beta, omega1, gamma1 = par[0], par[1], par[2]
        mu, omega2, gamma2 = 0.0, 0.0, 0.0
    else:
        mu, beta, omega1, gamma1, omega2, gamma2 = par[0], par[1], par[2], par[3], par[4], par[5]

    n = len(data)
    sigma_sq = np.zeros(n)
    sigma_sq[0] = np.var(data)
    vol_vol = np.zeros(n)
    # sigma_sq[0] = (np.var(self.data) - 3 * (omega1 + 0.5 * omega2)) / (beta + 3 * (gamma1 + 0.5 * gamma2))
    log_f = np.zeros(n)
    y = np.zeros(n)
    LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)

    for t in range(1, n):
        # Adjust returns by its drift term
        y[t] = data[t] - mu * np.sqrt(sigma_sq[t-1])
        # Separate negative returns
        indicator = 1.0 if y[t] <= 0 else 0.0
        b_t = beta * sigma_sq[t - 1]
        a_t = omega1 + omega2 * indicator + (gamma1 + gamma2 * indicator) * sigma_sq[t - 1]
        if b_t <= 0 or a_t < 0:
            log_f[t] = 1e10
            continue

        # Conditional variance of volatility
        vol_vol[t] = np.sqrt(2 * (omega1 + gamma1 * sigma_sq[t - 1]) ** 2 + 2 * (omega1 + gamma1 * sigma_sq[t - 1]) *
                            (omega2 + gamma2 * sigma_sq[t - 1]) + 5 / 4 * (omega2 + gamma2 * sigma_sq[t - 1]) ** 2)

        if omega1 == 0 and omega2 == 0 and gamma1 == 0 and gamma2 == 0:
            # Simply becomes a scaled standard normal
            d_t = y[t] / np.sqrt(b_t)
            sigma_sq[t] = b_t
            log_f[t] = -0.5 * np.log(b_t) - 0.5 * (y[t]**2 / b_t)
        else:
            # d2(y) function in Theorem 3.1 of Ding, 2022
            term1Log_t = 0.5 * np.log(b_t ** 2 + 4 * a_t * (y[t] ** 2))
            dLog_t = 0.5 * (np.log(np.exp(term1Log_t) - b_t) - np.log(2 * a_t))
            d_t = np.sign(y[t]) * np.exp(dLog_t)
            # Filtered volatility
            sigma_sq[t] = b_t + a_t * (d_t ** 2)
            if y[t] == 0:
                log_f[t] = -LOG_SQRT_2PI - np.log(b_t ** 0.5)
            else:
                log_f[t] = np.log(y[t] / d_t) - term1Log_t - LOG_SQRT_2PI - 0.5 * (d_t**2)

    negloglike = -np.sum(log_f) / n

    if np.isnan(negloglike) or np.isinf(negloglike):
        negloglike = n ** 10

    # Standardised residuals
    res = y / np.sqrt(sigma_sq)
    return negloglike
"""

class Sharv():
    def __init__(self, data, asymmetry=False):
        self.data = data
        self.asymmetry = asymmetry

    def _initial_guess(self):
        if self.asymmetry:
            o = 1
        else:
            o = 0
        garch = arch_model(self.data, mean="Zero", p=1, o=o, q=1).fit(disp=0)

        if self.asymmetry:
            return [0, garch.params['beta[1]'], garch.params['omega'], garch.params['alpha[1]'], 0,
                    garch.params['gamma[1]']]
        else:
            return [garch.params['beta[1]'], garch.params['omega'], garch.params['alpha[1]']]

    def vol_forecast(self, par, step=1):
        temp_res = self.filter(par)
        if not self.asymmetry:
            beta, omega1, gamma1 = par[0], par[1], par[2]
            mu, omega2, gamma2 = 0, 0, 0
        else:
            mu, beta, omega1, gamma1, omega2, gamma2 = par[0], par[1], par[2], par[3], par[4], par[5]

        var = temp_res["volatility"] ** 2
        forecast = np.zeros(step)
        forecast[1] = omega1 + 0.5 * omega2 + (beta + gamma1 + 0.5 * gamma2) * var[-1]

        # For multistep forecast, use recursive formula with unconditional mean
        mean_vsq = (omega1 + 0.5 * omega2) / (1 - beta - gamma1 - 0.5 * gamma2)
        mean_rsq = (1 + 2 * gamma1 + gamma2) * mean_vsq + 2 * omega1 + omega2

        if step > 1:
            for k in range(1, step):
                forecast[k] = mean_vsq + (beta + gamma1 + 0.5 * gamma2) * (forecast[k - 1] - mean_vsq)

        return np.sqrt(forecast)

    def VaR_forecast(self, par, q=0.05):
        if not self.asymmetry:
            beta, omega1, gamma1 = par[0], par[1], par[2]
            mu, omega2, gamma2 = 0, 0, 0
        else:
            mu, beta, omega1, gamma1, omega2, gamma2 = par[0], par[1], par[2], par[3], par[4], par[5]

        vol = self.vol_forecast(par, step=1)

        # Replace the quantile with empirical quantile in the formula 3.11 in Ding, 2022
        autoregressive_part = beta * vol[-1] ** 2
        heteroskedastic_part = omega1 + omega2 + (gamma1 + gamma2) * vol[-1] ** 2
        drift = mu * vol[-1]

        return drift - np.sqrt((q ** 4) * heteroskedastic_part + (q ** 2) * autoregressive_part)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
        if start_params is None:
            start_params = self._initial_guess()

        if self.asymmetry:
            bounds = [(-np.inf, np.inf), (0.0, 1.0), (0.0, np.inf), (0.0, 1.0), (0.0, np.inf), (0.0, 1.0)]
            A = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 0.5]])
        else:
            bounds = [(1e-6, 0.99), (1e-6, np.inf), (1e-6, 0.99)]
            A = np.array([[1.0, 0.0, 1.0]])

        lin_constr = LinearConstraint(A, 1e-6, 0.99)
        res = minimize(compute_loglike, start_params, args=(self.data, int(self.asymmetry)),
                       method='SLSQP', bounds=bounds, constraints=lin_constr,
                       options={'ftol': 1e-8, 'maxiter': maxiter})

        """res = minimize(compute_loglike, start_params, args=(self.data, int(self.asymmetry)),
                       method='trust-constr', bounds=bounds, constraints=lin_constr, options={'xtol': 1e-8, 'gtol': 1e-8,
                                                                                       'maxiter': maxiter})"""
        return res
        #fit_result = super(Sharv, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs)

        """class SharvFitResult:
            def __init__(self, fit_result, model):
                self._fit_result = fit_result
                self._model = model
                self.params = fit_result.params

            def __getattr__(self, name):
                return getattr(self._fit_result, name)

            def filter(self):
                return self._model.filter(self._fit_result.params)

            def vol_forecast(self, step=1):
                return self._model.vol_forecast(self._fit_result.params, step=step)

            def VaR_forecast(self, q=0.05):
                return self._model.VaR_forecast(self._fit_result.params, q=q)

        return SharvFitResult(fit_result, self)"""
