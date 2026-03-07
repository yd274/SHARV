import numpy as np
import pandas as pd
from arch import arch_model
import scipy as sp
from scipy.optimize import minimize, LinearConstraint
from loglike import *
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable, default_txt_fmt
from Utilities import *


class Sharv():
    def __init__(self, data, asymmetry=False):
        if isinstance(data, pd.DataFrame):
            self.data = data.values.reshape(len(data))
            self.dates = data.index
        else:
            self.data = np.array(data).reshape(len(data))
            self.dates = None

        self.asymmetry = asymmetry

    def _initial_guess(self):
        if self.asymmetry:
            o = 1
        else:
            o = 0
        garch = arch_model(self.data, mean="Zero", p=1, o=o, q=1).fit(disp=0)

        if self.asymmetry:
            return [0, garch.params['beta[1]'], garch.params['omega'], garch.params['alpha[1]'], 0.001,
                    garch.params['gamma[1]']]
        else:
            return [garch.params['beta[1]'], garch.params['omega'], garch.params['alpha[1]']]

    def filter(self, par):
        if not self.asymmetry:
            beta, omega1, gamma1 = par[0], par[1], par[2]
            mu, omega2, gamma2 = 0.0, 0.0, 0.0
        else:
            mu, beta, omega1, gamma1, omega2, gamma2 = par[0], par[1], par[2], par[3], par[4], par[5]

        n = len(self.data)
        sigma_sq = np.zeros(n)
        sigma_sq[0] = np.var(self.data)
        vol_vol = np.zeros(n)
        # sigma_sq[0] = (np.var(self.data) - 3 * (omega1 + 0.5 * omega2)) / (beta + 3 * (gamma1 + 0.5 * gamma2))
        log_f = np.zeros(n)
        y = np.zeros(n)
        LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)

        for t in range(1, n):
            # Adjust returns by its drift term
            y[t] = self.data[t] - mu * np.sqrt(sigma_sq[t - 1])
            # Separate negative returns
            indicator = 1.0 if y[t] <= 0 else 0.0
            b_t = beta * sigma_sq[t - 1]
            a_t = omega1 + omega2 * indicator + (gamma1 + gamma2 * indicator) * sigma_sq[t - 1]
            if b_t <= 0 or a_t < 0:
                log_f[t] = 1e10
                continue

            # Conditional variance of volatility
            vol_vol[t] = np.sqrt(
                2 * (omega1 + gamma1 * sigma_sq[t - 1]) ** 2 + 2 * (omega1 + gamma1 * sigma_sq[t - 1]) *
                (omega2 + gamma2 * sigma_sq[t - 1]) + 5 / 4 * (omega2 + gamma2 * sigma_sq[t - 1]) ** 2)

            if omega1 == 0 and omega2 == 0 and gamma1 == 0 and gamma2 == 0:
                # Simply becomes a scaled standard normal
                d_t = y[t] / np.sqrt(b_t)
                sigma_sq[t] = b_t
                log_f[t] = -0.5 * np.log(b_t) - 0.5 * (y[t] ** 2 / b_t)
            else:
                # d2(y) function in Theorem 3.1 of Ding, 2022
                d1_t = np.sqrt(b_t * b_t + 4.0 * a_t * (y[t] * y[t]))
                # Use safe subtraction for dLog_t
                d2_t = np.sqrt((d1_t - b_t) / (2.0 * a_t))
                d_t = (1.0 if y[t] >= 0 else -1.0) * d2_t
                sigma_sq[t] = b_t + a_t * (d_t * d_t)

                if y[t] == 0:
                    log_f[t] = -LOG_SQRT_2PI - np.log(np.sqrt(b_t))
                else:
                    log_f[t] = np.log(y[t] / (d_t * d1_t)) - LOG_SQRT_2PI - 0.5 * (d_t * d_t)

        # Standardised residuals
        res = y / np.sqrt(sigma_sq)

        if self.dates is not None:
            sigma_sq = pd.DataFrame(sigma_sq, index=self.dates, columns=['Volatility'])
            vol_vol = pd.DataFrame(vol_vol, index=self.dates, columns=['Volatility of Volatility'])
            res = pd.DataFrame(res, index=self.dates, columns=['Standardized residuals'])

        return {'Loglikelihood vector': log_f, 'Volatility': np.sqrt(sigma_sq), 'Vol of vol': vol_vol,
                'Standardized residuals': res}

    def vol_forecast(self, par, step=1):
        temp_res = self.filter(par)
        if not self.asymmetry:
            beta, omega1, gamma1 = par[0], par[1], par[2]
            mu, omega2, gamma2 = 0, 0, 0
        else:
            mu, beta, omega1, gamma1, omega2, gamma2 = par[0], par[1], par[2], par[3], par[4], par[5]

        var = temp_res["Volatility"] ** 2
        forecast = np.zeros(step)
        forecast[0] = omega1 + 0.5 * omega2 + (beta + gamma1 + 0.5 * gamma2) * var[-1]

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
            bounds = [(-np.inf, np.inf), (1e-6, 0.99), (1e-6, np.inf), (1e-6, 0.99), (1e-6, np.inf), (1e-6, 0.99)]
            A = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 0.5]])
        else:
            bounds = [(1e-6, 0.99), (1e-6, np.inf), (1e-6, 0.99)]
            A = np.array([[1.0, 0.0, 1.0]])

        lin_constr = LinearConstraint(A, 1e-6, 0.99)
        fit_result = minimize(compute_loglike, start_params, args=(self.data, int(self.asymmetry)),
                       method='SLSQP', bounds=bounds, constraints=lin_constr,
                       options={'ftol': 1e-8, 'maxiter': maxiter})

        # Compute the QLM standard errors (Sandwich form, NOT the inverse of negative Hessian since we allow for
        # potential model misspecification and use quasi-maximum likelihood theory. Specifically,
        # Theta hat - Theta ~ Normal(0, V^-1 @ S @ V^-1), where V is the inverse negative Hessian, S = s @ s.T where
        # s is the score vector
        pars = fit_result.x
        hes = -finite_difference_cython(self.data, pars, asymmetry=self.asymmetry, dh=1e-4 if self.asymmetry else 1e-7)

        def input_fun(par, data):
            return self.filter(par)['Loglikelihood vector']

        score = score_vec(pars, self.data, input_fun)
        temp = np.linalg.inv(-hes) @ score @ np.linalg.inv(-hes) / len(self.data)
        std = np.sqrt(np.diag(temp))
        t_stats = np.array(pars) / np.array(std)
        p_values = [2 * (1 - sp.stats.norm.cdf(np.abs(t))) for t in t_stats]
        t_crit = sp.stats.norm.ppf(0.975)
        ci_lower = fit_result.x - t_crit * std
        ci_upper = fit_result.x + t_crit * std
        loglike = -compute_loglike(pars, self.data, self.asymmetry) * len(self.data)
        bic = len(pars) * np.log(len(self.data)) - 2 * loglike
        bic_str = f"{len(pars) * np.log(len(self.data)) - 2 * loglike:.4f}"
        par_name = ['mu', 'beta', 'alpha', 'psi', 'omega', 'phi'] if self.asymmetry else \
            ['beta', 'alpha', 'psi']

        # Create statsmodels-like summary table
        top_data = [
            ["Dep. Variable:", "y", "Model:", "Quasi-ML"],
            ["No. Observations:", str(len(self.data)), "Log-Likelihood:", f"{loglike:.4f}"],
            ["Date:", "Sat, 07 Mar 2026", "BIC:", f"{bic_str}"]
        ]
        params_data = [[pars[i], std[i], t_stats[i], p_values[i], ci_lower[i], ci_upper[i]]
                       for i in range(len(pars))]
        headers = ["coef", "std err", "t", "P>|t|", "[0.025", "0.975]"]
        param_fmt = default_txt_fmt.copy()
        param_fmt.update({
            'stubs_align': 'l',
            'stubs_fmts': ["%-10s"],  # Fixed 10 chars for names
            'data_aligns': 'r',
            'data_fmts': ["%10.4f"] * 2 + ["%10.3f"] * 4,  # 6 cols * 10 chars = 60
            'colsep': '',  # We handle spacing inside the %10 format
            'header_align': 'r'
        })
        top_fmt = default_txt_fmt.copy()
        top_fmt.update({
            'data_aligns': 'l',
            'data_fmts': ["%-18s", "%-17s", "%-17s", "%-18s"],
            'colsep': ''  # 18+17+17+18 = Exactly 70
        })
        top_table = SimpleTable(top_data, txt_fmt=top_fmt)
        param_table = SimpleTable(params_data, headers=headers, stubs=par_name, txt_fmt=param_fmt)
        table_width = 70

        sm_summary = Summary()
        model_prefix = "Asymmetric " if self.asymmetry else ""
        title = f"{model_prefix}Stochastic Heteroskedastic AutoRegressive Volatility Model Results"
        title_fmt = default_txt_fmt.copy()
        title_fmt.update({
            'data_aligns': 'c',    # Center the data
            'border_bottom': '',   # Remove bottom border so it flows into the next table
        })
        title_table = SimpleTable([[title.center(table_width)]], txt_fmt=title_fmt)
        sm_summary.tables.append(title_table)
        sm_summary.tables.append(top_table)
        sm_summary.tables.append(param_table)

        class SharvFitResult:
            def __init__(self, fit_result, model, std, p_values, loglike, bic, summary_obj):
                self._fit_result = fit_result
                self._model = model
                self.params = fit_result.x
                self.std = std
                self.bic = bic
                self.pvalues = p_values
                self.loglike = loglike
                self._summary = summary_obj

            def __getattr__(self, name):
                return getattr(self._fit_result, name)

            def summary(self):
                """Returns the statsmodels-like summary table"""
                return self._summary

            def filter(self):
                return self._model.filter(self.params)

            def vol_forecast(self, step=1):
                return self._model.vol_forecast(self.params, step=step)

            def VaR_forecast(self, q=0.05):
                return self._model.VaR_forecast(self.params, q=q)

        return SharvFitResult(fit_result, self, std, p_values, loglike, bic, sm_summary)
