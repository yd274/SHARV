# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport log, sqrt, M_PI, fabs

def compute_loglike(double[:] par,
                    double[:] data,
                    int asymmetry):
    cdef int n = data.shape[0]
    cdef double[:] sigma_sq = np.zeros(n)
    cdef double[:] y = np.zeros(n)
    cdef double[:] log_f = np.zeros(n)

    cdef double mu, beta, omega1, gamma1, omega2, gamma2
    cdef double LOG_SQRT_2PI = 0.5 * log(2 * M_PI)
    cdef double total_ll = 0.0
    cdef double b_t, a_t, indicator, term1Log_t, dLog_t, d_t, negloglike

    # Parameter Mapping
    if asymmetry == 0:
        beta, omega1, gamma1 = par[0], par[1], par[2]
        mu = omega2 = gamma2 = 0.0
    else:
        mu, beta, omega1, gamma1, omega2, gamma2 = par[0], par[1], par[2], par[3], par[4], par[5]

    # Initial Variance Calculation (mean of squares)
    cdef double sum_sq = 0.0
    for i in range(n): sum_sq += data[i] * data[i]
    sigma_sq[0] = sum_sq / n

    for t in range(1, n):
        y[t] = data[t] - mu * sqrt(sigma_sq[t-1])
        indicator = 1.0 if y[t] <= 0 else 0.0
        b_t = beta * sigma_sq[t - 1]
        a_t = omega1 + omega2 * indicator + (gamma1 + gamma2 * indicator) * sigma_sq[t - 1]

        # Stability Barrier - Crucial for fmincon parity
        if b_t <= 1e-12 or a_t < 0:
            return 1e15

        if omega1 == 0 and omega2 == 0 and gamma1 == 0 and gamma2 == 0:
            sigma_sq[t] = b_t
            log_f[t] = -0.5 * log(b_t) - 0.5 * (y[t]*y[t] / b_t) - LOG_SQRT_2PI
        else:
            # Main logic loop using C-level math functions
            d1_t = sqrt(b_t * b_t + 4.0 * a_t * (y[t] * y[t]))
            # Use safe subtraction for dLog_t
            d2_t = sqrt((d1_t - b_t) / (2.0 * a_t))
            d_t = (1.0 if y[t] >= 0 else -1.0) * d2_t
            sigma_sq[t] = b_t + a_t * (d_t * d_t)

            if y[t] == 0:
                log_f[t] = -LOG_SQRT_2PI - log(sqrt(b_t))
            else:
                log_f[t] = log(fabs(y[t] / (d_t * d1_t))) - LOG_SQRT_2PI - 0.5 * (d_t * d_t)

        total_ll += log_f[t]

    negloglike = -total_ll / n
    # Catch NaNs immediately
    if negloglike != negloglike:  # Standard C-check for NaN
        return 1e15

    return negloglike

def finite_difference_cython(double[:] data, double[:] par, double dh=1e-7, int asymmetry=0):
    cdef int n = par.shape[0]
    cdef int i, j

    # 1. Pre-calculate step sizes h
    cdef double[:] h = np.empty(n, dtype=np.float64)
    for i in range(n):
        h[i] = dh if (1e-4 * fabs(par[i])) < dh else (1e-4 * fabs(par[i]))

    # 2. Initialize Hessian
    cdef double[:, :] hessian = np.zeros((n, n), dtype=np.float64)

    cdef double f_pp, f_pm, f_mp, f_mm
    cdef double hi, hj

    for i in range(n):
        for j in range(i, n):
            par_ij_plus = np.copy(par)
            par_ij_minus = np.copy(par)
            par_ij_minus2 = np.copy(par)
            par_ij_minus3 = np.copy(par)

            # f_pp (par + hi, par + hj)
            par_ij_plus[i] += h[i]
            par_ij_plus[j] += h[j]
            f_pp = compute_loglike(par_ij_plus, data, asymmetry)

            # f_pm (par + hi, par - hj)
            par_ij_minus[i] += h[i]
            par_ij_minus[j] -= h[j]
            f_pm = compute_loglike(par_ij_minus, data, asymmetry)

            # f_mm (par - hi, par - hj)
            par_ij_minus2[i] -= h[i]
            par_ij_minus2[j] -= h[j]
            f_mm = compute_loglike(par_ij_minus2, data, asymmetry)

            # f_mp (par - hi, par + hj)
            par_ij_minus3[i] -= h[i]
            par_ij_minus3[j] += h[j]
            f_mp = compute_loglike(par_ij_minus3, data, asymmetry)

            # Calculate derivative
            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h[i] * h[j])
            if i != j:
                hessian[j, i] = hessian[i, j]

    return np.asarray(hessian)
