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
