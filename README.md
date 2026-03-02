This project demonstrate the empirical application of the Stochastic Heteroskedastic AutoRgressive
Volatility (SHARV) model and its asymmetric extension ASHARV from Ding, 2023, Journal of Econometrics

The main purpose is to show how to estimate the model parameters, as well as filter and forecast volatility
and VaR

In order to maximise the log likelihood function, Cython is used to compile the function in C. To make sure
it works, please follow the guide:

1. the loglikelihood functio is stored in loglike.pyx
2. the set up script is in setup.py
3. open project terminal and run "python setup.py build_ext --inplace"
4. check if the compilation is successful by checking if a loglike.c script is created

In the _SHARV_class.py_, the optimisation function will call the C file and get the estimated parameters, it then 
uses the estimated parameters to calculate filtered volatility, volatility of volatility and standardised residuals. 
Multi-perioid volatility forecast and one-period VaR forecast is also available

Requirements:

numpy == 2.1.3

scipy == 1.15.3

arch == 8.0.0

pandas == 2.3.1

statsmodels == 0.14.5

matplotlib == 3.10.0