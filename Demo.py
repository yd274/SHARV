from SHARV_class import *
from Utilities import *
import matplotlib.pyplot as plt
from arch import arch_model
from loglike import *
from forecast_funcs import *


data = pd.read_excel('Data/DJI.xlsx')
data['Date'] = pd.to_datetime(data['Date'], unit='D', origin='1899-12-30')
data.set_index('Date', inplace=True)

data = data.dropna()
# Log return
data['Close'] = np.log(data['Close']).diff() * 100

y = data[['Close']].dropna()

# Fit the sharv model
sharv_res = Sharv(y).fit()
print(sharv_res.summary())
# Asymmetric version
asharv_res = Sharv(y, asymmetry=True).fit()
print(asharv_res.summary())
hes = -finite_difference_cython(y.values.reshape(len(y)), asharv_res.params,
                               asymmetry=True, dh=1e-4)
std_1 = np.sqrt(np.diag(np.linalg.inv(-hes))/len(y))
score = score_vec(asharv_res.params, y, asymmetry=True)
std_2 = np.sqrt(np.diag(np.linalg.inv(score))/len(y))

temp = np.linalg.inv(-hes) @ score @ np.linalg.inv(-hes) / len(y)
std = np.sqrt(np.diag(temp))

# Benchmark model
garch = arch_model(y, mean="Zero", p=1, o=0, q=1).fit(disp=0)

# Observe that even with normal innovation, the conditional distribution of returns have heavy tails
sharv_pdf = np.vectorize(lambda x: pdf_sharv(sharv_res.params, x, model='SHARV'))
garch_pdf = np.vectorize(lambda x: pdf_sharv([garch.params['omega'], garch.params['beta[1]'],
                                              garch.params['alpha[1]']], x, model='GARCH'))
# Notice that for ASHARV, even with symmetric normal innovation, the conditional density is left skewed with fatter
# left tail, which captures stylized facts of financial returns
asharv_pdf = np.vectorize(lambda x: pdf_sharv(asharv_res.params, x, model='ASHARV'))
x_ax = np.linspace(-5, 5, 100)
plt.plot(x_ax,sharv_pdf(x_ax), label='SHARV', color='blue')
plt.plot(x_ax,garch_pdf(x_ax), label='GARCH', color='red', linestyle='dashed')
plt.plot(x_ax,asharv_pdf(x_ax), label='ASHARV', color='green')
plt.legend()
plt.show()

# Compare the volatility estimation with realized volatility (RV). RV for DJIA is obtained from a high frequency trading
# platform
RV = pd.read_excel('Data/rv_dj.xlsx')
RV['Date'] = pd.to_datetime(RV['DateTime'], unit='D', origin='1899-12-30')
RV.set_index('Date', inplace=True)
RV = np.sqrt(RV['Return'])

sharv_vol = sharv_res.filter()
asharv_vol = asharv_res.filter()
garch_vol = garch.conditional_volatility
combined_vol = pd.concat([sharv_vol['Volatility'], asharv_vol['Volatility'], garch_vol, RV], axis=1)
combined_vol.columns = ['SHARV', 'ASHARV', 'GARCH', 'RV']
combined_vol = combined_vol.dropna()

# Compute MSE for each model column
mse = ((combined_vol[['SHARV', 'ASHARV', 'GARCH']].subtract(combined_vol['RV'], axis=0))**2).mean()

# Convert results into a new DataFrame
mse = pd.DataFrame(mse, columns=['MSE'])

# Volatility estimate comparisons are often dominated by outliers, let's also use the QLIKE loss function to compare
# the accuracy, which is far less acute to the influence of outliers

# QLIKE formula: (y / y_hat) - log(y / y_hat) - 1
ratio = combined_vol[['SHARV', 'ASHARV', 'GARCH']].div(combined_vol['RV'], axis=0).pow(-1) # This is y / y_hat
qlike = (ratio - np.log(ratio) - 1).mean()

# Combine into a results DataFrame
qlike = pd.DataFrame(qlike, columns=['QLIKE'])

# Out of sample forecast (for 1-step, 5-step and 10-step ahead forecast)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sharv_forecast = pd.concat(out_of_sample(y))
ashatv_forecast = pd.concat(out_of_sample(y, model='ASHARV'))
garch_forecast = pd.concat(out_of_sample(y, model='GARCH'))
forecast = pd.concat([sharv_forecast, ashatv_forecast, garch_forecast, RV], axis=1)
forecast.columns = ['SHARV', 'ASHARV', 'GARCH', 'RV']
forecast = forecast.dropna()

# Compute MSE for each model column
mse_oos = ((forecast[['SHARV', 'ASHARV', 'GARCH']].subtract(forecast['RV'], axis=0))**2).mean()
mse_oos = pd.DataFrame(mse_oos, columns=['MSE'])
# Compute QLIKE for each model column
ratio = forecast[['SHARV', 'ASHARV', 'GARCH']].div(forecast['RV'], axis=0).pow(-1) # This is y / y_hat
qlike_oos = (ratio - np.log(ratio) - 1).mean()
qlike_oos = pd.DataFrame(qlike_oos, columns=['QLIKE'])

# 5-step
sharv_forecast_5 = pd.concat(out_of_sample(y, step=5))
asharv_forecast_5 = pd.concat(out_of_sample(y, step=5, model='ASHARV'))
garchforecast_5 = pd.concat(out_of_sample(y, step=5, model='GARCH'))
forecast_5 = pd.concat([sharv_forecast_5, asharv_forecast_5, garchforecast_5], axis=1)
forecast_5 = forecast_5.join(RV, how='left')
forecast_5.columns = ['SHARV', 'ASHARV', 'GARCH', 'RV']
forecast_5 = forecast_5.dropna()

# Compute MSE for each model column
mse_oos_5 = ((forecast_5[['SHARV', 'ASHARV', 'GARCH']].subtract(forecast_5['RV'], axis=0))**2).mean()
mse_oos_5 = pd.DataFrame(mse_oos_5, columns=['MSE'])
# Compute QLIKE for each model column
ratio = forecast_5[['SHARV', 'ASHARV', 'GARCH']].div(forecast_5['RV'], axis=0).pow(-1) # This is y / y_hat
qlike_oos_5 = (ratio - np.log(ratio) - 1).mean()
qlike_oos_5 = pd.DataFrame(qlike_oos_5, columns=['QLIKE'])

# 10-step
sharv_forecast_10 = pd.concat(out_of_sample(y, step=10))
asharv_forecast_10 = pd.concat(out_of_sample(y, step=10, model='ASHARV'))
garchforecast_10 = pd.concat(out_of_sample(y, step=10, model='GARCH'))
forecast_10 = pd.concat([sharv_forecast_10, asharv_forecast_10, garchforecast_10], axis=1)
forecast_10 = forecast_10.join(RV, how='left')
forecast_10.columns = ['SHARV', 'ASHARV', 'GARCH', 'RV']
forecast_10 = forecast_10.dropna()

# Compute MSE for each model column
mse_oos_10 = ((forecast_10[['SHARV', 'ASHARV', 'GARCH']].subtract(forecast_10['RV'], axis=0))**2).mean()
mse_oos_10 = pd.DataFrame(mse_oos_10, columns=['MSE'])
# Compute QLIKE for each model column
ratio = forecast_10[['SHARV', 'ASHARV', 'GARCH']].div(forecast_10['RV'], axis=0).pow(-1) # This is y / y_hat
qlike_oos_10 = (ratio - np.log(ratio) - 1).mean()
qlike_oos_10 = pd.DataFrame(qlike_oos_10, columns=['QLIKE'])
