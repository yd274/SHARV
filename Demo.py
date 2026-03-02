from SHARV_class import *
from Utilities import *
import matplotlib.pyplot as plt
from arch import arch_model

data = pd.read_excel('Data/DJI.xlsx')
data['Date'] = pd.to_datetime(data['Date'], unit='D', origin='1899-12-30')
data.set_index('Date', inplace=True)

data = data.dropna()
# Log return
data['Close'] = np.log(data['Close']).diff() * 100

y = data[['Close']].dropna()

# Fit the sharv model
sharv_res = Sharv(y).fit()
# Asymmetric version
asharv_res = Sharv(y, asymmetry=True).fit()

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
RV = pd.read_excel('rv_dj.xlsx')
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
mse_values = ((combined_vol[['SHARV', 'ASHARV', 'GARCH']].subtract(combined_vol['RV'], axis=0))**2).mean()

# Convert results into a new DataFrame
mse_df = pd.DataFrame(mse_values, columns=['MSE'])

# Volatility estimate comparisons are often dominated by outliers, let's also use the QLIKE loss function to compare
# the accuracy, which is far less acute to the influence of outliers

# QLIKE formula: (y / y_hat) - log(y / y_hat) - 1
ratio = combined_vol[['SHARV', 'ASHARV', 'GARCH']].div(combined_vol['RV'], axis=0).pow(-1) # This is y / y_hat
qlike_values = (ratio - np.log(ratio) - 1).mean()

# 3. Combine into a results DataFrame
qlike_df = pd.DataFrame(qlike_values, columns=['QLIKE'])