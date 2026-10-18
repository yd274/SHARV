import pandas as pd
import numpy as np

data = pd.read_excel('Data/DJI.xlsx')
data['Date'] = pd.to_datetime(data['Date'], unit='D', origin='1899-12-30')
data = data.set_index('Date')
data['Return'] = np.log(data['Close']).diff() * 100
data[data.index < pd.to_datetime('2018-01-01', format='%Y-%m-%d')]

temp = data.resample('ME').last()
data = data.dropna()
np.corrcoef(data['Return'].iloc[1:], data['Return'].iloc[:-1])