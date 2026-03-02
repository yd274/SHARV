import numpy as np
import pandas as pd
from SHARV_class import *

data = pd.read_excel('DJI.xlsx')
data['Date'] = pd.to_datetime(data['Date'], unit='D', origin='1899-12-30')
data.set_index('Date', inplace=True)

data = data.dropna()
data['Close'] = np.log(data['Close']).diff() * 100

y = data[['Close']].dropna()

res = Sharv(np.array(y).reshape(len(y))).fit()