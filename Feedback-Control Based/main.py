# from constants import ttingo_api_key
# from  ttingo_api import retrieve_stock

#stock = retrieve_stock(['TSLA'], "2023-01-01", "2023-10-01", ttingo_api_key, dataframe = True, save_file = True)

import DLP
import json
import numpy as np
import math

with open("/Users/tch/Documents/Algorithmic_stock_trading/Feedback-Control Based/tsla_dlptest1 2023-01-03 00:00:00 - 2023-09-29 00:00:00.json") as f:
    stock = json.load(f)


import pandas as pd

stock = pd.DataFrame(stock)


dlp_tsla = DLP.DLP(stock, tickers = ['TSLA'], w = 0.5, alpha = 0.5, init_value = 100)

dlp_result = (dlp_tsla.dlp())




import matplotlib.pyplot as plt


fig, ax = plt.subplots(2,2)

ax[0,0].plot(dlp_result.loc[:,'V'], label = 'V')
ax[0,0].legend()
ax[0,0].hlines(y = 0, color = 'r', linestyle = '--', xmin = 0, xmax = len(dlp_result))

ax[0,1].plot(dlp_result.loc[:,'V_L'], label = 'V_L')
ax[0,1].legend()

ax[1,0].plot(dlp_result.loc[:,'V_S'], label = 'V_S')
ax[1,0].legend()

ax[1,1].plot(stock.loc[:,'tsla'], label = 'Stock Price')
ax[1,1].legend()

fig.supxlabel('Time')
fig.supylabel('Value')
fig.suptitle('DLP Simulation for TSLA')

plt.tight_layout
plt.show()
#-------------------------------------------------------
# w as a function of time

def w(t):
    v = 0.1
    return v

dlp_tsla = DLP.DLP(stock, tickers = ['TSLA'], w = w, alpha = 0.5, init_value = 100)
dlp_result = (dlp_tsla.dlp())


fig, ax = plt.subplots(2,2)

ax[0,0].plot(dlp_result.loc[:,'V'], label = 'V')
ax[0,0].legend()

ax[0,1].plot(dlp_result.loc[:,'V_L'], label = 'V_L')
ax[0,1].legend()

ax[1,0].plot(dlp_result.loc[:,'V_S'], label = 'V_S')
ax[1,0].legend()

ax[1,1].plot(stock.loc[:,'tsla'], label = 'Stock Price')
ax[1,1].legend()

fig.supxlabel('Time')
fig.supylabel('Value')
fig.suptitle('DLP Simulation for TSLA')

plt.tight_layout
plt.show()