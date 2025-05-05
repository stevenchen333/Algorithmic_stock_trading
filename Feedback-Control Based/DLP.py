from typing import Callable
import pandas as pd

class DLP:
    '''
    This class implements the Dynamic Leverage Portfolio (DLP) strategy.
    It allows for the specification of a portfolio of stocks, their weights,
    and the leverage factor (alpha). The class provides methods to calculate
    the initial asset values, portfolio returns, and the leveraged portfolio
    values over time.
    param w: float or function
        The weight of the portfolio. Can be a constant or a function of time.
    param alpha: float or function
        The leverage factor. Can be a constant or a function of time.
    param stocks: DataFrame consisting of one stock price
    '''
    def __init__(self, stocks,w = 0.5,alpha = 0.5, init_value = 2000, return_weights = True):
        self.init_value = init_value   
        self.alpha = alpha
        self.w = w
        self.stocks = stocks
        self.tickers = self.stocks.columns.tolist()[1]
        self.return_weights = return_weights

    def get_stock(self):
        return self.stocks

    def get_w(self, timestep = None):
        if callable(self.w):
            return self.w(timestep)
        else:
            # If w is not a function, return it directly
            return self.w
    
    def get_alpha(self, timestep = None):
        if callable(self.alpha):
            return self.alpha(timestep)
        else:
            # If alpha is not a function, return it directly
            return self.alpha
        
        
    def initial_acc_value(self):
        alpha = self.get_alpha()
        v_0 = self.init_value
        vl0 = alpha*v_0
        vs0 = (1-alpha)*v_0

        return vl0, vs0
    
    def pi(self, w, v_l,v_s):
        pi_l = w*v_l
        pi_s = -w*v_s
        return pi_s, pi_l
    
        
    
    
    def returns(self):
        stock = self.get_stock()
        stock_value = stock[self.tickers].values
        stock_returns = (stock_value[1:] - stock_value[:-1]) / stock_value[:-1]
        return stock_returns
        
    def dlp(self):
        returns = self.returns()

        V_L0, V_S0 = self.initial_acc_value()
        V_L = [0] * (len(returns) + 1)
        V_S = [0] * (len(returns) + 1)
        V = [0] * (len(returns) + 1)

        V_L[0] = V_L0
        V_S[0] = V_S0
        V[0] = V_L0 + V_S0

        pi_L = [0] * (len(returns))
        pi_S = [0] * (len(returns))

        if self.return_weights == True:
            w = [0] * (len(returns) + 1)
            for i in range(len(returns)):
                w[i] = self.get_w(i)
        else:
            w = None


        for i in range(len(returns)):
            w_t = self.get_w(i)
            pi_L[i], pi_S[i] = self.pi(w_t, V_L[i], V_S[i])
            V_L[i + 1] = V_L[i] + (returns[i] * pi_L[i])
            V_S[i + 1] = V_S[i] + (returns[i] * pi_S[i])
            V[i + 1] = V_L[i + 1] + V_S[i + 1]

        # Fill shorter lists with NaN to align with length of V
        pi_L.append(float('nan'))
        pi_S.append(float('nan'))

        
        return V_L, V_S, V, pi_L, pi_S, w

#------------------------------------------------------


