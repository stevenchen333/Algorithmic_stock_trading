from typing import Callable
import pandas as pd
import numpy as np

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
    def __init__(self, stocks,w = 0.5, alpha = 0.5, init_value = 2000):
        self.init_value = init_value   
        self.alpha = alpha
        self.w = w
        self.stocks = stocks
        self.ticker = stocks.columns[1].lower() if isinstance(stocks, pd.DataFrame) else None

    def get_stock(self):
        return self.stocks

    def get_w(self, t = None):
        if callable(self.w):
            return self.w(t)
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
        stock_value = stock[self.ticker].values
        stock_returns = (stock_value[1:] - stock_value[:-1]) / stock_value[:-1]
        return stock_returns
    
    def dlp(self):
        """
        Calculate the portfolio values and positions for the entire time period.
        Returns:
            V_L: List of long account values
            V_S: List of short account values
            V: List of total portfolio values
            pi_L: List of long positions
            pi_S: List of short positions
            w: List of weights
        """
        returns = self.returns()
        
        # Initialize arrays
        V_L = [0] * (len(returns) + 1)
        V_S = [0] * (len(returns) + 1)
        V = [0] * (len(returns) + 1)
        w = [0] * (len(returns) + 1)
        pi_L = [0] * (len(returns))
        pi_S = [0] * (len(returns))
        
        # Set initial values (t=0)
        V_L[0], V_S[0] = self.initial_acc_value()
        V[0] = V_L[0] + V_S[0]
        w[0] = self.get_w(0)
        pi_L[0], pi_S[0] = self.pi(w[0], V_L[0], V_S[0])
        
        # Calculate values for t > 0
        for t in range(1, len(returns) + 1):
            # Get weight for current timestep
            w[t] = self.get_w(t)
            
            # Calculate positions
            if t < len(returns):
                pi_L[t], pi_S[t] = self.pi(w[t], V_L[t-1], V_S[t-1])
            
            # Update account values
            V_L[t] = V_L[t-1] + (returns[t-1] * pi_L[t-1])
            V_S[t] = V_S[t-1] + (returns[t-1] * pi_S[t-1])
            V[t] = V_L[t] + V_S[t]
        
        # Fill shorter lists with NaN to align with length of V
        pi_L.append(float('nan'))
        pi_S.append(float('nan'))
        
        return V_L, V_S, V, pi_L, pi_S, w

#------------------------------------------------------


