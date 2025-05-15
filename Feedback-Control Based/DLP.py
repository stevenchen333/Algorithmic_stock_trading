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
    
    Parameters:
    -----------
    stocks : DataFrame
        DataFrame containing stock prices with dates and at least one stock price column
    w : float or Callable, optional (default=0.5)
        The weight of the portfolio. Can be a constant or a function of time.
    alpha : float or Callable, optional (default=0.5)
        The leverage factor. Can be a constant or a function of time.
    init_value : float, optional (default=2000)
        Initial portfolio value
    return_weights : bool, optional (default=True)
        Whether to return weight values in the results
    risk_free_rate : float, optional (default=0.0)
        Risk-free rate for Sharpe ratio calculation
    '''
    
    def __init__(self, stocks, w=0.5, alpha=0.5, init_value=2000, 
                return_weights=True, risk_free_rate=0.0):
        self.init_value = init_value   
        self.alpha = alpha
        self.w = w
        self.stocks = stocks
        self.tickers = self.stocks.columns.tolist()[1]  # Assumes first column is date
        self.return_weights = return_weights
        self.risk_free_rate = risk_free_rate

    def get_stock(self):
        return self.stocks

    def get_w(self, timestep=None):
        if callable(self.w):
            return self.w(timestep)
        return self.w
    
    def get_alpha(self, timestep=None):
        if callable(self.alpha):
            return self.alpha(timestep)
        return self.alpha
        
    def initial_acc_value(self):
        alpha = self.get_alpha()
        v_0 = self.init_value
        vl0 = alpha * v_0
        vs0 = (1 - alpha) * v_0
        return vl0, vs0
    
    def pi(self, w, v_l, v_s):
        pi_l = w * v_l
        pi_s = -w * v_s
        return pi_s, pi_l
    
    def returns(self):
        stock = self.get_stock()
        stock_value = stock[self.tickers].values
        stock_returns = (stock_value[1:] - stock_value[:-1]) / stock_value[:-1]
        return stock_returns
    
    def calculate_sharpe_ratio(self, portfolio_returns):
        """Calculate annualized Sharpe ratio"""
        excess_returns = portfolio_returns - self.risk_free_rate/252  # Assuming daily data
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe_ratio * np.sqrt(252)  # Annualize
        
    def dlp(self):
        returns = self.returns()
        V_L0, V_S0 = self.initial_acc_value()
        
        # Initialize arrays
        n_periods = len(returns)
        V_L = np.zeros(n_periods + 1)
        V_S = np.zeros(n_periods + 1)
        V = np.zeros(n_periods + 1)
        kum_ret = np.zeros(n_periods + 1)
        portfolio_returns = np.zeros(n_periods)
        
        # Set initial values
        V_L[0] = V_L0
        V_S[0] = V_S0
        V[0] = V_L0 + V_S0
        kum_ret[0] = 0
        
        # Initialize weight and position arrays
        pi_L = np.zeros(n_periods + 1)
        pi_S = np.zeros(n_periods + 1)
        
        if self.return_weights:
            w = np.zeros(n_periods + 1)
            for i in range(n_periods):
                w[i] = self.get_w(i)
        else:
            w = None

        # Main calculation loop
        for i in range(n_periods):
            w_t = self.get_w(i)
            pi_S[i], pi_L[i] = self.pi(w_t, V_L[i], V_S[i])
            
            V_L[i + 1] = V_L[i] + (returns[i] * pi_L[i])
            V_S[i + 1] = V_S[i] + (returns[i] * pi_S[i])
            V[i + 1] = V_L[i + 1] + V_S[i + 1]
            
            # Calculate daily portfolio return
            portfolio_returns[i] = (V[i + 1] - V[i]) / V[i] if V[i] != 0 else 0
            
            # Calculate cumulative return (fixed this calculation)
            kum_ret[i + 1] = (V[i + 1] - V[0]) / V[0]
        
        # Calculate performance metrics
        sharpe_ratio = self.calculate_sharpe_ratio(portfolio_returns)
        
        # Calculate max drawdown
        cumulative_max = np.maximum.accumulate(V)
        drawdown = (V - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdown)
        
        return {
            'metrics' : {
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
            },
            'info' : {
            'V_L': V_L,
            'V_S': V_S,
            'V': V,
            'pi_L': pi_L,
            'pi_S': pi_S,
            'weights': w,
            'cumulative_returns': kum_ret,
            'daily_returns': portfolio_returns
            }
            
        }