import functions #contains stock-preprocessing functions and verify input for K and alpha and w
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp
import ttingo_api
from typing import Callable



#----------------------------------------------------

class K:
    #list all available methods
    methods_avail = ["custom","RL","static","DLP","adaptive"]

    def __init__(self, methods):
        if methods not in self.methods_avail:
            raise ValueError(f"Invalid method: {methods}. Choose from {self.methods_avail}.")
        self.method = methods

#------------------------------------------------------


class SLS:
    '''
    SLS: Simultaneous Long-Short Trading Algorithm
    Args:
        K(int, float, function): K can be constant or a function
        Stock:         
    
    Return:
        Dataframe: cumulative gains and their respective index
    
    '''
    def __init__(self,k,stocks, tickers):       
        self.k = k
        self.stocks = stocks
        self.tickers = tickers
        
    def get_k(self):
        return self.k
    # In this most basic SLS method, we only consider one stock
    def get_stock(self):
        return self.stocks
    
    #In gains function, stocks
    def gains_L(self, k, I_0, p, p_0):
        """Calculate long gains."""
        return (I_0 / k) * ((p / p_0) ** k - 1)

    def gains_S(self, k, I_0, p, p_0):
        """Calculate short gains."""
        return (I_0 / k) * ((p / p_0) ** (-k) - 1)

    def sls(self):
        """
        Calculate long and short returns for the given stock.

        Args:
            ticker (str): The stock ticker to process.
        
        Returns:
            tuple: Two lists (gL, gS) for long and short gains respectively.
        """
        stock = self.get_stock()  # Get the stock data
        k = self.get_k()  # Get the value of k
        I_0 = 2000  # Initial investment (arbitrary positive value)
        gL = [0] * stock.shape[0]  # Initialize list for long gains
        gS = [0] * stock.shape[0]  # Initialize list for short gains
        g = [0] * stock.shape[0] 
        p_0 = stock.iloc[0]  # The price of the stock at time 0 (initial price)

        # Loop through the stock prices, starting from the second time step (index 1)
        for i in range(1, stock.shape[0]):
            p = stock.iloc[i]  # Current stock price
            gL[i] = self.gains_L(k, I_0, p, p_0)  # Calculate long gains for this time step
            gS[i] = self.gains_S(k, I_0, p, p_0)  # Calculate short gains for this time step
            g[i] = gL[i] + gS[i]
        return gL, gS, g

#------------------------------------------------------

class DLP:
    '''
    SLS: Simultaneous Long-Short Trading Algorithm
    Args:
        w(int, float, function): w needs to be a function
        alpha(0,1): alpha needs to be a function
        epsilon(0,1): denotes transaction cost
        Stocks(JSON, Dict): Is stock prices in Json or dictionary format. The key value must include Time, and Assets 
        (e.g.,
        {Time: [t1, t2,t ..., tn]
        AAPL: [p1, p2, ..., pn]}
                    .
                    .
                    .
        )
    
    Return:
        Dataframe: cumulative gains and their respective index
    
    '''
    def __init__(self,w,alpha,stocks, tickers):
        if (functions.is_function(w) == False):
            raise ValueError("w must be a function or a numeric constant")     
        if (functions.is_function(alpha) == False):
            raise ValueError("alpha must be a function or a numeric constant")       
        self.alpha = alpha
        self.w = w
        self.stocks = stocks

    def get_stock(self):
        return self.stocks

    def get_w(self):
        return self.w
    
    def get_alpha(self):
        return self.alpha
    
    def initial_acc_value(self,v_0 = 2000):
        alpha = self.get_alpha()

        vl0 = alpha*v_0
        vs0 = (1-alpha)*v_0

        return vl0, vs0
    
    def pi_l(self, w, v_l):
        pi_l = w*v_l
        return pi_l
    def pi_l(self, w, v_s):
        pi_s = w*v_s
        return pi_s
    
    
    def returns(self):
        stock = self.get_stock()
        returns = stock.pct_change().dropna()
        return self.returns
        
    def dlp(self):
        returns = returns()
        w = self.get_w
        alpha = self.get_alpha

        V_L[0] = V_L0
        V_S[0] = V_S0

        for k in range(n):
            V_L[k + 1] = V_L[k] + X[k] * pi_L[k] - epsilon * pi_L[k]
            V_S[k + 1] = V_S[k] + X[k] * pi_S[k] - epsilon * abs(pi_S[k])
            return vL, vS, v





    
    

    

    
        
