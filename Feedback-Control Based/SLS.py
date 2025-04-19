import functions #contains stock-preprocessing functions and verify input for K and alpha and w
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp
import ttingo_api
from typing import Callable



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


    

    

    
        
