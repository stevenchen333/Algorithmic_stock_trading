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
    def __init__(self,w,alpha,stocks, tickers):
        self.tickers = tickers   
        self.alpha = alpha
        self.w = w
        self.stocks = stocks

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
   
    def initial_acc_value(self,v_0 = 2000):
        alpha = self.get_alpha()

        vl0 = alpha*v_0
        vs0 = (1-alpha)*v_0

        return vl0, vs0
    
    def pi(self, w, v_l,v_s):
        pi_l = w*v_l
        pi_s = w*v_s
        return pi_s, pi_l
    
        
    
    
    def returns(self):
        stock = self.get_stock()
        stock_returns = (stock[1:] - stock[:-1]) / stock[:-1]
        return stock_returns
        
    def dlp(self):
        returns = returns()
        w = self.get_w()
        alpha = self.get_alpha()

        V_L0, V_S0 = self.initial_acc_value()
        V_L = [0] * len(returns)
        V_S = [0] * len(returns)
        V = [0] * (len(returns))

        V_L[0] = V_L0
        V_S[0] = V_S0
        V[0] = V_L0 + V_S0

        pi_L = [0] * len(returns)
        pi_S = [0] * len(returns)

        for i in range(len(returns)):
            pi_L[i], pi_S[i] = self.pi(w, V_L[i], V_S[i])
            V_L[i+1] = V_L[i] +(returns[i] * pi_L[i])
            V_S[i+1] = V_S[i] +(returns[i] * pi_S[i])
            V[i+1] = V_L[i+1] + V_S[i+1]
        return V_L, V_S,V, pi_L, pi_S
#------------------------------------------------------
