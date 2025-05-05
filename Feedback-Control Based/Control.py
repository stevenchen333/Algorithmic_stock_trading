#Optimum weights for DLP using Deep Reinforcement Learning

'''
Let X_k be the return of asset S at time k
MDP formulation:
# Action space: w(k), w(k) in (0,1) 
# States s_t = [X_t-k, X_t-k+1, ..., X_t-1]
        #Drawdown
# Reward: R_t = V(t+1) - V(t)
'''

# we use Natural Actor Critic (NAC) algorithm

import numpy as np

class 

