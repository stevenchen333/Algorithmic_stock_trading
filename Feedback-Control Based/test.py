import json
import pandas as pd

import numpy as np
from AC_PG_BVTD import TradingEnvironment, LinearPolicyGradientAgent,train_agent, evaluate_agent, plot_results
from ttingo_api import retrieve_stock
from constants import ttingo_api_key
from DLP import DLP
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#retrieve_stock(tickers = ['AAPL'], start_date= '2015-01-01',  end_date='2025-01-01', save_file=True, token = ttingo_api_key)


with open("Feedback-Control Based/tsla_dlptest1 2023-01-03 00:00:00 - 2023-09-29 00:00:00.json", "r") as f:
    data = json.load(f)

data_df = pd.DataFrame(data)

def simulate_market_data(n_steps=500,mu = 0.0008, sigma = 0.02,
                        jump_intensity=0.02, jump_size=0.03,
                        initial_price=100.0):
    normal_returns = np.random.normal(mu, sigma, n_steps)
    
    # Generate jumps
    jumps = np.random.binomial(1, jump_intensity, n_steps) * \
            np.random.choice([-1, 1], n_steps) * jump_size
    
    # Combine normal returns with jumps
    returns = normal_returns + jumps
    returns = np.clip(returns, -0.2, 0.2)  # Bound returns
    
    # Convert returns to price path
    prices = initial_price * np.cumprod(1 + returns)
    
    return returns, prices


returns_gbm, price_gbm = simulate_market_data()




# Logarithmic weight function from your question
def w1(t):
    return 0.8
def w2(t):
    return np.exp(-(t - 252)**2 / 1000000)
def w3(t):
    denominator = (0.02 / 252) * t - 0.01
    if denominator == 0:
        return 0  # or handle the division by zero case as appropriate for your application
    return 0.5 * (np.sin(1 / denominator) + 1)








np.random.seed(42)  # For reproducibility

dlp1 = DLP(stocks = data_df, w = w1, init_value=200)
_,_, v1, _,_,w_1 = dlp1.dlp()



dlp2 = DLP(stocks = data_df, w = w2, init_value=200)
_,_, v2, _,_,w_2 = dlp2.dlp()


dlp3 = DLP(stocks = data_df, w = w3, init_value=200)
_,_, v3, _,_,_ = dlp3.dlp()







env = TradingEnvironment(
    returns_data=dlp1.returns(),  # Make sure you have returns data
    initial_balance=200,
    transaction_cost=0.0,
    alpha=0.5,
    window_size=10,
    max_steps=len(data_df) - 11
)

# Initialize the agent with the environment's state dimension
state_dim = env.observation_space.shape[0]  # Get state dimension from environment
agent = LinearPolicyGradientAgent(
    state_size=state_dim,
    action_size=0.001,
    action_bound=1.0  # Maximum allowed weight
)

try:
    agent.load("Feedback-Control Based/multi_regime_agent.npz")
    print("Loaded trained agent weights")
except FileNotFoundError:
    print("Trained weights not found. Using untrained agent.")
def evaluate_agent_performance(env, agent, initial_balance=200):
    """
    Evaluate the agent's performance and return V, returns, and actions
    """
    state = env.reset()
    done = False
    
    # Initialize tracking lists
    portfolio_values = [initial_balance]
    actions = []
    returns = []
    
    while not done:
        # Get action from agent
        action_mean = np.dot(state, agent.theta)
        action = np.clip(action_mean, 0, agent.action_bound)
        action = np.array([action])
        
        # Take step in environment
        next_state, reward, done, info = env.step(action)
        
        # Record metrics
        portfolio_values.append(info['account_value'])
        actions.append(action[0])
        returns.append(info['return'])
        
        # Update state
        state = next_state
    
    # Calculate performance metrics
    total_return = (portfolio_values[-1]/initial_balance - 1) * 100
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    results = {
        'portfolio_values': np.array(portfolio_values),
        'actions': np.array(actions),
        'returns': np.array(returns),
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio
    }
    
    return results



# Get RL agent's portfolio values
agent_results = evaluate_agent_performance(env, agent)
v_rl = agent_results['portfolio_values']

# Create figure with 4 subplots
plt.figure(figsize=(12, 9))

# Plot Buy & Hold Strategy (v1)
plt.subplot(2, 2, 1)
plt.plot(v1, label='Buy & Hold', color='blue')
plt.title('Buy & Hold Strategy (w1)')
plt.xlabel('Trading Days')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.legend()

# Plot Logarithmic Strategy (v2)
plt.subplot(2, 2, 2)
plt.plot(v2, label='Logarithmic', color='green')
plt.title('Logarithmic Strategy (w2)')
plt.xlabel('Trading Days')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.legend()

# Plot RL Agent Strategy
plt.subplot(2, 2, 3)
plt.plot(v_rl, label='RL Agent', color='red')
plt.title('RL Agent Strategy')
plt.xlabel('Trading Days')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.legend()

# Plot Stock Price
plt.subplot(2, 2, 4)
plt.plot(data_df['tsla'], label='TSLA Price', color='purple')
plt.title('AAPL Stock Price')
plt.xlabel('Trading Days')
plt.ylabel('Price ($)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Print performance metrics
print("\nPerformance Metrics:")
print("-" * 50)
print(f"Buy & Hold Final Value: ${v1[-1]:.2f}")
print(f"Logarithmic Final Value: ${v2[-1]:.2f}")
print(f"RL Agent Final Value: ${v_rl[-1]:.2f}")
print(f"RL Agent Sharpe Ratio: {agent_results['sharpe_ratio']:.2f}")

# Create figure for comparing all strategies
plt.figure(figsize=(15, 8))

# Create the main axis and its twin
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot strategies on main axis
l1 = ax1.plot(v1, label='Buy & Hold (w1)', color='blue', linewidth=2)
l2 = ax1.plot(v2, label='Logarithmic (w2)', color='green', linewidth=2)
l3 = ax1.plot(v3, label='Oscillating (w3)', color='red', linewidth=2)
l4 = ax1.plot(v_rl, label='RL Agent', color='purple', linewidth=2)

# Plot stock price on twin axis
l5 = ax2.plot(data_df['tsla'], label='TSLA Price', color='gray', linestyle='--', alpha=0.5)

# Set labels and title
ax1.set_xlabel('Trading Days')
ax1.set_ylabel('Portfolio Value ($)')
ax2.set_ylabel('Stock Price ($)', color='gray')

# Combine all lines and labels
lines = l1 + l2 + l3 + l4 + l5
labels = [l.get_label() for l in lines]

# Create legend outside the plot
plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1.15, 0.5))

plt.title('Portfolio Value Comparison Across All Strategies')
ax1.grid(True, alpha=0.3)

# Adjust layout to prevent legend cutoff
plt.tight_layout()
plt.show()

#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----

env = TradingEnvironment(
    returns_data=returns_gbm,  # Make sure you have returns data
    initial_balance=200,
    transaction_cost=0.0,
    alpha=0.5,
    window_size=10,
    max_steps=len(returns_gbm) - 11
)

# Initialize the agent with the environment's state dimension
state_dim = env.observation_space.shape[0]  # Get state dimension from environment
agent = LinearPolicyGradientAgent(
    state_size=state_dim,
    action_size=0.001,
    action_bound=1.0  # Maximum allowed weight
)



# Get RL agent's portfolio values
agent_results = evaluate_agent_performance(env, agent)
v_rl = agent_results['portfolio_values']
gbm_df = pd.DataFrame({'ret': price_gbm,'gbm' :price_gbm})

print(gbm_df)
dlp1 = DLP(stocks = gbm_df, w = w1, init_value=200)
_,_, v1, _,_,w_1 = dlp1.dlp()



dlp2 = DLP(stocks = gbm_df, w = w2, init_value=200)
_,_, v2, _,_,w_2 = dlp2.dlp()


dlp3 = DLP(stocks = gbm_df, w = w3, init_value=200)
_,_, v3, _,_,_ = dlp3.dlp()
# Create figure for comparing all strategies
plt.figure(figsize=(15, 8))

# Create the main axis and its twin
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot strategies on main axis
l1 = ax1.plot(v1, label='Buy & Hold (w1)', color='blue', linewidth=2)
l2 = ax1.plot(v2, label='Logarithmic (w2)', color='green', linewidth=2)
l3 = ax1.plot(v3, label='Oscillating (w3)', color='red', linewidth=2)
l4 = ax1.plot(v_rl, label='RL Agent', color='purple', linewidth=2)

# Plot stock price on twin axis
l5 = ax2.plot(price_gbm, label='AAPL Price', color='gray', linestyle='--', alpha=0.5)

# Set labels and title
ax1.set_xlabel('Trading Days')
ax1.set_ylabel('Portfolio Value ($)')
ax2.set_ylabel('Stock Price ($)', color='gray')

# Combine all lines and labels
lines = l1 + l2 + l3 + l4 + l5
labels = [l.get_label() for l in lines]

# Create legend outside the plot
plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1.15, 0.5))

plt.title('Portfolio Value Comparison Across All Strategies')
ax1.grid(True, alpha=0.3)

# Adjust layout to prevent legend cutoff
plt.tight_layout()
plt.show()