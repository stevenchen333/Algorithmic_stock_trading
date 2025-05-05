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
    return np.log(1+t/252*(np.exp(1)-1))
def w3(t):
    denominator = (0.02 / 252) * t - 0.01
    if denominator == 0:
        return 0  # or handle the division by zero case as appropriate for your application
    return 0.5 * (np.sin(1 / denominator) + 1)








np.random.seed(42)  # For reproducibility

dlp1 = DLP(stocks = data_df, w = w1, init_value=200, return_weights=True)
_,_, v1, _,_,w_1 = dlp1.dlp()


dlp2 = DLP(stocks = data_df, w = w2, init_value=200, return_weights=True)
_,_, v2, _,_,w_2 = dlp2.dlp()


dlp3 = DLP(stocks = data_df, w = w3, init_value=200, return_weights=True)
_,_, v3, _,_,w_3 = dlp3.dlp()







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
    agent.load("multi_regime_agent_final.npz")
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
        'V': np.array(portfolio_values),
        'actions': np.array(actions),
        'returns': np.array(returns),
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio
    }
    
    return results



# Get RL agent's portfolio values
agent_results = evaluate_agent_performance(env, agent)
v_rl = agent_results['V']
action_rl = agent_results['actions']

# Create figure with 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 9))

# Flatten axes for easier iteration
axs = axs.flatten()

# Plot data with twin axes
def plot_strategy(ax, portfolio_values, strategy_name, color, ticker='TSLA'):
    """
    Plot trading strategy with stock price on twin axes
    
    Args:
        ax: Matplotlib axis object
        portfolio_values: Array of portfolio values
        strategy_name: Name of the strategy for legend
        color: Color for portfolio value line
        ticker: Stock ticker symbol (default: 'TSLA')
    """
    # First axis for portfolio values
    ax1 = ax
    ax1.plot(portfolio_values, label=strategy_name, color=color)
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('V ($)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    # Second axis for stock prices
    ax2 = ax1.twinx()
    ax2.plot(data_df[ticker.lower()], color='grey', alpha=0.5, 
            label=f'{ticker.upper()} Price')
    ax2.set_ylabel('Stock Price ($)', color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Plot each strategy
plot_strategy(axs[0], w_1, 'Buy & Hold', 'blue')
plot_strategy(axs[1], w_2, 'Logarithmic', 'green')
plot_strategy(axs[2], w_3, 'Active day trading', 'orange')
plot_strategy(axs[3], action_rl, 'RL Agent', 'red')

# Set titles
titles = ['Buy & Hold Strategy (w1)', 'Logarithmic Strategy (w2)', 
        'Active day trading (w3)', 'RL Agent Strategy']
for ax, title in zip(axs, titles):
    ax.set_title(title)

plt.tight_layout()
plt.show()

# Print performance metrics
print("\nPerformance Metrics:")
print("-" * 50)
print(f"Buy & Hold Final Value: ${v1[-1]:.2f}")
print(f"Logarithmic Final Value: ${v2[-1]:.2f}")
print(f"RL Agent Final Value: ${v_rl[-1]:.2f}")
print(f"RL Agent Sharpe Ratio: {agent_results['sharpe_ratio']:.2f}")



def plot_strategy_comparison(portfolio_values_dict, stock_prices, figsize=(15, 8)):
    """
    Plot comparison of different trading strategies against stock price.
    
    Args:
        portfolio_values_dict (dict): Dictionary of strategy names and their portfolio values
        stock_prices (array-like): Stock price data
        figsize (tuple): Figure size in inches
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create main axis and its twin
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Color map for different strategies
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
    lines = []
    
    # Plot each strategy
    for (strategy_name, values), color in zip(portfolio_values_dict.items(), colors):
        l = ax1.plot(values, 
                    label=strategy_name, 
                    color=color, 
                    linewidth=2)
        lines.extend(l)
    
    # Plot stock price on twin axis
    l_stock = ax2.plot(stock_prices, 
                    label='Stock Price', 
                    color='gray', 
                    linestyle='--', 
                    alpha=0.5)
    lines.extend(l_stock)
    
    # Set labels and title
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Account Value ($)')
    ax2.set_ylabel('Stock Price ($)', color='gray')
    
    # Get labels for legend
    labels = [l.get_label() for l in lines]
    
    # Create legend outside the plot
    plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1.15, 0.5))
    
    plt.title('Account Value Comparison Across All Strategies')
    ax1.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return ax1, ax2

# Example usage:
    # Example data
strategies = {
    'Buy & Hold (w1)': v1,
    'Logarithmic (w2)': v2,
    'Oscillating (w3)': v3,
    'RL Agent': v_rl
}

# Plot comparison
ax1, ax2 = plot_strategy_comparison(
    portfolio_values_dict=strategies,
    stock_prices=data_df['tsla']
)
plt.show()

plt.close('all')

#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----

# TODO: Try other stocks/bitcoin
with open("Feedback-Control Based/nvda 2021-01-04 00:00:00 - 2024-11-29 00:00:00.json", "r") as f:
    data = json.load(f)

data_df = pd.DataFrame(data)[0:100]
dlp1 = DLP(stocks = data_df, w = w1, init_value=200, return_weights=True)
_,_, v1, _,_,w_1 = dlp1.dlp()


dlp2 = DLP(stocks = data_df, w = w2, init_value=200, return_weights=True)
_,_, v2, _,_,w_2 = dlp2.dlp()


dlp3 = DLP(stocks = data_df, w = w3, init_value=200, return_weights=True)
_,_, v3, _,_,w_3 = dlp3.dlp()

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
    agent.load("multi_regime_agent_final.npz")
    print("Loaded trained agent weights")
except FileNotFoundError:
    print("Trained weights not found. Using untrained agent.")


# Get RL agent's portfolio values
agent_results = evaluate_agent_performance(env, agent)
v_rl = agent_results['V']
action_rl = agent_results['actions']




fig, axs = plt.subplots(2, 2, figsize=(12, 9))
axs = axs.flatten()  # Flatten for easy indexing

# Now plot again (this will work)
plot_strategy(axs[0], w_1, 'Buy & Hold', 'blue', 'nvda')
plot_strategy(axs[1], w_2, 'Logarithmic', 'green', 'nvda')
plot_strategy(axs[2], w_3, 'Active day trading', 'orange', 'nvda')
plot_strategy(axs[3], action_rl, 'RL Agent', 'red', 'nvda')

# Set titles again
titles = ['Buy & Hold Strategy (w1)', 'Logarithmic Strategy (w2)', 
        'Active day trading (w3)', 'RL Agent Strategy']
for ax, title in zip(axs, titles):
    ax.set_title(title)

plt.tight_layout()
plt.show()


# Print performance metrics
print("\nPerformance Metrics:")
print("-" * 50)
print(f"Buy & Hold Final Value: ${v1[-1]:.2f}")
print(f"Logarithmic Final Value: ${v2[-1]:.2f}")
print(f"RL Agent Final Value: ${v_rl[-1]:.2f}")
print(f"RL Agent Sharpe Ratio: {agent_results['sharpe_ratio']:.2f}")




# Example usage:
    # Example data
strategies = {
    'Buy & Hold (w1)': v1,
    'Logarithmic (w2)': v2,
    'Oscillating (w3)': v3,
    'RL Agent': v_rl
}

# Plot comparison
ax1, ax2 = plot_strategy_comparison(
    portfolio_values_dict=strategies,
    stock_prices=data_df['nvda']
)
plt.show()


#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----






