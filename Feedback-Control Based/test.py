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
#TODO: Test RL Agent on SP 500, MSFT, GOOGL

with open("test_spystock_data_2024-01-02_to_2025-01-01.json", "r") as f:
    data = json.load(f)
data_df = pd.DataFrame({
    'times': data['SPY']['dates'],
    'spy': data['SPY']['close']
})
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
    excess_returns = np.array(returns) - 0  # Subtract risk-free rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    results = {
        'V': np.array(portfolio_values),
        'actions': np.array(actions),
        'returns': np.array(returns),
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio
    }
    
    return results
# Plot data with twin axes
def plot_strategy(ax, portfolio_values, strategy_name, color, ticker='SPY'):
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
def plot_strategy_comparison(portfolio_values_dict, stock_prices, figsize=(15, 8)):

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
    ax1.set_ylabel('Cumulative Return (%)')
    ax2.set_ylabel('Stock Price ($)', color='gray')
    
    # Get labels for legend
    labels = [l.get_label() for l in lines]
    
    # Create legend outside the plot
    plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1.15, 0.5))
    
    plt.title('Cumulative Return Comparison Across All Strategies')
    ax1.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return ax1, ax2






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
result_dlp1 = dlp1.dlp()


dlp2 = DLP(stocks = data_df, w = w2, init_value=200, return_weights=True)
result_dlp2 = dlp2.dlp()


dlp3 = DLP(stocks = data_df, w = w3, init_value=200, return_weights=True)
result_dlp3 = dlp3.dlp()





env = TradingEnvironment(
    data_dict=data,
    ticker = "SPY",    initial_balance=200,
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
    agent.load("multi_stock_agent_final.npz")
    print("Loaded trained agent weights")
except FileNotFoundError:
    print("Trained weights not found. Using untrained agent.")




# Get RL agent's portfolio values
agent_results = evaluate_agent_performance(env, agent)
v_rl = agent_results['V']
action_rl = agent_results['actions']
kum_ret_rl = [0] * (len(v_rl) + 1)

kum_ret_rl[0] = 0

for i in range(len(v_rl)):
    kum_ret_rl[i] = (v_rl[i] - v_rl[0]) / v_rl[0]

cumulative_max = np.maximum.accumulate(v_rl)
drawdown = (v_rl - cumulative_max) / cumulative_max
max_drawdown_rl = np.min(drawdown)




print(f"Performance of DLP: \n")
print(f"Buy and Hold{result_dlp1["metrics"]}, G&L: {result_dlp1["info"]['cumulative_returns'][-1]} ")
print(f"Logarithmic{result_dlp2["metrics"]}, G&L: {result_dlp2["info"]['cumulative_returns'][-1]}")
print(f"Active trading{result_dlp3["metrics"]}, G&L: {result_dlp3["info"]['cumulative_returns'][-1]}")

print(f"Performance of DLP w rl: \n")
print(f'max_drawdown: {max_drawdown_rl}, sharpe_ratio: {agent_results['sharpe_ratio']}, G&L: {kum_ret_rl[-2]}')




# Example usage:
    # Example data
strategies = {
    '(w1)': result_dlp1['info']['cumulative_returns'][0:len(kum_ret_rl)],
    '(w2)': result_dlp2['info']['cumulative_returns'][0:len(kum_ret_rl)],
    '(w3)': result_dlp3['info']['cumulative_returns'][0:len(kum_ret_rl)],
    'RL Agent': kum_ret_rl[0:len(kum_ret_rl)-1]
}

# Plot comparison
ax1, ax2 = plot_strategy_comparison(
    portfolio_values_dict=strategies,
    stock_prices=data_df['spy']
)
plt.show()

plt.close('all')
#---------#---------#---------#---------#---------#---------#---------#---------#---------#---------#---------#---------


returns_gbm, price_gbm = simulate_market_data()

np.random.seed(435)
# Parameters for bear market GBM with jumps
mu = -0.001  # Negative drift for bear market
sigma = 0.05
jump_intensity = 0.01  # Higher chance of jumps in volatile bear market
jump_size = 0.005
initial_price = 100.0
n_steps = 252

# Generate GBM with jumps
normal_returns = np.random.normal(mu, sigma, n_steps)
jumps = np.random.binomial(1, jump_intensity, n_steps) * np.random.choice([-1, 1], n_steps) * jump_size
returns = normal_returns + jumps
returns = np.clip(returns, -0.2, 0.2)  # Bound returns

# Generate price path
prices = initial_price * np.cumprod(1 + returns)

# Create data_dict in required format
dates = pd.date_range(start="2024-01-01", periods=n_steps).strftime('%Y-%m-%d').tolist()
data_dict = {
    'GBM': {
        'open': prices.tolist(),
        'close': prices.tolist(),
        'high': (prices * 1.01).tolist(),  # Add small variation for high/low
        'low': (prices * 0.99).tolist(),
        'volume': np.random.lognormal(8, 1, n_steps).astype(int).tolist(),
        'dates': dates
    }
}

# Create DataFrame for DLP
data_df = pd.DataFrame({
    'times': dates,
    'gbm': prices
})



dlp1 = DLP(stocks = data_df, w = w1, init_value=200, return_weights=True)
result_dlp1 = dlp1.dlp()


dlp2 = DLP(stocks = data_df, w = w2, init_value=200, return_weights=True)
result_dlp2 = dlp2.dlp()


dlp3 = DLP(stocks = data_df, w = w3, init_value=200, return_weights=True)
result_dlp3 = dlp3.dlp()





env = TradingEnvironment(
    data_dict=data_dict,
    ticker = "GBM",    initial_balance=200,
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
    agent.load("multi_stock_agent_final.npz")
    print("Loaded trained agent weights")
except FileNotFoundError:
    print("Trained weights not found. Using untrained agent.")




# Get RL agent's portfolio values
agent_results = evaluate_agent_performance(env, agent)
v_rl = agent_results['V']
action_rl = agent_results['actions']
kum_ret_rl = [0] * (len(v_rl) + 1)

kum_ret_rl[0] = 0

for i in range(len(v_rl)):
    kum_ret_rl[i] = (v_rl[i] - v_rl[0]) / v_rl[0]

cumulative_max = np.maximum.accumulate(v_rl)
drawdown = (v_rl - cumulative_max) / cumulative_max
max_drawdown_rl = np.min(drawdown)




print(f"Performance of DLP: \n")
print(f"Buy and Hold{result_dlp1["metrics"]}, G&L: {result_dlp1["info"]['cumulative_returns'][-1]} ")
print(f"Logarithmic{result_dlp2["metrics"]}, G&L: {result_dlp2["info"]['cumulative_returns'][-1]}")
print(f"Active trading{result_dlp3["metrics"]}, G&L: {result_dlp3["info"]['cumulative_returns'][-1]}")

print(f"Performance of DLP w rl: \n")
print(f'max_drawdown: {max_drawdown_rl}, sharpe_ratio: {agent_results['sharpe_ratio']}, G&L: {kum_ret_rl[-2]}')




# Example usage:
    # Example data
strategies = {
    '(w0)': result_dlp1['info']['cumulative_returns'][0:len(kum_ret_rl)],
    '(w1)': result_dlp2['info']['cumulative_returns'][0:len(kum_ret_rl)],
    '(w2)': result_dlp3['info']['cumulative_returns'][0:len(kum_ret_rl)],
    'RL Agent': kum_ret_rl[0:len(kum_ret_rl)-1]
}

# Plot comparison
ax1, ax2 = plot_strategy_comparison(
    portfolio_values_dict=strategies,
    stock_prices=data_df['gbm']
)
plt.show()

plt.close('all')



#----------
