import numpy as np
from SLS_AC_PG_BVTD import TradingEnvironment, LinearPolicyGradientAgent,train_agent, evaluate_agent, plot_results
import matplotlib.pyplot as plt
import json


#TODO: Instead of training in BGM, train on TOP 10 stocks in SPY from https://finance.yahoo.com/quote/SPY/
# def simulate_realistic_market_data(n_days=2000, stock='normal'):
#     """
#     Simulate realistic stock returns using stock-specific GBM parameters
    
#     Parameters:
#     - n_days: Number of trading days to simulate
#     - stock: Market stock ('bull', 'bear', 'normal', 'volatile', or 'random')
    
#     Returns:
#     - returns: Daily returns
#     - prices: Corresponding price series
#     - params: Dictionary of parameters used
#     """
#     # Define realistic parameters for different market stocks
#     stocks = {
#         'bull': {
#             'mu': 0.0007,           # ~18% annualized return
#             'sigma': 0.012,         # ~19% annualized volatility
#             'jump_intensity': 0.02, # Rare jumps
#             'jump_size': 0.02       # Small jumps
#         },
#         'bear': {
#             'mu': -0.0005,          # ~-12% annualized return
#             'sigma': 0.018,         # ~28% annualized volatility
#             'jump_intensity': 0.04, # More frequent jumps
#             'jump_size': 0.03       # Moderate jumps
#         },
#         'normal': {
#             'mu': 0.0003,           # ~8% annualized return
#             'sigma': 0.01,          # ~16% annualized volatility
#             'jump_intensity': 0.03, # Occasional jumps
#             'jump_size': 0.02       # Small jumps
#         },
#         'volatile': {
#             'mu': 0.0001,           # ~2.5% annualized return
#             'sigma': 0.025,         # ~40% annualized volatility
#             'jump_intensity': 0.06, # Frequent jumps
#             'jump_size': 0.04       # Larger jumps
#         }
#     }
    
#     # Choose stock parameters
#     if stock == 'random':
#         # Randomly select a stock for each segment
#         n_segments = np.random.randint(3, 7)  # 3-6 different stocks
#         segment_length = n_days // n_segments
#         returns = np.array([])
        
#         for i in range(n_segments):
#             chosen_stock = np.random.choice(['bull', 'bear', 'normal', 'volatile'])
#             stock_params = stocks[chosen_stock]
#             segment_returns = simulate_segment(
#                 segment_length, 
#                 stock_params['mu'],
#                 stock_params['sigma'],
#                 stock_params['jump_intensity'],
#                 stock_params['jump_size']
#             )
#             returns = np.append(returns, segment_returns)
        
#         # Trim or extend to exact length
#         if len(returns) > n_days:
#             returns = returns[:n_days]
#         elif len(returns) < n_days:
#             # Generate remaining days with last stock
#             remaining = n_days - len(returns)
#             extra_returns = simulate_segment(
#                 remaining,
#                 stock_params['mu'],
#                 stock_params['sigma'],
#                 stock_params['jump_intensity'],
#                 stock_params['jump_size']
#             )
#             returns = np.append(returns, extra_returns)
#     else:
#         # Use specific stock
#         stock_params = stocks[stock]
#         returns = simulate_segment(
#             n_days,
#             stock_params['mu'],
#             stock_params['sigma'],
#             stock_params['jump_intensity'],
#             stock_params['jump_size']
#         )
    
#     # Calculate price series (starting at 100)
#     prices = 100 * np.cumprod(1 + returns)
    
#     # Return both returns and prices
#     return returns, prices, stock_params

# def simulate_segment(n_days, mu, sigma, jump_intensity, jump_size):
"""Simulate a market segment with consistent parameters"""
    # Generate base returns
    # base_returns = np.random.normal(mu, sigma, n_days)
    
    # # Generate jumps
    # jumps = np.random.binomial(1, jump_intensity, n_days) * np.random.choice([-1, 1], n_days) * jump_size
    
    # # Combine and clip to avoid unrealistic returns
    # returns = base_returns + jumps
    # returns = np.clip(returns, -0.15, 0.15)  # Limit daily moves to ±15%
    
    # return returns

def days_randomizer(data_dict, stock, window_size=10):
    """
    Randomize the length of training data and return a sliced dataset
    
    Args:
        data_dict (dict): Dictionary containing stock data
        stock (str): Stock ticker
        window_size (int): Window size for observation
        
    Returns:
        dict: Sliced data dictionary with randomized length
    """
    # Get total available data length
    total_length = len(data_dict[stock]['open'])
    
    # Generate random number of days between 252 (1 year) and 700 (~ 2.8 years)
    n_days = np.random.randint(400, 701)
    
    # Ensure we have enough data
    if total_length <= n_days + window_size:
        n_days = total_length - window_size - 1
        start_idx = 0
    else:
        # Calculate valid starting points
        max_start_idx = total_length - n_days - window_size
        # Choose random starting point
        start_idx = np.random.randint(0, max_start_idx)
    
    # Create sliced dataset
    sliced_data = {
        stock: {
            'open': data_dict[stock]['open'][start_idx:start_idx + n_days + window_size],
            'close': data_dict[stock]['close'][start_idx:start_idx + n_days + window_size],
            'high': data_dict[stock]['high'][start_idx:start_idx + n_days + window_size],
            'low': data_dict[stock]['low'][start_idx:start_idx + n_days + window_size],
            'volume': data_dict[stock]['volume'][start_idx:start_idx + n_days + window_size],
            'dates': data_dict[stock]['dates'][start_idx:start_idx + n_days + window_size]
        }
    }
    
    return sliced_data, n_days, start_idx



def train_agent_across_stocks(training_path, agent_class, epochs_per_stock=1000):
    with open(training_path, "r") as f:
        training_data = json.load(f)

    stocks_curriculum = list(training_data.keys())
    
    # Initialize storage
    all_training_results = {}
    agent = None
    action_size = 1  # K is a single value
    
    for stock in stocks_curriculum:
        print(f"\n===== Training on {stock.upper()} stock =====")
        print(f"Training for {epochs_per_stock} epochs")
        training_data_slice, n_days, start_idx = days_randomizer(training_data, stock)
        
        # Create environment with SLS parameters
        env = TradingEnvironment(
            data_dict=training_data,
            ticker=stock,
            initial_balance=1000,
            transaction_cost=0.0,
            alpha=0.5,  # Split between long and short
            window_size=10,
            max_steps=n_days - 11
        )
        
        # Initialize agent if first stock
        if agent is None:
            state_size = env.observation_space.shape[0]
            agent = agent_class(state_size, action_size, env.K_max)  # Use K_max instead of w_max
        
        # Train agent on this stock
        rewards, avg_rewards, asset_values = train_agent(
            env, agent, episodes=epochs_per_stock
        )
        
        # Collect K values by running one episode
        K_values = []
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            K_values.append(action)
            next_state, reward, done, info = env.step(action)
            state = next_state
        
        # Store results for this stock
        all_training_results[stock] = {
            'rewards': rewards,
            'avg_rewards': avg_rewards,
            'asset_values': asset_values,
            'K_values': K_values  # Store K values instead of actions
        }
        
        # Final evaluation
        eval_rewards, eval_values, K_history, returns_history = evaluate_agent(env, agent, episodes=100)
        print(f"\nFinal evaluation on {stock} stock - "
            f"Avg Final Value: {np.mean(eval_values):.2f}")
        
        # Save stock-specific model
        agent.save(f"sls_agent_{stock}_final.npz")
    
    # Save final model
    agent.save("sls_multi_stock_agent_final.npz")
    
    return agent, all_training_results

agent, training_results = train_agent_across_stocks(training_path="trainstock_data_2020-01-01_to_2024-01-01.json",
    agent_class=LinearPolicyGradientAgent, 
    epochs_per_stock=1000
)

def plot_training_across_stocks(training_results):
    stocks = list(training_results.keys())
    fig, axs = plt.subplots(len(stocks), 4, figsize=(20, 4*len(stocks)))
    
    for i, stock in enumerate(stocks):
        # Plot rewards
        axs[i, 0].plot(training_results[stock]['avg_rewards'])
        axs[i, 0].set_title(f"Training Rewards - {stock.capitalize()}")
        axs[i, 0].set_xlabel('Episodes')
        axs[i, 0].set_ylabel('Average Reward')
        axs[i, 0].grid(True)
        
        # Plot final values
        final_values = [values[-1] for values in training_results[stock]['asset_values']]
        axs[i, 1].plot(final_values)
        axs[i, 1].set_title(f"Final Asset Values - {stock.capitalize()}")
        axs[i, 1].set_xlabel('Episodes')
        axs[i, 1].set_ylabel('Final Value')
        axs[i, 1].grid(True)
        
        # Plot K value histogram
        if 'K_values' in training_results[stock] and len(training_results[stock]['K_values']) > 0:
            K_values = np.array(training_results[stock]['K_values']).flatten()
            axs[i, 2].hist(K_values, bins=20, density=True)
            axs[i, 2].set_title(f"K Distribution - {stock.capitalize()}")
            axs[i, 2].set_xlabel('K Value')
            axs[i, 2].set_ylabel('Density')
            axs[i, 2].grid(True)
        
        # Plot cumulative returns
        if 'asset_values' in training_results[stock]:
            values = np.array(training_results[stock]['asset_values'])
            initial_value = values[0]
            cum_returns = (values - initial_value) / initial_value * 100
            axs[i, 3].plot(cum_returns)
            axs[i, 3].set_title(f"Cumulative Returns % - {stock.capitalize()}")
            axs[i, 3].set_xlabel('Episodes')
            axs[i, 3].set_ylabel('Return %')
            axs[i, 3].grid(True)
    
    plt.tight_layout()
    plt.show()


plot_training_across_stocks(training_results)
def test_on_real_data(agent, real_returns_data):
    """Test trained agent on real market data"""
    # Create environment with real data
    env = TradingEnvironment(
        returns_data=real_returns_data,
        initial_balance=100000,
        transaction_cost=0.0,
        alpha=0.5,
        window_size=10,
        max_steps=len(real_returns_data) - 11
    )
    
    # Evaluate agent
    total_rewards, final_values, weight_history, returns_history = evaluate_agent(env, agent, episodes=1)
    
    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    
    # Plot price trajectory (approximated from returns)
    prices = 100 * np.cumprod(1 + np.array(returns_history[0]))
    axs[0].plot(prices)
    axs[0].set_title('Market Prices')
    axs[0].grid(True)
    
    # Plot agent weights over time
    axs[1].plot(weight_history[0])
    axs[1].set_title('Agent Position Weights')
    axs[1].grid(True)
    
    # Plot portfolio value
    account_values = np.array(final_values) * env.initial_balance
    axs[2].plot(account_values)
    axs[2].set_title('Portfolio Value')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return total_rewards, final_values, weight_history, returns_history