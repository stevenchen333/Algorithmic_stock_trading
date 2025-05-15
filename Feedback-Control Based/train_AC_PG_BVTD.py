import numpy as np
from AC_PG_BVTD import TradingEnvironment, LinearPolicyGradientAgent,train_agent, evaluate_agent, plot_results
import matplotlib.pyplot as plt
import json
import pandas as pd





def train_agent_across_stocks(training_path, agent_class, epochs_per_stock=1000, include_gbm=True):
    # Load real stock data
    with open(training_path, "r") as f:
        training_data = json.load(f)
    
    stocks_curriculum = list(training_data.keys())
    
    # Add simulated GBM with jumps if requested
    if include_gbm:
        # Define different market conditions
        market_scenarios = [
            {
                "name": "bull",
                "mu": 0.08,
                "sigma": 0.015,
                "jump_intensity": 0.01,
                "jump_size": 0.02,
            },
            {
                "name": "bear",
                "mu": -0.05,
                "sigma": 0.02,
                "jump_intensity": 0.05,
                "jump_size": 0.04,
            },
            {
                "name": "volatile",
                "mu": 0.01,
                "sigma": 0.05,
                "jump_intensity": 0.1,
                "jump_size": 0.06,
            },
            {
                "name": "calm",
                "mu": 0.03,
                "sigma": 0.01,
                "jump_intensity": 0.005,
                "jump_size": 0.01,
            },
        ]

        for i, scenario in enumerate(market_scenarios):
            # Generate GBM with jumps data
            gbm_data = generate_gbm_with_jumps(
                n_steps=500,
                mu=scenario["mu"],
                sigma=scenario["sigma"],
                jump_intensity=scenario["jump_intensity"],
                jump_size=scenario["jump_size"],
                initial_price=100.0,
            )
            stock_name = f"GBM_{scenario['name']}_{i}"
            training_data[stock_name] = gbm_data
            stocks_curriculum.append(stock_name)

    # Initialize storage
    all_training_results = {}
    agent = None
    action_size = 1
    
    for stock in stocks_curriculum:
        print(f"\n===== Training on {stock.upper()} =====")
        print(f"Training for {epochs_per_stock} epochs")
        
        # Create environment - handle GBM differently if needed
        if stock == 'GBM':
            # For GBM, we might want different parameters
            env = TradingEnvironment(
                data_dict=training_data,
                ticker=stock,
                initial_balance=1000,
                transaction_cost=0.001,  # Slightly higher cost for simulated data
                alpha=0.5,
                window_size=5,
                max_steps=252-6
            )
        else:
            # For real stocks
            env = TradingEnvironment(
                data_dict=training_data,
                ticker=stock,
                initial_balance=1000,
                transaction_cost=0.0,
                alpha=0.5,
                window_size=5,
                max_steps=252-6
            )
        
        # Initialize agent if first stock
        if agent is None:
            state_size = env.observation_space.shape[0]
            agent = agent_class(state_size, action_size, env.w_max)
        
        # Train agent on this stock/GBM
        rewards, avg_rewards, asset_values = train_agent(
            env, agent, episodes=epochs_per_stock
        )
        
        # Collect actions by running one episode
        actions_list = []
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            actions_list.append(action)
            next_state, reward, done, _ = env.step(action)
            state = next_state
        
        # Store results
        all_training_results[stock] = {
            'rewards': rewards,
            'avg_rewards': avg_rewards,
            'asset_values': asset_values,
            'actions': actions_list
        }
        
        # Evaluation
        eval_rewards, eval_values, weights_history, env_returns = evaluate_agent(env, agent, episodes=10)
        print(f"\nFinal evaluation on {stock} - "
            f"Avg Final Value: {np.mean(eval_values):.2f}")
        
        # Save model
        agent.save(f"agent_{stock}_final.npz")
    
    # Save final model
    agent.save("multi_stock_agent_final.npz")
    
    return agent, all_training_results

def generate_gbm_with_jumps(n_steps=500, mu=-0.0005, sigma=0.02, 
                        jump_intensity=0.004, jump_size=0.004, 
                        initial_price=100.0):
    """Generate GBM with jumps data in the required dictionary format"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n_steps).strftime('%Y-%m-%d').tolist()
    
    # Generate returns
    normal_returns = np.random.normal(mu, sigma, n_steps)
    jumps = np.random.binomial(1, jump_intensity, n_steps) * \
            np.random.choice([-1, 1], n_steps) * jump_size
    returns = normal_returns + jumps
    returns = np.clip(returns, -0.2, 0.2)
    
    # Generate prices
    prices = initial_price * np.cumprod(1 + returns)
    
    return {
        'open': prices.tolist(),
        'close': prices.tolist(),
        'high': (prices * 1.01).tolist(),  # Add small variation
        'low': (prices * 0.99).tolist(),
        'volume': np.random.lognormal(8, 1, n_steps).astype(int).tolist(),
        'dates': dates
    }

# Usage
agent, training_results = train_agent_across_stocks(
    training_path="trainstock_data_2015-01-01_to_2024-01-01.json",
    agent_class=LinearPolicyGradientAgent, 
    epochs_per_stock=100,
    include_gbm=True  # Now includes GBM with jumps in training
)

# Visualize training results
def plot_training_across_stocks(training_results, save_path = None):
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
        
        # Plot action histogram
        if 'actions' in training_results[stock] and len(training_results[stock]['actions']) > 0:
            actions = np.array(training_results[stock]['actions']).flatten()
            axs[i, 2].hist(actions, bins=20, density=True)
            axs[i, 2].set_title(f"Action Distribution - {stock.capitalize()}")
            axs[i, 2].set_xlabel('Action Value')
            axs[i, 2].set_ylabel('Density')
            axs[i, 2].grid(True)
        
        # Plot cumulative returns
        if 'asset_values' in training_results[stock]:
            # Convert to numpy array first
            all_values = np.array(training_results[stock]['asset_values'])
            
            # Determine sample size (min of 100 or length of data)
            sample_size = min(100, len(all_values))
            
            # Sample values using linear indices to maintain temporal order
            indices = np.linspace(0, len(all_values)-1, sample_size, dtype=int)
            values = all_values[indices]
            
            # Calculate cumulative returns
            initial_value = values[0]
            cum_returns = (values - initial_value) / initial_value * 100
            
            # Plot with proper x-axis
            x_axis = np.linspace(0, len(all_values), sample_size)
            axs[i, 3].plot(x_axis, cum_returns)
            axs[i, 3].set_title(f"Cumulative Returns % - {stock.capitalize()}")
            axs[i, 3].set_xlabel('Episodes')
            axs[i, 3].set_ylabel('Return %')
            axs[i, 3].grid(True)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
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
        max_steps=252
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