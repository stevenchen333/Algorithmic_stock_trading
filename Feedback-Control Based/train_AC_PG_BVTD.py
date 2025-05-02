import numpy as np
from AC_PG_BVTD import TradingEnvironment, LinearPolicyGradientAgent,train_agent, evaluate_agent, plot_results
import matplotlib.pyplot as plt

def simulate_realistic_market_data(n_days=2000, regime='normal'):
    """
    Simulate realistic stock returns using regime-specific GBM parameters
    
    Parameters:
    - n_days: Number of trading days to simulate
    - regime: Market regime ('bull', 'bear', 'normal', 'volatile', or 'random')
    
    Returns:
    - returns: Daily returns
    - prices: Corresponding price series
    - params: Dictionary of parameters used
    """
    # Define realistic parameters for different market regimes
    regimes = {
        'bull': {
            'mu': 0.0007,           # ~18% annualized return
            'sigma': 0.012,         # ~19% annualized volatility
            'jump_intensity': 0.02, # Rare jumps
            'jump_size': 0.02       # Small jumps
        },
        'bear': {
            'mu': -0.0005,          # ~-12% annualized return
            'sigma': 0.018,         # ~28% annualized volatility
            'jump_intensity': 0.04, # More frequent jumps
            'jump_size': 0.03       # Moderate jumps
        },
        'normal': {
            'mu': 0.0003,           # ~8% annualized return
            'sigma': 0.01,          # ~16% annualized volatility
            'jump_intensity': 0.03, # Occasional jumps
            'jump_size': 0.02       # Small jumps
        },
        'volatile': {
            'mu': 0.0001,           # ~2.5% annualized return
            'sigma': 0.025,         # ~40% annualized volatility
            'jump_intensity': 0.06, # Frequent jumps
            'jump_size': 0.04       # Larger jumps
        }
    }
    
    # Choose regime parameters
    if regime == 'random':
        # Randomly select a regime for each segment
        n_segments = np.random.randint(3, 7)  # 3-6 different regimes
        segment_length = n_days // n_segments
        returns = np.array([])
        
        for i in range(n_segments):
            chosen_regime = np.random.choice(['bull', 'bear', 'normal', 'volatile'])
            regime_params = regimes[chosen_regime]
            segment_returns = simulate_segment(
                segment_length, 
                regime_params['mu'],
                regime_params['sigma'],
                regime_params['jump_intensity'],
                regime_params['jump_size']
            )
            returns = np.append(returns, segment_returns)
        
        # Trim or extend to exact length
        if len(returns) > n_days:
            returns = returns[:n_days]
        elif len(returns) < n_days:
            # Generate remaining days with last regime
            remaining = n_days - len(returns)
            extra_returns = simulate_segment(
                remaining,
                regime_params['mu'],
                regime_params['sigma'],
                regime_params['jump_intensity'],
                regime_params['jump_size']
            )
            returns = np.append(returns, extra_returns)
    else:
        # Use specific regime
        regime_params = regimes[regime]
        returns = simulate_segment(
            n_days,
            regime_params['mu'],
            regime_params['sigma'],
            regime_params['jump_intensity'],
            regime_params['jump_size']
        )
    
    # Calculate price series (starting at 100)
    prices = 100 * np.cumprod(1 + returns)
    
    # Return both returns and prices
    return returns, prices, regime_params

def simulate_segment(n_days, mu, sigma, jump_intensity, jump_size):
    """Simulate a market segment with consistent parameters"""
    # Generate base returns
    base_returns = np.random.normal(mu, sigma, n_days)
    
    # Generate jumps
    jumps = np.random.binomial(1, jump_intensity, n_days) * np.random.choice([-1, 1], n_days) * jump_size
    
    # Combine and clip to avoid unrealistic returns
    returns = base_returns + jumps
    returns = np.clip(returns, -0.15, 0.15)  # Limit daily moves to ±15%
    
    return returns

def train_agent_across_regimes(agent_class, epochs_per_regime=1000):
    """Train agent across different market regimes"""
    # Market regimes to train on
    regimes = ['normal', 'bull', 'bear', 'volatile', 'random']
    
    # Initialize agent
    state_size = None  # Will initialize after first environment creation
    action_size = 1
    agent = None
    
    all_training_results = {}
    
    for regime in regimes:
        print(f"\n===== Training on {regime.upper()} market regime =====")
        
        # Generate market data for this regime
        returns_data, prices_data, params = simulate_realistic_market_data(
            n_days=5000, regime=regime
        )
        
        # Create environment
        env = TradingEnvironment(
            returns_data=returns_data,
            initial_balance=1000,
            transaction_cost=0.0,
            alpha=0.5,
            window_size=10,
            max_steps=len(returns_data) - 11
        )
        
        # Initialize agent if first regime
        if agent is None:
            state_size = env.observation_space.shape[0]
            agent = agent_class(state_size, action_size, env.w_max)
        
        # Train agent on this regime
        rewards_history, avg_rewards_history, asset_values_history = train_agent(
            env, agent, episodes=epochs_per_regime
        )
        
        # Save results for this regime
        all_training_results[regime] = {
            'rewards': rewards_history,
            'avg_rewards': avg_rewards_history,
            'asset_values': asset_values_history,
            'params': params
        }
        
        # Evaluate on this regime
        eval_results = evaluate_agent(env, agent, episodes=3)
        print(f"Evaluation on {regime} regime - Avg Final Value: {np.mean(eval_results[1]):.2f}")
    
    # Save the trained model
    agent.save("multi_regime_agent.npz")
    
    return agent, all_training_results


# First, visualize the simulations to verify realism

# Train across multiple regimes
agent, training_results = train_agent_across_regimes(
    agent_class=LinearPolicyGradientAgent, 
    epochs_per_regime=2000
)

# Visualize training results
def plot_training_across_regimes(training_results):
    regimes = list(training_results.keys())
    fig, axs = plt.subplots(len(regimes), 2, figsize=(15, 4*len(regimes)))
    
    for i, regime in enumerate(regimes):
        # Plot rewards
        axs[i, 0].plot(training_results[regime]['avg_rewards'])
        axs[i, 0].set_title(f"Training Rewards - {regime.capitalize()} Market")
        axs[i, 0].set_xlabel('Episodes')
        axs[i, 0].set_ylabel('Average Reward')
        axs[i, 0].grid(True)
        
        # Plot final values
        final_values = [values[-1] for values in training_results[regime]['asset_values']]
        axs[i, 1].plot(final_values)
        axs[i, 1].set_title(f"Final Asset Values - {regime.capitalize()} Market")
        axs[i, 1].set_xlabel('Episodes')
        axs[i, 1].set_ylabel('Final Value')
        axs[i, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

plot_training_across_regimes(training_results)
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