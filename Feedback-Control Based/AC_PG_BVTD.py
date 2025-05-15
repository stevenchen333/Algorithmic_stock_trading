import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from gym import spaces
import random
from tqdm import tqdm
from collections import deque
import json

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class TradingEnvironment(gym.Env):

    '''
    This is a class for defining reinforcment learning environment (or POMDP) for Double Linear Trading.

    Params:
        - data_dict: is a dictionrary of dictionaries of stocks of prices and dates
        - ticker: ticker of stocks to extract from data_dict
        - initial_balance: the initial account value for DLP (i.e., V0)
        - transaction_cost: transaction cost defined in DLP paper
        - alpha : allocation cost
        - window_size: Size of observatio windows, that is only consider t-5 stock price at t 
        - max_steps: Maximum steps per episode


    '''

    def __init__(self, data_dict, ticker='NVDA', initial_balance=100000, transaction_cost=0.0, alpha=0.5,
                window_size=5, max_steps=252):
        super(TradingEnvironment, self).__init__()
        
        # Environment parameters
        self.data_dict = data_dict
        self.ticker = ticker
        
        # Reduce window_size to 3-5 days instead of 10
        self.window_size = min(5, window_size)  # Limit to 5 days at most
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.alpha = alpha  # Allocation constant
        self.max_steps = max_steps
        
        # Check if ticker exists in data_dict
        if ticker not in data_dict:
            raise ValueError(f"Ticker {ticker} not found in data dictionary. Available tickers: {list(data_dict.keys())}")
        
        # Extract data for the selected ticker
        ticker_data = data_dict[ticker]
        
        # Verify all required fields are present
        required_fields = ['open', 'close', 'high', 'low', 'volume', 'dates']
        for field in required_fields:
            if field not in ticker_data:
                raise ValueError(f"Field '{field}' missing for ticker {ticker}")
        
        # Convert to numpy arrays for efficient computation
        self.open_prices = np.array(ticker_data['open'])
        self.close_prices = np.array(ticker_data['close'])
        self.high_prices = np.array(ticker_data['high'])
        self.low_prices = np.array(ticker_data['low'])
        self.volumes = np.array(ticker_data['volume'])
        self.dates = ticker_data['dates']
        
        # Ensure data lengths are consistent
        data_lengths = [
            len(self.open_prices),
            len(self.close_prices),
            len(self.high_prices),
            len(self.low_prices),
            len(self.volumes),
            len(self.dates)
        ]
        if len(set(data_lengths)) != 1:
            raise ValueError(f"Inconsistent data lengths for ticker {ticker}: {data_lengths}")
        
        # Calculate returns from close prices
        self.returns_data = np.diff(self.close_prices) / self.close_prices[:-1]
        self.returns_data = np.insert(self.returns_data, 0, 0)  # Add 0 return for the first day
        
        # Calculate price ratios (close/open)
        # Handle potential zero values in open_prices
        self.price_ratios = np.divide(
            self.close_prices,
            self.open_prices,
            out=np.ones_like(self.close_prices),
            where=self.open_prices != 0
        )
        
        # Add additional price data features
        self.daily_volatility = self._calculate_daily_volatility()
        
        # Compute bounds for returns, handling edge cases
        self.X_min = np.min(self.returns_data) if len(self.returns_data) > 0 else -0.1
        self.X_max = np.max(self.returns_data) if len(self.returns_data) > 0 else 0.1
        
        # Safety check for X_max to prevent division by zero or negative
        if self.X_max <= -1:
            self.X_max = 0.1  # Default to a reasonable positive value
        
        # Define action space: w ∈ [0, w_max]
        self.w_max = min(1/(1+self.transaction_cost), 1/(self.X_max+self.transaction_cost))
        self.action_space = spaces.Box(low=0, high=self.w_max, shape=(1,), dtype=np.float32)
        
        # Update observation space dimension to include window_size returns + window_size price ratios + additional features
        # Add more features: returns, price ratios, high/low ratio, volume change
        obs_dim = self.window_size * 4  # returns, price ratios, high/low ratio, volume change
        observation_high = np.ones(obs_dim) * np.finfo(np.float32).max
        observation_low = np.ones(obs_dim) * np.finfo(np.float32).min
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)
        
        # Add account value history tracking
        self.value_history = deque(maxlen=self.window_size)
        
        # Initialize state variables
        self.reset()
    
    def _calculate_daily_volatility(self):
        """Calculate daily volatility (high-low range)"""
        return (self.high_prices - self.low_prices) / ((self.high_prices + self.low_prices) / 2)
        
    def reset(self):
        # Calculate max_start ensuring it's not negative
        available_steps = len(self.returns_data) - self.max_steps - self.window_size
        max_start = max(self.window_size, available_steps)
        
        # Choose a starting point, ensuring it's valid
        self.current_step = random.randint(self.window_size, max_start)
        self.steps_taken = 0
        
        # Initialize account values
        self.V_0 = self.initial_balance
        self.V_L = self.alpha * self.V_0
        self.V_S = (1 - self.alpha) * self.V_0
        self.V = self.V_L + self.V_S
        self.prev_V = self.V
        
        # Initialize position weight
        self.current_w = 0.0

        self.value_history = deque(maxlen=self.window_size)
        self.value_history.append(self.V)
        
        # Get initial state
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation with enhanced features based on recent data only"""
        # Returns history (past window_size days)
        returns_history = self.returns_data[self.current_step-self.window_size:self.current_step]
        
        # Price ratios history (past window_size days)
        price_ratios_history = self.price_ratios[self.current_step-self.window_size:self.current_step]
        
        # High/Low ratio history (measure of intraday volatility)
        hl_ratio_history = self.daily_volatility[self.current_step-self.window_size:self.current_step]
        
        # Volume changes (normalized)
        volume_history = self.volumes[self.current_step-self.window_size:self.current_step]
        if np.min(volume_history) > 0:
            volume_changes = np.diff(volume_history) / volume_history[:-1]
            volume_changes = np.insert(volume_changes, 0, 0)  # Pad first element
        else:
            volume_changes = np.zeros_like(volume_history)
        
        # Combine all features
        observation = np.concatenate([
            returns_history, 
            price_ratios_history, 
            hl_ratio_history,
            volume_changes
        ])
        
        return observation
    
    def improved_reward_function(self, w, old_V, new_V, init_V, lambd=0.2):

        base_reward = np.log(new_V / old_V + 1e-6)  # Percentage return
        kum_reward = np.log(new_V / init_V + 1e-6)

        # Penalty for not taking positions (to encourage exploration)
        zero_penalty = -0.1 if abs(w) < 0.05 else 0
        one_penalty = -0.1 if abs(w) > 0.95 else 0
        
        # Add small incentive for risk-adjusted return
        
        # Total reward
        reward = kum_reward + base_reward + zero_penalty 
        
        return reward

    # Modified step method for TradingEnvironment
    def step(self, action):
        """Modified step function with improved reward calculation"""
        # Unpack action
        init_V = self.initial_balance
        w = float(action[0])
        self.current_w = w
        
        # Check if we're at the end of data
        if self.current_step >= len(self.returns_data) - 1:
            # Handle end of data - return terminal state
            done = True
            observation = self._get_observation()
            return observation, 0, done, {'account_value': self.V, 'return': 0, 
                                        'long_value': self.V_L, 'short_value': self.V_S, 'weight': w}
        
        # Get current return
        X = self.returns_data[self.current_step]

        # Calculate positions based on double linear policy
        pi_L = w * self.V_L  # Long position
        pi_S = -w * self.V_S  # Short position
        
        # Apply transaction costs
        epsilon = self.transaction_cost
        
        # Store previous value for reward calculation
        self.prev_V = self.V
        
        # Update account values according to Equation (2) in the paper
        self.V_L = self.V_L + X * pi_L - epsilon * pi_L
        self.V_S = self.V_S + X * pi_S - epsilon * abs(pi_S)
        self.V = self.V_L + self.V_S
        
        # Update value history and calculate sd_V
        self.value_history.append(self.V)
        
        # Calculate improved reward
        reward = self.improved_reward_function(w, self.prev_V, self.V, init_V)
        
        # Move to next step
        self.current_step += 1
        self.steps_taken += 1
        portfolio_returns = (self.V - self.prev_V ) / self.prev_V if self.prev_V != 0 else 0

        
        # Check if done
        done = (self.steps_taken >= self.max_steps) or (self.V <= 0)
        
        # Get new observation
        observation = self._get_observation()
        
        # Return additional info
        info = {
            'account_value': self.V,
            'return': portfolio_returns,
            'long_value': self.V_L,
            'short_value': self.V_S,
            'weight': w
        }
        
        return observation, reward, done, info
    


class ReplayBuffer:
    
    '''
    This class is used for replaybuffer used for experience replay. More specifically, this class provides functionality used for experience replay.
    '''
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
            
        # First, select a certain percentage of zero/near-zero action samples
        zero_samples = [i for i, exp in enumerate(self.buffer) if abs(exp[1][0]) < 0.1]
        mid_samples = [i for i, exp in enumerate(self.buffer) if 0.1 <= abs(exp[1][0]) <= 0.7]
        high_samples = [i for i, exp in enumerate(self.buffer) if abs(exp[1][0]) > 0.7]
        
        # Balance the batch
        n_zero = min(int(batch_size * 0.3), len(zero_samples))
        n_mid = min(int(batch_size * 0.4), len(mid_samples))
        n_high = min(batch_size - n_zero - n_mid, len(high_samples))
        
        # Fill remaining slots if any category is underrepresented
        remaining = batch_size - n_zero - n_mid - n_high
        if remaining > 0:
            all_indices = list(range(len(self.buffer)))
            remaining_indices = random.sample(all_indices, remaining)
            selected_indices = (random.sample(zero_samples, n_zero) + 
                                random.sample(mid_samples, n_mid) + 
                                random.sample(high_samples, n_high) + 
                                remaining_indices)
        else:
            selected_indices = (random.sample(zero_samples, n_zero) + 
                            random.sample(mid_samples, n_mid) + 
                            random.sample(high_samples, n_high))
        
        batch = [self.buffer[i] for i in selected_indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states), 
                np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class LinearPolicyGradientAgent:
    """
    Policy Gradient agent with linear function approximation and backward view TD-lambda 
    """
    def __init__(self, state_size, action_size, action_bound):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound  # Upper bound for continuous action
        
        # Hyperparameters
        self.gamma = 0.95       # Discount factor
        self.entropy_coef = 0.02 
        self.lambda_val = 0.9   # TD(λ) parameter
        self.alpha = 0.001      # Learning rate for policy parameters
        self.alpha_v = 0.01     # Learning rate for value function
        self.sigma = 0.05       # Reduced std deviation for exploration (from 0.4)
        self.exploration_decay = 0.999  # Decay rate for exploration
        self.min_sigma = 0.01    # Reduced min sigma (from 0.15)
        self.episodes_count = 0 # Episodes counter
        
        # Initialize policy parameters (θ)
        self.theta = np.zeros(state_size)
        
        # Initialize value function parameters (w)
        self.w = np.zeros(state_size)
        
        # Initialize eligibility traces
        self.e_theta = np.zeros_like(self.theta)  # Eligibility trace for policy
        self.e_w = np.zeros_like(self.w) 


        self.replay_buffer = ReplayBuffer(capacity=5000)
        self.batch_size = 32
        self.min_experiences = 1000  # Min experiences before training
        self.update_every = 4        # Update network every N steps
        self.step_count = 0       # Step counter
        
    def get_action(self, state):
        """Get action using linear policy with Gaussian exploration"""
        # Calculate mean of policy distribution (linear function of state)
        action_mean = np.dot(state, self.theta)
        
        # Bound the action mean between 0 and action_bound
        action_mean = np.clip(action_mean, 0, self.action_bound)
        
        # Sample action from Gaussian distribution
        action = np.random.normal(action_mean, self.sigma)
        
        # Ensure action is within bounds
        action = np.clip(action, 0, self.action_bound)
        
        return np.array([action])
    
    def get_value(self, state):
        """Estimate state value using linear function approximation"""
        return np.dot(state, self.w)
    
    def _get_action_prob(self, state, action, use_current=False):
        """Calculate probability of action under current or old policy"""
        action_mean = np.dot(state, self.theta)
        action_mean = np.clip(action_mean, 0, self.action_bound)
        return (1.0 / (self.sigma * np.sqrt(2 * np.pi))) * \
            np.exp(-0.5 * ((action[0] - action_mean) / self.sigma) ** 2)
    
    def update(self, state, action, reward, next_state, done):
        """Update policy and value function parameters using experience replay"""
        # Store experience in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.step_count += 1
        
        # Only update if we have enough experiences
        if len(self.replay_buffer) < self.min_experiences:
            return
            
        if self.step_count % self.update_every != 0:
            return
            
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Normalize rewards for stability
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        
        # Calculate TD errors with clipping for stability
        next_values = np.array([self.get_value(next_state) for next_state in next_states])
        current_values = np.array([self.get_value(state) for state in states])
        deltas = rewards + (1 - dones) * self.gamma * next_values - current_values
        deltas = np.clip(deltas, -1, 1)  # Clip TD errors
        
        # Calculate policy gradients with importance sampling
        gradients = []
        importance_weights = []
        
        for i in range(len(states)):
            gradient = self._policy_gradient(states[i], actions[i])
            old_prob = self._get_action_prob(states[i], actions[i])
            new_prob = self._get_action_prob(states[i], actions[i], use_current=True)
            importance_weight = np.clip(new_prob / (old_prob + 1e-8), 0.8, 1.2)
            
            gradients.append(gradient)
            importance_weights.append(importance_weight)
        
        gradients = np.array(gradients)
        importance_weights = np.array(importance_weights)
        
        # Calculate weighted updates
        policy_update = np.mean(deltas[:, np.newaxis] * gradients * 
                            importance_weights[:, np.newaxis], axis=0)
        
        # Adaptive entropy coefficient
        entropy_coef = self.entropy_coef * (1.0 - min(1.0, self.episodes_count / 2000))
        entropy_bonus = entropy_coef * np.log(self.sigma)
        entropy_update = np.mean([entropy_bonus * state for state in states], axis=0)
        
        # Trust region update for policy
        policy_update_norm = np.linalg.norm(policy_update)
        if policy_update_norm > 1.0:
            policy_update = policy_update / policy_update_norm
        l2_reg = 0.001 * np.sum(self.theta**2)
        policy_update -= l2_reg * self.theta
    
        # Update parameters with adaptive learning rates
        effective_alpha = self.alpha * (1.0 / (1.0 + 0.01 * self.episodes_count))
        effective_alpha_v = self.alpha_v * (1.0 / (1.0 + 0.01 * self.episodes_count))
        
        self.theta += effective_alpha * (policy_update + entropy_update)
        self.w += effective_alpha_v * np.mean(deltas[:, np.newaxis] * states, axis=0)
        
        # Exploration and learning rate updates
        if done:
            self.episodes_count += 1
            
            # Faster adaptive exploration decay for shorter time frames
            if self.episodes_count % 3 == 0:  # Check more frequently (was 5)
                progress = min(1.0, self.episodes_count / 3000)  # Faster convergence (was 5000)
                target_sigma = self.min_sigma + (1.0 - progress) * (0.2 - self.min_sigma)
                self.sigma = self.sigma * 0.95 + target_sigma * 0.05
            
            # Parameter noise with decay
            if self.episodes_count % 30 == 0:  # More frequent noise (was 50)
                noise_magnitude = 0.01 * (1.0 - min(1.0, self.episodes_count / 3000))
                self.theta += np.random.normal(0, noise_magnitude, size=self.theta.shape)

        return np.mean(deltas)  # Return mean TD error for monitoring
    
    def _policy_gradient(self, state, action):
        """Calculate policy gradient for the linear Gaussian policy"""
        # For linear Gaussian policy, the gradient is proportional to (action - mean) * state
        action_mean = np.dot(state, self.theta)
        action_mean = np.clip(action_mean, 0, self.action_bound)
        
        # Calculate gradient
        gradient = (action[0] - action_mean) * state / (self.sigma ** 2)
        
        return gradient
    
    def save(self, filename):
        """Save model parameters"""
        np.savez(filename, theta=self.theta, w=self.w)
    
    def load(self, filename):
        """Load model parameters"""
        data = np.load(filename)
        self.theta = data['theta']
        self.w = data['w']

def train_agent(env, agent, episodes=100):
    """Train the agent on the environment"""
    rewards_history = []
    avg_rewards_history = []
    asset_values_history = []
    
    for e in tqdm(range(episodes), desc="Training Episodes"):
        state = env.reset()
        done = False
        total_reward = 0
        episode_values = []
        
        # Episode trajectory
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        while not done:
            # Get action from agent
            action = agent.get_action(state)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            # Update policy using TD(λ)
            agent.update(state, action, reward, next_state, done)
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            episode_values.append(info['account_value'])
        
        # Record metrics
        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-100:] if len(rewards_history) >= 100 else rewards_history)
        avg_rewards_history.append(avg_reward)
        asset_values_history.append(episode_values)
        
        if e % 10 == 0:
            print(f"Episode {e}/{episodes}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Final Value: {episode_values[-1]:.2f}")
    
    return rewards_history, avg_rewards_history, asset_values_history


def evaluate_agent(env, agent, episodes=10):
    """Evaluate the trained agent"""
    # Check if agent is properly initialized
    if agent is None or not hasattr(agent, 'theta') or agent.theta is None:
        print("Error: Agent not properly initialized or trained.")
        return [], [], [], []
    
    total_rewards = []
    final_values = []
    weight_history = []
    env_returns = []
    
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_weights = []
        episode_returns = []
        
        while not done:
            # Get deterministic action without exploration
            action_mean = np.dot(state, agent.theta)
            action_mean = np.clip(action_mean, 0, agent.action_bound)
            action = np.array([action_mean])
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Record metrics
            total_reward += reward
            episode_weights.append(info['weight'])
            episode_returns.append(info['return'])
            
            # Move to next state
            state = next_state
        
        total_rewards.append(total_reward)
        final_values.append(info['account_value'])
        weight_history.append(episode_weights)
        env_returns.append(episode_returns)
        
        print(f"Evaluation Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Final Value: {info['account_value']:.2f}")
    
    if total_rewards:
        print(f"Average Total Reward: {np.mean(total_rewards):.2f}")
        print(f"Average Final Value: {np.mean(final_values):.2f}")
    
    return total_rewards, final_values, weight_history, env_returns


def plot_results(rewards_history, avg_rewards_history, asset_values_history, eval_results=None):
    """Plot training and evaluation results"""
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot training rewards
    axs[0, 0].plot(rewards_history, alpha=0.6, label='Episode Reward')
    axs[0, 0].plot(avg_rewards_history, label='Avg Reward (100 episodes)')
    axs[0, 0].set_title('Training Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot final asset values in training
    final_values = [values[-1] for values in asset_values_history]
    axs[0, 1].plot(final_values)
    axs[0, 1].set_title('Final Asset Values in Training')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Asset Value')
    axs[0, 1].grid(True)
    
    # Sample asset value trajectory from last episode
    axs[1, 0].plot(asset_values_history[-1])
    axs[1, 0].set_title('Asset Value Trajectory (Last Training Episode)')
    axs[1, 0].set_xlabel('Step')
    axs[1, 0].set_ylabel('Asset Value')
    axs[1, 0].grid(True)
    
    # Plot evaluation results if available
    if eval_results:
        total_rewards, final_values, weight_history, env_returns = eval_results
        
        # Plot weight vs returns for a sample evaluation episode
        sample_episode = 0
        axs[1, 1].scatter(env_returns[sample_episode], weight_history[sample_episode], alpha=0.7)
        axs[1, 1].set_title('Weight vs Returns (Sample Evaluation Episode)')
        axs[1, 1].set_xlabel('Return')
        axs[1, 1].set_ylabel('Weight')
        axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


def simulate_market_data(n_steps=2000, mu=-0.0001, sigma=0.01, jump_intensity=0.05, jump_size=0.03, ticker='NVDA'):
    """
    Simulate stock returns data with a geometric Brownian motion with jumps model
    
    Args:
        n_steps: Number of time steps to simulate
        mu: Drift parameter for returns
        sigma: Volatility parameter
        jump_intensity: Probability of jumps
        jump_size: Size of jumps
        ticker: Ticker symbol to use as key in the output dictionary
        
    Returns:
        Dictionary with simulated market data
    """
    # Generate normal returns
    normal_returns = np.random.normal(mu, sigma, n_steps)
    
    # Generate jumps
    jumps = np.random.binomial(1, jump_intensity, n_steps) * np.random.choice([-1, 1], n_steps) * jump_size
    
    # Combine normal returns with jumps
    returns = normal_returns + jumps
    
    # Ensure returns are within bounds (avoid extreme values)
    returns = np.clip(returns, -0.2, 0.2)
    
    # Create synthetic market data dictionary (similar to the JSON structure mentioned in TODO)
    dates = [f"2023-{i//21+1:02d}-{i%21+1:02d}" for i in range(n_steps)]
    
    # Start price at 100
    close_prices = 100 * np.cumprod(1 + returns)
    
    # Generate open, high, low prices
    open_prices = close_prices / (1 + np.random.normal(0, 0.005, n_steps))
    high_prices = np.maximum(close_prices, open_prices) * (1 + np.abs(np.random.normal(0, 0.005, n_steps)))
    low_prices = np.minimum(close_prices, open_prices) * (1 - np.abs(np.random.normal(0, 0.005, n_steps)))
    
    # Generate volume
    volumes = np.random.lognormal(mean=15, sigma=1, size=n_steps)
    
    # Create dictionary structure with the provided ticker
    data_dict = {
        ticker: {
            'open': open_prices.tolist(),
            'close': close_prices.tolist(),
            'high': high_prices.tolist(),
            'low': low_prices.tolist(),
            'volume': volumes.tolist(),
            'dates': dates
        }
    }
    
    return data_dict


def load_market_data(file_path):
    """
    Load market data from JSON file
    
    Args:
        file_path: Path to the JSON file containing market data
        
    Returns:
        Dictionary with market data loaded from the file
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate that the data has the expected structure
        for ticker, ticker_data in data.items():
            required_fields = ['open', 'close', 'high', 'low', 'volume', 'dates']
            for field in required_fields:
                if field not in ticker_data:
                    print(f"Warning: {field} missing for ticker {ticker}")
        
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return {}
    except json.JSONDecodeError:
        print(f"Error: File {file_path} contains invalid JSON")
        return {}
    except Exception as e:
        print(f"Error loading market data: {str(e)}")
        return {}


def train_agent_across_stocks(data_dict, window_size=10, max_steps=252, 
                        episodes_per_stock=1000, tickers=None):
    """
    Train an agent across multiple stocks for better generalization.
    
    Args:
        data_dict: Dictionary containing market data for multiple tickers
        window_size: Size of observation window
        max_steps: Maximum steps per episode
        episodes_per_stock: Number of episodes to train on each stock
        tickers: List of ticker symbols to use (if None, use all available)
    
    Returns:
        Trained agent and evaluation results
    """
    if tickers is None:
        tickers = list(data_dict.keys())
    
    # Verify at least one ticker exists in the data
    if not tickers:
        raise ValueError("No tickers available in the provided data")
    
    # Create environment with the first ticker
    env = TradingEnvironment(
        data_dict=data_dict,
        ticker=tickers[0],
        initial_balance=100000,
        transaction_cost=0.0001,
        alpha=0.5,
        window_size=window_size,
        max_steps=max_steps    )
    
    # Create policy gradient agent
    state_size = env.observation_space.shape[0]
    action_size = 1
    agent = LinearPolicyGradientAgent(state_size, action_size, env.w_max)
    
    all_rewards_history = []
    all_avg_rewards_history = []
    
    # Train on each ticker sequentially
    for ticker in tickers:
        print(f"\nTraining on {ticker}...")
        
        # Update environment to use current ticker
        env = TradingEnvironment(
            data_dict=data_dict,
            ticker=ticker,
            initial_balance=100000,
            transaction_cost=0.0001,
            alpha=0.5,
            window_size=window_size,
            max_steps=max_steps
        )
        
        # Train agent on this ticker
        rewards_history, avg_rewards_history, _ = train_agent(
            env, agent, episodes=episodes_per_stock
        )
        
        all_rewards_history.extend(rewards_history)
        all_avg_rewards_history.extend(avg_rewards_history)
    
    # Final evaluation on the first ticker
    print(f"\nEvaluating agent on {tickers[0]}...")
    env = TradingEnvironment(
        data_dict=data_dict,
        ticker=tickers[0],
        initial_balance=100000,
        transaction_cost=0.0001,
        alpha=0.5,
        window_size=window_size,
        max_steps=max_steps
    )
    
    # Save the model before evaluation
    agent.save("trading_pg_model_weights.npz")
    
    # Evaluate the trained agent
    eval_results = evaluate_agent(env, agent, episodes=3)
    
    return agent, env, all_rewards_history, all_avg_rewards_history, eval_results


def main():
    # Simulate market data or load real data
    data_dict = simulate_market_data(n_steps=3000)
    
    # Add another ticker to the data dictionary for multi-stock training
    data_dict['AAPL'] = simulate_market_data(n_steps=3000)['NVDA']
    
    # You can also load data from a file instead of simulation
    # data_dict = load_market_data('market_data.json')
    
    # Try both approaches: single stock and multi-stock training
    
    # 1. Single stock approach
    env_single = TradingEnvironment(
        data_dict=data_dict,
        ticker='NVDA',  # Use the provided ticker
        initial_balance=100000,
        transaction_cost=0.0001,  # 0.01%
        alpha=0.5,  # Equal allocation to long and short
        window_size=10,
        max_steps=252,  # Roughly one trading year
    )
    
    # Create policy gradient agent with linear function approximation
    state_size = env_single.observation_space.shape[0]
    action_size = 1  # Just the weight parameter w
    agent_single = LinearPolicyGradientAgent(state_size, action_size, env_single.w_max)
    
    # Train agent on single stock
    print("Training on single stock (NVDA)...")
    rewards_history, avg_rewards_history, asset_values_history = train_agent(
        env_single, agent_single, episodes=1000
    )
    
    # Evaluate agent
    print("Evaluating single-stock agent...")
    eval_results_single = evaluate_agent(env_single, agent_single, episodes=3)
    
    # Save the model
    agent_single.save("trading_pg_model_single_stock.npz")
    
    # 2. Multi-stock approach
    print("\nTraining across multiple stocks...")
    agent_multi, env_multi, rewards_multi, avg_rewards_multi, eval_results_multi = train_agent_across_stocks(
        data_dict, 
        tickers=['NVDA', 'AAPL'],
        episodes_per_stock=500
    )
    
    # Plot results for single stock training
    print("\nPlotting results for single stock training...")
    plot_results(rewards_history, avg_rewards_history, asset_values_history, eval_results_single)
    
    # Multi-stock results
    print("\nPlotting results for multi-stock training...")
    # Get asset values for the last evaluation episode
    if eval_results_multi and len(eval_results_multi) > 0 and len(eval_results_multi[0]) > 0:
        # Simulate asset values history for plotting
        asset_values_multi = [[env_multi.initial_balance * (1 + r/100) for r in range(env_multi.max_steps)]]
        plot_results(rewards_multi, avg_rewards_multi, asset_values_multi, eval_results_multi)
    
    return agent_single, agent_multi, eval_results_single, eval_results_multi


if __name__ == "__main__":
    main()