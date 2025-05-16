import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from gym import spaces
import random
from tqdm import tqdm
from collections import deque
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

class TradingEnvironment(gym.Env):
    '''
    This is a class for defining reinforcement learning environment (or POMDP) for Double Linear Trading.

    Params:
        - data_dict: is a dictionary of dictionaries of stocks of prices and dates
        - ticker: ticker of stocks to extract from data_dict
        - initial_balance: the initial account value for DLP (i.e., V0)
        - transaction_cost: transaction cost defined in DLP paper
        - alpha : allocation cost
        - window_size: Size of observation windows, that is only consider t-5 stock price at t 
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
        
        # Total reward
        reward = kum_reward + base_reward + zero_penalty 
        
        return reward

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
            selected_indices = (random.sample(zero_samples, n_zero if n_zero > 0 else 0) + 
                                random.sample(mid_samples, n_mid if n_mid > 0 else 0) + 
                                random.sample(high_samples, n_high if n_high > 0 else 0) + 
                                remaining_indices)
        else:
            selected_indices = (random.sample(zero_samples, n_zero if n_zero > 0 else 0) + 
                            random.sample(mid_samples, n_mid if n_mid > 0 else 0) + 
                            random.sample(high_samples, n_high if n_high > 0 else 0))
        
        batch = [self.buffer[i] for i in selected_indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states), 
                np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


# Neural network policy model
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, action_bound=1.0):
        super(PolicyNetwork, self).__init__()
        self.action_bound = action_bound
        
        # Policy network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, 1)  # Output action mean
        self.fc_std = nn.Linear(hidden_dim, 1)   # Output action std
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Action mean bounded between 0 and action_bound
        action_mean = torch.sigmoid(self.fc_mean(x)) * self.action_bound
        
        # Log std with minimum to prevent collapse
        log_std = torch.clamp(self.fc_std(x), min=-20, max=2)
        action_std = torch.exp(log_std)
        
        return action_mean, action_std


# Value network for critic
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        
        # Value network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        
        return value


class ActorCriticAgent:
    """
    Actor-Critic agent with neural network function approximation
    """
    def __init__(self, state_size, action_size, action_bound, hidden_dim=64):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound  # Upper bound for continuous action
        self.hidden_dim = hidden_dim
        
        # Hyperparameters
        self.gamma = 0.99         # Discount factor
        self.tau = 0.005          # Target network update rate
        self.entropy_coef = 0.01  # Entropy coefficient
        self.lr_actor = 0.0003    # Learning rate for actor
        self.lr_critic = 0.001    # Learning rate for critic
        self.min_sigma = 0.01     # Minimum exploration noise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        # Neural Networks
        self.policy_net = PolicyNetwork(state_size, hidden_dim, action_bound).to(self.device)
        self.value_net = ValueNetwork(state_size, hidden_dim).to(self.device)
        
        # Create target networks
        self.target_value_net = ValueNetwork(state_size, hidden_dim).to(self.device)
        self.update_target_network(tau=1.0)  # Hard update to initialize
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr_actor)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr_critic)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.batch_size = 64
        self.min_experiences = 1000  # Min experiences before training
        self.update_every = 4        # Update network every N steps
        self.step_count = 0          # Step counter
        self.episodes_count = 0      # Episodes counter
        
    def get_action(self, state, deterministic=False):
        """Get action using policy network with Gaussian policy"""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
            
            # Get mean and std from policy network
            action_mean, action_std = self.policy_net(state)
            
            if deterministic:
                # Use mean for deterministic action
                action = action_mean.cpu().numpy()
            else:
                # Sample from Gaussian distribution
                normal_dist = torch.distributions.Normal(action_mean, action_std)
                action = normal_dist.sample().cpu().numpy()
            
            # Ensure action is within bounds and reshape to match environment expectation
            action = np.clip(action, 0, self.action_bound)
            return action.reshape(1)
    
    def update_target_network(self, tau=None):
        """Update target network parameters"""
        if tau is None:
            tau = self.tau
            
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def update(self, state, action, reward, next_state, done):
        """Update policy and value networks using experience replay"""
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
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Normalize rewards for stability
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Calculate TD target using target network
        with torch.no_grad():
            next_values = self.target_value_net(next_states)
            td_target = rewards + (1 - dones) * self.gamma * next_values
        
        # Calculate current value estimates
        values = self.value_net(states)
        
        # Calculate value loss (MSE)
        value_loss = F.mse_loss(values, td_target)
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()
        
        # Calculate advantage
        advantage = td_target - values.detach()
        
        # Get action distribution parameters
        action_means, action_stds = self.policy_net(states)
        normal_dist = torch.distributions.Normal(action_means, action_stds)
        # Calculate log probabilities of actions
        log_probs = normal_dist.log_prob(actions)
        
        # Calculate entropy for exploration bonus
        entropy = normal_dist.entropy().mean()
        
        # Calculate policy loss (negative of objective function)
        policy_loss = -(log_probs * advantage).mean() - self.entropy_coef * entropy
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        # Update target networks
        self.update_target_network()
        
        # Update episode counter if done
        if done:
            self.episodes_count += 1
            
            # Dynamic entropy coefficient decay
            if self.episodes_count % 10 == 0:
                self.entropy_coef = max(0.001, self.entropy_coef * 0.995)
                
        return value_loss.item(), policy_loss.item()
    
    def save(self, filename):
        """Save model parameters"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'target_value_state_dict': self.target_value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, filename)
    
    def load(self, filename):
        """Load model parameters"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.target_value_net.load_state_dict(checkpoint['target_value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])


def train_agent(env, agent, episodes=100):
    """Train the agent on the environment"""
    rewards_history = []
    avg_rewards_history = []
    asset_values_history = []
    value_losses = []
    policy_losses = []
    
    for e in tqdm(range(episodes), desc="Training Episodes"):
        state = env.reset()
        done = False
        total_reward = 0
        episode_values = []
        episode_value_losses = []
        episode_policy_losses = []
        
        while not done:
            # Get action from agent
            action = agent.get_action(state)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Update policy using experience replay
            update_result = agent.update(state, action, reward, next_state, done)
            
            if update_result:
                value_loss, policy_loss = update_result
                episode_value_losses.append(value_loss)
                episode_policy_losses.append(policy_loss)
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            episode_values.append(info['account_value'])
        
        # Record metrics
        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-100:] if len(rewards_history) >= 100 else rewards_history)
        avg_rewards_history.append(avg_reward)
        asset_values_history.append(episode_values)
        
        if episode_value_losses:
            value_losses.append(np.mean(episode_value_losses))
        if episode_policy_losses:
            policy_losses.append(np.mean(episode_policy_losses))
        
        if e % 10 == 0:
            print(f"Episode {e}/{episodes}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Final Value: {episode_values[-1]:.2f}")
            if episode_value_losses and episode_policy_losses:
                print(f"Value Loss: {np.mean(episode_value_losses):.4f}, Policy Loss: {np.mean(episode_policy_losses):.4f}")
    
    return rewards_history, avg_rewards_history, asset_values_history, value_losses, policy_losses


def evaluate_agent(env, agent, episodes=10):
    """Evaluate the trained agent"""
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
            # Get deterministic action (using mean of policy)
            action = agent.get_action(state, deterministic=True)
            
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
    agent = ActorCriticAgent(state_size, action_size, env.w_max)
    
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
    agent_single = ActorCriticAgent(state_size, action_size, env_single.w_max)
    
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