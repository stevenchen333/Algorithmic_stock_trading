import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from gym import spaces
import random
from tqdm import tqdm
from collections import deque

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class TradingEnvironment(gym.Env):
    """
    A trading environment that implements the double linear policy from the paper.
    """
    def __init__(self, returns_data, initial_balance=100000, transaction_cost=0.0001, alpha=0.5,
                window_size=10, max_steps=252):
        super(TradingEnvironment, self).__init__()
        
        # Environment parameters
        self.returns_data = returns_data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.alpha = alpha  # Allocation constant
        self.max_steps = max_steps
        
        # Compute bounds for returns
        self.X_min = min(self.returns_data)
        self.X_max = max(self.returns_data)
        
        # Define action space: w ∈ [0, w_max]
        self.w_max = min(1/(1+self.transaction_cost), 1/(self.X_max+self.transaction_cost))
        self.action_space = spaces.Box(low=0, high=self.w_max, shape=(1,), dtype=np.float32)
        
        # Define observation space (returns history, account values, current w)
        observation_high = np.ones(self.window_size + 4) * np.finfo(np.float32).max
        observation_low = np.ones(self.window_size + 4) * np.finfo(np.float32).min
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)
        
        # Initialize state variables
        self.reset()
        
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
        
        # Get initial state
        return self._get_observation()
    
    def _get_observation(self):
        #TODO: add stock price as feature

        # Returns history features
        returns_history = self.returns_data[self.current_step-self.window_size:self.current_step]
        
        # Add account values and current w to state
        additional_features = np.array([
            self.V_L / self.V_0,        # Normalized long account value
            self.V_S / self.V_0,        # Normalized short account value
            self.V / self.V_0,          # Normalized total account value
            self.current_w / self.w_max # Normalized weight
        ])
        
        # Combine into observation
        observation = np.concatenate([returns_history, additional_features])
        return observation
    
    # Improved reward function for the TradingEnvironment
    def improved_reward_function(self, X, w, old_V, new_V):
        """
        Compute a reward that better incentivizes risk-adjusted returns.
        
        Args:
            X: Current market return
            w: Current weight (action)
            old_V: Previous account value 
            new_V: New account value
            
        Returns:
            reward: Calculated reward value
        """
        # 1. Base reward for account value change (main objective)
        base_reward = ((new_V - old_V) / old_V) * 100  # Percentage return
        
        # 2. Sharpe ratio component (risk-adjusted return)
        # Using a simple approximation - positive if return exceeds the "risk" taken
        risk_adjustment = 0
        if abs(w) > 0.01:  # Only if there is a significant position
            expected_risk = abs(w) * 0.01  # Simplification: risk is proportional to position size
            risk_adjustment = 2 * (base_reward / 100) / expected_risk if expected_risk > 0 else 0
            risk_adjustment = max(-5, min(risk_adjustment, 5))  # Clipping to prevent extreme values
        
        # 3. Penalty for zero action (to prevent the policy from always choosing w=0)
        zero_action_penalty = -0.5 if abs(w) < 0.01 else 0
        
        # 4. Market prediction bonus
        # Reward for correctly predicting market direction (positive w when X>0, negative w when X<0)
        prediction_bonus = 0.5 * np.sign(w * X) if abs(w) > 0.01 else 0
        
        # 5. Position sizing bonus - reward for using appropriate position sizes
        # Encourage positions that are proportional to the opportunity
        # This will help prevent reward converging to zero as the agent learns
        sizing_bonus = 0
        if abs(X) > 0.001:  # Only if there's a significant market move
            optimal_w = min(0.5, abs(X) * 5)  # Simplified optimal position size
            sizing_error = abs(abs(w) - optimal_w)
            sizing_bonus = 0.5 * (1 - sizing_error/optimal_w) if abs(w) > 0.01 else -0.5
        
        # Total reward
        reward = base_reward + risk_adjustment + zero_action_penalty + prediction_bonus + sizing_bonus
        
        return reward


    # Modified step method for TradingEnvironment
    def step(self, action):
        """Modified step function with improved reward calculation"""
        # Unpack action
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
        
        # Calculate improved reward
        reward = self.improved_reward_function( X, w, self.prev_V, self.V)
        
        # Move to next step
        self.current_step += 1
        self.steps_taken += 1
        
        # Check if done
        done = (self.steps_taken >= self.max_steps) or (self.V <= 0)
        
        # Get new observation
        observation = self._get_observation()
        
        # Return additional info
        info = {
            'account_value': self.V,
            'return': X,
            'long_value': self.V_L,
            'short_value': self.V_S,
            'weight': w
        }
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        pass  # We'll implement visualization separately
# Add this class before LinearPolicyGradientAgent
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states), 
                np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class LinearPolicyGradientAgent:
    """
    Policy Gradient agent with linear function approximation and TD-lambda for trading
    """
    def __init__(self, state_size, action_size, action_bound):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound  # Upper bound for continuous action
        
        # Hyperparameters
        self.gamma = 0.95       # Discount factor
        self.entropy_coef = 0.01 
        self.lambda_val = 0.9   # TD(λ) parameter
        self.alpha = 0.001      # Learning rate for policy parameters
        self.alpha_v = 0.01     # Learning rate for value function
        self.sigma = 0.2       # Standard deviation for exploration
        self.exploration_decay = 0.995  # Decay rate for exploration
        self.min_sigma = 0.05
        self.episodes_count = 0   # Minimum exploration
        
        # Initialize policy parameters (θ)
        self.theta = np.zeros(state_size)
        
        # Initialize value function parameters (w)
        self.w = np.zeros(state_size)
        
        # Initialize eligibility traces
        self.e_theta = np.zeros_like(self.theta)  # Eligibility trace for policy
        self.e_w = np.zeros_like(self.w)   
        self.replay_buffer = ReplayBuffer(capacity=50000)
        self.batch_size = 32
        self.min_experiences = 1000  # Min experiences before training
        self.update_every = 4        # Update network every N steps
        self.step_count = 0       # Eligibility trace for value function
        
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
    
    def update(self, state, action, reward, next_state, done):
        # Calculate TD error
        delta = reward + (0 if done else self.gamma * self.get_value(next_state)) - self.get_value(state)
        
        # Update eligibility traces
        self.e_theta = self.gamma * self.lambda_val * self.e_theta + self._policy_gradient(state, action)
        self.e_w = self.gamma * self.lambda_val * self.e_w + state
        
        # Add entropy bonus
        entropy_bonus = self.entropy_coef * np.log(self.sigma)
        
        # Update parameters
        self.theta += self.alpha * (delta * self.e_theta + entropy_bonus * state)
        self.w += self.alpha_v * delta * self.e_w
        
        if done:
            self.episodes_count += 1
            
            # Reset eligibility traces
            self.e_theta = np.zeros_like(self.theta)
            self.e_w = np.zeros_like(self.w)
            
            # Periodic exploration decay
            if self.episodes_count % 5 == 0:  # Decay every 5 episodes
                self.sigma = max(self.min_sigma, self.sigma * self.exploration_decay)
            
            # Learning rate annealing - very slow decay
            if self.episodes_count % 100 == 0:
                self.alpha = max(0.0001, self.alpha * 0.999)
                self.alpha_v = max(0.001, self.alpha_v * 0.999)
            
            # Periodic parameter noise to escape local minima
            if self.episodes_count % 50 == 0:
                noise_magnitude = 0.01 * (1.0 - min(1.0, self.episodes_count / 5000))
                self.theta += np.random.normal(0, noise_magnitude, size=self.theta.shape)
    
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


def simulate_market_data(n_steps=2000, mu=-0.0001, sigma=0.01, jump_intensity=0.05, jump_size=0.03):
    """
    Simulate stock returns data with a geometric Brownian motion with jumps model
    """
    # Generate normal returns
    normal_returns = np.random.normal(mu, sigma, n_steps)
    
    # Generate jumps
    jumps = np.random.binomial(1, jump_intensity, n_steps) * np.random.choice([-1, 1], n_steps) * jump_size
    
    # Combine normal returns with jumps
    returns = normal_returns + jumps
    
    # Ensure returns are within bounds (avoid extreme values)
    returns = np.clip(returns, -0.2, 0.2)
    
    return returns


def main():
    # Simulate market data or load real data
    returns_data = simulate_market_data(n_steps=3000)
    
    # Create environment
    env = TradingEnvironment(
        returns_data=returns_data,
        initial_balance=100000,
        transaction_cost=0.0001,  # 0.01%
        alpha=0.5,  # Equal allocation to long and short
        window_size=10,
        max_steps=252  # Roughly one trading year
    )
    
    # Create policy gradient agent with linear function approximation
    state_size = env.observation_space.shape[0]
    action_size = 1  # Just the weight parameter w
    agent = LinearPolicyGradientAgent(state_size, action_size, env.w_max)
    
    # Train agent
    rewards_history, avg_rewards_history, asset_values_history = train_agent(env, agent, episodes=10000)
    
    # Evaluate agent
    eval_results = evaluate_agent(env, agent, episodes=5)
    
    # Plot results
    plot_results(rewards_history, avg_rewards_history, asset_values_history, eval_results)
    
    # Save the trained model
    agent.save("trading_pg_model_weights.npz")
    
    return env, agent, rewards_history, eval_results


if __name__ == "__main__":
    main()