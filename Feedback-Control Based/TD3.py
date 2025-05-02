'''
Key Components of TD3 Algorithm
1. Replay Buffer (D)
2. Actor Network (πφ) with parameters φ
3. Two Critic Networks (Qθ1, Qθ2) with parameters θ1, θ2
4. Target Critic Networks (Qθ'1, Qθ'2) with parameters θ'1, θ'2
5. Temperature parameter (α) - can be learned or fixed


Hyperparameters
batch_size = 256        # Minibatch size
γ = 0.99                # Discount factor
τ = 0.005               # Target network update rate
lr = 3e-4               # Learning rate
α = 0.2                # Initial temperature (if not learned)
buffer_size = 1e6      # Replay buffer size
start_steps = 10000     # Random action steps before training
update_freq = 1         # Network update frequency
'''



# Initialize:
#     - Replay buffer D with capacity buffer_size
#     - Actor network πφ with random weights φ
#     - Two critic networks Qθ1, Qθ2 with random weights θ1, θ2
#     - Target critic networks Qθ'1, Qθ'2 with weights θ'1 ← θ1, θ'2 ← θ2
#     - If automatic entropy tuning:
#         - Initialize α with target entropy H̄ (typically -dim(action))
#         - Initialize optimizer for α
#     - Optimizers for actor and critics



# for episode = 1 to M do:
#     Reset environment, get initial state s
    
#     for t = 1 to T do:
#         if t < start_steps:
#             a ∼ random_action()
#         else:
#             a ∼ πφ(s)  # Sample action from policy
            
#         Execute a, observe r, s', done
        
#         Store (s, a, r, s', done) in D
        
#         s ← s'
        
#         if t > start_steps and t % update_freq == 0:
#             Sample batch {(s, a, r, s', d)} ∼ D
            
#             # Critic Update
#             with stop_gradient:
#                 a' ∼ πφ(s')  # Sample next action
#                 target_Q1 = Qθ'1(s', a')
#                 target_Q2 = Qθ'2(s', a')
#                 target_Q = min(target_Q1, target_Q2)
#                 target = r + γ * (1 - d) * (target_Q - α * log πφ(a'|s'))
                
#             # Update both critics
#             Q1_loss = MSE(Qθ1(s, a), target)
#             Q2_loss = MSE(Qθ2(s, a), target)
#             Update θ1, θ2 to minimize Q1_loss + Q2_loss
            
#             # Actor Update
#             a_new ∼ πφ(s)  # Resample action for current state
#             Q_new = min(Qθ1(s, a_new), Qθ2(s, a_new))
#             policy_loss = α * log πφ(a_new|s) - Q_new
#             Update φ to minimize policy_loss
            
#             # Temperature Update (if automatic tuning)
#             if learning α:
#                 α_loss = -α * (log πφ(a_new|s) + H̄)
#                 Update α to minimize α_loss
                
#             # Update target networks
#             θ'1 ← τθ1 + (1 - τ)θ'1
#             θ'2 ← τθ2 + (1 - τ)θ'2
            
#     end for
# end for



# Actor Network πφ(s):
#     Input: state s
#     Hidden layers: 2 fully connected layers (256 units each) with ReLU
#     Output: mean μ and log_std logσ (for diagonal Gaussian)
#     Action: a = tanh(μ + σ·ε), where ε ∼ N(0, I)
    
# Critic Networks Qθ(s, a):
#     Input: state s and action a concatenated
#     Hidden layers: 2 fully connected layers (256 units each) with ReLU
#     Output: Q-value (single scalar)