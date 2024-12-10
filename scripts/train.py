import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from scripts.agent import ReplayBuffer

def train_actor_critic(
    env, 
    agent,
    actor_lr=1e-3, 
    critic_lr=1e-3,
    decay_wt = 1, 
    gamma=0.99
):  
    # Separate optimizers for actor and critic
    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=critic_lr)

    # Initialize optimizer
    rewards_per_episode = []
    epsilon = agent.epsilon

    for episode in range(0, 5000): #24 * 2016 + 1
        
        total_reward = 0
        total_actor_loss = 0
        total_critic_loss = 0
        done = False
        time_step = 0
        
        # # Initial state
        # tag = episode//2016 + 1
        # t_point = episode % 2016

        tag = 1
        t_point = 0
        state = env.topology.state_graph(filetag = tag, point = t_point)
        state = env.nx_to_pyg_graph(state)
        
        # Lists to store episode information
        episode_states = []
        episode_next_states = []
        episode_actions = []
        episode_rewards = []
        episode_targets = []
        episode_state_values = []
        episode_actor_losses = []
        
        for time_step in range(100):

            # Select action using actor (policy)
            # current_time = time.time()
            # action = agent.select_action(state, decay_coef = decay_wt)
            
            action = agent.actor(state)
            # duration = time.time() - current_time 
            # print(f"Time spent by model: {duration} sec")
            action = action_stochacity(env = env, 
                                       action = action, 
                                       decay_coef = decay_wt, 
                                       epsilon = epsilon)
            
            # Perform action in environment
            with torch.no_grad():
                new_state, reward, done = env.rl_step(action, time_step=t_point) # or 0
            new_state = env.nx_to_pyg_graph(new_state)

            # Store episode information
            episode_states.append(state)
            episode_next_states.append(new_state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            # Compute value of next state using the critic
            action = env.nx_to_pyg_graph(action)
            next_state_value = agent.critic(new_state, action)

            # Compute temporal difference (TD) error for critic
            state_value = agent.critic(state, action)  # Get value of the current state
            td_error = reward + gamma * next_state_value.detach() - state_value

            # Critic Loss: MSE between target and current Q values
            # total_critic_loss += td_error

            # Compute losses
            episode_state_values.append(state_value)
            episode_targets.append(td_error)
            episode_actor_losses.append(-state_value)

            # Update total reward and state
            total_reward += reward
            state = new_state
            decay_wt -= ((episode + 1)*800)**-1
            epsilon = max(0.01, agent.epsilon - 0.01 * episode)


        # Compute mean losses for the episode
        episode_states_tensor = torch.stack(episode_state_values)
        episode_targets_tensor = torch.stack(episode_targets)
        mean_critic_loss = F.mse_loss(episode_states_tensor, episode_targets_tensor)
        mean_actor_loss = torch.stack(episode_actor_losses).mean()

        # Reset gradients before backward passes
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        # # Optimize Actor
        # mean_actor_loss.backward()
        # actor_optimizer.zero_grad()
        # actor_optimizer.step()     

        # # Optimize Critic
        # mean_critic_loss.backward()
        # critic_optimizer.zero_grad()
        # critic_optimizer.step()

        # Backward passes with retain_graph
        mean_actor_loss.backward(retain_graph=True)
        actor_optimizer.step()     

        mean_critic_loss.backward()
        critic_optimizer.step()

        # Log rewards
        rewards_per_episode.append(total_reward)

        # --- Update Target Networks ---
        # Soft update for the target networks
        agent.update_target_networks()

        # Optional: Add to replay buffer
        for state, action, reward, next_state in zip(
            episode_states, 
            episode_actions, 
            episode_rewards, 
            episode_states[1:] + [new_state], 
        ):
            agent.replay_buffer.push(state, action, reward, next_state, done)

        print(
            f"File {tag}, Time point {t_point}, "
            f"Episode {episode}, "
            f"Critic Loss: {mean_critic_loss.item():.4f}, "
            f"Actor Loss: {mean_actor_loss.item():.4f}, " 
            f"Total Reward: {total_reward:.4f}, "
            f"Epsilon: {epsilon}")

    return agent

def action_stochacity(env, action, decay_coef, epsilon):
   
    """
    Select an action based on the current state of the environment.
    Implements epsilon-greedy strategy for exploration/exploitation.
    """
    # # Use the policy network to get action probabilities and state value
    # action_probs = self.actor(state)  # Get action probabilities from the actor part of the model
    
    # Convert action_probs to numpy for easier manipulation
    action_probs = action.detach().numpy()
    
    if np.random.rand() < epsilon:
        # Exploration: add random noise to actions
        exploration_noise = np.random.normal(0, 0.2, action_probs.shape) * decay_coef
        
        # Combine network's action with exploration noise
        # Clip to ensure we stay within [0, 1] range of tanh
        action = np.clip(action_probs + exploration_noise, 0, 1)
    else:
        # Exploitation: use the network's suggested actions directly
        action = np.clip(action_probs, 0, 1)

    
    action = env.np_to_nx(action) # convert it to graph

    return action
