import torch
import numpy as np
from scripts.model.GCN import ActorGCN, CriticGCN
from scripts.model.baseline import ActorFC, CriticFC

class TopologyAgent:
    def __init__(self, env, epsilon, tau=0.001):
        self.env = env  # The network environment (Mininet or other)
        self.replay_buffer = ReplayBuffer(capacity = 10000)
        
        num_nodes = env.topology.num_node
        num_edges = env.topology.num_link

        # Actor and Critic networks
        self.actor = ActorGCN(num_nodes, num_edges)  # The neural network that approximates the policy
        self.critic = CriticGCN(num_nodes, num_edges, num_edges)  # The neural network that approximates the value function

        # self.actor = ActorFC(num_edges)  # The neural network that approximates the policy
        # self.critic = CriticFC(num_edges, num_edges)  # The neural network that approximates the value function
        
        # Target Actor and Target Critic networks (copy of the original networks)
        self.target_actor = ActorGCN(num_nodes, num_edges)
        self.target_critic = CriticGCN(num_nodes, num_edges, num_edges)

        # self.target_actor = ActorFC(num_edges)
        # self.target_critic = CriticFC(num_edges, num_edges)
        
        # Initialize the target networks' weights to be the same as the current networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Hyper-parameters
        self.epsilon = epsilon
        self.tau = tau  # Soft update factor (for soft updates)

    def select_action(self, state, decay_coef):
        """
        Select an action based on the current state of the environment.
        Implements epsilon-greedy strategy for exploration/exploitation.
        """
        # Use the policy network to get action probabilities and state value
        action_probs = self.actor(state)  # Get action probabilities from the actor part of the model
        
        # Convert action_probs to numpy for easier manipulation
        action_probs = action_probs.detach().numpy()
        
        if np.random.rand() < self.epsilon:
            # Exploration: add random noise to actions
            exploration_noise = np.random.normal(0, 0.2, action_probs.shape) * decay_coef
            
            # Combine network's action with exploration noise
            # Clip to ensure we stay within [0, 1] range of tanh
            action = np.clip(action_probs + exploration_noise, 0, 1)
        else:
            # Exploitation: use the network's suggested actions directly
            action = np.clip(action_probs, 0, 1)

        
        action = self.env.np_to_nx(action) # convert it to graph

        return action

    def update_target_networks(self):
        """
        Softly update the target networks using Polyak averaging.
        The target network gets updated by a fraction of the current network's weights.
        """
        # Soft update: Update the target networks' weights slowly (Polyak averaging)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        # Sample a batch of experiences
        batch = np.random.choice(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
