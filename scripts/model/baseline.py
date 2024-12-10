import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

class ActorFC(nn.Module):
    def __init__(self, edge_input_dim, hidden_dim=128):
        super(ActorFC, self).__init__()
        
        # Update edge processing layers
        self.edge_fc1 = nn.Linear(edge_input_dim, hidden_dim)
        self.edge_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # Further reduce dimensions

        self.actor_fc = nn.Linear(hidden_dim // 2, 64)
        self.actor_out = nn.Linear(64, edge_input_dim)  # Output is per graph's action space

    def forward(self, data):
        nodes, edge_attr = data.x, data.edge_attr
      
        # Process edge attributes
        edge_features = F.relu(self.edge_fc1(edge_attr))
        edge_features = F.relu(self.edge_fc2(edge_features))

        # Actor network (final layers for action output)
        actor_x = F.relu(self.actor_fc(edge_features))
        action_probs = self.actor_out(actor_x)
        
        return action_probs


class CriticFC(nn.Module):
    def __init__(self, edge_input_dim, action_dim, hidden_dim=128):
        super(CriticFC, self).__init__()
        
        # Fully connected layers for edge attributes
        self.edge_fc1 = nn.Linear(edge_input_dim, hidden_dim)
        self.edge_fc2 = nn.Linear(hidden_dim, 64)

        # Fully connected layers for combined state and action
        self.state_fc = nn.Linear(hidden_dim // 2, 64)
        self.action_fc = nn.Linear(action_dim, 64)
        self.q_fc1 = nn.Linear(64 + 64, 64)  # Combine state and action embeddings
        self.q_out = nn.Linear(64, 1)  # Output a single Q-value

    def forward(self, data, action):
        """
        data: A batch of graphs (Batch object).
        action: Continuous action tensor (batch_size x action_dim).
        """
        nodes, edge_attr = data.x, data.edge_attr
        _, action_attr = action.x, action.edge_attr
        
        # Process edge attributes
        edge_features = F.relu(self.edge_fc1(edge_attr))
        edge_features = F.relu(self.edge_fc2(edge_features))

        # Process state and action separately
        state_features = F.relu(self.state_fc(edge_features))
        action_features = F.relu(self.action_fc(action_attr))

        # Combine state and action features
        combined_features = torch.cat([state_features, action_features], dim=-1)
        q_value = F.relu(self.q_fc1(combined_features))
        q_value = self.q_out(q_value)

        return q_value