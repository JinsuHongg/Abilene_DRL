import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool 
from torch_geometric.utils import scatter

class ActorGCN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim=128):
        super(ActorGCN, self).__init__()
        
        self.gcn1 = GCNConv(node_input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim // 2)

        # Update edge_fc1 to match the size of edge features
        self.edge_fc1 = nn.Linear(edge_input_dim, hidden_dim)
        self.edge_fc2 = nn.Linear(hidden_dim, 30)  # Further reduce dimensions

        self.actor_fc = nn.Linear(hidden_dim // 2, 64)
        self.actor_out = nn.Linear(64, edge_input_dim)  # Output is per graph's action space

    def forward(self, data):
        nodes, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
      
        # Check if edge_attr has the expected number of features
        # Process edge attributes (adjusting this layer to match input)
        edge_features = F.relu(self.edge_fc1(edge_attr))  # Input dimension: edge_input_dim
        edge_features = F.relu(self.edge_fc2(edge_features))  # Output dimension: hidden_dim // 2

        # Aggregate edge features to node features
        edge_aggregated = scatter(edge_features, edge_index[0], dim=0, reduce='mean')
        x = nodes + edge_aggregated  # Combine node and edge features

        # Apply GCN layers
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))

        # Actor network (final layers for action output)
        actor_x = F.relu(self.actor_fc(x))
        action_probs = self.actor_out(actor_x)

        global_action = torch.mean(action_probs, dim=0, keepdim=True)
        return global_action[0]


class CriticGCN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, action_dim, hidden_dim=128):
        super(CriticGCN, self).__init__()
        
        # GCN layers for state processing
        self.gcn1 = GCNConv(node_input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim // 2)

        # Fully connected layers for edge attributes
        self.edge_fc1 = nn.Linear(edge_input_dim, hidden_dim)
        self.edge_fc2 = nn.Linear(hidden_dim, 30)

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
        nodes, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        _, _, action_attr = action.x, action.edge_index, action.edge_attr
        
        # Process edge attributes
        edge_features = F.relu(self.edge_fc1(edge_attr))
        edge_features = F.relu(self.edge_fc2(edge_features))

        # Aggregate edge features to node features
        edge_aggregated = scatter(edge_features, edge_index[0], dim=0, reduce='mean')
        x = nodes + edge_aggregated  # Combine node and edge features

        # Apply GCN layers
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))

        # Graph-level pooling (optional, depending on your use case)
        graph_embedding = global_mean_pool(x, data.batch)

        # Process state and action separately
        state_features = F.relu(self.state_fc(graph_embedding))
        action_features = F.relu(self.action_fc(action_attr))

        # Combine state and action features
        combined_features = torch.cat([state_features, action_features.reshape(1,64)], dim=-1)
        q_value = F.relu(self.q_fc1(combined_features))
        q_value = self.q_out(q_value)

        return q_value

