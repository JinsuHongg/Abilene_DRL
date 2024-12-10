import time
import torch
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
import networkx as nx
from torch_geometric.data import Data

class MininetGraphEnv(Env):

    def __init__(self, topology):
        super(MininetGraphEnv, self).__init__()
        self.topology = topology
        # self.topology.mininet() # setup mininet network
        self.current_interval = 0
    
    def nx_to_pyg_graph(self, G):
        """
        Convert a networkx graph to a PyTorch Geometric Data object.
        """
        # Node features (optional, here we use identity)
        num_nodes = G.number_of_nodes()
        x = torch.eye(num_nodes, dtype=torch.float32)  # Identity matrix as node features
        
        # Extract edges and weights
        edge_index = torch.tensor(list(G.edges)).t().contiguous()  # Shape: [2, num_edges]
        edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges]  # Edge weights, default is 1.0
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)  # Shape: [num_edges]
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
    
    def np_to_nx(self, arr):
        """
        Convert numpy arr to a networkx graph.
        """
        G = nx.DiGraph()
        for id, (src, dst) in self.topology.link_idx_to_sd.items():
            G.add_weighted_edges_from([(int(src), int(dst),  arr[id])])
        
        return G 

    def rl_step(self, new_action: nx.DiGraph, time_step):
        """
        Perform an RL step by applying the action and observing the next state.
        :param action: RL agent's action (new routing weights).
        :param time_step: Current time step in the simulation.
        :return: next_state, reward, done
        """
        # Update routing weights based on the action
        self._apply_action(new_action)  # Convert action to weight updates

        # update state
        # crr_time = time.time()
        new_state = self.calculate_new_state(point = time_step)
        # duration = time.time() - crr_time
        # print(f"Time spend by update: {duration}")
      
        # Compute reward (negative of total travel time)
        reward = self.calculate_total_traveling_time()
        
        # Determine if the episode is done
        done = (time_step >= len(self.topology.traffic_matrices) * 24 - 1) # becuase we have total 24 weeks data

        return new_state, reward, done
    
    def _apply_action(self, new_action: nx.DiGraph):
        """
        Update the routing weights dynamically.
        :param action: Array of weight updates.
        """
        for src, dst in new_action.edges:
            self.topology.action[src][dst]['weight'] = new_action[src][dst]['weight']  # Update graph weights
            

    def step_traffic_matrix(self, point: int, channel: int = 0):
        """
        Update the network's traffic matrix to the next time interval.
        :param point: Index of the traffic matrix (time step).
        :param channel: Traffic matrix channel to use.
        """
        if point >= len(self.topology.traffic_matrices):
            raise ValueError("Traffic matrix point exceeds available data.")

        # Update the traffic matrix
        self.current_traffic_matrix = self.topology.traffic_matrices[point][channel]
        print(f"Traffic matrix updated to time point {point}, channel {channel}.")

    
    def calculate_total_traveling_time(self):
        """
        Estimate total traveling time without simulation
        
        Parameters:
        - traffic_matrix: Traffic volume between nodes
        - topology: Network graph
        - link_weights: Weights affecting path selection
        
        Returns:
        - Total traveling time
        """
        total_traveling_time = 0
        # max_val = 0
        
        for src, dst in self.topology.state.edges:
            if src == dst:
                continue  # Skip self-connections
            
            # Find shortest path considering link weights
            # try:
            
            #     # Find shortest path
            #     path = nx.shortest_path(
            #         self.topology.action, 
            #         source=src, 
            #         target=dst, 
            #         weight='weight'
            #     )
                
                # # Calculate path length (travel time)
                # path_length = nx.path_weight(self.topology.action, path, weight='weight')
                
                # # Estimate traveling time based on path length and traffic volume
                # traveling_time = path_length * self.topology.state[src][dst]['weight']

                # # if traveling_time > max_val:
                # #     max_val = traveling_time
                
                # total_traveling_time += traveling_time

            # Compute path travel time considering congestion
            link_id = self.topology.link_sd_to_idx[src, dst]
            capacity = self.topology.link_capacities[link_id]
            weight = self.topology.action[src][dst]['weight']
            traffic_volume = self.topology.state[src][dst]['weight']
            # print(traffic_volume, capacity, weight)
            # Apply a congestion model
            link_travel_time = weight * (1 + 0.15 * (traffic_volume / capacity) ** 4)
            
            
            # Scale by traffic volume
            total_traveling_time += link_travel_time

            
            # except nx.NetworkXNoPath:
            #     # Handle cases where no path exists
            #     print(f"No path between {src} and {dst}")
        
        return (10000 / (total_traveling_time))
    
    def calculate_new_state(self, point, channel: int = 0):
        """
        Calculate new state after link weight modification
        Returns:
        - new_traffic_matrix: Redistributed traffic based on new routing paths
        """
    
        # New traffic matrix to store redistributed traffic
        new_state = nx.DiGraph()
        for id, (src, dst) in self.topology.link_idx_to_sd.items():
            new_state.add_weighted_edges_from([(src, dst,  0)])

        # Recalculate routing for each traffic flow
        for src in range(self.topology.num_node):
            for dst in range(self.topology.num_node):
                if src == dst:
                    continue  # Skip self-connections
                
                try:
                    # Find shortest path using Dijkstra with new link weights
                    shortest_path = nx.shortest_path(
                        self.topology.action, 
                        source=src, 
                        target=dst, 
                        weight='weight'
                    )
                    
                    # Calculate path links
                    path_links = list(zip(shortest_path[:-1], shortest_path[1:]))
                    
                    # Distribute traffic along the new path
                    for x, y in path_links:
                        # You might want to distribute traffic proportionally
                        # This is a simple equal distribution approach
                        
                        new_state[x][y]['weight'] += self.topology.traffic_matrices[point][channel][src, dst] # traffic
            
                except nx.NetworkXNoPath:
                    print(f"No path between {src} and {dst}")
        
        return new_state
