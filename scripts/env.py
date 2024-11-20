import torch
import numpy as np
import networkx as nx
from .dataloader import Topology_Traffic


class Environment(object):
    def __init__(self):
        
        self.total_info = Topology_Traffic()
        # self.traffic = Traffic(config, self.topology.num_nodes, self.data_dir, is_training=is_training)
        
        # traffic information
        self.traffic_matrices = self.total_info.traffic_matrices #kbps
        self.tm_cnt = self.total_info.tm_cnt
        # self.traffic_file = self.total_info.traffic_file

        # topology information
        self.num_node = self.total_info.num_node
        self.num_link = self.total_info.num_link
        self.link_idx_to_sd = self.total_info.link_idx_to_sd
        self.link_sd_to_idx = self.total_info.link_sd_to_idx
        self.link_capacities = self.total_info.link_capacities
        self.link_weights = self.total_info.link_weights
  
    def __len__(self):
        return len(self.traffic_matrices)

    def __getitem__(self, idx):
        demand = self.traffic_matrices[idx][0][:] # TM type: 0
        return {"topology": self.total_info.DG, "demand": torch.tensor(demand, dtype=torch.float)}
    
    def compute_link_utilization(self, channel):
        # change 100bytes/5min to kilobits/sec, 300 = (5min x 60sec)
        self.traffic_matrices = self.traffic_matrices * 8 * 100 / (1024 * 300)

        self.traffic = np.zeros(self.num_link)
        for src in range(self.num_node):
            for dst in range(self.num_node):
                traffic = self.traffic_matrices[0][channel][src, dst] # time interval, channel, source, destination
                if traffic > 0:
                    path = nx.shortest_path(self.total_info.DG, src, dst, weight='weight')
                    for i in range(len(path) - 1):
                        link = self.link_sd_to_idx[(path[i], path[i + 1])]
                        self.traffic[link] += (traffic) # because of 5min interval (5min x 60sec)
        self.traffic
        # return self.utilization
    
    def calculate_total_traveling_time(self):
        total_traveling_time = 0
        for i, trf in enumerate(self.traffic):
            # if trf < 1:  # Avoid division by zero
            link_delay = 1 / (self.link_capacities[i] - trf)  # Delay proportional to 1/(1 - utilization)
            # else:
                # print(i, utilization)
                # link_delay = float('inf')  # Highly congested link
            # traffic = self.utilization[i] * self.link_capacities[i]
            total_traveling_time += link_delay  # Weight delay by traffic volume
        
        return total_traveling_time

    def reward(self):
        total_time = self.calculate_total_traveling_time()
        return -total_time  # Negative reward to minimize total traveling time