
import os
import glob
import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt

class Topology_Traffic():
    def __init__(self):
        
        self.path = os.getcwd()
        # print(path)
        # parentpath = os.path.abspath(os.path.join(path, os.pardir))
        file = self.path + '/data_processed/node_list_data.npy'
        print('Loading topology data...', file)

        # node info: [id, name, city, latitude, longitude], 
        # link info: [start, desination, start_id, destination_id, link_id, capacity (kbps), OSPF]
        with open(file, 'rb') as f:
            header = np.load(f)
            node = np.load(f)
            link = np.load(f)

        self.num_node = node.shape[0]
        self.num_link = link.shape[0]

        self.DG = nx.DiGraph()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((link.shape[0]+1))
        self.link_weights = np.empty((link.shape[0]+1))

        for each_info in link:
            start, destination, s, d, i, c, w = each_info
            self.link_idx_to_sd[int(i)] = (int(s),int(d))
            self.link_sd_to_idx[(int(s),int(d))] = int(i)
            self.link_capacities[int(i)] = float(c)
            self.link_weights[int(i)] = int(w)
            self.DG.add_weighted_edges_from([(int(s), int(d), int(w))])

        assert len(self.DG.nodes()) == self.num_node and len(self.DG.edges()) == self.num_link

        self.Load_traffic()


    def Load_traffic(self):
        
        self.traffic_file = glob.glob(self.path + '/data/X01*') 
        assert len(self.traffic_file) > 0

        print('Loading traffic matrices...', self.traffic_file[0])
        f = open(self.traffic_file[0], 'r')


        # Total 5 TM in each line <realOD>, <simpleGravityOD>, <simpleTomogravityOD>, <generalGravityOD>, <generalTomogravityOD>
        #  Check readme file here (https://sndlib.put.poznan.pl/home.action)
        # final shape of traffic matrices: 2016 lines x (5 channel x 12 nodes x 12 nodes) 
        self.traffic_matrices = []
        for line in f:
            volumes = line.strip().split(' ')
            total_volume_cnt = len(volumes)
            # print(total_volume_cnt, self.num_node)
            assert total_volume_cnt == self.num_node * self.num_node * 5
            matrix = np.zeros((5, self.num_node, self.num_node))
            for v in range(total_volume_cnt):
                
                channel = v % 5
                row = int( ((v - channel) / 5) // 12 )
                column = int( ((v - channel) / 5) % 12 )
                # change 100bytes/5min to kilobits/sec, and 5 min interval, 300 = (5min x 60sec)
                matrix[channel][row, column] = float(volumes[v]) * 8 * 100 / (1024 * 300) 
            
            self.traffic_matrices.append(matrix)

        f.close()
        
        self.traffic_matrices = np.array(self.traffic_matrices)
        self.tms_shape = self.traffic_matrices.shape
        self.tm_cnt = self.tms_shape[0]
        print(f'Traffic matrices dims: {self.tms_shape}')
