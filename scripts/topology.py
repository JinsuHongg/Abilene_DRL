#!/usr/bin/env python

import os
import sys
import time
import glob
import random
import threading
import subprocess
import numpy as np
import networkx as nx

# from mininet.cli import CLI
# from mininet.link import TCLink
# from mininet.net import Mininet
# from mininet.clean import cleanup
# from mininet.log import setLogLevel, info
# from mininet.node import Controller, OVSSwitch

from mininet.log import info
from mininet.cli import CLI
from mininet.net import Mininet
from mininet.clean import cleanup
from mininet.node import OVSController, Controller, OVSSwitch

from scripts.monitor import TrafficMonitor 

# def cleanup():
#     """Perform cleanup of existing network interfaces"""
#     info('*** Cleaning up existing Mininet interfaces\n')
#     # Kill any existing Mininet processes
#     os.system('sudo mn -c')
    
#     # Additional cleanup for Open vSwitch
#     os.system('sudo ovs-vsctl del-br br0')
#     os.system('sudo pkill -f "ovs-"')

#     # os.system("sudo ovs-vsctl -- --all destroy QoS -- --all destroy Queue")
#     # os.system("sudo ovs-vsctl list-br | xargs -n 1 sudo ovs-vsctl del-br")


class topology_mininet():
    def __init__(self):
        
        # 1) define initial topology from the dataset.
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
            self.link = np.load(f)

        self.num_node = node.shape[0]
        self.num_link = self.link.shape[0]

        self.state = nx.DiGraph()
        self.action = nx.DiGraph()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.link.shape[0]))
        # self.flow_completion_times = {}
        # self.real_time_traffic_matrix = np.zeros((self.num_node, self.num_node))
        
        for each_info in self.link:
            start, destination, s, d, i, c, w = each_info
            capacity = float(c) / (10**(4))
            self.link_idx_to_sd[int(i)] = (int(s),int(d))
            self.link_sd_to_idx[(int(s),int(d))] = int(i)
            self.link_capacities[int(i)] = capacity / (10**4) # scaling 9920000 to 922 because of the mininet limits
            # self.link_weights[int(i)] = int(w)
            self.action.add_weighted_edges_from([(int(s), int(d),  int(w) / random.random())]) # low weight prefered
            # self.action.add_weighted_edges_from([(int(s), int(d),  int(w) / np.sum(self.link[:, -1].astype('int')))]) # low weight prefered

        assert len(self.action.nodes()) == self.num_node and len(self.action.edges()) == self.num_link

    def Load_TM(self, filetag):
        
        self.traffic_file = glob.glob(self.path + f'/data/X{filetag:02d}*') 
        assert len(self.traffic_file) > 0

        # print('Loading traffic matrices...', self.traffic_file[0])
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
                if row != column:
                    matrix[channel][row, column] = float(volumes[v]) 
            
            self.traffic_matrices.append(matrix)

        f.close()
        
        # change 100bytes/5min to kilobits/sec, 300 = (5min x 60sec) and divide 10^4 because of the mininet 1GB
        self.traffic_matrices = np.array(self.traffic_matrices) * 8 * 100 / (1024 * 300 * 10**4)
    
    def compute_link_utilization(self):

        self.traffic_arr = np.zeros(self.num_link)
        for src, dst in self.state.edges:
            traffic = self.state[src, dst] # time interval(2016), channel, source, destination
            if traffic > 0:
                path = nx.shortest_path(self.DG, src, dst, weight='weight')
                for i in range(len(path) - 1):
                    link = self.link_sd_to_idx[(path[i], path[i + 1])]
                    self.traffic_arr[link] += (traffic) # because of 5min interval (5min x 60sec)

        # Compute utilization for each link
        self.link_utilization = {}
        for link, traffic in enumerate(self.traffic_arr):
            capacity = self.link_capacities[link]
            utilization = traffic / capacity
            self.link_utilization[link] = utilization
    
    def state_graph(self, filetag, point, channel = 0):

        self.Load_TM(filetag)

        for each_info in self.link:
            start, destination, s, d, i, c, w = each_info
            self.state.add_weighted_edges_from([(int(s), int(d),  self.traffic_matrices[point][channel][int(s)][int(d)])])

        return self.state

    # def mininet(self):
    #     """
    #     Setup a Mininet network using the provided graph (self.DG).
    #     """
        
    #     """
    #     Setup a Mininet network using the provided graph (self.DG).
    #     """
    
    #     # Initialize Mininet
    #     self.net = Mininet(controller=OVSController, cleanup=True)

    #     # Add controller
    #     info('*** Adding controller\n')
    #     self.net.addController('c0')

    #     # Add switches and hosts
    #     switches = [self.net.addSwitch(f's{i}') for i in range(12)]
    #     hosts = [self.net.addHost(f'h{i}', ip=f"10.0.0.{i+1}/24") for i in range(12)]

    #     # Link switches and hosts
    #     for i, switch in enumerate(switches):
    #         self.net.addLink(switch, hosts[i])

    #     # Add inter-switch links based on your graph
    #     # Example: You'll want to replace this with actual topology from your NumPy data
    #     for sr, dt in self.DG.edges:
    #         self.net.addLink(switches[sr], switches[dt])

    #     # Optionally, add some additional cross-links to create a more connected topology
    #     # self.net.addLink(switches[0], switches[5])
    #     # self.net.addLink(switches[3], switches[8])

    #     # for host in hosts:
    #     #     # Ensure IP configuration
    #     #     host.cmd(f'ifconfig {host.name}-eth0 {host.IP()}')
    #     #     host.cmd(f'ip link set {host.name}-eth0 up')

    #     # Start the network
    #     info('*** Starting network\n')
    #     self.net.start()

    #     # Explicitly configure interfaces
    #     for host in self.net.hosts:
    #         # Ensure interface is properly configured
    #         host.configDefault()

    #     # Optional: Configure routing
    #     # info('*** Configuring routes\n')
    #     # for host in hosts:
    #     #     host.cmd('ip route add default via 10.0.0.1')

    #     # # Add inter-switch links (example, customize as needed)
    #     # self.net.addLink(switches[0], switches[1])

    #     # # Start the network
    #     # info('*** Starting network\n')
    #     # self.net.start()

    #     # # Run ping test
    #     # info('*** Testing connectivity\n')
    #     # self.net.pingAll()

    #     # # Start CLI for manual interactions
    #     # info('*** Running CLI\n')
    #     # CLI(self.net)

    #     # # Cleanup on exit
    #     # info('*** Stopping network\n')
    #     # self.net.stop()

    # def generate_traffic(self, point: int, channel: int = 0, duration=10):
    #     """
    #     Generate traffic between hosts in the network based on the traffic matrix.
    #     Measures flow completion times and real-time traffic
        
    #     :param channel: Traffic matrix channel to use
    #     :param duration: Traffic generation duration in seconds
    #     :return: Comprehensive traffic generation results
    #     """
    #     # Reset tracking data structures
    #     self.flow_completion_times.clear()
    #     # self.real_time_traffic_matrix = np.zeros((self.num_node, self.num_node))

    #     # Parallel flow generation and measurement
    #     flow_threads = []
    #     flow_results = {}

    #     # Create the lock once before starting threads
    #     lock = threading.Lock()
    #     # Iterate through traffic matrix
    #     for src, dst in self.DG.edges:
    #         traffic = self.traffic_matrices[channel][point][src, dst]
            
    #         if traffic > 0 and src != dst:
    #             thread = threading.Thread(
    #                 target=self._measure_flow_thread,
    #                 args=(src, dst, traffic, flow_results, lock)
    #             )
    #             thread.start()
    #             flow_threads.append(thread)

    #     # Wait for all flows to complete
    #     for thread in flow_threads:
    #         thread.join(timeout=10)

    #     # Compute network-level statistics
    #     self.network_metrics = self.compute_network_metrics(flow_results)

    #     return {
    #         'flow_completion_times': flow_results,
    #         'network_metrics': self.network_metrics
    #     }
        
    
    # def measure_flow_completion_time(self, src_idx, dst_idx, traffic_volume, duration = 10):
    #     """
    #     Measure completion time for a specific flow
        
    #     :param src_idx: Source host index
    #     :param dst_idx: Destination host index
    #     :param traffic_volume: Traffic volume in Mbps
    #     :param duration: Maximum duration to wait for flow completion
    #     :return: Dictionary with flow completion metrics
    #     """
        
    #     # Skip self-connections and zero traffic
    #     if src_idx == dst_idx or traffic_volume <= 0:
    #         return None

    #     print(f"Processing Source: {src_idx}, Destination {dst_idx}, Traffic volume {traffic_volume}")

    #     src_host = self.net.get(f'h{src_idx}')
    #     dst_host = self.net.get(f'h{dst_idx}')
    #     dst_ip = dst_host.IP()

    #     # Prepare flow tracking
    #     flow_key = (src_idx, dst_idx)
    #     start_time = time.perf_counter()

    #     # Setup iperf server on destination
    #     server_output = dst_host.cmd(f'iperf -s -u -y C &')
    #     # print(f"Server output: {server_output}")
        
    #     # Generate traffic with precise monitoring
    #     iperf_cmd = f'iperf -c {dst_ip} -u -b {traffic_volume}M -t {duration} -y C'
    #     result = src_host.cmd(iperf_cmd)
    #     if result is None:
    #         raise RuntimeError(f"iperf command failed for {src_idx} -> {dst_idx}")

    #     # Compute completion time
    #     end_time = time.perf_counter()
    #     actual_duration = end_time - start_time

    #     # Parse iperf result (CSV format)
    #     try:
    #         # Typical CSV format: timestamp,source_ip,source_port,destination_ip,destination_port,interval,transferred_bytes,bandwidth
    #         parsed_result = result.strip().split('\n')[-1].split(',')

    #         # Correct field indexing for transferred bytes and bandwidth
    #         transferred_bytes = float(parsed_result[7]) / 1e6  # Convert to Mbps # Transferred bytes
    #         actual_bandwidth = float(parsed_result[8]) / 1e6  # Convert to Mbps

    #     except Exception as e:
    #         print(f"Parsing error: {e}")
    #         transferred_bytes = traffic_volume * duration  # Estimated
    #         actual_bandwidth = traffic_volume
        
    #     # Initialize traffic monitor
    #     self.traffic_monitor = TrafficMonitor(src_host, dst_host)
        
    #     # Attempt to measure traffic
    #     measured_traffic = self.traffic_monitor.measure_traffic_alternative()

    #     # Compute metrics
    #     flow_metrics = {
    #         'start_time': start_time,
    #         'end_time': end_time,
    #         'duration': actual_duration,
    #         'traffic_volume': traffic_volume,
    #         'transferred_bytes': transferred_bytes,
    #         'actual_bandwidth': actual_bandwidth,
    #         'measured_traffic': measured_traffic,
    #         'completion_time': actual_duration,
    #         'is_completed': actual_duration <= duration
    #     }

    #     self.flow_completion_times[flow_key] = flow_metrics
    #     return flow_metrics

    # def safe_send_cmd(self, node, cmd):
    #     # Retry sending the command if the shell isn't ready
    #     for _ in range(3):  # Try 3 times
    #         try:
    #             return node.cmd(cmd)
    #         except Exception as e:
    #             print(f"Error executing command on node {node}: {e}")
    #             return None

    # def _measure_flow_thread(self, src, dst, traffic, flow_results, lock):
        
    #     with lock:
    #         dst_host = self.net.get(f'h{dst}')  # Assuming a method to get the node
    #         self.safe_send_cmd(dst_host, f'iperf -s -u -y C &')  # Start iperf server
    #         result = self.measure_flow_completion_time(src, dst, traffic)
    #         # print(f"iperf raw result: {result}")
    #         if result:
    #             flow_results[(src, dst)] = result
            
    # def parse_traffic_output(self, output):
    #     """
    #     Parse nethogs output to extract traffic volume
        
    #     Args:
    #         output (str): Raw nethogs output
        
    #     Returns:
    #         float: Parsed traffic volume in Mbps
    #     """
    #     try:
    #         # Example parsing logic - adjust based on actual nethogs output format
    #         lines = output.strip().split('\n')
    #         for line in lines:
    #             # Look for lines with traffic volume
    #             if 'sent' in line and 'received' in line:
    #                 # Extract numeric values
    #                 parts = line.split()
    #                 sent = float(parts[1])
    #                 received = float(parts[3])
    #                 return (sent + received) / 2  # Average traffic
            
    #         return None
        
    #     except (ValueError, IndexError) as e:
    #         print(f"Error parsing traffic output: {e}")
    #         return None

    # def _measure_flow_thread(self, src, dst, traffic, results_dict, lock):
    #     """
    #     Thread-safe flow measurement method
    #     """
    #     result = self.measure_flow_completion_time(src, dst, traffic)
    #     if result:
    #         with lock:  # Protect access to shared dictionary
    #             results_dict[(src, dst)] = result

    # def compute_network_metrics(self, flow_results):
    #     """
    #     Compute overall network-level metrics from flow results
    #     """
    #     if not flow_results:
    #         return None

    #     completion_times = [result['completion_time'] for result in flow_results.values()]
        
    #     return {
    #         'total_flows': len(flow_results),
    #         'max_completion_time': max(completion_times),
    #         'min_completion_time': min(completion_times),
    #         'mean_completion_time': np.mean(completion_times),
    #         'median_completion_time': np.median(completion_times),
    #         'std_completion_time': np.std(completion_times)
    #     }
    
    # def reward(self):
    #     # total_time = self.calculate_total_traveling_time()
    #     # return -total_time  # Negative reward to minimize total traveling time
    #     """
    #     Compute a reward based on network performance
        
    #     Lower completion times and more completed flows result in higher rewards
    #     """
    #     if not self.network_metrics:
    #         return 0

    #     # Example reward calculation
    #     # You can customize this based on your specific RL requirements
    #     # max_time_penalty = 1 / (self.network_metrics['max_completion_time'] + 1)
    #     mean_time_reward = 1 / (self.network_metrics['mean_completion_time'] + 1)
        
    #     # Combine rewards (adjust weights as needed)
    #     # total_reward = 0.6 * max_time_penalty + 0.4 * mean_time_reward
        
    #     return mean_time_reward
    
    # def calculate_total_traveling_time(self, point: int, channel: int):
    #     # total_traveling_time = 0
    #     # for i, trf in enumerate(self.traffic):
    #     #     remaining_capacity = self.link_capacities[i] - trf
    #     #     if remaining_capacity <= 0:
    #     #         link_delay = 1e6  # Assign high delay for congested links
    #     #     else:
    #     #         link_delay = 1 / (remaining_capacity + 1e-6)
    #     #     total_traveling_time += link_delay  # Weight delay by traffic volume
        
    #     # return total_traveling_time
    #     total_traveling_time = 0
    #     for src in self.num_node:
    #         for dst in self.num_node:
    #             if src == dst:
    #                 continue  # Skip self-loops

    #             # Find the traffic demand between src and dst
    #             traffic_demand = self.real_time_traffic_matrix[src, dst]  # in Mbps
                
    #             if traffic_demand > 0:
    #                 # Get the shortest path using Dijkstra's algorithm
    #                 path = nx.shortest_path(self.DG, source=src, target=dst, weight='weight')

    #                 # Distribute traffic demand across links in the path
    #                 for i in range(len(path) - 1):
    #                     link = (path[i], path[i + 1])

    #                     # Update link utilization
    #                     self.link_utilization[link] = self.link_utilization[link] + traffic_demand

    #                     # Compute remaining capacity
    #                     remaining_capacity = self.link_capacities[link] - self.link_utilization[link]

    #                     # Compute delay for the link
    #                     if remaining_capacity <= 0:
    #                         link_delay = 1e6  # Assign high delay for congested links
    #                     else:
    #                         link_delay = 1 / (remaining_capacity + 1e-6)  # Inverse of remaining capacity

    #                     # Add to total traveling time (weighted by traffic)
    #                     total_traveling_time += link_delay * traffic_demand

    #     return total_traveling_time
        