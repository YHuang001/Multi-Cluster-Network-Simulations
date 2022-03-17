import math
import networkx as nx
import numpy as np
import heapq
from collections import defaultdict
from itertools import islice
from scipy import spatial
from scipy.optimize import fsolve


def ComputeSingleTwoErResults(k_i, k_p, similarity, beta, gamma):
    i_v, i_m=-np.log(1 - beta), -np.log(1 - gamma)

    coeff_a_1=k_i * beta + k_p * gamma - np.sqrt(k_i * k_p) * similarity * beta * gamma 
    coeff_a_2=(k_i - np.sqrt(k_i * k_p) * similarity) * beta + np.sqrt(k_i * k_p) * similarity * (i_v/(i_v + i_m)) * (beta + gamma - beta * gamma)

    func_1 = lambda r_inf: r_inf - (coeff_a_2 / coeff_a_1) * (1 - np.exp(-coeff_a_1 * r_inf))      #Theoretical function for R(\infty) of case 1 with a layer similarity
    r_inf_1 = fsolve(func_1, 1.0)[0] 
    p_inf_1 = (1 - coeff_a_2 / coeff_a_1) * (1 - np.exp(-coeff_a_1 * r_inf_1))

    coeff_b_1 = k_i * beta + k_p * gamma - np.sqrt(k_i * k_p) * similarity * beta * gamma
    coeff_b_2 = k_i * i_v
    coeff_b_3 = k_i * i_v + k_p * i_m
    func_2 = lambda r_inf: r_inf - (coeff_b_2 / coeff_b_3) * (1 - np.exp(-coeff_b_1 * r_inf))
    r_inf_2 = fsolve(func_2, 1.0)[0]
    p_inf_2 = (1 - coeff_b_2 / coeff_b_3) * (1 - np.exp(-coeff_b_1 * r_inf_2))

    if r_inf_1 > r_inf_2:
        return r_inf_1 * (r_inf_1 + p_inf_1), p_inf_1 * (r_inf_1 + p_inf_1)
    return r_inf_2 * (r_inf_2 + p_inf_2), p_inf_2 * (r_inf_2 + p_inf_2)

def PoiSample(rate):
    """Generates a Poisson distributed random value.
    
    Args:
        rate: The rate for the Poisson distribution.
    
    Returns:
        A Poisson distributed random variable.
    """
    return -np.log(np.random.rand()) / rate

def SetPairTimes(node_pairs, rate):
    """Set random times for events on the corresponding node pairs based on the Poisson distribution.

    Args:
        node_pairs: The given node pairs.
        rate: The rate for the Poisson distribution.
    
    Returns:
        A dictionary maps the node pair to its event time.
    """
    t_dict = defaultdict(float)
    for node_pair in node_pairs:
        t_dict[node_pair] = PoiSample(rate)
    
    return t_dict

def ConstructER(num_of_nodes, avg_degree):
    """Constructs a ER network.

    Args:
        num_of_nodes: number of nodes in the network.
        avg_degree: average node degree of the network.
    
    Returns:
        node pairs of the ER network.
    """
    graph = nx.fast_gnp_random_graph(num_of_nodes, avg_degree / (num_of_nodes - 1))
    return list(graph.edges())

def ConstructTwoER(num_of_nodes, avg_degree_1, avg_degree_2, similarity):
    """Constructs a two-layer ER network.

    Args:
        num_of_nodes: Number of nodes in the network.
        avg_degree_1: Average number of degree in layer 1.
        avg_degree_2: Average number of degree in layer 2.
        similarity: The edge similarity between layer 1 and layer 2.

    Returns:
        Node pairs of two ER layers.
    
    Raises:
        A ValueError if the inputs are invalid.
    """
    p_1, p_2 = avg_degree_1 / num_of_nodes, avg_degree_2 / num_of_nodes
    if (p_1 > 0 and p_2 > 0) and (np.sqrt(p_1 / p_2) < similarity or np.sqrt(p_2 / p_1) < similarity):
        raise ValueError('Invalid two-layer ER network inputs!')
    graph_1 = nx.fast_gnp_random_graph(num_of_nodes, p_1 - np.sqrt(p_1 * p_2) * similarity)
    graph_2 = nx.fast_gnp_random_graph(num_of_nodes, p_2 - np.sqrt(p_1 * p_2) * similarity)
    graph_mutual = nx.fast_gnp_random_graph(num_of_nodes, np.sqrt(p_1 * p_2) * similarity)
    mutual_node_pairs = set(graph_mutual.edges())
    node_pairs_1 = list(mutual_node_pairs.union(set(graph_1.edges())))
    node_pairs_2 = list(mutual_node_pairs.union(set(graph_2.edges())))

    return node_pairs_1, node_pairs_2

def ConstructRGG(num_of_nodes, avg_degree):
    """Constructs a RGG network.

    Args:
        num_of_nodes: Number of nodes in the network.
        avg_degree: Average number of degree.
    
    Returns:
        Node pairs of a RGG.
    """ 
    node_positions = np.random.rand(num_of_nodes, 2)
    kd_tree = spatial.KDTree(node_positions)
    radius = np.sqrt(avg_degree / num_of_nodes / math.pi)
    node_pairs = list(kd_tree.query_pairs(radius))

    return node_pairs

def ConstructTwoRGG(num_of_nodes, avg_degree_1, avg_degree_2):
    """Constructs a two-layer RGG network.

    Args:
        num_of_nodes: Number of nodes in the network.
        avg_degree_1: Average number of degree in layer 1.
        avg_degree_2: Average number of degree in layer 2.
    
    Returns:
        Node pairs of two RGG layers.
    """
    node_positions = np.random.rand(num_of_nodes, 2)
    kd_tree = spatial.KDTree(node_positions)
    radius_1, radius_2 = np.sqrt(avg_degree_2 / num_of_nodes / math.pi), np.sqrt(avg_degree_2 / num_of_nodes / math.pi)
    node_pairs_1, node_pairs_2 = list(kd_tree.query_pairs(radius_1)), list(kd_tree.query_pairs(radius_2))

    return node_pairs_1, node_pairs_2

def ShiftEdge(node_pairs, shift_index):
    """Shifts the edges' labels based on the shift index.

    Args:
        node_pairs: Given node pairs (edges).
        shift_index: Node indexes in all clusters.
    
    Returns:
        Node paris with labels shifted to the shift_index.
    """
    return [(e1 + shift_index, e2 + shift_index) for e1, e2 in node_pairs]

def ConstructInterClusterConfigs(ic_topology, num_of_clusters, ic_config, random_cluster_prob=None):
    """Generates the inter cluster link configurations.
    Args:
        ic_topology: The inter-cluster level topology.
        num_of_clusters: Total number of clusters.
        ic_config: The configuration for inter-cluster links including the number of inter-cluster links
            in layer 1, the number of links in layer 2, and the number of shared links in both layers.
        random_cluster_prob: The probability of connecting two clusters in the random cluster case.

    Returns:
        A tuple that includes:
            cluster_pairs: List of cluster pairs.
            num_of_ic_links: Number of inter-cluster links in this cluster pair.
            ic_similarity: The portion of shared inter-cluster links in this cluster pair.
    """
    cluster_pairs = []
    if ic_topology == 'line':
        num = list(range(num_of_clusters))
        for cluster_pair in tuple(zip(num[:-1], num[1:])):
            cluster_pairs.append(cluster_pair)
    elif ic_topology == 'star':
        num = list(range(1, num_of_clusters))
        for cluster_pair in tuple(zip([0] * (num_of_clusters - 1), num)):
            cluster_pairs.append(cluster_pair)
    elif ic_topology == 'random':
        graph = nx.fast_gnp_random_graph(num_of_clusters, random_cluster_prob)
        for cluster_pair in graph.edges:
            cluster_pairs.append(cluster_pair)
    else:
        raise TypeError(f'{ic_topology} not supported!')

    return cluster_pairs, ic_config

class MultiClusterNetworks(object):
    """A multi-cluster network object with two layers."""
    def __init__(self, num_of_clusters, cluster_sizes, cluster_types, cluster_paras,
                 infection_rate, protection_rate, inter_cluster_link_paras):
        self._num_of_clusters = num_of_clusters
        self._cluster_sizes = cluster_sizes
        self._cluster_types = cluster_types
        self._cluster_paras = cluster_paras
        self._inf_rate = infection_rate
        self._pro_rate = protection_rate
        self._ic_link_paras = inter_cluster_link_paras
        self._total_nodes = np.arange(sum(self._cluster_sizes))
        nodes_to_slice = iter(self._total_nodes)
        self._cluster_nodes = [np.array(list(islice(nodes_to_slice, size))) for size in self._cluster_sizes]
        self._graph_1 = nx.Graph()
        self._graph_1.add_nodes_from(self._total_nodes)
        self._graph_2 = nx.Graph()
        self._graph_2.add_nodes_from(self._total_nodes)
        self._pair_event_times_1 = None
        self._pair_event_times_2 = None

    def ConstructClusters(self):
        for node in self._cluster_nodes:
            self._graph_1.add_nodes_from(node)
            self._graph_2.add_nodes_from(node)
        # Construct each cluster based on the input parameters.
        for cluster in range(self._num_of_clusters):
            num_of_nodes = self._cluster_sizes[cluster]
            c_paras = self._cluster_paras[cluster]
            c_type = self._cluster_types[cluster]
            if c_type == 'ER':
                if len(c_paras) < 3:
                    raise ValueError(f'Expected 3 parameters, but got {len(c_paras)}.')
                node_pairs_1, node_pairs_2 = ConstructTwoER(
                    num_of_nodes, c_paras[0], c_paras[1], c_paras[2])
                node_pairs_1 = ShiftEdge(node_pairs_1, self._cluster_nodes[cluster][0])
                node_pairs_2 = ShiftEdge(node_pairs_2, self._cluster_nodes[cluster][0])
            elif c_type == 'RGG':
                if len(c_paras) < 2:
                    raise ValueError(f'Expected 2 parameters, but got {len(c_paras)}.')
                node_pairs_1, node_pairs_2 = ConstructTwoRGG(
                    num_of_nodes, c_paras[0], c_paras[1])
                node_pairs_1 = ShiftEdge(node_pairs_1, self._cluster_nodes[cluster][0])
                node_pairs_2 = ShiftEdge(node_pairs_2, self._cluster_nodes[cluster][0])
            else:
                raise TypeError('Graph type' + c_type + 'not implemented yet!')
            self._graph_1.add_edges_from(node_pairs_1)
            self._graph_2.add_edges_from(node_pairs_2)
    
    def ConstructInterClusterLinks(self):
        # We assume the same inter-cluster topology in both layers. The difference is the portion
        # of different inter-cluster links between the same cluster pair.
        ic_links_1, ic_links_2 = set(), set()
        cluster_pairs, ic_config = self._ic_link_paras
        num_of_ic_links_1, num_of_ic_links_2, shared_links = ic_config['num_of_ic_links_1'], ic_config['num_of_ic_links_2'], ic_config['shared_links']
        if shared_links > num_of_ic_links_1 or shared_links > num_of_ic_links_2:
            raise ValueError(
                f'The number of shared links {shared_links} is greater than num_of_ic_links_1 {num_of_ic_links_1} or greater than num_of_ic_links_2 {num_of_ic_links_2}!')
        total_num_ic_links = int(num_of_ic_links_1 + num_of_ic_links_2 - shared_links)
        for cluster_1, cluster_2 in cluster_pairs:
            total_ic_links = set()
            for _ in range(total_num_ic_links):
                node_1 = np.random.choice(self._cluster_nodes[cluster_1])
                node_2 = np.random.choice(self._cluster_nodes[cluster_2])
                while (node_1, node_2) in total_ic_links:
                    node_2 = np.random.choice(self._cluster_nodes[cluster_2])
                total_ic_links.add((node_1, node_2))
            total_ic_links = list(total_ic_links)
            ic_links_1 = ic_links_1.union(set(total_ic_links[:num_of_ic_links_1]))
            ic_links_2 = ic_links_2.union(set(total_ic_links[total_num_ic_links - num_of_ic_links_2:]))
        self._graph_1.add_edges_from(ic_links_1)
        self._graph_2.add_edges_from(ic_links_2)
    
    def SetLinkTimes(self):
        # Set up the event times happening on all links.
        self._pair_event_times_1 = SetPairTimes(list(self._graph_1.edges()), self._inf_rate)
        self._pair_event_times_2 = SetPairTimes(list(self._graph_2.edges()), self._pro_rate)
    
    def GetNetworkInfo(self):
        neighbor_list_1 = [set(self._graph_1.neighbors(i)) for i in self._total_nodes]
        neighbor_list_2 = [set(self._graph_2.neighbors(i)) for i in self._total_nodes]
        return neighbor_list_1, neighbor_list_2, self._pair_event_times_1, self._pair_event_times_2


class MultiClusterNetworksSingleLayer(object):
    """A multi-cluster network object with a single layer."""
    def __init__(self, num_of_clusters, cluster_sizes, cluster_types, cluster_paras,
                 infection_rate, inter_cluster_link_paras):
        self._num_of_clusters = num_of_clusters
        self._cluster_sizes = cluster_sizes
        self._cluster_types = cluster_types
        self._cluster_paras = cluster_paras
        self._inf_rate = infection_rate
        self._ic_link_paras = inter_cluster_link_paras
        self._total_nodes = np.arange(sum(self._cluster_sizes))
        nodes_to_slice = iter(self._total_nodes)
        self._cluster_nodes = [np.array(list(islice(nodes_to_slice, size))) for size in self._cluster_sizes]
        self._graph = nx.Graph()
        self._graph.add_nodes_from(self._total_nodes)
        self._pair_event_times = None

    def ConstructClusters(self):
        for node in self._cluster_nodes:
            self._graph.add_nodes_from(node)
        # Construct each cluster based on the input parameters.
        for cluster in range(self._num_of_clusters):
            num_of_nodes = self._cluster_sizes[cluster]
            c_paras = self._cluster_paras[cluster]
            c_type = self._cluster_types[cluster]
            if c_type == 'ER':
                node_pairs = ConstructER(num_of_nodes, c_paras)
                node_pairs = ShiftEdge(node_pairs, self._cluster_nodes[cluster][0])
            elif c_type == 'RGG':
                node_pairs = ConstructRGG(num_of_nodes, c_paras)
                node_pairs = ShiftEdge(node_pairs, self._cluster_nodes[cluster][0])
            else:
                raise TypeError(f'Graph type {c_type} not implemented yet!')
            self._graph.add_edges_from(node_pairs)
    
    def ConstructInterClusterLinks(self):
        ic_links = set()
        cluster_pairs, ic_config = self._ic_link_paras
        num_of_ic_links = ic_config['num_of_ic_links']
        for cluster_1, cluster_2 in cluster_pairs:
            total_ic_links = set()
            for _ in range(num_of_ic_links):
                node_1 = np.random.choice(self._cluster_nodes[cluster_1])
                node_2 = np.random.choice(self._cluster_nodes[cluster_2])
                while (node_1, node_2) in total_ic_links:
                    node_2 = np.random.choice(self._cluster_nodes[cluster_2])
                total_ic_links.add((node_1, node_2))
            ic_links = ic_links.union(set(total_ic_links))
        self._graph.add_edges_from(ic_links)
    
    def SetLinkTimes(self):
        # Set up the event times happening on all links.
        self._pair_event_times = SetPairTimes(list(self._graph.edges()), self._inf_rate)
    
    def GetNetworkInfo(self):
        neighbor_list = [set(self._graph.neighbors(i)) for i in self._total_nodes]
        return neighbor_list, self._pair_event_times

def PushInfectionEvent(event_heap, neighbor_node, infected_node, time):
    """Pushes the infection event into the event heap.

    Args:
        event_heap: A heap recording all events happing during the whole process.
        neighbor_node: The chosen neighbor node by the source to interact.
        infected_node: The infected node who is about to infect another node.
        time: The time for the infection event. 
    """
    heapq.heappush(event_heap, (time, (neighbor_node, infected_node, 'I')))

def PushProtectionEvent(event_heap, neighbor_node, infected_node, time):
    """Pushes the protection event into the event heap.

    Args:
        event_heap: A heap recording all events during the whole process.
        neighbor_node: The chosen neighbor node by the source to interact.
        infected_node: The infected node who is about to protect another node.
        time: The time for the protection event.
    """
    heapq.heappush(event_heap, (time, (neighbor_node, infected_node, 'P')))

def PushRecoveryEvent(event_heap, infected_node, time):
    """Pushes the recovery event into the event heap.

    Args:
        event_heap: A heap recording all events during the whole process.
        infected_node: The infected node who is about to recover.
        time: The time for the recovery event.
    """
    heapq.heappush(event_heap, (time, (infected_node, infected_node, 'R')))   

def DetermineNodeSusceptible(node, status_record):
    """Determines if a node is susceptible.
    
    Args:
        node: The node whose status to be determined.
        status_record: The status record.
    
    Returns:
        A Boolean value determines whether a node is susceptible or not.
    """
    if node not in status_record['R'] and node not in status_record['I'] and node not in status_record['P']:
        return True
    return False

def UpdateEventHeapSingle(event_heap, event_times, neighbor_list, source_node, status_record, current_time):
    """Update events in the event heap for a single spreading process.

    Args:
        event_heap: A heap recording all events during the whole process.
        event_times: Event times for all links.
        neighbor_list: Neighbor list.
        source_node: The new events triggered by this source node.
        status_record: The status record.
        current_time: The current time.
    """
    for neighbor in neighbor_list[source_node]:
        # If the neighbor node is susceptible, it can be infected in the future,
        # store the corresponding time for the infection event in the event heap.
        if DetermineNodeSusceptible(neighbor, status_record):
            if (neighbor, source_node) in event_times:
                PushInfectionEvent(event_heap, neighbor, source_node,
                                   current_time + event_times[(neighbor, source_node)])
            elif (source_node, neighbor) in event_times:
                PushInfectionEvent(event_heap, neighbor, source_node,
                                   current_time + event_times[(source_node, neighbor)])

def UpdateEventHeap(event_heap, event_times_1, event_times_2,
                    neighbor_list_1, neighbor_list_2, source_node, status_record, current_time):
    """Update events in the event heap.

    Args:
        event_heap: A heap recording all events during the whole process.
        event_times_1: Event times for all links in layer 1.
        event_times_2: Event times for all links in layer 2.
        neighbor_list_1: Neighbor list in layer 1.
        neighbor_list_2: Neighbor list in layer 2.
        source_node: The new events triggered by this source node.
        status_record: The status record.
        current_time: The current time.
    """
    for neighbor in neighbor_list_1[source_node]:
        # If the neighbor node is susceptible, it can be infected in the future,
        # store the corresponding time for the infection event in the event heap.
        if DetermineNodeSusceptible(neighbor, status_record):
            if (neighbor, source_node) in event_times_1:
                PushInfectionEvent(event_heap, neighbor, source_node,
                                   current_time + event_times_1[(neighbor, source_node)])
            elif (source_node, neighbor) in event_times_1:
                PushInfectionEvent(event_heap, neighbor, source_node,
                                   current_time + event_times_1[(source_node, neighbor)])
    for neighbor in neighbor_list_2[source_node]:
        # If the neighbor node is susceptible, it can be protected in the future,
        # store the corresponding time for the infection event in the event heap.
        if DetermineNodeSusceptible(neighbor, status_record):
            if (neighbor, source_node) in event_times_2:
                PushProtectionEvent(event_heap, neighbor, source_node,
                                    current_time + event_times_2[(neighbor, source_node)])
            elif (source_node, neighbor) in event_times_2:
                PushProtectionEvent(event_heap, neighbor, source_node,
                                    current_time + event_times_2[(source_node, neighbor)])

def SimulationSingle(num_of_nodes, event_times, neighbor_list, recovery_time, monte_runs=100):
    """Simulates the SIR process.

    Args:
        num_of_nodes: Number of nodes in the network.
        event_times: Event times for all links.
        neighbor_list: Neighbor list.
        recovery_time: Time for an infected node to recover
        monte_runs: The number of Monte Carlo runs.
    
    Returns:
        Portion of recovered nodes in the final state.
    """
    portion_of_recovered = 0
    for _ in range(monte_runs):
        status_record = defaultdict(set)
        # Source node is randonly chosen among all nodes.
        source_node = int(num_of_nodes * np.random.rand())
        status_record['I'].add(source_node)
        event_heap = []
        heapq.heapify(event_heap)
        # Update the future events in the heap with respect to the newly infected node.
        UpdateEventHeapSingle(event_heap, event_times, neighbor_list,source_node, status_record, 0)
        # Push the recovery event for the newly infectede node.
        PushRecoveryEvent(event_heap, source_node, recovery_time)
        while len(event_heap):
            current_time, new_event = heapq.heappop(event_heap)
            target_node, infected_node, event_type = new_event
            if event_type == 'R':
                # The infected node recovers.
                status_record['R'].add(infected_node)
            elif event_type == 'I':
                # Check the status of the two nodes of the incoming infection event.
                if (infected_node not in status_record['R'] and
                    DetermineNodeSusceptible(target_node, status_record)):
                    status_record['I'].add(target_node)
                    # Update the future events in the heap with respect to the newly infected node.
                    UpdateEventHeapSingle(event_heap, event_times, neighbor_list, target_node, status_record, current_time)
                    # Push the recovery event for the newly infectede node.
                    PushRecoveryEvent(event_heap, target_node, current_time + recovery_time)
        portion_of_recovered += len(status_record['R']) / num_of_nodes
    
    portion_of_recovered /= monte_runs

    return portion_of_recovered

def Simulation(num_of_nodes, event_times_1, event_times_2,
               neighbor_list_1, neighbor_list_2, recovery_time, monte_runs=100):
    """Simulates the SIPR process.

    Args:
        num_of_nodes: Number of nodes in the network.
        event_times_1: Event times for all links in layer 1.
        event_times_2: Event times for all links in layer 2.
        neighbor_list_1: Neighbor list in layer 1.
        neighbor_list_2: Neighbor list in layer 2.
        recovery_time: Time for an infected node to recover
        monte_runs: The number of Monte Carlo runs.
    
    Returns:
        Portion of recovered and protected nodes in the final state.
    """
    portion_of_recovered, portion_of_protected = 0, 0
    for _ in range(monte_runs):
        status_record = defaultdict(set)
        # Source node is randonly chosen among all nodes.
        source_node = int(num_of_nodes * np.random.rand())
        status_record['I'].add(source_node)
        event_heap = []
        heapq.heapify(event_heap)
        # Update the future events in the heap with respect to the newly infected node.
        UpdateEventHeap(
            event_heap, event_times_1, event_times_2, neighbor_list_1, neighbor_list_2,
            source_node, status_record, 0)
        # Push the recovery event for the newly infectede node.
        PushRecoveryEvent(event_heap, source_node, recovery_time)
        while len(event_heap):
            current_time, new_event = heapq.heappop(event_heap)
            target_node, infected_node, event_type = new_event
            if event_type == 'R':
                # The infected node recovers.
                status_record['R'].add(infected_node)
            elif event_type == 'P':
                # Check the status of the two nodes of the incoming protection event.
                if (infected_node not in status_record['R'] and
                    DetermineNodeSusceptible(target_node, status_record)):
                        status_record['P'].add(target_node)
            elif event_type == 'I':
                # Check the status of the two nodes of the incoming infection event.
                if (infected_node not in status_record['R'] and
                    DetermineNodeSusceptible(target_node, status_record)):
                    status_record['I'].add(target_node)
                    # Update the future events in the heap with respect to the newly infected node.
                    UpdateEventHeap(
                        event_heap, event_times_1, event_times_2, neighbor_list_1, neighbor_list_2,
                        target_node, status_record, current_time)
                    # Push the recovery event for the newly infectede node.
                    PushRecoveryEvent(event_heap, target_node, current_time + recovery_time)
        portion_of_recovered += len(status_record['R']) / num_of_nodes
        portion_of_protected += len(status_record['P']) / num_of_nodes
    
    portion_of_recovered /= monte_runs
    portion_of_protected /= monte_runs

    return portion_of_recovered, portion_of_protected

def GetSingleTheoreticalResultsWithInputs(r_inf_t, beta, num_of_ic_links, average_cluster_degree, num_of_clusters, cluster_type):
    beta_t = 1 - (1 - beta * r_inf_t) ** num_of_ic_links
    if cluster_type == 'random':
        func_s = lambda s: s - (1 - np.exp(-beta_t * average_cluster_degree * s))
        r_inf = fsolve(func_s, 2.0)[0]
    elif cluster_type == 'star':
        r_inf = ((num_of_clusters - 1) * beta_t * (1 - (1 - beta_t)**(num_of_clusters - 2)) + 1 - (1 - beta_t)**(num_of_clusters - 1)) / num_of_clusters
    else:
        ValueError(cluster_type + 'not implemented!')
    return r_inf**2 * r_inf_t

def GetSingleTheoreticalResults(beta, average_degree, num_of_ic_links, average_cluster_degree, num_of_clusters, cluster_type):
    func = lambda r_t_inf: r_t_inf - (1 - np.exp(-beta * r_t_inf * average_degree))
    r_inf_t = fsolve(func, 2.0)[0]
    r_inf_t = r_inf_t**2
    beta_t = 1 - (1 - beta * r_inf_t) ** num_of_ic_links
    r_inf = 0
    if cluster_type == 'random':
        func_s = lambda s: s - (1 - np.exp(-beta_t * average_cluster_degree * s))
        r_inf = fsolve(func_s, 2.0)[0]
    elif cluster_type == 'star':
        r_inf = ((num_of_clusters - 1) * beta_t * (1 - (1 - beta_t)**(num_of_clusters - 2)) + 1 - (1 - beta_t)**(num_of_clusters - 1)) / num_of_clusters
    else:
        ValueError(cluster_type + 'not implemented!')
    return r_inf**2 * r_inf_t

def GetMultiplexTheoreticalResults(beta, gamma, average_degree_i, average_degree_p, num_of_ic_links_i,
                                   num_of_ic_links_i_p, similarity, average_cluster_degree, num_of_clusters,
                                   cluster_type):
    r_inf_t, p_inf_t = ComputeSingleTwoErResults(average_degree_i, average_degree_p, similarity, beta, gamma)
    beta_t = 1 - (1 - r_inf_t * beta)**num_of_ic_links_i * (1 - r_inf_t * (beta + gamma - beta * gamma))**num_of_ic_links_i_p
    r_inf = 0
    if cluster_type == 'random':
        func_s = lambda s: s - (1 - np.exp(-beta_t * average_cluster_degree * s))
        r_inf = fsolve(func_s, 2.0)[0]
    elif cluster_type == 'star':
        r_inf = ((num_of_clusters - 1) * beta_t * (1 - (1 - beta_t)**(num_of_clusters - 2)) + 1 - (1 - beta_t)**(num_of_clusters - 1)) / num_of_clusters
    else:
        ValueError(cluster_type + 'not implemented!')
    return r_inf**2 * r_inf_t, r_inf**2 * p_inf_t    

def GetMultiplexGiantTheoreticalResults(beta, gamma, average_degree_i, average_degree_p, num_of_ic_links_i,
                                   num_of_ic_links_i_p, similarity, average_cluster_degree, num_of_clusters,
                                   cluster_sizes):
    r_inf_t, p_inf_t = ComputeSingleTwoErResults(average_degree_i, average_degree_p, similarity, beta, gamma)
    beta_t = 1 - (1 - r_inf_t * beta)**num_of_ic_links_i * (1 - r_inf_t * (beta + gamma - beta * gamma))**num_of_ic_links_i_p
    func_s = lambda s: s - (1 - np.exp(-beta_t * average_cluster_degree * s))
    r_inf = fsolve(func_s, 2.0)[0]

    final_r_inf = 0
    final_p_inf = 0
    total_cluster_size = np.sum(cluster_sizes)
    for cluster_size in cluster_sizes:
        final_r_inf += cluster_size / total_cluster_size * (cluster_size * r_inf_t + (total_cluster_size - cluster_size) * (r_inf**2 - 1 / num_of_clusters) * r_inf_t) / total_cluster_size
        final_p_inf += cluster_size / total_cluster_size * (cluster_size * p_inf_t + (total_cluster_size - cluster_size) * (r_inf**2 - 1 / num_of_clusters) * p_inf_t) / total_cluster_size
    return final_r_inf, final_p_inf