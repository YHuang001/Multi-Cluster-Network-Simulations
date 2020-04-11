import math
import networkx as nx
import numpy as np
from collections import defaultdict
from itertools import islice
from scipy import spatial


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
    if np.sqrt(p_1 / p_2) < similarity or np.sqrt(p_2 / p_1) < similarity:
        raise ValueError("Invalid two-layer ER network inputs!")
    graph_1 = nx.fast_gnp_random_graph(num_of_nodes, p_1 - np.sqrt(p_1 * p_2) * similarity)
    graph_2 = nx.fast_gnp_random_graph(num_of_nodes, p_2 - np.sqrt(p_1 * p_2) * similarity)
    graph_mutual = nx.fast_gnp_random_graph(num_of_nodes, np.sqrt(p_1 * p_2) * similarity)
    mutual_node_pairs = set(graph_mutual.edges())
    node_pairs_1 = list(mutual_node_pairs.union(set(graph_1.edges())))
    node_pairs_2 = list(mutual_node_pairs.union(set(graph_2.edges())))

    return node_pairs_1, node_pairs_2

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
    node_pairs_1, node_pairs_2 = list(kd_tree.query(radius_1)), list(kd_tree.query(radius_2))

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

def ConstructInterClusterConfigs(ic_topology, num_of_clusters, num_of_ic_links, ic_similarity, random_cluster_prob=None):
    """Generates the inter cluster link configurations.
    Args:
        ic_topology: The inter-cluster level topology.
        num_of_clusters: Total number of clusters.
        num_of_ic_links: Number of inter-cluster links between any two cluster pair.
        ic_similarity: The portion of shared inter-cluster links in a cluster pair.

    Returns:
        A tuple that includes:
            cluster_pairs: List of cluster pairs.
            num_of_ic_links: Number of inter-cluster links in this cluster pair.
            ic_similarity: The portion of shared inter-cluster links in this cluster pair.
    """
    cluster_pairs = []
    if ic_topology == "line":
        num = list(range(num_of_clusters))
        for cluster_pair in tuple(zip(num[:-1], num[1:])):
            cluster_pairs.append(cluster_pair)
    elif ic_topology == "star":
        num = list(range(1, num_of_clusters))
        for cluster_pair in tuple(zip([0] * (num_of_clusters - 1), num)):
            cluster_pairs.append(cluster_pair)
    elif ic_topology == "random":
        graph = nx.fast_gnp_random_graph(num_of_clusters, random_cluster_prob)
        for cluster_pair in graph.edges:
            cluster_pairs.append(cluster_pair)
    else:
        raise TypeError(f"{ic_topology} not supported!")

    return cluster_pairs, num_of_ic_links, ic_similarity

class MultiClusterNetworks(object):
    """A multi-cluster network object."""
    def __init__(self, num_of_clusters, cluster_sizes, cluster_types, cluster_paras, infection_rate, protection_rate, inter_cluster_link_paras):
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
        cluster_size, cluster_type, cluster_para = None, None, None
        if len(self._cluster_sizes) == 1:
            cluster_size = self._cluster_sizes
        if len(self._cluster_types) == 1:
            cluster_type = self._cluster_types
        if len(self._cluster_paras) == 1:
            cluster_para = self._cluster_paras
        for cluster in range(self._num_of_clusters):
            num_of_nodes = cluster_size if cluster_size else self._cluster_sizes[cluster]
            c_paras = cluster_para if cluster_para else self._cluster_paras[cluster]
            c_type = cluster_type if cluster_type else self._cluster_types[cluster]
            if c_type == 'ER':
                if len(c_paras) < 3:
                    raise ValueError(f"Expected 3 parameters, but got {len(c_paras)}.")
                node_pairs_1, node_pairs_2 = ConstructTwoER(
                    num_of_nodes, c_paras[0], c_paras[1], c_paras[2])
                node_pairs_1 = ShiftEdge(node_pairs_1, self._cluster_nodes[cluster][0])
                node_pairs_2 = ShiftEdge(node_pairs_2, self._cluster_nodes[cluster][0])
            elif c_type == 'RGG':
                if len(c_paras) < 2:
                    raise ValueError(f"Expected 2 parameters, but got {len(c_paras)}.")
                node_pairs_1, node_pairs_2 = ConstructTwoRGG(
                    num_of_nodes, c_paras[0], c_paras[1])
                node_pairs_1 = ShiftEdge(node_pairs_1, self._cluster_nodes[cluster][0])
                node_pairs_2 = ShiftEdge(node_pairs_2, self._cluster_nodes[cluster][1])
            else:
                raise TypeError(f"Graph type {c_type} does not exist!")
            self._graph_1.add_edges_from(node_pairs_1)
            self._graph_2.add_edges_from(node_pairs_2)
    
    def ConstructInterClusterLinks(self):
        ic_links_1, ic_links_2 = set(), set()
        cluster_pairs, num_of_ic_links, ic_similarity = self._ic_link_paras
        total_num_ic_links = num_of_ic_links + (1 - ic_similarity) * num_of_ic_links
        for cluster_1, cluster_2 in cluster_pairs:
            total_ic_links = set()
            for _ in range(total_num_ic_links):
                node_1 = np.random.choice(self._cluster_nodes[cluster_1])
                node_2 = np.random.choice(self._cluster_nodes[cluster_2])
                while (node_1, node_2) in total_ic_links:
                    node_2 = np.random.choice(self._cluster_nodes[cluster_2])
                total_ic_links.add((node_1, node_2))
            total_ic_links = list(total_ic_links)
            ic_links_1 = ic_links_1.union(set(total_ic_links[:num_of_ic_links]))
            ic_links_2 = ic_links_2.union(set(total_ic_links[(1 - ic_similarity) * num_of_ic_links:]))
        self._graph_1.add_edges_from(ic_links_1)
        self._graph_2.add_edges_from(ic_links_2)
    
    def SetLinkTimes(self):
        self._pair_event_times_1 = SetPairTimes(self._graph_1, self._inf_rate)
        self._pair_event_times_2 = SetPairTimes(self._graph_2, self._pro_rate)
    
    def GetNetworkInfo(self):
        neighbor_list_1 = [set(self._graph_1.neighbors(i)) for i in range(self._total_nodes)]
        neighbor_list_2 = [set(self._graph_2.neighbors(i)) for i in range(self._total_nodes)]
        return neighbor_list_1, neighbor_list_2, self._pair_event_times_1, self._pair_event_times_2

