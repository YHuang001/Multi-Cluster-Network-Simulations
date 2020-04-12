import numpy as np
import heapq
import time
import network_simulation_lib as ns_lib
from collections import defaultdict


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
        portion_of_recovered += len(status_record['P']) / num_of_nodes
        portion_of_protected += len(status_record['R']) / num_of_nodes
    
    portion_of_recovered /= monte_runs
    portion_of_protected /= monte_runs

    return portion_of_recovered, portion_of_protected


if __name__ == '__main__':
    num_of_clusters = 7
    num_of_ic_links = 3
    ic_similarity = 1.0
    cluster_sizes = [100] * num_of_clusters
    cluster_types = ['ER'] * num_of_clusters
    cluster_paras = [[10, 10, 1]] * num_of_clusters
    # Set up parameters for the inter-cluster links.
    inter_cluster_link_paras = ns_lib.ConstructInterClusterConfigs('line', num_of_clusters, num_of_ic_links, ic_similarity)
    beta, gamma, recovery_time = 0.5, 0.3, 1.0
    infection_rate, protection_rate = -np.log(1 - beta) / recovery_time, -np.log(1 - gamma) / recovery_time
    # Create the multi-cluster network object
    multi_cluster_network = ns_lib.MultiClusterNetworks(
        num_of_clusters, cluster_sizes, cluster_types,
        cluster_paras, infection_rate, protection_rate, inter_cluster_link_paras)
    # Construct the multi-cluster network based on the input parameters.
    multi_cluster_network.ConstructClusters()
    multi_cluster_network.ConstructInterClusterLinks()
    multi_cluster_network.SetLinkTimes()
    neighbor_list_1, neighbor_list_2, event_times_1, event_times_2 = multi_cluster_network.GetNetworkInfo()
    portion_of_recovered, portion_of_protected = Simulation(
        sum(cluster_sizes), event_times_1, event_times_2, neighbor_list_1, neighbor_list_2, recovery_time, monte_runs=10)
    print(portion_of_recovered, portion_of_protected)


