
import network_simulation_lib as ns_lib


if __name__ == '__main__':
    num_of_clusters = 100
    single_cluster_size = 200
    num_of_ic_links_i = 2
    num_of_ic_links_p = 2
    num_of_shared_ic_links = 0
    intra_cluster_avg_degree_i = 5
    intra_cluster_avg_degree_p = 5
    intra_cluster_similarity = 0
    inter_cluster_avg_degree = 4
    cluster_type = 'random'
    ic_config = {
        'num_of_ic_links_1': num_of_ic_links_i,
        'num_of_ic_links_2': num_of_ic_links_p,
        'shared_links': num_of_shared_ic_links
    }
    cluster_sizes = [single_cluster_size] * num_of_clusters
    cluster_types = ['ER'] * num_of_clusters
    cluster_paras = [[intra_cluster_avg_degree_i, intra_cluster_avg_degree_p, intra_cluster_similarity]] * num_of_clusters
    # Set up parameters for the inter-cluster links.
    inter_cluster_link_paras = ns_lib.ConstructInterClusterConfigs(cluster_type, num_of_clusters, ic_config, inter_cluster_avg_degree / num_of_clusters)
    betas = list(np.arange(0.1, 0.51, 0.01)) + list(np.arange(0.6, 1.0, 0.1))
    gamma, recovery_time = 0.7, 1.0
    protection_rate = -np.log(1 - gamma) / recovery_time
    recovered_results, protected_results = [], []
    cluster_level_monte_runs = 10
    graph_level_monte_runs = 100
    saved_path = 'multi_er_random_results_1.csv'

    for _, beta in enumerate(betas):
        infection_rate = -np.log(1 - beta) / recovery_time
        portion_of_recovered, portion_of_protected= 0, 0

        for _ in range(cluster_level_monte_runs):
            # Create the multi-cluster network object
            multi_cluster_network = ns_lib.MultiClusterNetworks(
                num_of_clusters, cluster_sizes, cluster_types,
                cluster_paras, infection_rate, protection_rate, inter_cluster_link_paras)
            # Construct the multi-cluster network based on the input parameters.
            multi_cluster_network.ConstructClusters()
            multi_cluster_network.ConstructInterClusterLinks()
            multi_cluster_network.SetLinkTimes()
            neighbor_list_1, neighbor_list_2, event_times_1, event_times_2 = multi_cluster_network.GetNetworkInfo()
            portion_of_recovered_per_run, portion_of_protected_per_run = ns_lib.Simulation(
                sum(cluster_sizes), event_times_1, event_times_2, neighbor_list_1, neighbor_list_2, recovery_time, monte_runs=graph_level_monte_runs)
            portion_of_recovered += portion_of_recovered_per_run
            portion_of_protected += portion_of_protected_per_run
        recovered_results.append(portion_of_recovered / cluster_level_monte_runs)
        protected_results.append(portion_of_protected / cluster_level_monte_runs)
    
    df = pd.DataFrame({'betas': list(betas), 'recovered_portion': recovered_results, 'protected_portion': protected_results})
    df.to_csv(saved_path)



