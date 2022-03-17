from email.policy import default
import network_simulation_lib as ns_lib
import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import numpy as np

os.chdir('G:\\work\\Multi-cluster Simulation\\multiplex_simulations\\results')
plt.rcParams.update({'font.size': 50})

def RetrieveDataDict(file_path):
    data_dict = defaultdict(list)
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for index, row in enumerate(csv_reader):
            if index >= 1:
                data_dict['betas'].append(round(float(row[1]), 3))
                data_dict['recovered_portion'].append(round(float(row[2]), 10))
                data_dict['protected_portion'].append(round(float(row[3]), 10))
    return data_dict

def RetrieveMutiDatas(base_file_path, segments, betas):
    final_results = defaultdict(list)
    for seg in segments:
        actual_file_path = base_file_path + str(seg) + '.csv'
        actual_data_dict = RetrieveDataDict(actual_file_path)
        if not final_results['betas']:
            final_results['betas'] = actual_data_dict['betas']
        if not final_results['recovered_portion']:
            final_results['recovered_portion'] = actual_data_dict['recovered_portion']
        else:
            for index in range(len(final_results['recovered_portion'])):
                final_results['recovered_portion'][index] += actual_data_dict['recovered_portion'][index]
        if not final_results['protected_portion']:
            final_results['protected_portion'] = actual_data_dict['protected_portion']
        else:
            for index in range(len(final_results['protected_portion'])):
                final_results['protected_portion'][index] += actual_data_dict['protected_portion'][index]
    final_results['recovered_portion'] = [val / len(segments) for val in final_results['recovered_portion']]
    final_results['protected_portion'] = [val / len(segments) for val in final_results['protected_portion']]
    return final_results

betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
segments = [1, 2, 3, 4, 5]
final_results_sim0 = RetrieveMutiDatas('multi_er_random_giant_d5_results_', segments, betas)
final_results_sim1 = RetrieveMutiDatas('multi_er_random_giant_d5_sim1_results_', segments, betas)

num_of_clusters = 100
single_cluster_size = 200
giant_cluster_size = 10100
cluster_sizes = [single_cluster_size] * (num_of_clusters - 1)
cluster_sizes.append(giant_cluster_size)
for beta in betas:
    theo_recovered_sim0, theo_protected_sim0 = ns_lib.GetMultiplexGiantTheoreticalResults(beta, 0.7, 5, 5, 2, 0, 0, 5, 100, cluster_sizes)
    theo_recovered_sim1, theo_protected_sim1 = ns_lib.GetMultiplexGiantTheoreticalResults(beta, 0.7, 5, 5, 2, 0, 1, 5, 100, cluster_sizes)
    # theo_recovered_sim0, theo_protected_sim0 = ns_lib.GetMultiplexTheoreticalResults(beta, 0.7, 5, 5, 2, 0, 0, 4, 100, 'star')
    # theo_recovered_sim1, theo_protected_sim1 = ns_lib.GetMultiplexTheoreticalResults(beta, 0.7, 5, 5, 2, 0, 1, 4, 100, 'star')
    final_results_sim0['theo_recovered_portion'].append(theo_recovered_sim0)
    final_results_sim0['theo_protected_portion'].append(theo_protected_sim0)
    final_results_sim1['theo_recovered_portion'].append(theo_recovered_sim1)
    final_results_sim1['theo_protected_portion'].append(theo_protected_sim1)
    index_0 = final_results_sim0['betas'].index(beta)
    final_results_sim0['simu_recovered_portion'].append(final_results_sim0['recovered_portion'][index_0])
    final_results_sim0['simu_protected_portion'].append(final_results_sim0['protected_portion'][index_0])
    index_1 = final_results_sim1['betas'].index(beta)
    final_results_sim1['simu_recovered_portion'].append(final_results_sim1['recovered_portion'][index_1])
    final_results_sim1['simu_protected_portion'].append(final_results_sim1['protected_portion'][index_1])

plt.plot(betas, final_results_sim0['simu_recovered_portion'], label='simulation-recovered-sim0', linestyle='None', marker='o', markersize=32, markerfacecolor='None', markeredgecolor='k', markeredgewidth=8)
plt.plot(betas, final_results_sim0['theo_recovered_portion'], label='theo-recovered-sim0', color='r', linewidth=10)
plt.plot(betas, final_results_sim1['simu_recovered_portion'], label='simulation-recovered-sim1', linestyle='None', marker='d', markersize=32, markerfacecolor='None', markeredgecolor='k', markeredgewidth=8)
plt.plot(betas, final_results_sim1['theo_recovered_portion'], label='theo-recovered-sim1', color='b', linewidth=10)
# plt.plot(betas, final_results_sim0['simu_protected_portion'], label='simulation-protected-sim0', linestyle='None', marker='o', markersize=32, markerfacecolor='None', markeredgecolor='k', markeredgewidth=8)
# plt.plot(betas, final_results_sim0['theo_protected_portion'], label='theo-protected-sim0', color='r', linewidth=10)
# plt.plot(betas, final_results_sim1['simu_protected_portion'], label='simulation-protected-sim1', linestyle='None', marker='d', markersize=32, markerfacecolor='None', markeredgecolor='k', markeredgewidth=8)
# plt.plot(betas, final_results_sim1['theo_protected_portion'], label='theo-protected-sim1', color='b', linewidth=10)
plt.arrow(0.62, 0.006, -0.08, 0.01, color='b', linewidth=6)
plt.arrow(0.34, 0.066, -0.018, -0.028, color='b', linewidth=6)
plt.arrow(0.52, 0.065, -0.08, -0.028, color='r', linewidth=6)
plt.arrow(0.26, 0.05, -0.035, -0.028, color='r', linewidth=6)
plt.xlabel(r'$\beta$')
plt.ylabel(r'P($\infty$)')
plt.grid()
plt.legend(fontsize=24)

fig = plt.gcf()
fig.set_size_inches(20, 16)
# plt.show()
plt.savefig('multi_er_random_giant_d5_recovered.eps', format='eps')
