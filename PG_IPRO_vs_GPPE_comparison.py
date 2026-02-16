from ipro.ipro_test import TestOracle, TestSolver
from ipro.ipro_dfs import DFSOracle, DijkstraSolver

import matplotlib.pyplot as plt
import numpy as np

import geopandas as gpd
import pickle
import networkx as nx
from ipro import helpers as h

from interactive_ipro_VS_gp_pref_elicit_experiment import InteractiveIproVsGPE

import json

'''''''''''''''''''''''''''''''''''''''''''''
'       EXPERIMENT (test pareto_front)      '
'''''''''''''''''''''''''''''''''''''''''''''
number_of_users = 300
number_of_queries = 15

'''
    PARETO FRONT Problem Setting (artificial data)
'''

pf_not_so_balanced = [
    (2.0, 8.0), (2.21, 7.38), (2.41, 6.76), (2.62, 6.19), (2.83, 5.65),
    (3.03, 5.17), (3.24, 4.99), (3.45, 4.81), (3.66, 4.63), (3.86, 4.39),
    (4.07, 4.03), (4.28, 3.74), (4.48, 3.59), (4.69, 3.43), (4.90, 3.28),
    (5.10, 3.16), (5.31, 3.08), (5.52, 3.01), (5.72, 2.93), (5.93, 2.83),
    (6.14, 2.71), (6.34, 2.59), (6.55, 2.47), (6.76, 2.35), (6.97, 2.23),
    (7.17, 2.12), (7.38, 2.08), (7.59, 2.05), (7.79, 2.03), (8.0, 2.0)
]

balanced_pf_30 = [(2.00, 8.00), (2.10, 7.71), (2.17, 7.39), (2.25, 7.08), (2.34, 6.78), (2.45, 6.48),
                  (2.57, 6.19), (2.70, 5.91), (2.85, 5.64), (3.01, 5.37), (3.18, 5.12), (3.36, 4.87),
                  (3.55, 4.63), (3.75, 4.40), (3.96, 4.17), (4.18, 3.96), (4.41, 3.75), (4.65, 3.56),
                  (4.89, 3.37), (5.15, 3.19), (5.41, 3.02), (5.67, 2.87), (5.95, 2.72), (6.23, 2.58),
                  (6.51, 2.45), (6.80, 2.33), (7.10, 2.23), (7.40, 2.13), (7.70, 2.04), (8.00, 2.00)]

balanced_pf_10 = [
    (2.00, 8.00),
    (2.34, 6.78),
    (2.85, 5.64),
    (3.36, 4.87),
    (3.96, 4.17),
    (4.89, 3.37),
    (5.67, 2.87),
    (6.51, 2.45),
    (7.40, 2.13),
    (8.00, 2.00)
]

concave_pf_30 = [
    (2.00, 8.00), (2.29, 7.90), (2.61, 7.83), (2.92, 7.75), (3.22, 7.66), (3.52, 7.55),
    (3.81, 7.43), (4.09, 7.30), (4.36, 7.15), (4.63, 6.99), (4.88, 6.82), (5.13, 6.64),
    (5.37, 6.45), (5.60, 6.25), (5.83, 6.04), (6.04, 5.82), (6.25, 5.59), (6.44, 5.35),
    (6.63, 5.11), (6.81, 4.85), (6.98, 4.59), (7.13, 4.33), (7.28, 4.05), (7.42, 3.77),
    (7.55, 3.49), (7.67, 3.20), (7.77, 2.90), (7.87, 2.60), (7.96, 2.30), (8.00, 2.00)
]

concave_pf_10 = [
    (2.00, 8.00),
    (3.22, 7.66),
    (4.36, 7.15),
    (5.13, 6.64),
    (5.83, 6.04),
    (6.25, 5.59),
    (6.63, 5.11),
    (7.13, 4.33),
    (7.67, 3.20),
    (8.00, 2.00)
]
pf_4obj = [
 [4, 3, 3, 2], [3, 4, 5, 7], [6, 2, 5, 3], [4, 4, 2, 8], [2, 8, 8, 2], [2, 7, 8, 8],
 [5, 2, 2, 5], [7, 8, 2, 4], [8, 6, 2, 3], [4, 2, 8, 2], [4, 2, 3, 7], [7, 2, 3, 4],
 [2, 8, 6, 8], [3, 5, 8, 3], [3, 8, 5, 3], [3, 4, 6, 4], [3, 8, 3, 5], [3, 6, 5, 5],
 [2, 8, 7, 3], [8, 5, 2, 4], [8, 2, 3, 2], [4, 7, 2, 7], [3, 3, 6, 8], [3, 6, 8, 2],
 [3, 8, 7, 2], [3, 7, 6, 3], [3, 2, 8, 5], [3, 4, 4, 8], [4, 2, 6, 6], [5, 2, 4, 4]
]

'''
linear_pf_30 = [
    (2.00, 8.00), (2.21, 7.79), (2.41, 7.59), (2.62, 7.38), (2.83, 7.17),
    (3.03, 6.97), (3.24, 6.76), (3.45, 6.55), (3.66, 6.34), (3.86, 6.14),
    (4.07, 5.93), (4.28, 5.72), (4.48, 5.52), (4.69, 5.31), (4.90, 5.10),
    (5.10, 4.90), (5.31, 4.69), (5.52, 4.48), (5.72, 4.28), (5.93, 4.07),
    (6.14, 3.86), (6.34, 3.66), (6.55, 3.45), (6.76, 3.24), (6.97, 3.03),
    (7.17, 2.83), (7.38, 2.62), (7.59, 2.41), (7.79, 2.21), (8.00, 2.00)
]'''

# separate objectives to show scatterplot
obj1_values, obj2_values = zip(*balanced_pf_30)
# create the scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(obj1_values, obj2_values, label='non-dominated solution')
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('Pareto front')
plt.legend()
plt.grid(True)
plt.show()

'''
    PARETO FRONT Problem Setting (real data Amsterdam)
'''

adj_matrix_path = "amsterdam_data/adjacency_matrix_osdpm.txt"
gdf_path = "amsterdam_data/gdf_osdpm_connected_pt.gpkg"
node_dict_path = "amsterdam_data/node_dict_osdpm.pickle"

gdf = gpd.read_file(gdf_path)
with open(node_dict_path, 'rb') as f:
    node_dict = pickle.load(f)

graph = h.load_adjacency_matrix_safe(adj_matrix_path)

### source and target
origin_lat = 52.353005 #52.350758!   #52.353662   #52.351699          #52.352000  #mvp1 - 52.350016
origin_lon = 4.794515 #4.798904!   #4.793257   #4.800223       #4.799500     #mvp1 - 4.797826
destination_lat = 52.351712 #52.357348!  #52.357892  #52.352097             #52.352700 #mvp1 - 52.362954
destination_lon = 4.800207 #4.793762!  #4.795246  #4.798464          #4.798500 #mvp1- 4.793823

source, distance = h.find_nearest_node(origin_lat, origin_lon, node_dict)
target, distance = h.find_nearest_node(destination_lat, destination_lon, node_dict)
print("source-node:", source)
print("target-node:", target)


'''
    Clean graph
'''
for origin in graph:
    for destination in graph[origin]:
        original_edge = graph[origin][destination]
        rounded_edge = tuple(int(np.ceil(val)) if isinstance(val, (int, float)) else val for val in original_edge)
        graph[origin][destination] = rounded_edge
# Remove self-loops
def remove_self_loops(graph):
    removed = []
    for node in list(graph.keys()):
        if node in graph[node]:
            del graph[node][node]
            removed.append(node)
    return removed
remove_self_loops(graph)

gdf_viz = gpd.read_file('amsterdam_data/gdf_osdpm_connected.gpkg')

G = nx.Graph()
for node1 in graph:
    for node2 in graph[node1]:
        edge = graph[node1][node2]
        G.add_edge(
            node1,
            node2,
            length=edge[0],
            crossing=edge[1],
            #walk=edge[2],
            #bike=edge[3],
            vec=np.array([edge[0], edge[1]])
        )


''''''''''''''''''''
'     IPRO setup   '
''''''''''''''''''''
dimensions = 4
utility_noise = .01
artificial_data = True

if artificial_data:
    # upper bounds calculations: taking the max values of the pareto optimal solutions calculated for each objective
    # individually
    upper_bnds = {'obj1': 8.0, 'obj2': 8.0, 'obj3': 8.0, 'obj4': 8.0}  # specified/defined as is for this test case
    linear_solver = TestSolver(pf_4obj, upper_bnds)
    oracle = TestOracle(pf_4obj, upper_bnds, heuristic='chebyshev')
else:
    oracle = DFSOracle(graph, source, target, dimensions, lower_bounds_algorithm="reverse_dijkstra")

    linear_solver = DijkstraSolver(G, source, target, dimensions, upper_bounds=None, lower_bounds=None,
                                   shortest_paths=None)
    # upper bounds calculations: taking the max values of the pareto optimal paths calculated for each objective
    # individually with Dijkstra
    ideals = [linear_solver.solve(weight_vec, precomputed=False) for weight_vec in np.eye(dimensions)]
    objectives_ideals = [ideal_vec for ideal_vec, ideal_sol in ideals]
    ideal = [min(values) for values in zip(*objectives_ideals)]
    print("ideal: ", ideal)

    # get ideal values for objective_i and as small as possible for other objectives (as tie-breaker)
    shortest_paths_solutions_to_ideal = [oracle.dijkstra_shortest_path_with_ideal_guidance(obj_i, ideal) for obj_i in
                                         range(dimensions)]
    objectives_ideals = [ideal_cost for ideal_cost, ideal_sol in shortest_paths_solutions_to_ideal]
    paths_ideals = [ideal_sol for ideal_vec, ideal_sol in shortest_paths_solutions_to_ideal]
    linear_solver.lower_bounds = objectives_ideals
    linear_solver.shortest_paths = paths_ideals
    print("objectives_ideals: ", objectives_ideals)

    upper_bounds = [max(values) for values in zip(*objectives_ideals)]
    print("upper_bounds: ", upper_bounds)
    linear_solver.upper_bounds = upper_bounds


''''''''''''''''''''''''''
'       EXPERIMENT       '
''''''''''''''''''''''''''
PGIPRO_vs_GPPE_experiment = InteractiveIproVsGPE(num_queries=number_of_queries,
                                                 experiment_iterations=number_of_users,
                                                 dimensions=dimensions,
                                                 ipro_oracle=oracle,
                                                 ipro_linear_solver=linear_solver,
                                                 gppe_query_type='pairwise',
                                                 utility_noise=utility_noise,
                                                 direction='minimize',
                                                 referent_selection_heuristic='middle_distance')
average_utility_gppe_exp, average_utility_interactive_ipro_exp, average_max_utility_gppe_exp, \
    average_max_utility_interactive_ipro_exp, std_dev_utility_gppe_exp, std_dev_utility_interactive_ipro_exp, \
    std_dev_max_utility_gppe_exp, std_dev_max_utility_interactive_ipro_exp, ipro_generate_pf_time_exp, \
    mean_interactive_ipro_generate_subsolution_time_exp, std_dev_interactive_ipro_generate_subsolution_time_exp = PGIPRO_vs_GPPE_experiment.run()


# save results as json:
def convert_ndarrays(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(i) for i in obj]
    else:
        return obj

results = {
    "average_utility_gppe_exp": average_utility_gppe_exp,
    "average_utility_interactive_ipro_exp": average_utility_interactive_ipro_exp,
    "average_max_utility_gppe_exp": average_max_utility_gppe_exp,
    "average_max_utility_interactive_ipro_exp": average_max_utility_interactive_ipro_exp,
    "std_dev_utility_gppe_exp": std_dev_utility_gppe_exp,
    "std_dev_utility_interactive_ipro_exp": std_dev_utility_interactive_ipro_exp,
    "std_dev_max_utility_gppe_exp": std_dev_max_utility_gppe_exp,
    "std_dev_max_utility_interactive_ipro_exp": std_dev_max_utility_interactive_ipro_exp,
    "ipro_generate_pf_time_exp": ipro_generate_pf_time_exp,
    "mean_interactive_ipro_generate_subsolution_time_exp": mean_interactive_ipro_generate_subsolution_time_exp,
    "std_dev_interactive_ipro_generate_subsolution_time_exp": std_dev_interactive_ipro_generate_subsolution_time_exp
}
serializable_results = convert_ndarrays(results)
with open("pgipro_vs_gppe_4objPF_results_300users.json", "w") as f:
    json.dump(serializable_results, f, indent=4)

# plotting Gaussian Process Preference Elicitation versus preference guided (interactive) IPRO results
plt.figure(figsize=(10, 5))
plt.errorbar(
    np.arange(1, len(average_utility_gppe_exp) + 1),
    average_utility_gppe_exp,
    yerr=std_dev_utility_gppe_exp,
    fmt='-o',
    capsize=5,
    color='b',
    label='GPPE'
)
plt.errorbar(
    np.arange(1, len(average_utility_interactive_ipro_exp) + 1),
    average_utility_interactive_ipro_exp,
    yerr=std_dev_utility_interactive_ipro_exp,
    fmt='-o',
    capsize=5,
    color='g',
    label='PG-IPRO'
)
plt.ylim(-0.1, 1.1)
plt.xticks(np.arange(1, len(average_utility_interactive_ipro_exp) + 1))
plt.xlabel("Solution (querying between points)")
plt.ylabel("Utility")
plt.title("Average solution utility comparison for four-objectives PF with 30 solutions")
#plt.title("Average route utility comparison for Osdorp-Midden route planning problem, PF with 7 solutions")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
plt.errorbar(
    np.arange(1, len(average_max_utility_gppe_exp) + 1),
    average_max_utility_gppe_exp,
    yerr=std_dev_max_utility_gppe_exp,
    fmt='-o',
    capsize=5,
    color='b',
    label='GPPE'
)
plt.errorbar(
    np.arange(1, len(average_max_utility_interactive_ipro_exp) + 1),
    average_max_utility_interactive_ipro_exp,
    yerr=std_dev_max_utility_interactive_ipro_exp,
    fmt='-o',
    capsize=5,
    color='g',
    label='PG-IPRO'
)
plt.ylim(-0.1, 1.1)
plt.xticks(np.arange(1, len(average_max_utility_interactive_ipro_exp) + 1))
plt.xlabel("Solution (querying between points)")
plt.ylabel("Utility")
plt.title("Average maximum utility comparison for four-objectives PF with 30 solutions")
#plt.title("Average maximum utility comparison for Osdorp-Midden route planning problem, PF with 7 solutions")
plt.legend()
plt.grid(True)
plt.show()


# Show time comparison
plt.figure(figsize=(5, 5))
bars = plt.bar(['Generate PF', 'Generate PO route'],
               [ipro_generate_pf_time_exp, mean_interactive_ipro_generate_subsolution_time_exp],
               yerr=[0, std_dev_interactive_ipro_generate_subsolution_time_exp],
               capsize=16,
               color=['blue', 'green'])
for bar, mean in zip(bars, [ipro_generate_pf_time_exp, mean_interactive_ipro_generate_subsolution_time_exp]):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height - 0.01,
             f'{mean:.2f}',
             ha='center',
             va='top',
             fontsize=10,
             color='white' if height > 0.01 else 'black')
plt.ylabel('Time (in seconds)')
plt.title('Generating approximated PF vs Pareto Optimal route time')
plt.show()
