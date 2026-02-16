import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import networkx as nx

from ipro.ipro_dfs import DFSOracle, DijkstraSolver
#from ipro.ipro_dfs_igraph import DijkstraSolver
from ipro.outer_loops.ipro import IPRO

from ipro import helpers as h
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import pickle

import matplotlib.patches as mpatches
import contextily as ctx

import heapq


'''
    IPRO testing on Amsterdam Data
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

#check data
# Number of nodes
num_nodes = len(graph)
# Number of edges
num_edges = sum(len(neighbors) for neighbors in graph.values())
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")

for node, neighbors in list(graph.items())[:10]:  # Show first 3 nodes
    print(f"Node {node} has edges to:")
    for dest, props in neighbors.items():
        print(f"  → {dest}, with properties: {props}")


print(gdf.info())        # Overview of column types and nulls
print(gdf.head())        # First few rows
print(gdf.columns)       # Column names
print(gdf.crs)           # Coordinate reference system

gdf.plot()
print(gdf.head())

'''
def is_reachable(graph, start, target):
    from collections import deque
    visited = set()
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == target:
            return True
        for neighbor in graph.get(current, {}):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


print("are source and target connected?: ", is_reachable(graph, source, target))

def check_bidirectionality(adj_matrix):
    missing = []
    for u, nbrs in adj_matrix.items():
        for v in nbrs:
            # If v isn’t in the dict at all, or if u isn’t listed as its neighbor
            if v not in adj_matrix or u not in adj_matrix[v]:
                missing.append((u, v))
    return missing

missing_edges = check_bidirectionality(graph)

if not missing_edges:
    print("All edges are bidirectional!!!")
else:
    print("Missing reverse edges!!!")

def check_self_loops(graph):
    """Return list of nodes that have self-loops."""
    return [(u, u) for u in graph if u in graph[u]]

def check_node_consistency(graph):
    """Return set of destination nodes that are used but not defined as keys."""
    undefined_nodes = set()
    for u, nbrs in graph.items():
        for v in nbrs:
            if v not in graph:
                undefined_nodes.add(v)
    return undefined_nodes

def check_edge_weights(graph):
    """Return list of edges with any negative weight in their cost tuple."""
    bad_edges = []
    for u, nbrs in graph.items():
        for v, cost_tuple in nbrs.items():
            if any(weight < 0 for weight in cost_tuple):
                bad_edges.append((u, v, cost_tuple))
    return bad_edges

self_loops = check_self_loops(graph)
undefined_nodes = check_node_consistency(graph)
bad_edge_weights = check_edge_weights(graph)
'''

gdf_viz = gpd.read_file('amsterdam_data/gdf_osdpm_connected.gpkg')


if "graph_osdpm.gpickle" in os.listdir('amsterdam_data/'):
    with open('amsterdam_data/graph_osdpm.gpickle', 'rb') as f:
        G = pickle.load(f)
else:
    G = nx.Graph()
    for node1 in graph:
        for node2 in graph[node1]:
            edge = graph[node1][node2]
            #G.add_vertex(node1)
            #G.add_vertex(node2)
            G.add_edge(
                node1,
                node2,
                length=edge[0],
                crossing=edge[1],
                walk=edge[2],
                bike=edge[3],
                vec=np.array([edge[0], edge[1], edge[2], edge[3]])
            )

with open('amsterdam_data/graph_osdpm.gpickle', 'wb') as f:
    pickle.dump(G, f)

print(type(G))

# Initialize DFS oracle for amsterdam data graph
problem_id = 'graph_amsterdam'
dimensions = 4
oracle = DFSOracle(graph, source, target, dimensions, lower_bounds_algorithm="reverse_dijkstra")

linear_solver = DijkstraSolver(G, source, target, dimensions, upper_bounds=None, lower_bounds=None, shortest_paths=None)
# upper bounds calculations: taking the max values of the pareto optimal paths calculated for each objective
# individually with Dijkstra
for weight_vec in np.eye(dimensions):
    print(weight_vec)
ideals = [linear_solver.solve(weight_vec, precomputed=False) for weight_vec in np.eye(dimensions)]
objectives_ideals = [ideal_vec for ideal_vec, ideal_sol in ideals]
ideal = [min(values) for values in zip(*objectives_ideals)]
#print("ideal: ", ideal)

# get ideal values for objective_i and as small as possible for other objectives (as tie-breaker)
paths_ideals = [ideal_sol for ideal_vec, ideal_sol in ideals]
linear_solver.lower_bounds = objectives_ideals
linear_solver.shortest_paths = paths_ideals
#print("objectives_ideals: ", objectives_ideals)

upper_bounds = [max(values) for values in zip(*objectives_ideals)]
#print("upper_bounds: ", upper_bounds)
linear_solver.upper_bounds = upper_bounds

# show start and destination nodes:
fig, ax = plt.subplots()
gdf_viz.plot(ax=ax, color='grey', alpha=0.001)
ctx.add_basemap(ax, crs=gdf_viz.crs, source=ctx.providers.CartoDB.Voyager, zoom=15)
ax.scatter(node_dict[source][0], node_dict[source][1], color='blue', s=20, zorder=5, label="Source")
ax.scatter(node_dict[target][0], node_dict[target][1], color='red', s=20, zorder=5, label="Target")
origin = mpatches.Patch(color='blue', label='origin')
dest = mpatches.Patch(color='red', label='destination')
plt.rcParams["legend.fontsize"] = 4
origin_point = gpd.GeoSeries([Point(origin_lon, origin_lat)], crs="EPSG:4326").to_crs(gdf_viz.crs)
destination_point = gpd.GeoSeries([Point(destination_lon, destination_lat)], crs="EPSG:4326").to_crs(gdf_viz.crs)
all_points = pd.concat([origin_point, destination_point])
minx, miny, maxx, maxy = all_points.total_bounds
padding = 100  # meters if using EPSG:28992 or Web Mercator
plt.xlim(minx - padding, maxx + padding)
plt.ylim(miny - padding, maxy + padding)
#plt.xlim(114000, 115500)
#plt.ylim(484600, 486400)
plt.axis('off')
plt.tight_layout()
plt.show()


ipro_amsterdam = IPRO(
    problem_id=problem_id,
    dimensions=dimensions,
    oracle=oracle,
    linear_solver=linear_solver,
    direction='minimize',
    max_iterations=100000,
    tolerance=0,
    user_interaction_loop=True,  # IPRO with user input interaction
    referent_selection_heuristic='middle_distance'
)

processed_solutions = []
user_input = None
while True:
    print("Enter ipro solve")
    subsolution = ipro_amsterdam.solve(return_inter_sol=False, user_input=user_input) #  subsolution of type list[tuple[np.ndarray, Any]],
                               #  but only returns 1 element list from solve for IPRO with user input interaction
    if subsolution is None:
        print("User preferences can't be improved, no better solution exists.")
        break
    elif len(subsolution) != 1:  # at end of IPRO solve method it returns the whole pareto front,
                               # this marks the end of the iteration/interaction loop since all possibilities have been processed
        # this marks the end of the iteration/interaction loop since all possibilities have been processed
        current_pf = subsolution
        print("pf:", current_pf)
        print("pf size:", len(current_pf))
        objectives_vectors = [pf_sol[0] * -1 for pf_sol in current_pf]
        print("pf objective vectors: ", objectives_vectors)
        print("Done!")

        # the figure
        fig, ax = plt.subplots()
        # Plot base map and background
        gdf_viz.plot(ax=ax, color='grey', alpha=0.001)
        ctx.add_basemap(ax, crs=gdf_viz.crs, source=ctx.providers.CartoDB.Voyager, zoom=15)
        ax.scatter(node_dict[source][0], node_dict[source][1], color='blue', s=20, zorder=5,label="Source")
        ax.scatter(node_dict[target][0], node_dict[target][1], color='red', s=20, zorder=5,label="Destination")
        colors = cm.get_cmap('tab20', len(current_pf))  # You can increase this if needed
        legend_handles = [
            mpatches.Patch(color='blue', label='Origin'),
            mpatches.Patch(color='red', label='Destination')
        ]
        # plot each route
        for idx, pf_sol in enumerate(current_pf):
            cost_vector = pf_sol[0] * -1 # negative values due to minimization
            path = pf_sol[1]
            color = colors(idx)
            h.get_route_gdf(path, gdf_viz).plot(ax=ax, color=color, linewidth=2, alpha=0.8)
            # add path info to legend
            label = f"Pareto optimal route {idx + 1}: {np.round(cost_vector, 2)}"
            legend_handles.append(mpatches.Patch(color=color, label=label))
        # show the legend with costs
        plt.legend(handles=legend_handles, loc='lower right', fontsize=6)

        plt.rcParams["legend.fontsize"] = 5
        plt.xlim(114000, 115500)
        plt.ylim(484600, 486400)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        processed_solutions = [pf_sol[0] * -1 for pf_sol in current_pf]
        # separate objectives to show scatterplot
        obj1_values, obj2_values = zip(*processed_solutions)
        # draw the scatterplot
        plt.figure(figsize=(6, 6))
        plt.scatter(obj1_values, obj2_values, label='processed non-dominated solutions')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Pareto front')
        plt.xlim(min(obj1_values) - 100, max(obj1_values) + 100)
        plt.ylim(min(obj2_values) - 1, max(obj2_values) + 1)
        plt.legend()
        plt.grid(True)
        plt.show()
        break
    else:
        pareto_sol_objectives = subsolution[0][0] * -1
        path = subsolution[0][1]
        print("route: ", path)
        if path is None:
           print("Current referent can't find a subsolution, try again for a next_referent...")
           continue

        # Visualize the current solution ***
        print(f"Current solution objectives: {pareto_sol_objectives}")

        fig, ax = plt.subplots(figsize=(10, 8))
        gdf_viz.plot(ax=ax, color='grey', alpha=0.001)
        ctx.add_basemap(ax, crs=gdf_viz.crs, source=ctx.providers.CartoDB.Voyager, zoom=15)
        ax.scatter(node_dict[source][0], node_dict[source][1], color='blue', s=50, zorder=5,
                  label="Origin")
        ax.scatter(node_dict[target][0], node_dict[target][1], color='red', s=50, zorder=5,
                  label="Destination")

        # Plot the current route
        h.get_route_gdf(path, gdf_viz).plot(ax=ax, color='green', linewidth=3, alpha=0.8,
                                           label='Current Route')

        plt.legend(loc='lower right', fontsize=8)
        plt.title(f'Current Route - Objectives: {np.round(pareto_sol_objectives, 2)}',
                 fontsize=10)
        plt.xlim(114000, 115500)
        plt.ylim(484600, 486400)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    # Query user input for next objective binary preference
    while True:
        objective = input("Choose which objective to decrease or increase:")
        if not objective.isdigit():
            print("Invalid input! Please enter a valid objective (integer).")
            continue
        objective = int(objective)
        if not 0 <= objective < dimensions:  # check bounds of the present dimensions (objectives)
            print("Invalid input! Please enter a valid objective.")
            continue
        print("You chose objective :", objective)
        direction = input("Enter '-' to decrease or '+' to increase the objective: ")
        if direction not in ['-', '+']:
            print("Incorrect direction. Please enter either '-' or '+'.")
            continue
        print("You chose direction :", direction)
        user_input = (objective, direction)
        break



'''
    IPRO initialization + testing on small artificial graph
'''
#
# node_coords = {
#     'A': (0, 3), 'B': (1, 3), 'C': (2, 3), 'D': (3, 3),
#     'E': (0, 2), 'F': (1, 2), 'G': (2, 2), 'H': (3, 2),
#     'I': (0, 1), 'J': (1, 1), 'K': (2, 1), 'L': (3, 1),
#     'M': (0, 0), 'N': (1, 0), 'O': (2, 0), 'P': (3, 0)
# }
# # make graph
# '''graph = {
#     'A': {'B': (10, 1), 'E': (1, 10)},
#     'B': {'A': (10, 1), 'C': (10, 1), 'F': (20, 20)},
#     'C': {'B': (10, 1), 'D': (10, 1), 'G': (20, 20)},
#     'D': {'C': (10, 1), 'H': (10, 1)},
#     'E': {'A': (1, 10), 'F': (2, 2), 'I': (1, 10)},
#     'F': {'B': (20, 20), 'E': (2, 2), 'G': (2, 1), 'J': (1, 2)},
#     'G': {'C': (20, 20), 'F': (2, 1), 'H': (20, 20), 'K': (2, 1)},
#     'H': {'D': (10, 1), 'G': (20, 20), 'L': (10, 1)},
#     'I': {'E': (1, 10), 'J': (20, 20), 'M': (1, 10)},
#     'J': {'F': (1, 2), 'I': (20, 20), 'K': (1, 2), 'N': (20, 20)},
#     'K': {'G': (2, 1), 'J': (1, 2), 'L': (2, 2), 'O': (20, 20)},
#     'L': {'H': (10, 1), 'K': (2, 2), 'P': (10, 1)},
#     'M': {'I': (1, 10), 'N': (1, 10)},
#     'N': {'J': (20, 20), 'M': (1, 10), 'O': (1, 10)},
#     'O': {'K': (20, 20), 'N': (1, 10), 'P': (1, 10)},
#     'P': {'L': (10, 1), 'O': (1, 10)}
# }'''
# '''graph = {
#     'A': {'B': (10, 1), 'E': (1, 10)},
#     'B': {'A': (10, 1), 'C': (10, 1), 'F': (20, 20)},
#     'C': {'B': (10, 1), 'D': (10, 1), 'G': (20, 20)},
#     'D': {'C': (10, 1), 'H': (10, 1)},
#     'E': {'A': (1, 10), 'F': (30, 30), 'I': (1, 10)},
#     'F': {'B': (20, 20), 'E': (30, 30), 'G': (2, 1), 'J': (1, 2)},
#     'G': {'C': (20, 20), 'F': (2, 1), 'H': (20, 20), 'K': (2, 1)},
#     'H': {'D': (10, 1), 'G': (20, 20), 'L': (10, 1)},
#     'I': {'E': (1, 10), 'J': (20, 20), 'M': (1, 10)},
#     'J': {'F': (1, 2), 'I': (20, 20), 'K': (1, 2), 'N': (20, 20)},
#     'K': {'G': (2, 1), 'J': (1, 2), 'L': (30, 30), 'O': (20, 20)},
#     'L': {'H': (10, 1), 'K': (30, 30), 'P': (10, 1)},
#     'M': {'I': (1, 10), 'N': (1, 10)},
#     'N': {'J': (20, 20), 'M': (1, 10), 'O': (1, 10)},
#     'O': {'K': (20, 20), 'N': (1, 10), 'P': (1, 10)},
#     'P': {'L': (10, 1), 'O': (1, 10)}
# }'''
# '''graph = {
#     'A': {'B': (10, 1), 'E': (1, 10)},
#     'B': {'A': (10, 1), 'C': (10, 1), 'F': (20, 20)},
#     'C': {'B': (10, 1), 'D': (10, 1), 'G': (20, 20)},
#     'D': {'C': (10, 1), 'H': (10, 1)},
#     'E': {'A': (1, 10), 'F': (30, 30), 'I': (1, 10)},
#     'F': {'B': (20, 20), 'E': (30, 30), 'G': (30, 30), 'J': (30, 30)},
#     'G': {'C': (20, 20), 'F': (30, 30), 'H': (20, 20), 'K': (30,30)},
#     'H': {'D': (10, 1), 'G': (20, 20), 'L': (10, 1)},
#     'I': {'E': (1, 10), 'J': (20, 20), 'M': (1, 10)},
#     'J': {'F': (30, 30), 'I': (20, 20), 'K': (30, 30), 'N': (20, 20)},
#     'K': {'G': (30, 30), 'J': (30, 30), 'L': (30, 30), 'O': (20, 20)},
#     'L': {'H': (10, 1), 'K': (30, 30), 'P': (10, 1)},
#     'M': {'I': (1, 10), 'N': (1, 10)},
#     'N': {'J': (20, 20), 'M': (1, 10), 'O': (1, 10)},
#     'O': {'K': (20, 20), 'N': (1, 10), 'P': (1, 10)},
#     'P': {'L': (10, 1), 'O': (1, 10)}
# }'''
#
# graph = {
#     'A': {'B': (10, 1), 'E': (1, 10)},
#     'B': {'A': (10, 1), 'C': (10, 1), 'F': (20, 20)},
#     'C': {'B': (10, 1), 'D': (10, 1), 'G': (20, 20)},
#     'D': {'C': (10, 1), 'H': (10, 1)},
#     'E': {'A': (1, 10), 'F': (15, 15), 'I': (1, 10)},
#     'F': {'B': (20, 20), 'E': (15, 15), 'G': (2, 1), 'J': (1, 2)},
#     'G': {'C': (20, 20), 'F': (2, 1), 'H': (15, 15), 'K': (2, 1)},
#     'H': {'D': (10, 1), 'G': (15, 15), 'L': (10, 1)},
#     'I': {'E': (1, 10), 'J': (20, 20), 'M': (1, 10)},
#     'J': {'F': (1, 2), 'I': (20, 20), 'K': (1, 2), 'N': (15, 15)},
#     'K': {'G': (2, 1), 'J': (1, 2), 'L': (15, 15), 'O': (20, 20)},
#     'L': {'H': (10, 1), 'K': (15, 15), 'P': (10, 1)},
#     'M': {'I': (1, 10), 'N': (1, 10)},
#     'N': {'J': (15, 15), 'M': (1, 10), 'O': (1, 10)},
#     'O': {'K': (20, 20), 'N': (1, 10), 'P': (1, 10)},
#     'P': {'L': (10, 1), 'O': (1, 10)}
# }
#
# source = "A"
# target = "P"
#
# G = nx.Graph()
# for node1 in graph:
#     for node2 in graph[node1]:
#         edge = graph[node1][node2]
#         G.add_edge(
#             node1,
#             node2,
#             weight=edge[0],
#             crosswalk=edge[1],
#             #walk=edge[2],
#             #bike=edge[3],
#             vec=np.array(edge)
#         )
#
#
# def single_objective_value_iteration(objective):
#     """Calculate lower bounds for the objective for each node in the graph."""
#     node_values = {target: 0}  # lower_bound per node
#     done = False
#     while not done:
#         done = True
#         for node in graph.keys():
#             if node == target:
#                 continue
#             new_cost = min(
#                 graph[node][neighbor][objective] + node_values.get(neighbor, float('inf')) for neighbor in
#                 graph.get(node, {}))
#             if new_cost < node_values.get(node, float('inf')):
#                 done = False
#                 node_values[node] = new_cost
#     return node_values
#
#
# def reverse_dijkstra(objective):
#     """Calculate lower bounds for the objective for each node in the graph, starting from the target node."""
#     node_values = {}  # lower_bound per node
#     pq = [(0, target)]
#     while pq:
#         cost, current = heapq.heappop(pq)
#         if current in node_values:
#             continue
#         node_values[current] = cost
#         for neighbor in graph.get(current, {}):
#             if neighbor not in node_values:
#                 edge_costs = graph[current][neighbor]
#                 new_cost = cost + edge_costs[objective]
#                 heapq.heappush(pq, (new_cost, neighbor))
#     return node_values
#
# nodes_sobi0 = single_objective_value_iteration(0)
# nodes_rev_dijkstra0 = reverse_dijkstra(0)
# print("nodes_sobi0: ", nodes_sobi0)
# print("nodes_rev_dijkstra0: ", nodes_rev_dijkstra0)
# nodes_sobi1 = single_objective_value_iteration(1)
# nodes_rev_dijkstra1 = reverse_dijkstra(1)
# print("nodes_sobi1: ", nodes_sobi1)
# print("nodes_rev_dijkstra1: ", nodes_rev_dijkstra1)
#
# '''
#     IPRO initialization with Depth First Search multi-objective path finding Oracle
# '''
# problem_id = 'graph'
# dimensions = 2
# oracle = DFSOracle(graph, source, target, dimensions, lower_bounds_algorithm="reverse_dijkstra")
#
# linear_solver = DijkstraSolver(G, source, target, dimensions, upper_bounds=None, lower_bounds=None, shortest_paths=None)
# # upper bounds calculations: taking the max values of the pareto optimal paths calculated for each objective
# # individually with Dijkstra
# ideals = [linear_solver.solve(weight_vec, precomputed=False) for weight_vec in np.eye(dimensions)]
# objectives_ideals = [ideal_vec for ideal_vec, ideal_sol in ideals]
# ideal = [min(values) for values in zip(*objectives_ideals)]
# print("ideal: ", ideal)
#
# # get ideal values for objective_i and as small as possible for other objectives (as tie-breaker)
# shortest_paths_solutions_to_ideal = [oracle.dijkstra_shortest_path_with_ideal_guidance(obj_i, ideal) for obj_i in range(dimensions)]
# objectives_ideals = [ideal_cost for ideal_cost, ideal_sol in shortest_paths_solutions_to_ideal]
# paths_ideals = [ideal_sol for ideal_vec, ideal_sol in shortest_paths_solutions_to_ideal]
# linear_solver.lower_bounds = objectives_ideals
# linear_solver.shortest_paths = paths_ideals
# print("objectives_ideals: ", objectives_ideals)
#
# upper_bounds = [max(values) for values in zip(*objectives_ideals)]
# print("upper_bounds: ", upper_bounds)
# linear_solver.upper_bounds = upper_bounds
#
#
# ipro = IPRO(
#     problem_id=problem_id,
#     dimensions=dimensions,
#     oracle=oracle,
#     linear_solver=linear_solver,
#     direction='minimize',
#     max_iterations=100,
#     tolerance=0,
#     user_interaction_loop=False,
# )
#
# pf = ipro.solve()
# print(f'Full Pareto front: {pf}')
#
# processed_solutions = [pf_sol[0] * -1 for pf_sol in pf]
# # separate objectives to show scatterplot
# obj1_values, obj2_values = zip(*processed_solutions)
# # draw the scatterplot
# plt.figure(figsize=(6, 6))
# plt.scatter(obj1_values, obj2_values, label='processed non-dominated solutions')
# plt.xlabel('Objective 1')
# plt.ylabel('Objective 2')
# plt.title('Pareto front')
# plt.xlim(1, 100)
# plt.ylim(1, 100)
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
#
# # Assign colors to paths
# path_colors = ['green', 'red', "blue", "orange", "yellow", "purple", "cyan", "pink"]  # different colors for each path
# edge_colors = {}
# default_edge_color = "gray"
# # Assign colors to edges
# for i, path in enumerate([path for _, path in pf]):
#     for u, v in zip(path[:-1], path[1:]):
#         edge_colors[(u, v)] = path_colors[i]
#         edge_colors[(v, u)] = path_colors[i]
# # draw the graph
# plt.figure(figsize=(8, 8))
# nx.draw(G, node_coords, with_labels=True, node_size=2000, node_color="lightblue", font_size=12, font_weight="bold")
# edge_labels = {(u, v): f"{d['weight']}, {d['crosswalk']}" for u, v, d in G.edges(data=True)}
# for (u, v) in G.edges():
#     color = edge_colors.get((u, v), default_edge_color)
#     nx.draw_networkx_edges(G, node_coords, edgelist=[(u, v)], edge_color=color, width=2)
# nx.draw_networkx_edge_labels(G, node_coords, edge_labels=edge_labels, font_size=10)
# nx.draw_networkx_nodes(G, node_coords, nodelist=['A'], node_color='yellow', node_size=2500)
# nx.draw_networkx_nodes(G, node_coords, nodelist=['P'], node_color='orange', node_size=2500)
#
# plt.title("4x4 grid graph")
# plt.show()
#
#
#
#
# '''
#     IPRO with user input interaction
# '''
#
# ipro = IPRO(
#     problem_id=problem_id,
#     dimensions=dimensions,
#     oracle=oracle,
#     linear_solver=linear_solver,
#     direction='minimize',
#     max_iterations=100,
#     tolerance=0,
#     user_interaction_loop=True,  # IPRO with user input interaction
# )
#
# processed_solutions = []
# user_input = None
# while True:
#     print("Enter ipro solve")
#     subsolution = ipro.solve(user_input=user_input) #  subsolution of type list[tuple[np.ndarray, Any]],
#                                #  but only returns 1 element list from solve for IPRO with user input interaction
#     if subsolution is None:
#         print("User preferences can't be improved, no better solution exists.")
#         #break
#     elif len(subsolution) != 1:  # at end of IPRO solve method it returns the whole pareto front,
#                                # this marks the end of the iteration/interaction loop since all possibilities have been processed
#         print("pf:", subsolution)
#         print("Done!")
#         #break
#     else:
#         pareto_sol_objectives = subsolution[0][0] * -1  # * -1 because values returned are negative due to minimization problem semantics in IPRO
#         route = subsolution[0][1]
#         print("route: ", route)
#         if route is None:  # sol returned by oracle should be None if no pareto_sol can be found
#             print("Current referent can't find a subsolution, try again for a next_referent...")
#             continue
#         else:
#             processed_solutions.append(pareto_sol_objectives)
#             print("subsolution in user loop:", pareto_sol_objectives)
#             print("processed_sols: ", processed_solutions)
#             # separate objectives to show scatterplot
#             obj1_values, obj2_values = zip(*processed_solutions)
#             # draw the scatterplot
#             plt.figure(figsize=(6, 6))
#             plt.scatter(obj1_values, obj2_values, label='processed non-dominated solutions')
#             # show current solution
#             plt.scatter(pareto_sol_objectives[0], pareto_sol_objectives[1], color='green', label='Current non-dominated solution')
#             plt.xlabel('Objective 1')
#             plt.ylabel('Objective 2')
#             plt.title('Pareto front')
#             plt.xlim(1, 100)
#             plt.ylim(1, 100)
#             plt.legend()
#             plt.grid(True)
#             plt.show()
#             # show route
#             # assign colors
#             edge_colors = {}
#             for u, v in zip(route[:-1], route[1:]):
#                 edge_colors[(u, v)] = 'red'
#                 edge_colors[(v, u)] = 'red'
#             default_edge_color = "gray"
#             # draw the graph
#             plt.figure(figsize=(8, 8))
#             nx.draw(G, node_coords, with_labels=True, node_size=2000, node_color="lightblue", font_size=12,
#                     font_weight="bold")
#             edge_labels = {(u, v): f"{d['weight']}, {d['crosswalk']}" for u, v, d in G.edges(data=True)}
#             for (u, v) in G.edges():
#                 color = edge_colors.get((u, v), default_edge_color)
#                 nx.draw_networkx_edges(G, node_coords, edgelist=[(u, v)], edge_color=color, width=2)
#             nx.draw_networkx_edge_labels(G, node_coords, edge_labels=edge_labels, font_size=10)
#             nx.draw_networkx_nodes(G, node_coords, nodelist=['A'], node_color='yellow', node_size=2500)  # source
#             nx.draw_networkx_nodes(G, node_coords, nodelist=['P'], node_color='orange', node_size=2500)  # target
#             plt.title("4x4 grid graph with highlighted Pareto-optimal paths")
#             plt.show()
#     # Query user input for next objective binary preference
#     while True:
#         objective = input("Choose which objective to decrease or increase: ")
#         if not objective.isdigit():
#             print("Invalid input! Please enter a valid objective (integer).")
#             continue
#         objective = int(objective)
#         if not 0 <= objective < dimensions:  # check bounds of the present dimensions (objectives)
#             print("Invalid input! Please enter a valid objective.")
#             continue
#         print("You chose objective :", objective)
#         direction = input("Enter '-' to decrease or '+' to increase the objective: ")
#         if direction not in ['-', '+']:
#             print("Incorrect direction. Please enter either '-' or '+'.")
#             continue
#         print("You chose direction :", direction)
#         user_input = (objective, direction)
#         break
