import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import networkx as nx

from ipro.ipro_dfs import DFSOracle, DijkstraSolver
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
user_input = [0,'']
all_solutions_found = []

while True:
    print("\n=== Starting new IPRO iteration ===")
    print(f"Current number of lower points to explore: {len(ipro_amsterdam.lower_points)}")

    # Check if there are still lower points to explore
    #if len(ipro_amsterdam.lower_points) == 0:
        #print("No more lower points to explore. All regions exhausted!")
        #break

    subsolution = ipro_amsterdam.solve(return_inter_sol=False, user_input=user_input)

    if subsolution is None:
        print("User preferences can't be improved, no better solution exists in this direction.")
        print("Trying a different region...")
        user_input = None  # Reset to allow exploring a different region
        continue

    elif len(subsolution) != 1:
        # Full Pareto front returned - but we can continue exploring
        current_pf = subsolution
        print(f"Current Pareto front size: {len(current_pf)}")
        objectives_vectors = [pf_sol[0] * -1 for pf_sol in current_pf]
        print("Pareto front objective vectors: ", objectives_vectors)

        # Store all solutions found so far
        all_solutions_found.extend(current_pf)

        # Visualize current state
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf_viz.plot(ax=ax, color='grey', alpha=0.001)
        ctx.add_basemap(ax, crs=gdf_viz.crs, source=ctx.providers.CartoDB.Voyager, zoom=15)
        ax.scatter(node_dict[source][0], node_dict[source][1], color='blue', s=20, zorder=5, label="Source")
        ax.scatter(node_dict[target][0], node_dict[target][1], color='red', s=20, zorder=5, label="Destination")
        colors = cm.get_cmap('tab20', len(current_pf))
        legend_handles = [
            mpatches.Patch(color='blue', label='Origin'),
            mpatches.Patch(color='red', label='Destination')
        ]

        for idx, pf_sol in enumerate(current_pf):
            cost_vector = pf_sol[0] * -1
            path = pf_sol[1]
            color = colors(idx)
            h.get_route_gdf(path, gdf_viz).plot(ax=ax, color=color, linewidth=2, alpha=0.8)
            label = f"Route {idx + 1}: {np.round(cost_vector, 2)}"
            legend_handles.append(mpatches.Patch(color=color, label=label))

        plt.legend(handles=legend_handles, loc='lower right', fontsize=6)
        plt.rcParams["legend.fontsize"] = 5
        plt.xlim(114000, 115500)
        plt.ylim(484600, 486400)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Show scatter plot
        processed_solutions = [pf_sol[0] * -1 for pf_sol in current_pf]
        if len(processed_solutions) >= 2:
            obj1_values, obj2_values = zip(*[(sol[0], sol[1]) for sol in processed_solutions])
            plt.figure(figsize=(6, 6))
            plt.scatter(obj1_values, obj2_values, label='Current solutions')
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            plt.title(f'Current Pareto front ({len(processed_solutions)} solutions)')
            plt.legend()
            plt.grid(True)
            plt.show()

        print(f"\nRemaining lower points to explore: {len(ipro_amsterdam.lower_points)}")

        # Ask if user wants to continue exploring other regions
        if len(ipro_amsterdam.lower_points) > 0:
            continue_choice = input("\nContinue exploring other regions? (yes/no): ").strip().lower()
            if continue_choice != 'yes':
                break
            user_input = None  # Reset for new exploration
            continue
        else:
            print("All regions explored!")
            break

    else:
        # Single solution found
        pareto_sol_objectives = subsolution[0][0] * -1
        path = subsolution[0][1]
        print("route: ", path)

        if path is None:
            print("Current referent can't find a subsolution, trying next referent...")
            user_input = None  # Reset to try different direction
            continue

        # Visualize the current solution
        print(f"Current solution objectives: {pareto_sol_objectives}")

        fig, ax = plt.subplots(figsize=(10, 8))
        gdf_viz.plot(ax=ax, color='grey', alpha=0.001)
        ctx.add_basemap(ax, crs=gdf_viz.crs, source=ctx.providers.CartoDB.Voyager, zoom=15)
        ax.scatter(node_dict[source][0], node_dict[source][1], color='blue', s=50, zorder=5, label="Origin")
        ax.scatter(node_dict[target][0], node_dict[target][1], color='red', s=50, zorder=5, label="Destination")
        h.get_route_gdf(path, gdf_viz).plot(ax=ax, color='green', linewidth=3, alpha=0.8, label='Current Route')

        plt.legend(loc='lower right', fontsize=8)
        plt.title(f'Current Route - Objectives: {np.round(pareto_sol_objectives, 2)}', fontsize=10)
        plt.xlim(114000, 115500)
        plt.ylim(484600, 486400)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Query user input for next objective binary preference
    while True:
        print(f"\nCurrent objectives: {pareto_sol_objectives if 'pareto_sol_objectives' in locals() else 'N/A'}")
        print("Available objectives: 0 (length), 1 (crossing), 2 (walk), 3 (bike)")
        objective_input = input(
            "Choose which objective to decrease/increase (or 'skip' for different region, 'exit' to quit): ")

        if objective_input.lower() == 'exit':
            print("Exiting interaction loop.")
            user_input = None
            break

        if objective_input.lower() == 'skip':
            print("Skipping to explore a different region...")
            user_input = None
            break

        if not objective_input.isdigit():
            print("Invalid input! Please enter a valid objective (integer), 'skip', or 'exit'.")
            continue

        objective = int(objective_input)
        if not 0 <= objective < dimensions:
            print(f"Invalid input! Please enter an objective between 0 and {dimensions - 1}.")
            continue

        print(f"You chose objective: {objective}")

        direction = input("Enter '-' to decrease or '+' to increase the objective: ")
        if direction not in ['-', '+']:
            print("Incorrect direction. Please enter either '-' or '+'.")
            continue

        print(f"You chose direction: {direction}")
        user_input = (objective, direction)
        break

    # Exit if user chose to exit
    if objective_input.lower() == 'exit':
        break

# Final summary
print("\n=== FINAL SUMMARY ===")
print(f"Total unique solutions found: {len(all_solutions_found)}")
if all_solutions_found:
    print("All solution objectives:")
    for idx, (vec, path) in enumerate(all_solutions_found):
        print(f"  Solution {idx + 1}: {vec * -1}")