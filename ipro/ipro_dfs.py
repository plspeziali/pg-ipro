import heapq
import time

from ipro.oracles.oracle import Oracle

from ipro.linear_solvers.linear_solver import LinearSolver




import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

from ipro import helpers as h
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import pickle

import matplotlib.patches as mpatches
import contextily as ctx

adj_matrix_path = "amsterdam_data/adjacency_matrix_osdpm.txt"
gdf_path = "amsterdam_data/gdf_osdpm_connected_pt.gpkg"
node_dict_path = "amsterdam_data/node_dict_osdpm.pickle"

graph = h.load_adjacency_matrix_safe(adj_matrix_path)
gdf = gpd.read_file(gdf_path)
with open(node_dict_path, 'rb') as f:
    node_dict = pickle.load(f)

gdf_viz = gpd.read_file('amsterdam_data/gdf_osdpm_connected.gpkg')

origin_lat = 52.351699 #52.352000 #52.350758  #mvp1 - 52.350016
origin_lon = 4.800223 #4.799500 #4.798904    #mvp1 - 4.797826
destination_lat = 52.352097 #52.352700 #52.357348 #mvp1 - 52.362954
destination_lon = 4.798464 #4.798500 #4.793762 #mvp1- 4.793823




class DijkstraSolver(LinearSolver):
    def __init__(self, graph, source, target, dimensions, upper_bounds=None, lower_bounds=None, shortest_paths=None):
        super().__init__(problem=graph, direction="minimize")
        self.graph = graph
        self.source = source
        self.target = target
        self.dimensions = dimensions
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.shortest_paths = shortest_paths

    def weighter(self, weight_vec, edge_dict: dict):
        return np.dot(edge_dict['vec'], weight_vec)

    def solve(self, weight_vec, precomputed=True):
        """Every edge has a cost, let's make the dot product of the vector cost to make it into a scalar cost"""
        if np.sum(weight_vec) < 0:
            bounds = np.array(list(self.upper_bounds))
            return bounds, []
        elif precomputed and np.any(weight_vec == 1):
            bounds = np.array(list(next(lb for lb, w in zip(self.lower_bounds, weight_vec) if w == 1)))
            shortest_path = next(sp for sp, w in zip(self.shortest_paths, weight_vec) if w == 1)
            return bounds, shortest_path
        else:  # a more balanced weighted approach to find a solution
            path = nx.dijkstra_path(self.graph, self.source, self.target, weight=lambda u, v, d: self.weighter(weight_vec, d))
            vec_return = np.sum([self.graph[path[i]][path[i + 1]]['vec'] for i in range(len(path) - 1)], axis=0)
            print("Linear solver vec_return: ", vec_return)
            return vec_return, path


class DFSOracle(Oracle):
    def __init__(self, graph, source, target, nr_objectives, lower_bounds_algorithm="reverse_dijkstra", problem=None,
                 seed=None):
        super().__init__(problem=problem, seed=seed)
        self.graph = graph
        self.source = source
        self.target = target
        self.nr_objectives = nr_objectives  # IPRO dimensions
        self.lower_bounds_algorithm = lower_bounds_algorithm

    def manhattan_distance(self, objectives1, objectives2):
        """Calculate the Manhattan distance of the values between objective vectors."""
        return sum(abs(obj1 - obj2) for obj1, obj2 in zip(objectives1, objectives2))

    def single_objective_value_iteration(self, objective):
        """Calculate lower bounds for the objective for each node in the graph."""
        node_values = {self.target: 0} #  lower_bound per node
        done = False
        while not done:
            done = True
            for node in self.graph.keys():
                if node == self.target:
                    continue
                new_cost = min(self.graph[node][neighbor][objective] + node_values.get(neighbor, float('inf')) for neighbor in self.graph.get(node, {}))
                if new_cost < node_values.get(node, float('inf')):
                    done = False
                    node_values[node] = new_cost
        return node_values

    def reverse_dijkstra(self, objective):
        """Calculate lower bounds for the objective for each node in the graph, starting from the target node."""
        node_values = {} #  lower_bound per node
        pq = [(0, self.target)]
        while pq:
            cost, current = heapq.heappop(pq)
            if current in node_values:
                continue
            node_values[current] = cost
            for neighbor in self.graph.get(current, {}):
                if neighbor not in node_values:
                    edge_costs = self.graph[current][neighbor]
                    new_cost = cost + edge_costs[objective]
                    heapq.heappush(pq, (new_cost, neighbor))
        return node_values

    def dijkstra_shortest_path_with_ideal_guidance(self, objective_i, ideal):
        """Dijkstra's algorithm for shortest path with primary objective_i.
        Uses Manhattan distance to ideal as tie-breaker among equal-cost options."""

        match self.lower_bounds_algorithm:  # calculate the lowest value for an objective to get to the goal node from a specific node
            case "reverse_dijkstra":
                lower_bnds = [self.reverse_dijkstra(objective) for objective in range(self.nr_objectives)]
            case "single_objective_value_iteration":
                lower_bnds = [self.single_objective_value_iteration(objective) for objective in range(self.nr_objectives)]
            case _:
                raise ValueError(f'Unknown lower_bounds_algorithm: {self.lower_bounds_algorithm}')

        initial_path_cost = np.zeros(self.nr_objectives)
        lower_bounds_cost = np.array([lower_bnd[self.source] for lower_bnd in lower_bnds])
        distance_to_ideal = np.sum(np.abs(ideal - (initial_path_cost + lower_bounds_cost)))

        # Priority queue: (objective_i_cost, distance_to_ideal, current, path, full_cost_vec)
        pq = [(0, distance_to_ideal, self.source, [self.source], initial_path_cost)]
        heapq.heapify(pq)
        visited = {}

        while pq:
            # Always sort by objective_i first, then distance to ideal (acts like min-heap)
            #pq.sort(key=lambda x: (x[0], x[1]))
            #obj_cost, dist, current, path, cost_vec = pq.pop(0)
            obj_cost, dist, current, path, cost_vec = heapq.heappop(pq)

            if current == self.target:
                return cost_vec, path

            if current in visited:
                prev_cost_vec = visited[current]
                if np.all(cost_vec >= prev_cost_vec):
                    continue  # Already visited with a better or equal cost vector

            visited[current] = cost_vec

            for neighbor in self.graph.get(current, {}):
                edge_cost = np.array(self.graph[current][neighbor][:self.nr_objectives])
                new_cost_vec = cost_vec + edge_cost

                # Get lower bound estimate from neighbor to target
                lower_bounds_cost = np.array([lower_bnd[neighbor] for lower_bnd in lower_bnds])
                estimate_total_cost = new_cost_vec + lower_bounds_cost

                # Manhattan distance to ideal (heuristic)
                distance = np.sum(np.abs(ideal - estimate_total_cost))
                obj_cost = new_cost_vec[objective_i]

                heapq.heappush(pq, (obj_cost, distance, neighbor, path + [neighbor], new_cost_vec))
                #pq.append((obj_cost, distance, neighbor, path + [neighbor], new_cost_vec))

        return np.full(self.nr_objectives, np.inf), []

    ''' def calculate_objectives(self, path):
        """Calculate the objectives for a given path."""
        totals = [0] * self.nr_objectives
        for i in range(len(path) - 1):
            edge = self.graph[path[i]][path[i+1]]
            length = edge[2] + edge[3]  # Recalculate length as walk + bike
            adjusted_edge = (length,) + edge[1:]
            for j in range(len(adjusted_edge)):
                totals[j] += adjusted_edge[j]
        return totals
    '''
    def calculate_objectives(self, path):
        """Calculate the objectives for a given path."""
        totals = [0] * self.nr_objectives
        for i in range(len(path) - 1):
            edge = self.graph[path[i]][path[i + 1]]#['vec'] # edge format: (length, crosswalk, walk, bike)
            for j in range(self.nr_objectives):
                totals[j] += edge[j]
        return totals

    def pareto_dominate(self, objectives1, objectives2):
        """Returns True if objectives1 pareto-dominates objectives2."""
        dominates = False
        for obj1, obj2 in zip(objectives1, objectives2):
            if obj1 > obj2:
                return False
            if obj1 < obj2:
                dominates = True
        return dominates

    def strict_pareto_dominate(self, objectives1, objectives2):
        """Returns True if objectives1 strictly pareto-dominates objectives2."""
        for obj1, obj2 in zip(objectives1, objectives2):
            if obj1 >= obj2:
                return False
        return True

    def mo_depth_first_search(self, lower_bnds, target_obj, upper_bnd, target_region_ideal):
        """Multi-objective DFS with lower bound pruning and Pareto-aware visited control."""

        current_best_path = []
        current_best_cost = np.array(upper_bnd)
        initial_path_cost = np.zeros(self.nr_objectives)
        stack = [(initial_path_cost, self.source, [self.source])]
        visited = {self.source: [initial_path_cost]}

        while stack:
            cost, current, path = stack.pop()

            if current == self.target:
                if np.all(np.less_equal(cost, current_best_cost)):
                    current_best_cost = cost.copy()
                    current_best_path = path.copy()
                continue

            neighbor_list = []
            for neighbor in self.graph.get(current, {}):
                edge_cost = np.array(self.graph[current][neighbor][:self.nr_objectives])
                new_cost = cost + edge_cost

                # compute lower bound estimate from neighbor to goal
                lower_bounds_cost = np.array([lower_bnd[neighbor] for lower_bnd in lower_bnds])
                estimate_total_cost = new_cost + lower_bounds_cost

                # prune paths that can't beat the current best cost, and are strict better than the initial Referent/upper_bnd
                if np.any(np.greater(estimate_total_cost, current_best_cost)) or np.any(upper_bnd == estimate_total_cost):
                    continue

                # pareto check (skip if new_cost is dominated or a duplicate)
                is_dominated = False
                for prev_cost in visited.get(neighbor, []):
                    if np.all(np.less_equal(prev_cost, new_cost)):
                        is_dominated = True
                        break
                if is_dominated:
                    continue

                # remove worse ones
                visited.setdefault(neighbor, [])
                visited[neighbor] = [c for c in visited[neighbor] if not np.all(np.less_equal(new_cost, c))]
                visited[neighbor].append(new_cost)

                # Manhattan distance heuristic to guide DFS toward ideal point
                #denom = np.maximum(nadir - target_obj, 1e-8)  # avoid division by zero
                #normalized_diff = np.abs(estimate_total_cost - target_obj) / denom
                #distance = np.sum(normalized_diff)

                # Chebyshev scalarization heuristic to guide DFS with balance toward ideal point
                denom = np.maximum(upper_bnd - target_region_ideal, 1e-8)  # avoid division by zero
                normalized_diff = np.abs(estimate_total_cost - target_region_ideal) / denom
                distance = np.max(normalized_diff)
                neighbor_list.append((new_cost, neighbor, distance, path + [neighbor]))

            # Sort neighbors by estimated closeness to ideal (the lowest distance first → stack top)
            neighbor_list.sort(key=lambda x: x[2], reverse=True)
            for new_cost, neighbor, _, new_path in neighbor_list:
                stack.append((new_cost, neighbor, new_path))

        return current_best_cost, current_best_path

    def solve(self, referent, nadir, ideal, upper_point_target_region):
        """The inner loop solver for the multi-objective path finding setting.

                Args:
                    referent (np.ndarray): The reference vector.

                Returns:
                    np.ndarray: The Pareto optimal vector.
                    list: A new optimal route.
                """
        start_time = time.time()

        match self.lower_bounds_algorithm: # calculate the lowest value for an objective to get to the goal node from a specific node
            case "reverse_dijkstra":
                lower_bounds = [self.reverse_dijkstra(objective) for objective in range(self.nr_objectives)]
            case "single_objective_value_iteration":
                lower_bounds = [self.single_objective_value_iteration(objective) for objective in
                                range(self.nr_objectives)]
            case _:
                raise ValueError(f'Unknown lower_bounds_algorithm: {self.lower_bounds_algorithm}')
        #print("Lower bounds calculation done.")

        objectives_cost, optimal_route = self.mo_depth_first_search(lower_bounds, ideal, referent, upper_point_target_region)
        #print("Done Oracle search!")

        # IPRO expects None from Oracle if no Pareto optimal solution exists in the target region of a given referent:
        optimal_route = None if optimal_route == [] else optimal_route

        duration = time.time() - start_time

        return np.array(objectives_cost), optimal_route



'''
    def old_dfs_path(self, lower_bnds, target_obj, upper_bnd):
        """Depth first search algorithm for finding a pareto-optimal path/route between target (ideal) and upper bound (referent) objective."""
        #visited = set()
        #visited_costs = {}
        current_best_path = []  # current best path
        current_best_cost = upper_bnd
        stack = [([0] * self.nr_objectives, self.source, [self.source])]
        while stack:
            print("stack size: ", len(stack))
            cost, current, path = stack.pop()

            #if current in visited:
            #    continue
            #existing_costs = visited_costs.get(current, [])
            #if any(self.pareto_dominate(existing_cost, cost) for existing_cost in existing_costs):
            #    continue

            # Update visited_costs at current node with Pareto filtering
            new_pareto_set = []
            dominated = False
            for existing in existing_costs:
                if self.pareto_dominate(cost, existing):
                    continue  # new cost dominates existing, skip it
                if self.pareto_dominate(existing, cost):
                    dominated = True
                    break  # new cost is dominated, skip adding it
                new_pareto_set.append(existing)
            if dominated:
                continue
            new_pareto_set.append(cost)
            visited_costs[current] = new_pareto_set

            if current == self.target:
                print("Target reached!", path)
                if self.strict_pareto_dominate(cost, current_best_cost):
                    print("Target reached and FOUND NEW strict pareto dominant PATH!", path)
                    current_best_cost = cost
                    current_best_path = path.copy()
                continue
            #visited.add(current)
            #       visited_costs.setdefault(current, []).append(cost)
            neighbor_list = []
            for neighbor in self.graph.get(current, {}):
                #if neighbor not in visited:
                new_edge_cost = self.graph[current][neighbor]
                lower_bounds_cost = (lower_bnd[neighbor] for lower_bnd in lower_bnds)
                # calculate result cost indicating if this is a viable option to still become a (strict) pareto dominant optimal solution
                result = tuple(c1 + c2 + c3 for c1, c2, c3 in zip(cost, tuple(new_edge_cost), lower_bounds_cost))  # this is new lower bound
                # Prune paths that will not improve the current upper bound/current_best_cost:
                if not self.strict_pareto_dominate(result, current_best_cost):
                    continue
                distance = self.manhattan_distance(target_obj, result)
                new_cost = tuple(c1 + c2 for c1, c2 in zip(cost, tuple(new_edge_cost)))
                # Optional: prune dominated neighbor expansions
                #if any(self.pareto_dominate(existing_cost, new_cost) for existing_cost in visited_costs.get(neighbor, [])):
                #    continue
                neighbor_list.append((new_cost, neighbor, distance))
            # Add neighbours (new paths) on stack based on lowest distance at the top
            neighbor_list_sorted = sorted(neighbor_list, key=lambda x: x[2], reverse=True)
            for curr_new_cost, neighbor, distance in neighbor_list_sorted:
                new_path = path + [neighbor]
                stack.append((curr_new_cost, neighbor, new_path))
        return current_best_cost, current_best_path
'''

