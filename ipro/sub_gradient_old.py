import heapq
import math

def subgradient_multiple_routes(graph, node_coords, source, target, constraints, weights, max_routes=5, max_iter=100, tol=1e-3, turn_penalty=1.0):

    """
    Find multiple feasible routes in a graph optimizing multiple objectives
    with user-defined constraints and weights using a subgradient method,
    including a penalty for turns.

    Parameters:
        graph (dict): A dictionary representing the graph.
                      Example: {node1: {node2: (length, crosswalk, walk, bike)}, ...}
        node_coords (dict): A dictionary with node coordinates.
                            Example: {node: (x, y), ...}
        source (str): The starting node.
        target (str): The target node.
        constraints (list): Upper bounds for objectives [length, crosswalk, walk, bike].
        weights (list): Weights for objectives [length, crosswalk, walk, bike].
        max_routes (int): Maximum number of feasible routes to find.
        max_iter (int): Maximum number of iterations for optimization.
        tol (float): Tolerance for constraint violations.
        turn_penalty (float): Penalty multiplier for turn angles.

    Returns:
        list: Feasible routes as tuples (cost, path, objectives).
    """
    def calculate_angle(node1, node2, node3):
        """Calculate the angle (in degrees) between the vectors (node1 -> node2) and (node2 -> node3)."""

        x1, y1 = node_coords[node1]
        x2, y2 = node_coords[node2]
        x3, y3 = node_coords[node3]
        # vectors
        v1 = (x2 - x1, y2 - y1)
        v2 = (x3 - x2, y3 - y2)
        # dot product and magnitudes
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 == 0 or mag2 == 0:  # avoid division by zero
            return 0
        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        angle = math.acos(cos_angle)
        return math.degrees(angle)

    def turn_cost(path):
        """Calculate the total turn penalty for a path."""
        total_penalty = 0
        for i in range(1, len(path) - 1):  # at least 3 nodes to calculate an angle
            angle = calculate_angle(path[i-1], path[i], path[i+1])
            penalty = turn_penalty * (180 - angle) / 180  # normalize penalty to [0, 1]
            total_penalty += penalty
        return total_penalty

    def weighted_cost(node1, node2):
        """Compute the weighted cost of an edge between two nodes."""
        edge = graph[node1][node2]  # edge format: (length, crosswalk, walk, bike)
        return sum(w * e for w, e in zip(weights, edge))

    def shortest_path():
        """Modified Dijkstra's algorithm with current weights"""
        pq = [(0, source, [source])]
        visited = {source: 0}  # Keep track of best costs
        paths = {source: [source]}

        while pq:
            cost, current, path = heapq.heappop(pq)

            if current == target:
                return cost, path

            # Skip if we've found a better path to this node
            if cost > visited[current]:
                continue

            for neighbor in graph.get(current, {}):
                new_cost = cost + weighted_cost(current, neighbor)
                if neighbor not in visited or new_cost < visited[neighbor]:
                    visited[neighbor] = new_cost
                    new_path = path + [neighbor]
                    paths[neighbor] = new_path
                    heapq.heappush(pq, (new_cost, neighbor, new_path))

        return float('inf'), []

    def calculate_objectives(path):
        """Calculate the objectives for a given path."""
        totals = [0] * len(constraints)
        for i in range(len(path) - 1):
            edge = graph[path[i]][path[i+1]]  # edge format: (length, crosswalk, walk, bike)
            for j in range(len(edge)):
                totals[j] += edge[j]
        return totals

    def check_constraints(totals):
        """Check if the path satisfies user-defined constraints."""
        return [totals[i] - constraints[i] for i in range(len(constraints))]

    alpha = 1.0  # initial step size
    feasible_routes = []  # to store feasible routes

    for iteration in range(max_iter):
        cost, path = shortest_path()
        if not path:
            print("No feasible path found.")
            break

        turn_penalty_cost = turn_cost(path)
        total_cost = cost + turn_penalty_cost  # add turn penalty to total cost

        objectives = calculate_objectives(path)
        objectives.append(turn_penalty_cost)  # include turn penalty as part of the objective

        violations = check_constraints(objectives)

        if all(v <= tol for v in violations):
            # store feasible route if it meets constraints
            feasible_routes.append((total_cost, path, objectives))
            feasible_routes = sorted(feasible_routes, key=lambda x: x[0])[:max_routes]

        # update weights using subgradient for violated constraints
        for i in range(len(weights)):
            weights[i] += alpha * max(0, violations[i])

        # includes a penalty for turn cost in the weight update (turn_penalty can be adjusted)
        weights[-1] += alpha * max(0, violations[-1])  # Update the weight for the turn penalty

        alpha *= 0.9  # reduce step size for convergence

        # stop if we've found enough feasible routes
        if len(feasible_routes) >= max_routes:
            break

    if not feasible_routes:
        print("No feasible paths found after optimization.")
        return None
    return feasible_routes
