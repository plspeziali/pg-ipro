import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import heapq
from pprint import pprint

from sub_gradient_old import subgradient_multiple_routes


def create_test_graph_b(num_nodes=4):
    """
    Creates a test graph with more varied edge characteristics to demonstrate
    different optimal paths under different constraints.
    """
    # Create nodes in a num_nodes x num_nodes grid
    nodes = [f"{i}{j}" for i in range(num_nodes) for j in range(num_nodes)]

    # Generate node coordinates (grid layout)
    node_coords = {
        node: (int(node[0]) * 100, int(node[1]) * 100)
        for node in nodes
    }

    # Initialize empty graph
    graph = {node: {} for node in nodes}

    # Connect nodes horizontally and vertically with significantly different characteristics
    for i in range(num_nodes):
        for j in range(num_nodes):
            current = f"{i}{j}"

            # Horizontal connections
            if j < 3:
                right = f"{i}{j + 1}"
                # Create very different edge characteristics based on position
                if i == 0:  # Top row - good for walking
                    graph[current][right] = (
                        150,  # longer length
                        2,  # more crosswalks
                        0.2,  # excellent walk score
                        1.5  # poor bike score
                    )
                elif i == 3:  # Bottom row - good for biking
                    graph[current][right] = (
                        100,  # standard length
                        0,  # no crosswalks
                        0.5,  # good walk score
                        0.2  # excellent bike score
                    )
                else:  # Middle rows - balanced
                    graph[current][right] = (
                        120,  # medium length
                        1,  # some crosswalks
                        0.8,  # decent walk score
                        0.8  # decent bike score
                    )
                graph[right][current] = graph[current][right]  # Make bidirectional

            # Vertical connections
            if i < 3:
                down = f"{i + 1}{j}"
                # Create very different edge characteristics based on position
                if j == 0:  # Leftmost column - balanced but longer
                    graph[current][down] = (
                        200,  # very long
                        1,  # some crosswalks
                        0.5,  # medium walk score
                        0.5  # medium bike score
                    )
                elif j == 3:  # Rightmost column - short but risky
                    graph[current][down] = (
                        80,  # very short
                        0,  # no crosswalks
                        1.8,  # very poor walk score
                        1.8  # very poor bike score
                    )
                else:  # Middle columns - varied
                    graph[current][down] = (
                        150,  # longer length
                        2 if j == 1 else 0,  # crosswalks only in one column
                        0.3 if j == 1 else 1.2,  # good walk score in one column
                        1.2 if j == 1 else 0.3  # good bike score in other
                    )
                graph[down][current] = graph[current][down]  # Make bidirectional

    return graph, node_coords


# Test cases with adjusted constraints
test_cases = [
    {
        'name': 'Less Crossings Route',
        'constraints': [900, 3, 3, 2.1],
        'weights': [0.5, 2.0, 0.5, 0.5],
    },
    {
        'name': 'Shortest Path',
        'constraints': [900, 20, 20, 20],  # More relaxed on everything except length
        'weights': [2.0, 0.5, 0.5, 0.5],
    },
    {
        'name': 'Pedestrian Friendly',
        'constraints': [1500, 8, 3, 20],  # Strict on walk score, relaxed on bike
        'weights': [1.0, 1.5, 2.0, 0.5],
    },
    {
        'name': 'Bike Friendly',
        'constraints': [1500, 8, 20, 3],  # Strict on bike score, relaxed on walk
        'weights': [1.0, 1.5, 0.5, 2.0],
    }
]


def visualize_graph_with_path(graph, node_coords, path=None, title="Graph Visualization"):
    """
    Display the graph and optionally highlight a path using matplotlib.

    Parameters:
        graph (dict): The graph structure
        node_coords (dict): Coordinates of each node
        path (list): Optional list of nodes representing the path to highlight
        title (str): Title for the plot
    """
    # Create a NetworkX graph
    G = nx.Graph()

    # Add edges with their properties
    for node1 in graph:
        for node2 in graph[node1]:
            edge = graph[node1][node2]
            G.add_edge(
                node1,
                node2,
                weight=edge[0],
                crosswalk=edge[1],
                walk=edge[2],
                bike=edge[3],
                vec=np.array(edge)
            )

    # Create the plot
    plt.figure(figsize=(10, 10))

    # Draw all edges
    nx.draw_networkx_edges(G, node_coords, edge_color='lightgray', width=1)

    # Draw the highlighted path if provided
    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, node_coords, edgelist=path_edges,
                               edge_color='red', width=2)

    # Draw all nodes
    nx.draw_networkx_nodes(G, node_coords, node_color='white',
                           node_size=500, edgecolors='black')

    # Highlight path nodes if provided
    if path:
        nx.draw_networkx_nodes(G, node_coords, nodelist=path,
                               node_color='lightcoral',
                               node_size=500, edgecolors='black')

    # Add node labels
    nx.draw_networkx_labels(G, node_coords)

    # Add edge labels
    edge_labels = {}
    for node1 in graph:
        for node2 in graph[node1]:
            edge = graph[node1][node2]
            edge_labels[(node1, node2)] = f'L: {edge[0]:.0f}\nC: {edge[1]:.0f}\nW: {edge[2]:.1f}\nB: {edge[3]:.1f}'

    nx.draw_networkx_edge_labels(G, node_coords, edge_labels, font_size=10)

    # Set title and remove axes
    plt.title(title, fontsize=16)
    plt.axis('off')


def visualize_test_results(graph, node_coords, routes, test_case_name):
    """
    Visualize each route found for a test case in separate matplotlib windows
    """
    if not routes:
        return

    for i, (cost, path, objectives) in enumerate(routes, 1):
        title = (f"{test_case_name} - Cost: {cost:.2f}\n"
                 f"Cost Vector: [L: {objectives[0]:.2f}, C: {objectives[1]:.2f}, W: {objectives[2]:.2f}, B: {objectives[3]:.2f}]")
        visualize_graph_with_path(graph, node_coords, path, title)
        return


def test_subgradient_algorithm():
    """
    Test the subgradient_multiple_routes algorithm with different constraints
    and visualize the results
    """
    # Create test graph
    graph, node_coords = create_test_graph_b()

    source = "00"
    target = "33"

    visualize_graph_with_path(graph, node_coords, [], "")

    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print("=" * 50)

        routes = subgradient_multiple_routes(
            graph=graph,
            node_coords=node_coords,
            source=source,
            target=target,
            constraints=test_case['constraints'],
            weights=test_case['weights'].copy(),
            max_routes=3,
            max_iter=200,  # Increased iterations
            tol=1e-3,
            turn_penalty=0.5  # Reduced turn penalty to allow more path variation
        )

        if routes:
            for i, (cost, path, objectives) in enumerate(routes, 1):
                print(f"\nRoute {i}:")
                print(f"Path: {' -> '.join(path)}")
                print(f"Total Cost: {cost:.2f}")
                print("Objectives:")
                print(f"  Length: {objectives[0]:.1f}")
                print(f"  Crosswalks: {objectives[1]:.1f}")
                print(f"  Walk Score: {objectives[2]:.1f}")
                print(f"  Bike Score: {objectives[3]:.1f}")
                print(f"  Turn Penalty: {objectives[4]:.1f}")

            # Visualize the routes
            visualize_test_results(graph, node_coords, routes, test_case['name'])
        else:
            print("No feasible routes found")

    plt.show()


if __name__ == "__main__":
    test_subgradient_algorithm()