import itertools
import networkx as nx
from pyproj import Transformer
import random
import hdbscan
import numpy as np

def convert_to_epsg4326(lat, lon):
    # Create a transformer object for converting WGS84 to EPSG:28992
    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    x, y = transformer.transform(lon, lat)  # Note: (lon, lat) order
    return x, y


def get_node_couples(gdf, max_nodes=1000):
    """
    Extract random non-adjacent nodes from a GeoDataFrame.

    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with line geometries

    Returns:
    --------
    list of dict
        List of 10,000 random non-adjacent node couples with their coordinates
    """
    # Create a graph from the GeoDataFrame
    G = nx.MultiGraph()
    for _, row in gdf.iterrows():
        G.add_edge(row['origin'], row['destination'], length=row['length'])

    # Create a mapping of nodes to their coordinates
    node_coords = {}
    for idx, row in gdf.iterrows():
        for node_col in ['origin', 'destination']:
            node = row[node_col]
            if node not in node_coords:
                line = row['geometry']
                point = line.coords[0] if node_col == 'origin' else line.coords[-1]
                x, y = convert_to_epsg4326(point[1], point[0])
                node_coords[node] = {'x': x, 'y': y}

    # Get list of all nodes
    all_nodes = list(G.nodes())
    if not all_nodes:
        return []

    # Generate random non-adjacent node couples
    unique_couples = set()  # Use set to ensure uniqueness
    couples_list = []
    max_attempts = len(all_nodes) * 100  # Prevent infinite loop
    attempts = 0

    while len(couples_list) < max_nodes and attempts < max_attempts:
        # Randomly select two nodes
        node1 = random.choice(all_nodes)
        node2 = random.choice(all_nodes)

        attempts += 1

        # Skip if same node or adjacent nodes
        if (node1 == node2) or (node2 in G.neighbors(node1)):
            continue

        # Create a unique identifier for this couple (sort to avoid duplicates in different order)
        couple_id = tuple(sorted([node1, node2]))

        # Only add if this is a new unique couple
        if couple_id not in unique_couples:
            unique_couples.add(couple_id)
            couples_list.append({
                'node1_id': node1,
                'node1_x': node_coords[node1]['x'],
                'node1_y': node_coords[node1]['y'],
                'node2_id': node2,
                'node2_x': node_coords[node2]['x'],
                'node2_y': node_coords[node2]['y']
            })
            attempts = 0  # Reset attempts counter after successful addition

    return couples_list


def get_significant_node_couples(gdf, max_nodes=10000, min_cluster_size=5, min_samples=1):
    """
    Extract random non-adjacent node couples from a GeoDataFrame after clustering with HDBSCAN.

    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with line geometries.
    max_nodes : int, optional
        Maximum number of node couples to return (default is 10,000).
    min_cluster_size : int, optional
        Minimum cluster size for HDBSCAN (default is 5).
    min_samples : int, optional
        Minimum samples for HDBSCAN clustering (default is 1).

    Returns:
    --------
    list of dict
        List of random non-adjacent node couples with their coordinates.
    """
    # Create a graph from the GeoDataFrame
    G = nx.MultiGraph()
    for _, row in gdf.iterrows():
        G.add_edge(row['origin'], row['destination'], length=row['length'])

    # Create a mapping of nodes to their coordinates
    node_coords = {}
    for idx, row in gdf.iterrows():
        for node_col in ['origin', 'destination']:
            node = row[node_col]
            if node not in node_coords:
                line = row['geometry']
                point = line.coords[0] if node_col == 'origin' else line.coords[-1]
                x, y = convert_to_epsg4326(point[1], point[0])
                node_coords[node] = {'x': x, 'y': y}

    # Perform HDBSCAN clustering on node coordinates
    coordinates = np.array([[v['x'], v['y']] for v in node_coords.values()])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(coordinates)

    # Assign clusters to nodes
    node_clusters = {node: cluster_labels[idx] for idx, node in enumerate(node_coords.keys())}

    # Filter out noise points (label == -1)
    clustered_nodes = [node for node, cluster in node_clusters.items() if cluster != -1]

    # Generate random non-adjacent node couples from different clusters
    unique_couples = set()  # Use set to ensure uniqueness
    couples_list = []
    max_attempts = len(clustered_nodes) * 100  # Prevent infinite loop
    attempts = 0

    while len(couples_list) < max_nodes and attempts < max_attempts:
        # Randomly select two nodes from the clustered nodes
        node1, node2 = random.sample(clustered_nodes, 2)

        attempts += 1

        # Skip if nodes belong to the same cluster or are adjacent in the graph
        if (node_clusters[node1] == node_clusters[node2]) or (node2 in G.neighbors(node1)):
            continue

        # Create a unique identifier for this couple (sort to avoid duplicates in different order)
        couple_id = tuple(sorted([node1, node2]))

        # Only add if this is a new unique couple
        if couple_id not in unique_couples:
            unique_couples.add(couple_id)
            couples_list.append({
                'node1_id': node1,
                'node1_x': node_coords[node1]['x'],
                'node1_y': node_coords[node1]['y'],
                'node2_id': node2,
                'node2_x': node_coords[node2]['x'],
                'node2_y': node_coords[node2]['y'],
                'node1_cluster': node_clusters[node1],
                'node2_cluster': node_clusters[node2]
            })
            attempts = 0  # Reset attempts counter after successful addition

    return couples_list
