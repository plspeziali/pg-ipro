import pandas as pd
from shapely.geometry import MultiPoint
import networkx as nx
from geopandas import GeoDataFrame
import momepy
import ast
from pyproj import Transformer

def calculate_centroid(group):
    # Combine points into a MultiPoint
    multi_point = MultiPoint(group.geometry.tolist())
    return multi_point.centroid

def shortest_distance(row, centroids):
    distances = centroids.distance(row.geometry)  # Calculate distances to all centroids
    return distances.min()  # Keep the shortest distance

def add_node_ids (gdf):
        
    origin = []
    destination = []

    for index, row in gdf.iterrows():

            first = row.geometry.coords[0]
            last = row.geometry.coords[-1]

            origin.append(first)
            destination.append(last)

    gdf['origin'] = origin
    gdf['destination'] = destination

    nodes_df = pd.concat([pd.DataFrame(origin), pd.DataFrame(destination)],axis=0).drop_duplicates(subset=[0,1]).reset_index(drop=True)
    nodes_df = nodes_df.rename(columns={0:"x", 1:"y"})
    nodes_df['node_geometry'] = [(row['x'], row['y']) for index, row in nodes_df.iterrows()]
    nodes_df = nodes_df.reset_index(names=['node'])
    nodes_df['node'] = 'n_'+nodes_df['node'].astype(str)
    node_ids = pd.Series(nodes_df.node.values, index =nodes_df.node_geometry ).to_dict()

    return node_ids

def return_connected_networks (graph):

    #sub_graph = [graph.subgraph(c).copy() for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
    sub_graph = [graph.subgraph(c).copy() for c in sorted(nx.weakly_connected_components(graph), key=len, reverse=True)]
    sub_graph_0 = nx.to_pandas_edgelist(sub_graph[0])
    #sub_graph_1 = nx.to_pandas_edgelist(sub_graph[1]).rename(columns={'source':'origin', 'target':'destination'})
    #sub_loc = pd.concat([sub_graph_0, sub_graph_1]).reset_index(drop=True)
    gdf_sub_loc = GeoDataFrame(sub_graph_0, crs="EPSG:28992", geometry='geometry')

    node_ids = add_node_ids(gdf_sub_loc)

    gdf_sub_loc['origin'] = gdf_sub_loc['origin'].map(node_ids)
    gdf_sub_loc['destination'] = gdf_sub_loc['destination'].map(node_ids)

    gdf_sub_loc = gdf_sub_loc.drop_duplicates(subset=['origin','destination'])
    gdf_sub_loc = gdf_sub_loc.drop(columns=['source','target'])
    gdf_sub_loc = gdf_sub_loc.reset_index(drop=True)

    gdf_sub_loc = gdf_sub_loc[['origin','destination' ,'length','crossing', 'walk','bike', 'walk_bike_connection','walk_pt_connection','distance_to_pt_stops', 'oneway', 'geometry']]

    G_subset = momepy.gdf_to_nx(gdf_sub_loc)
    G_subset = nx.relabel_nodes(G_subset, node_ids, copy=False)

    return gdf_sub_loc, G_subset

def create_node_dict (gdf):
    
    node_dict = {}

    # Step 2: Iterate over each row in the GeoDataFrame to populate the dictionary
    for _, row in gdf.iterrows():
        origin = row['origin']
        destination = row['destination']
        line = row['geometry']
        
        # Extract coordinates of origin and destination from the LineString
        origin_coords = line.coords[0]  # First coordinate in LineString
        destination_coords = line.coords[-1]  # Last coordinate in LineString
        
        # Add the origin node to the dictionary if it's not already there
        if origin not in node_dict:
            node_dict[origin] = origin_coords
        
        # Add the destination node to the dictionary if it's not already there
        if destination not in node_dict:
            node_dict[destination] = destination_coords

    return node_dict

def create_adj_matrix (gdf):

    adj_matrix = {}

    for _, row in gdf.iterrows():
        
        origin = row['origin']
        destination = row['destination']
        #properties = (row['length'], row['crossing'], row['walk'], row['bike'], row['distance_to_pt_stops_band'])
        properties = (row['length'], row['crossing'], row['walk'], row['bike'])
        if origin not in adj_matrix:
            adj_matrix[origin] = {}
        if destination not in adj_matrix:
            adj_matrix[destination] = {}
        
        adj_matrix[origin][destination] = properties
        adj_matrix[destination][origin] = properties

    return adj_matrix

def load_adjacency_matrix_safe(filename):
    adj_matrix = {}
    
    with open(filename, 'r') as f:
        for line_number, line in enumerate(f, start=1):  # Track line numbers for better error messages
            line = line.strip()
            parts = line.split(' ')
            
            if len(parts) >= 3:
                origin = parts[0]
                destination = parts[1]
                
                # Combine the remaining parts into a properties string
                properties_str = ' '.join(parts[2:])
                
                try:
                    # Safely evaluate the string into a Python literal
                    properties = ast.literal_eval(properties_str)
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing properties on line {line_number}: {line}")
                    print(f"Exception: {e}")
                    continue  # Skip this line and proceed to the next
                
                # Add the parsed data to the adjacency matrix
                if origin not in adj_matrix:
                    adj_matrix[origin] = {}
                adj_matrix[origin][destination] = properties

    return adj_matrix


# Convert the coordinates from WGS84 (EPSG:4326) to EPSG:28992
def convert_to_epsg28992(lat, lon):
    # Create a transformer object for converting WGS84 to EPSG:28992
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    x, y = transformer.transform(lon, lat)  # Note: (lon, lat) order
    return x, y

# Find the nearest node
def find_nearest_node(lat, lon, nodes):
    # Geocode the address to get its coordinates (latitude, longitude)

    
    # Convert the coordinates to EPSG:28992
    x, y = convert_to_epsg28992(lat, lon)
    
    # Find the nearest node by calculating the Euclidean distance
    min_distance = float("inf")
    nearest_node = None
    
    for node, coords in nodes.items():
        dist = ((x - coords[0]) ** 2 + (y - coords[1]) ** 2) ** 0.5
        if dist < min_distance:
            min_distance = dist
            nearest_node = node
    
    return nearest_node, min_distance

import networkx as nx
import geopandas as gpd
from shapely.geometry import Point

def convert_to_multidigraph(gdf):
    """
    Convert a GeoDataFrame with edge attributes into a MultiDiGraph.
    
    :param gdf: GeoDataFrame with 'origin', 'destination', 'length', 'crossing', 'bike', 'walk', 'geometry', 'oneway'.
    :return: MultiDiGraph
    """
    G = nx.MultiDiGraph()
    
    # Iterate over rows in the GeoDataFrame
    for _, row in gdf.iterrows():
        u = row['origin']
        v = row['destination']
        length = row['length']
        crossing = row['crossing']
        bike = row['bike']
        walk = row['walk']
        geometry = row['geometry']
        oneway = row['oneway'] == 'true'  # Assuming 'true'/'false' as strings
        
        # Add edges in both directions if oneway is False
        G.add_edge(u, v, length=length, crosswalk=crossing, bike=bike, walk=walk, geometry=geometry, oneway=oneway)
        
        if not oneway:
            G.add_edge(v, u, length=length, crosswalk=crossing, bike=bike, walk=walk, geometry=geometry, oneway=False)
    
    return G

def get_route_gdf(node_list, gdf):

    nodes_df = pd.DataFrame(node_list, columns=['origin'])

    nodes_df['destination'] = nodes_df['origin'].shift(-1)
    nodes_df = nodes_df.dropna()

    gdf_copy = gdf.rename(columns={'destination':'origin', 'origin':'destination'})[gdf.columns]
    route_gdf = pd.concat([gdf, gdf_copy], ignore_index=True).merge(nodes_df, how='right').drop_duplicates(subset=['origin','destination'])

    return route_gdf


def convert_to_epsg4326(lat, lon):
    # Create a transformer object for converting WGS84 to EPSG:28992
    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    x, y = transformer.transform(lon, lat)  # Note: (lon, lat) order
    return x, y


# Find the nearest node
def find_nearest_node(lat, lon, nodes):
    # Geocode the address to get its coordinates (latitude, longitude)

    # Convert the coordinates to EPSG:28992
    x, y = convert_to_epsg28992(lat, lon)

    # Find the nearest node by calculating the Euclidean distance
    min_distance = float("inf")
    nearest_node = None

    for node, coords in nodes.items():
        dist = ((x - coords[0]) ** 2 + (y - coords[1]) ** 2) ** 0.5
        if dist < min_distance:
            min_distance = dist
            nearest_node = node

    return nearest_node, min_distance


import itertools

def get_node_couples(gdf):
    """
        Extract unique nodes with their coordinates from a GeoDataFrame.

        Parameters:
        -----------
        gdf : gpd.GeoDataFrame
            Input GeoDataFrame with line geometries

        Returns:
        --------
        list of dict
            List of unique node couples with their coordinates
        """
    # Create a graph from the GeoDataFrame
    G = nx.MultiGraph()
    for _, row in gdf.iterrows():
        G.add_edge(row['origin'], row['destination'], length=row['length'])

    # Extract unique nodes from origin and destination columns
    unique_nodes = set(gdf['origin'].tolist() + gdf['destination'].tolist())

    # Create a mapping of nodes to their coordinates
    node_coords = {}

    for idx, row in gdf.iterrows():
        # Process both origin and destination for the same row
        for node_col in ['origin', 'destination']:
            node = row[node_col]

            # If node not already mapped, extract its coordinates
            if node not in node_coords:
                # Get the LineString geometry
                line = row['geometry']

                # Determine coordinates based on node type
                if node_col == 'origin':
                    point = line.coords[0]
                else:
                    point = line.coords[-1]

                x, y = convert_to_epsg4326(point[1], point[0])

                node_coords[node] = {
                    'x': x,
                    'y': y
                }

    # Generate unique node couples
    unique_couples = []
    for node1, node2 in itertools.combinations(unique_nodes, 2):
        try:
            shortest_path = nx.shortest_path(G, node1, node2)

            if len(shortest_path) > 10:
                unique_couples.append({
                    'node1_id': node1,
                    'node1_x': node_coords[node1]['x'],
                    'node1_y': node_coords[node1]['y'],
                    'node2_id': node2,
                    'node2_x': node_coords[node2]['x'],
                    'node2_y': node_coords[node2]['y']
                })
            #if len(unique_couples) >= 100:
            #    break
        except nx.NetworkXNoPath:
            continue

    return unique_couples

