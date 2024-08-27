# %%
import numpy as np 
import networkx as nx
import pickle
import cv2
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance_matrix

from sys import stdout

from fatbox.preprocessing import *
from fatbox.metrics import *
from fatbox.edits import *
from fatbox.plots import *

import cartopy.crs as ccrs
import pyvista as pv
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cmcrameri.cm as cmc
import geovista as gv
from geovista.common import to_cartesian
from skimage import measure
from scipy.interpolate import griddata
from cartopy.util import add_cyclic_point

from joblib import Parallel, delayed
import networkx as nx
import cartopy.util as cutil
from itertools import cycle

import os
import re

import matplotlib.patches as patches

from ipywidgets import Layout, interactive, widgets
from tqdm import tqdm

from fatbox.metrics import total_length, get_fault, get_fault_labels
from fatbox.plots import plot_components, plot_attribute, plot_faults
from matplotlib.patches import Polygon


def latlon_to_cartesian(lat, lon):
    """
    Convert latitude and longitude to Cartesian coordinates.
    """
    lat, lon = np.radians(lat), np.radians(lon)
    x = 6371000 * np.cos(lat) * np.cos(lon)
    y = 6371000 * np.cos(lat) * np.sin(lon)
    z = 6371000 * np.sin(lat)
    return x, y, z

# Function to create a line for each edge
def create_line(edge, cartesian_positions):
    pos1 = cartesian_positions[edge[0]]
    pos2 = cartesian_positions[edge[1]]
    return pv.Line(pos1, pos2)

# Function to connect components across the dateline
def connect_across_dateline(G, threshold=2):
    pos = nx.get_node_attributes(G, 'pos')
    components = nx.get_node_attributes(G, 'component')

    positions = np.array([pos[node] for node in G.nodes])
    nodes = np.array(list(G.nodes))

    left_edge_nodes = nodes[positions[:, 0] < 5]  # Near -180 degrees
    right_edge_nodes = nodes[positions[:, 0] > 355]  # Near 180 degrees

    left_positions = positions[positions[:, 0] < 5]
    right_positions = positions[positions[:, 0] > 355]

    if len(left_positions) > 0 and len(right_positions) > 0:
        left_positions[:, 0] += 360  # Shift to the right side for distance calculation
        dm = distance_matrix(left_positions, right_positions)

        for i, left_node in enumerate(left_edge_nodes):
            for j, right_node in enumerate(right_edge_nodes):
                if dm[i, j] < threshold:
                    G.add_edge(left_node, right_node)
                    print(f"Connecting {left_node} to {right_node} across the dateline with distance {dm[i, j]}")

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth using the Haversine formula.
    """
    # Earth radius in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance

# Check the existence of a specific edge
def check_edge_existence(G, u, v):
    if u in G.nodes and v in G.nodes:
        if (u, v) in G.edges:
            return G.edges[(u, v)]
        else:
            return f"Edge ({u}, {v}) does not exist."
    else:
        return f"One or both of the nodes {u} and {v} do not exist."
    

def create_line_with_length(edge, cartesian_positions, edge_length):
    start, end = cartesian_positions[edge[0]], cartesian_positions[edge[1]]
    line = pv.Line(start, end)
    line['length'] = edge_length
    return line

# Function to calculate the strike of an edge on a sphere
def calculate_strike(pos1, pos2):
    # Convert positions from Cartesian to spherical coordinates
    lat1 = np.degrees(np.arcsin(pos1[2] / 6371000))
    lon1 = np.degrees(np.arctan2(pos1[1], pos1[0]))
    lat2 = np.degrees(np.arcsin(pos2[2] / 6371000))
    lon2 = np.degrees(np.arctan2(pos2[1], pos2[0]))

    # Calculate the strike angle
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    angle_rad = math.atan2(d_lon, d_lat)
    angle_deg = math.degrees(angle_rad)

    # Normalize the strike to be within [0, 360) degrees
    strike = angle_deg if angle_deg >= 0 else angle_deg + 360

    # Normalize the strike to be within [0, 180) degrees with 0 degrees meaning north
    if strike > 180:
        strike -= 180

    return strike


def plot_edge_attribute_3d(G, attribute,cartesian_positions):

    # Calculate the attribute values for each edge
    attributes = np.array([G.edges[edge][attribute] for edge in G.edges])

    # Normalize attribute values for colormap
    norm = plt.Normalize(vmin=attributes.min(), vmax=attributes.max())
    cmap = plt.cm.viridis

    # Generate colors based on attributes
    edge_colors = [cmap(norm(attr)) for attr in attributes]

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Add a sphere to represent the Earth
    sphere = pv.Sphere(radius=6371000, theta_resolution=360, phi_resolution=180)
    plotter.add_mesh(sphere, color='white', opacity=0.6, style='wireframe')

    # Add nodes to the plotter
    points = np.array([cartesian_positions[node] for node in G.nodes])
    point_cloud = pv.PolyData(points)
    plotter.add_mesh(point_cloud, color='red', point_size=0.1, render_points_as_spheres=True)

    # Parallelize the creation of lines for edges
    lines = Parallel(n_jobs=num_cores)(
        delayed(create_line)(edge, cartesian_positions) for edge in G.edges
    )

    # Add the lines to the plotter with corresponding colors
    for i, line in enumerate(lines):
        plotter.add_mesh(line, color=edge_colors[i], line_width=5.0)

    # Add a scalar bar for attributes
    dummy_scalar_array = np.linspace(attributes.min(), attributes.max(), len(G.edges))
    dummy_points = np.random.rand(len(dummy_scalar_array), 3)  # Dummy points just for the scalar bar
    plotter.add_mesh(pv.PolyData(dummy_points), scalars=dummy_scalar_array, cmap='viridis', show_scalar_bar=True)

    # Show the plot
    plotter.show()


# def plot_edge_attribute(G, attribute, ax=None):
#         if ax is None:
#             fig, ax = plt.subplots()

#         pos = nx.get_node_attributes(G, 'pos')
#         edges = np.array([G.edges[edge][attribute] for edge in G.edges])

#         cmap = plt.cm.twilight_shifted
#         # norm = plt.Normalize(vmin=compute_edge_values(G, attribute, 'min'), vmax=compute_edge_values(G, attribute, 'max'))
#         norm = plt.Normalize(vmin=edges.min(), vmax=edges.max())

#         nx.draw(G, pos=pos, node_size=0.001, ax=ax)
#         nx.draw_networkx_edges(G, pos=pos, edge_color=edges, edge_cmap=cmap, edge_vmin=norm.vmin, edge_vmax=norm.vmax, ax=ax)
#         ax.axis('equal')

#         # Colorbar
#         sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#         sm.set_array([])

#         cbar = plt.colorbar(sm, ax=
#         ax, fraction=0.046, pad=0.04)
#         cbar.ax.set_ylabel(attribute, rotation=270)
    
def plot_edge_attribute(G, attribute, current_time, lon_grid_shape, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))  # Define the figure size

    # Extract positions from the graph nodes and convert to latitude/longitude
    pos = nx.get_node_attributes(G, 'pos')
    cartesian_positions = {}
    points = []

    for node in G.nodes:
        lon, lat = G.nodes[node]['pos']
        lon = (lon / lon_grid_shape[1]) * 360 - 180
        lat = (lat / lon_grid_shape[0]) * 180 - 90
        cartesian_positions[node] = (lon, lat)
        points.append((lon, lat))

    edges = np.array([G.edges[edge][attribute] for edge in G.edges])

    cmap = plt.cm.twilight_shifted
    norm = plt.Normalize(vmin=edges.min(), vmax=edges.max())

    # Plot latitude and longitude grid
    ax.set_xticks(np.linspace(-180, 180, num=7))
    ax.set_yticks(np.linspace(-90, 90, num=7))
    ax.set_xlim([-180, 180])  # Set longitude limits
    ax.set_ylim([-90, 90])    # Set latitude limits
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)

    # Draw the graph edges with strike attribute
    for edge in G.edges:
        x = [cartesian_positions[edge[0]][0], cartesian_positions[edge[1]][0]]
        y = [cartesian_positions[edge[0]][1], cartesian_positions[edge[1]][1]]
        ax.plot(x, y, color=cmap(norm(G.edges[edge][attribute])), lw=0.5)

    ax.axis('equal')
    title_str = f"Plate boundaries strike\nTime: {current_time / 1e6:.2f} My"
    ax.set_title(title_str, fontsize=18)

    # Colorbar with better positioning and size
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.set_ylabel(attribute, rotation=270, labelpad=20, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()  # Adjust the layout to ensure everything fits well



def plot_rose(strikes, lengths=[], ax=[]):    
    
    if lengths ==[]:
        lengths = np.ones_like(np.array(strikes))

    bin_edges = np.arange(-5, 366, 10)
    number_of_strikes, bin_edges = np.histogram(strikes, bin_edges, weights = lengths)           
    number_of_strikes[0] += number_of_strikes[-1]
    half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
    two_halves = np.concatenate([half, half])
    
    cmap = plt.cm.twilight_shifted(np.concatenate((np.linspace(0, 1, 18), np.linspace(0, 1, 18)), axis=0))
    
    if ax==[]:
        fig = plt.figure(figsize=(8,8))
            
        ax = fig.add_subplot(111, projection='polar')
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))
    
    ax.bar(np.deg2rad(np.arange(0, 360, 10)), two_halves, 
        width=np.deg2rad(10), bottom=0.0, color=cmap, edgecolor='k')
    
    #    ax.set_rgrids(np.arange(1, two_halves.max() + 1, 2), angle=0, weight= 'black')
    ax.set_title('Rose Diagram', y=1.10, fontsize=15)
    
    # fig.tight_layout()


def get_nodes(G):
    labels = get_fault_labels(G)
    point_set=[]
    for label in labels:            
        G_fault = get_fault(G, label)
        points = []
        for node in G_fault:
            points.append(G_fault.nodes[node]['pos'])
        point_set.append(points)
    return point_set


def compute_similarity(set_A, set_B):
      distances = np.zeros((len(set_A), len(set_B)))
      for n, pt_0 in enumerate(set_A):
          for m, pt_1 in enumerate(set_B):
              distances[n,m] = math.sqrt((pt_0[0]-pt_1[0])**2 + (pt_0[1]-pt_1[1])**2)
      return np.mean(np.min(distances, axis=1))

 
def correlation_slow(G_0, G_1, R):
    # A function which labels the faults in G_1 according to G_0 using the 
    # minimum radius R
    
    
    # Get labels and nodes
    fault_labels_0 = get_fault_labels(G_0)
    fault_labels_1 = get_fault_labels(G_1)
    
    nodes_0 = get_nodes(G_0)
    nodes_1 = get_nodes(G_1) 


    # Compute similarities    
    smf = np.zeros((len(fault_labels_0), len(fault_labels_1)))
    smb = np.zeros((len(fault_labels_1), len(fault_labels_0)))    
    
    
    for n in tqdm(range(len(fault_labels_0)), desc='   Compute similarities'):
        for m in range(len(fault_labels_1)):
            smf[n,m] = compute_similarity(nodes_0[n], nodes_1[m])
            smb[m,n] = compute_similarity(nodes_1[m], nodes_0[n])
            
            
    # Determine correlations
    correlations = set()
    for n in tqdm(range(len(fault_labels_0)), desc='   Find correlations'):
        for m in range(len(fault_labels_1)):
            if smf[n,m] < R:
                correlations.add((fault_labels_0[n], fault_labels_1[m]))
            if smb[m,n] < R:
                correlations.add((fault_labels_0[n], fault_labels_1[m]))                 

    return correlations, smf, smb

##This count and assign the total number of fault
def assign_fault_labels(G):
    # Iterate over each connected component and assign a unique fault label
    for jj, component in enumerate(nx.connected_components(G)):
        for node in component:
            G.nodes[node]['fault'] = jj  # Assign the component index as the fault label


# Function to count the number of unique faults
def count_faults(G):
    fault_labels = set(nx.get_node_attributes(G, 'fault').values())
    return len(fault_labels)


# Function to calculate direction vectors using geographic positions
def calculate_direction(G, cutoff,geographic_positions, normalize=True):
    for node in G.nodes:
        length = nx.single_source_shortest_path_length(G, node, cutoff=cutoff)
        keys = [key for key, value in length.items() if value == max(length.values())]

        if len(keys) > 2:
            node_0, node_1 = keys[:2]
        elif len(keys) == 2:
            node_0, node_1 = keys
        elif len(keys) == 1:
            node_0 = keys[0]
            length = nx.single_source_shortest_path_length(G, node, cutoff=cutoff-1)
            keys = [key for key, value in length.items() if value == max(length.values())]
            node_1 = keys[0]

        # Extract geographic positions
        pt_0 = geographic_positions[node_0]  # (lon, lat)
        pt_1 = geographic_positions[node_1]  # (lon, lat)

        # Convert lat/lon from degrees to radians
        lat1 = np.radians(pt_0[1])
        lon1 = np.radians(pt_0[0])
        lat2 = np.radians(pt_1[1])
        lon2 = np.radians(pt_1[0])

        # Calculate differences
        dlon = lon2 - lon1

        # Calculate the bearing
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))

        bearing = np.arctan2(x, y)  # Bearing in radians

        # Convert bearing to degrees and normalize
        bearing_degrees = np.degrees(bearing)
        bearing_degrees = (bearing_degrees + 360) % 360  # Normalize to 0-360 degrees

        # Optionally normalize to a unit vector
        if normalize:
            dx = np.cos(np.radians(bearing_degrees))
            dy = np.sin(np.radians(bearing_degrees))
        else:
            dx = x
            dy = y

        # Write to graph
        G.nodes[node]['dx'] = dx
        G.nodes[node]['dy'] = dy

    return G

# Function to calculate pickup points using geographic positions
def calculate_pickup_points(G, factor,geographic_positions):
    H = nx.Graph()

    for node in G.nodes:
        lon, lat = geographic_positions[node]  # Current position of the node (longitude, latitude)

        dx = G.nodes[node]['dx']  # Direction vector dx
        dy = G.nodes[node]['dy']  # Direction vector dy

        # Scale the direction by the factor
        dx = factor * dx
        dy = factor * dy

        # Calculate the new positions
        lat_p = lat + dy
        lon_p = lon + dx

        lat_n = lat - dy
        lon_n = lon - dx

        # Ensure the latitude and longitude stay within valid bounds
        lat_p = np.clip(lat_p, -90, 90)
        lat_n = np.clip(lat_n, -90, 90)
        lon_p = (lon_p + 180) % 360 - 180
        lon_n = (lon_n + 180) % 360 - 180

        # Handle potential discontinuities near the poles
        if abs(lat_p) == 90:
            lon_p = lon  # Keep longitude unchanged at the poles
        if abs(lat_n) == 90:
            lon_n = lon  # Keep longitude unchanged at the poles

        # Create nodes and assign positions
        node_mid = (node, 0)
        H.add_node(node_mid)
        H.nodes[node_mid]['pos'] = (lon, lat)
        H.nodes[node_mid]['component'] = -1

        node_p = (node, 1)
        H.add_node(node_p)
        H.nodes[node_p]['pos'] = (lon_p, lat_p)
        H.nodes[node_p]['component'] = -2

        node_n = (node, 2)
        H.add_node(node_n)
        H.nodes[node_n]['pos'] = (lon_n, lat_n)
        H.nodes[node_n]['component'] = -3

        # Add an edge between the two pickup points
        H.add_edge(node_n, node_p)

    return H

# Function to prepare edge data
def plot_edge(H,edge):
    pos1 = H.nodes[edge[0]]['pos']
    pos2 = H.nodes[edge[1]]['pos']
    return ([pos1[0], pos2[0]], [pos1[1], pos2[1]])

# Function to prepare node data
def plot_node(H,node, pos):
    component = H.nodes[node]['component']
    color = 'yellow' if component == -1 else 'blue' if component == -2 else 'red'
    return (pos[0], pos[1], color, str(node))


def geographic_to_image_coordinates(lon, lat, x_max, y_max, lon_min, lon_max, lat_min, lat_max):
    """
    Converts geographic coordinates to image coordinates.
    """
    x_img = int((lon - lon_min) / (lon_max - lon_min) * x_max)
    y_img = int((lat_max - lat) / (lat_max - lat_min) * y_max)  # Flip latitude since images are top-down
    return x_img, y_img

# Function to extract attributes from the image and assign them to the graph nodes
def extract_attribute(G, image, name, lon_min, lon_max, lat_min, lat_max, channel=None):
    """
    Extracts attribute from image and assigns to graph nodes.
    
    Parameters:
    - G: Graph
    - image: 2D or 3D image array
    - name: Attribute name to assign
    - lon_min, lon_max, lat_min, lat_max: Geographic bounds of the image
    - channel: If the image is 3D, specify the channel to extract (e.g., 0 for R in RGB)
    """
    # Check image shape and handle accordingly
    # print(f"Image shape: {image.shape}")  # Debugging line to print the shape of the image
    
    if len(image.shape) == 3:
        if channel is None:
            raise ValueError("Image has multiple channels; please specify the 'channel' parameter.")
        image = image[:, :, channel]
    elif len(image.shape) != 2:
        raise ValueError("Image must be either a 2D array or a 3D array with a specified channel.")
    
    (y_max, x_max) = image.shape  # y_max corresponds to the height (rows), x_max to width (columns)
    
    for node in G.nodes:
        lon, lat = G.nodes[node]['pos']
        x_img, y_img = geographic_to_image_coordinates(lon, lat, x_max, y_max, lon_min, lon_max, lat_min, lat_max)

        if x_img < 0 or y_img < 0 or x_img >= x_max or y_img >= y_max:
            G.nodes[node][name] = float('nan')  # Handle out-of-bounds indices
        else:
            G.nodes[node][name] = image[y_img, x_img]  # Access image using y (row) and x (column)
    
    return G

def filter_pickup_points_sphere(G, H):    
    for node in G:
        # Check if the latitude of the pickup points is below 0 (Southern Hemisphere)
        if H.nodes[(node, 1)]['pos'][1] < 0 or H.nodes[(node, 2)]['pos'][1] < 0:
            # Set v_x, v_y, and v_z to 0 for the current node and its pickup points
            H.nodes[(node, 0)]['v_x'] = 0
            H.nodes[(node, 0)]['v_y'] = 0
            H.nodes[(node, 0)]['v_z'] = 0
    
            H.nodes[(node, 1)]['v_x'] = 0
            H.nodes[(node, 1)]['v_y'] = 0
            H.nodes[(node, 1)]['v_z'] = 0
    
            H.nodes[(node, 2)]['v_x'] = 0
            H.nodes[(node, 2)]['v_y'] = 0
            H.nodes[(node, 2)]['v_z'] = 0

    return H

def calculate_slip_rate_sphere(G, H, dim):
    for node in H.nodes:
        if node[1] == 0:  # Center point
            
            if dim == 2:
                # Check if either of the pickup points has zero velocity
                if H.nodes[(node[0], 1)]['v_x'] == 0 or H.nodes[(node[0], 2)]['v_x'] == 0:
                    G.nodes[node[0]]['slip_rate_x'] = 0
                    G.nodes[node[0]]['slip_rate_z'] = 0
                    G.nodes[node[0]]['slip_rate'] = 0
                else:
                    G.nodes[node[0]]['slip_rate_x'] = abs(H.nodes[(node[0], 1)]['v_x'] - H.nodes[(node[0], 2)]['v_x'])
                    G.nodes[node[0]]['slip_rate_z'] = abs(H.nodes[(node[0], 1)]['v_z'] - H.nodes[(node[0], 2)]['v_z'])
                    G.nodes[node[0]]['slip_rate'] = math.sqrt(G.nodes[node[0]]['slip_rate_x']**2 + G.nodes[node[0]]['slip_rate_z']**2)
            
            elif dim == 3:
                # Check if either of the pickup points has zero velocity
                if (H.nodes[(node[0], 1)]['v_x'] == 0 or H.nodes[(node[0], 2)]['v_x'] == 0 or 
                    H.nodes[(node[0], 1)]['v_y'] == 0 or H.nodes[(node[0], 2)]['v_y'] == 0 or
                    H.nodes[(node[0], 1)]['v_z'] == 0 or H.nodes[(node[0], 2)]['v_z'] == 0):
                    G.nodes[node[0]]['slip_rate_x'] = 0
                    G.nodes[node[0]]['slip_rate_y'] = 0
                    G.nodes[node[0]]['slip_rate_z'] = 0
                    G.nodes[node[0]]['slip_rate'] = 0
                else:
                    G.nodes[node[0]]['slip_rate_x'] = abs(H.nodes[(node[0], 1)]['v_x'] - H.nodes[(node[0], 2)]['v_x'])
                    G.nodes[node[0]]['slip_rate_y'] = abs(H.nodes[(node[0], 1)]['v_y'] - H.nodes[(node[0], 2)]['v_y'])
                    G.nodes[node[0]]['slip_rate_z'] = abs(H.nodes[(node[0], 1)]['v_z'] - H.nodes[(node[0], 2)]['v_z'])
                    G.nodes[node[0]]['slip_rate'] = math.sqrt(G.nodes[node[0]]['slip_rate_x']**2 + 
                                                              G.nodes[node[0]]['slip_rate_y']**2 + 
                                                              G.nodes[node[0]]['slip_rate_z']**2)
    
    return G

def calculate_slip_sphere(G, H, dt, dim):
    if dim == 2:
        for node in H.nodes:
            if node[1] == 0:
                if H.nodes[(node[0], 1)]['v_x'] == 0 or H.nodes[(node[0], 2)]['v_x'] == 0:
                    G.nodes[node[0]]['slip_x'] = 0
                    G.nodes[node[0]]['slip_z'] = 0
                    G.nodes[node[0]]['slip'] = 0
                else:            
                    G.nodes[node[0]]['slip_x'] = abs((H.nodes[(node[0], 1)]['v_x'] - H.nodes[(node[0], 2)]['v_x'])) * dt
                    G.nodes[node[0]]['slip_z'] = abs((H.nodes[(node[0], 1)]['v_z'] - H.nodes[(node[0], 2)]['v_z'])) * dt
                    G.nodes[node[0]]['slip'] = math.sqrt(G.nodes[node[0]]['slip_x']**2 + G.nodes[node[0]]['slip_z']**2)
                                                            
    if dim == 3:
        for node in H.nodes:
            if node[1] == 0:
                if H.nodes[(node[0], 1)]['v_x'] == 0 or H.nodes[(node[0], 2)]['v_x'] == 0:
                    G.nodes[node[0]]['slip_x'] = 0
                    G.nodes[node[0]]['slip_y'] = 0
                    G.nodes[node[0]]['slip_z'] = 0
                    G.nodes[node[0]]['slip'] = 0
                else:            
                    G.nodes[node[0]]['slip_x'] = abs((H.nodes[(node[0], 1)]['v_x'] - H.nodes[(node[0], 2)]['v_x'])) * dt
                    G.nodes[node[0]]['slip_y'] = abs((H.nodes[(node[0], 1)]['v_y'] - H.nodes[(node[0], 2)]['v_y'])) * dt
                    G.nodes[node[0]]['slip_z'] = abs((H.nodes[(node[0], 1)]['v_z'] - H.nodes[(node[0], 2)]['v_z'])) * dt
                    G.nodes[node[0]]['slip'] = math.sqrt(G.nodes[node[0]]['slip_x']**2 + 
                                                        G.nodes[node[0]]['slip_y']**2 + 
                                                        G.nodes[node[0]]['slip_z']**2)
    return G


def plot_attribute_zero(G, attribute, ax=None, node_size=20, cmap='viridis', show_colorbar=True):
    pos = nx.get_node_attributes(G, 'pos')
    values = nx.get_node_attributes(G, attribute)
    nodes = G.nodes()
    node_colors = [values[node] for node in nodes]

    if ax is None:
        fig, ax = plt.subplots()

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, cmap=cmap, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)
    
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax)


# Function to plot an attribute with a colormap
def plot_attribute_one(G, attribute, cmap='viridis', vmin=None, vmax=None, ax=None, **kwargs):
    pos = nx.get_node_attributes(G, 'pos')
    values = np.array([G.nodes[node][attribute] for node in G.nodes()])
    
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
    
    if ax is None:
        ax = plt.gca()
    
    nodes = nx.draw_networkx_nodes(G, pos, node_color=values, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax,node_size=5, **kwargs)
    
    return nodes


# def get_fault_labels(G):
#     labels = set()
#     for node in G:
#         labels.add(G.nodes[node]['fault'])
#     return sorted(list(labels))

# def get_fault(G, n):
#     nodes = [node for node in G if G.nodes[node]['fault'] == n]
#     return G.subgraph(nodes)

# def total_length(fault):
#     length = 0
#     for u, v in fault.edges():
#         pos_u = fault.nodes[u]['pos']
#         pos_v = fault.nodes[v]['pos']
#         length += np.sqrt((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)
#     return length


def write_slip_to_displacement(G, dim):    
    if dim == 2:
        for node in G:
            G.nodes[node]['heave'] = G.nodes[node]['slip_x']
            G.nodes[node]['throw'] = G.nodes[node]['slip_z']
            G.nodes[node]['displacement'] = G.nodes[node]['slip']

    if dim == 3:
        for node in G:
            G.nodes[node]['heave']   = G.nodes[node]['slip_x']            
            G.nodes[node]['lateral'] = G.nodes[node]['slip_y']
            G.nodes[node]['throw']   = G.nodes[node]['slip_z']
            G.nodes[node]['displacement'] = G.nodes[node]['slip']
    return G


def common_faults(G, H):
    C_G = get_fault_labels(G)
    C_H = get_fault_labels(H)
    return list(set(C_G) & set(C_H))

def get_displacement_sphere(G, dim):
    if dim == 2:
        points = np.zeros((len(list(G.nodes)), 6))
        for n, node in enumerate(G.nodes):
            points[n, 0] = node
            points[n, 1] = G.nodes[node]['pos'][0]
            points[n, 2] = G.nodes[node]['pos'][1]
            points[n, 3] = G.nodes[node].get('heave', 0)  # Use .get to avoid KeyErrors
            points[n, 4] = G.nodes[node].get('throw', 0)
            points[n, 5] = G.nodes[node].get('displacement', 0)
    elif dim == 3:
        points = np.zeros((len(list(G.nodes)), 7))
        for n, node in enumerate(G.nodes):
            points[n, 0] = node
            points[n, 1] = G.nodes[node]['pos'][0]
            points[n, 2] = G.nodes[node]['pos'][1]
            points[n, 3] = G.nodes[node].get('heave', 0)
            points[n, 4] = G.nodes[node].get('lateral', 0)
            points[n, 5] = G.nodes[node].get('throw', 0)
            points[n, 6] = G.nodes[node].get('displacement', 0)
    return points

def extract_output_number_from_filename(file_name):
    """Extract the time step from the filename, assuming the format is solution_surface-XXXXX.XXXX.vtu"""
    base_name = os.path.basename(file_name)  # Get the base filename without the directory path
    number = base_name.split('-')[1].split('.')[0]  # Extract the numerical part
    return number

def extract_output_number_from_pickle_file(file_name):
    """Extract the output number from the pickle filename, assuming the format is G_XXXXX.p"""
    base_name = os.path.basename(file_name)
    
    # Check if the filename follows the expected pattern
    if base_name.startswith('G_') and base_name.endswith('.p'):
        output_number = base_name.split('_')[1].split('.')[0]  # Extract the numerical part between 'g_' and '.p'
        return output_number
    
    # If the filename doesn't match the expected format, raise an error or handle it
    print(f"Warning: Unexpected filename format for {file_name}. Skipping this file.")
    return None  # You can also raise an exception here if that's preferred

def get_times_solution_surface(file_name):
    # Extract variables
    # read entire file and store each line
    with open(file_name) as f:
        header = f.readlines()
    # remove all lines that do not start with "#" (starting from the back)
    # nonheaderID = [x[0] != '#' for x in header]
    for index, linecontent in reversed(list(enumerate(header))):
        if linecontent[0] != '#':
            del header[index]
    # remove whitespace characters like `\n` at the end of each line
    header = [x.strip() for x in header]

    # EXTRACT DATA
    df = pd.read_csv(file_name, comment='#', header=None, delim_whitespace=True)

    # Find index of column containing output files
    for col in range(df.shape[1]):
        if 'solution_surface/solution_surface-00000' in str(df[col][0]):
            index = col 

    # EXTRACT TIMES
    times = []    
    for n in range(df.shape[0]):           
        if pd.notnull(df.iloc[n,index]):
            times.append(df.iloc[n,1])
    return times


def get_fault_labels_for_sphere(G):
    """
    Extracts unique fault labels from a graph G for spherical coordinates.
    
    Parameters:
    G (networkx.Graph): The input graph with fault labels as node attributes.
    
    Returns:
    list: A list of unique fault labels.
    """
    # Extract fault labels from all nodes
    fault_labels = set(nx.get_node_attributes(G, 'fault').values())
    
    # Return the unique fault labels as a sorted list
    return sorted(fault_labels)


def get_fault_for_sphere(G, fault_label):
    """
    Extracts a subgraph of the graph G that corresponds to a specific fault label for spherical coordinates.
    Also calculates the total length and maximum slip for that fault.
    
    Parameters:
    G (networkx.Graph): The input graph with fault labels as node attributes.
    fault_label (int or str): The fault label for which the subgraph is to be extracted.
    
    Returns:
    tuple: A tuple containing the subgraph of G for the specified fault, total length, and maximum slip.
    """
    # Extract the nodes associated with the specified fault label
    nodes = [node for node, attr in G.nodes(data=True) if attr.get('fault') == fault_label]
    
    # Create a subgraph containing only these nodes
    fault_subgraph = G.subgraph(nodes).copy()
    
    # Initialize total length and maximum slip
    total_length = 0
    max_slip = 0
    
    # Iterate over edges in the fault subgraph to accumulate length and find maximum slip
    for u, v, attr in fault_subgraph.edges(data=True):
        if 'length' in attr:
            total_length += attr['length']
        
        slip_u = G.nodes[u].get('slip', 0)
        slip_v = G.nodes[v].get('slip', 0)
        max_slip = max(max_slip, slip_u, slip_v)
    
    return fault_subgraph, total_length, max_slip

def common_faults_sphere(G, H):
    C_G = get_fault_labels_for_sphere(G)
    C_H = get_fault_labels_for_sphere(H)
    return list(set(C_G) & set(C_H))

import numpy as np

def plot_displacement_sphere(G, title, plot_figures):
    geographic_positions = {}
    
    # Check if positions are already in lat/lon or Cartesian coordinates
    for node in G.nodes:
        pos = G.nodes[node]['pos']
        if len(pos) == 3:
            # If position is in Cartesian coordinates (x, y, z)
            x, y, z = pos
            
            # Convert Cartesian coordinates to geographic coordinates (lat/lon)
            lon = np.arctan2(y, x) * 180 / np.pi  # Convert to degrees
            lat = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)) * 180 / np.pi  # Convert to degrees
            
        elif len(pos) == 2:
            # If position is already in (lon, lat)
            lon, lat = pos
            
        else:
            raise ValueError("Position should be in either (x, y, z) or (lon, lat) format.")
        
        geographic_positions[node] = (lon, lat)

    lon_min, lon_max = -180, 180  # Longitude limits
    lat_min, lat_max = -90, 90    # Latitude limits
    
    fig, ax = plt.subplots(figsize=(16, 8))  # Adjusted figure size

    # Update graph nodes with geographic positions for correct visualization
    for node, pos in geographic_positions.items():
        G.nodes[node]['pos'] = pos

    # Plot the 'displacement' attribute without a colorbar
    plot_attribute_zero(G, 'displacement', ax=ax, node_size=20, cmap='viridis',show_colorbar=False)

    # Set the limits for the plot based on latitude and longitude ranges
    plt.xlim([lon_min, lon_max])
    plt.ylim([lat_min, lat_max])

    ax.set_title(title, fontsize=16)

    # Add colorbar with title 'Displacement'
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(nx.get_node_attributes(G, 'displacement').values()), vmax=max(nx.get_node_attributes(G, 'displacement').values())))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Displacement (meters)', rotation=270, labelpad=20, fontsize=14)

    if plot_figures.lower() == 'true':
        plt.show()

    return fig  # Return the figure object


def plot_fault_age_map(G, title, plot_figures, fault_ages):
    geographic_positions = {}
    
    # Check if positions are already in lat/lon or Cartesian coordinates
    for node in G.nodes:
        pos = G.nodes[node]['pos']
        if len(pos) == 3:
            # If position is in Cartesian coordinates (x, y, z)
            x, y, z = pos
            
            # Convert Cartesian coordinates to geographic coordinates (lat/lon)
            lon = np.arctan2(y, x) * 180 / np.pi  # Convert to degrees
            lat = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)) * 180 / np.pi  # Convert to degrees
            
        elif len(pos) == 2:
            # If position is already in (lon, lat)
            lon, lat = pos
            
        else:
            raise ValueError("Position should be in either (x, y, z) or (lon, lat) format.")
        
        geographic_positions[node] = (lon, lat)

    lon_min, lon_max = -180, 180  # Longitude limits
    lat_min, lat_max = -90, 90    # Latitude limits
    
    fig, ax = plt.subplots(figsize=(16, 8))  # Adjusted figure size

    # Update graph nodes with geographic positions for correct visualization
    for node, pos in geographic_positions.items():
        G.nodes[node]['pos'] = pos

    # Plot the fault ages on the map
    fault_labels = nx.get_node_attributes(G, 'fault')
    node_ages = {node: fault_ages[fault_labels[node]] for node in G.nodes() if fault_labels[node] in fault_ages}

    # Use plot_attribute_zero to plot the fault ages
    plot_attribute_zero(G, attribute='fault_age', ax=ax, node_size=20, cmap='viridis',show_colorbar=False)

    # Set the limits for the plot based on latitude and longitude ranges
    plt.xlim([lon_min, lon_max])
    plt.ylim([lat_min, lat_max])

    ax.set_title(title, fontsize=16)

    # Add colorbar with title 'Fault Age'
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(node_ages.values()), vmax=max(node_ages.values())))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Fault Age (years)', rotation=270, labelpad=20, fontsize=14)

    if plot_figures.lower() == 'true':
        plt.show()

    return fig  # Return the figure object

# def H_add_nodes(H, G, time, limited_faults=None):
#     labels = get_fault_labels_for_sphere(G)
    
#     # Limit the number of faults if limited_faults is specified
#     if limited_faults is not None:
#         labels = labels[:limited_faults]
    
#     for label in labels:
#         H.add_node((time, label))
#         H.nodes[(time, label)]['pos'] = (time, random.uniform(0, 1) * len(labels))  # Randomize the y-position
#         H.nodes[(time, label)]['time'] = time
#         H.nodes[(time, label)]['fault'] = label

#         fault, total_length, max_displacement = get_fault_for_sphere(G, label)
#         H.nodes[(time, label)]['length'] = total_length
#         H.nodes[(time, label)]['displacement'] = max_displacement
#     return H

# def plot_faults_evolution(Gs, limited_faults=None, file_suffix=""):
#     H = nx.Graph()

#     # Process each graph and update H with nodes and edges
#     for time in range(len(Gs) - 1):
#         print(time)

#         G_0 = Gs[time]
#         G_1 = Gs[time + 1]

#         if time == 0:
#             H = H_add_nodes(H, G_0, time, limited_faults)
#         H = H_add_nodes(H, G_1, time + 1, limited_faults)

#         for label_0 in get_fault_labels_for_sphere(G_0)[:limited_faults]:
#             for label_1 in get_fault_labels_for_sphere(G_1)[:limited_faults]:
#                 if label_0 == label_1:
#                     H.add_edge((time, label_0), (time + 1, label_1))

#     # Plot the fault evolution graph with diverse colors
#     fig, ax = plt.subplots(figsize=(20, 10))
#     node_pos = nx.get_node_attributes(H, 'pos')

#     # Get node colors based on the fault labels
#     node_colors = get_node_colors(H, 'fault')

#     # Plot using networkx draw function
#     nx.draw(
#         H,
#         pos=node_pos,
#         labels=nx.get_node_attributes(H, 'fault'),
#         with_labels=True,
#         node_color=node_colors,
#         node_size=300,  # Adjust size as necessary
#         edge_color='gray',
#         ax=ax,
#     )

#     ax.set_title(f"Fault Evolution Over Time {file_suffix}")

#     return fig  # Return the figure object for external saving

def get_dictionary_sphere(G):
    faults = get_fault_labels_for_sphere(G)  # Use the appropriate function for spherical graphs
    dic = {}
    for fault in faults:
        G_fault, _, _ = get_fault_for_sphere(G, fault)  # Get the fault subgraph
        components = set()
        for node in G_fault.nodes:
            if 'component' in G_fault.nodes[node]:  # Check if 'component' attribute exists
                components.add(G_fault.nodes[node]['component'])
        dic[fault] = sorted(list(components))
    return dic

def H_add_nodes_sphere(H, G, time, limited_faults=None):
    labels = get_fault_labels_for_sphere(G)
    
    # Limit the number of faults if limited_faults is specified
    if limited_faults is not None:
        labels = labels[:limited_faults]
    
    # Sort labels to ensure consistent vertical alignment
    sorted_labels = sorted(labels)
    
    for index, label in enumerate(sorted_labels):
        H.add_node((time, label))
        # Position nodes by time and their order in sorted labels
        H.nodes[(time, label)]['pos'] = (time, index)  # Use index instead of random y-position
        H.nodes[(time, label)]['time'] = time
        H.nodes[(time, label)]['fault'] = label

        fault, total_length, max_displacement = get_fault_for_sphere(G, label)
        H.nodes[(time, label)]['length'] = total_length
        H.nodes[(time, label)]['displacement'] = max_displacement
    return H

# # Function to add nodes to graph H
# def H_add_nodes_sphere(H, G, time, limited_faults=None):
#     labels = get_fault_labels_for_sphere(G)
    
#     # Limit the number of faults if limited_faults is specified
#     if limited_faults is not None:
#         labels = labels[:limited_faults]
    
#     for label in labels:
#         H.add_node((time, label))
#         H.nodes[(time, label)]['pos'] = (time, random.uniform(0, 1) * len(labels))  # Randomize the y-position
#         H.nodes[(time, label)]['time'] = time
#         H.nodes[(time, label)]['fault'] = label

#         fault, total_length, max_displacement = get_fault_for_sphere(G, label)
#         H.nodes[(time, label)]['length'] = total_length
#         H.nodes[(time, label)]['displacement'] = max_displacement
#     return H

def plot_faults_evolution(H, file_suffix=""):
    fig, ax = plt.subplots(figsize=(20, 10))
    node_pos = nx.get_node_attributes(H, 'pos')

    # Get node colors based on the fault labels
    node_colors = get_node_colors(H, 'fault')

    # Plot using networkx draw function
    nx.draw(
        H,
        pos=node_pos,
        labels=nx.get_node_attributes(H, 'fault'),
        with_labels=True,
        node_color=node_colors,
        node_size=300,  # Adjust size as necessary
        edge_color='gray',
        ax=ax,
    )

    ax.set_title(f"Fault Evolution Over Time {file_suffix}")
    return fig  # Return the figure object for external saving

def plot_width_sphere(G, ax, width, tips=True, plot=False):
    """ Plot edge width of fault network
    
    Parameters
    ----------
    G : nx.graph
        Graph
    ax : plt axis
        Axis
    width : np.array
        Width of network edges
    tips : boolean
        Plot tips
    plot : False
        Plot helper functions
    
    Returns
    -------  
    ax : plt axis
        Axis with the plotted graph
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    # Plotting
    pos = nx.get_node_attributes(G, 'pos')

    n_comp = 10000

    sns.color_palette(None, 2*n_comp)

    colors = get_node_colors(G, 'fault')

    def get_points(u):
        u0 = np.array(pos[u[0]])
        u1 = np.array(pos[u[1]])

        u_vec = u0-u1

        u_perp = np.array([-u_vec[1], u_vec[0]])
        u_perp = u_perp/np.linalg.norm(u_perp)

        u0a = u0 - u_perp*width[u[0]]
        u0b = u0 + u_perp*width[u[0]]

        u1a = u1 - u_perp*width[u[1]]
        u1b = u1 + u_perp*width[u[1]]

        return u0a, u0b, u1a, u1b

    def get_intersect(a1, a2, b1, b2):
        """
        Returns the point of intersection of the lines passing through a2,a1
        and b2,b1.
        a1: [x, y] a point on the first line
        a2: [x, y] another point on the first line
        b1: [x, y] a point on the second line
        b2: [x, y] another point on the second line
        """
        s = np.vstack([a1, a2, b1, b2])        # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        if z == 0:                          # lines are parallel
            return (float('inf'), float('inf'))
        return np.array([x/z, y/z])

    def clockwiseangle_and_distance(origin, point):
        refvec = [0, 1]
        vector = [point[0]-origin[0], point[1]-origin[1]]
        lenvector = math.hypot(vector[0], vector[1])
        if lenvector == 0:
            return -math.pi, 0
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod = normalized[0]*refvec[0] + normalized[1]*refvec[1]
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]
        angle = math.atan2(diffprod, dotprod)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        return angle, lenvector

    def get_edges(G, node):
        neighbors = list(G.neighbors(node))
        pts = [G.nodes[neighbor]['pos'] for neighbor in neighbors]
        pts, neighbors = zip(
            *sorted(
                zip(pts, neighbors),
                key=lambda x: clockwiseangle_and_distance(
                    G.nodes[node]['pos'], x[0])
                )
            )
        edges = [(node, neighbor) for neighbor in neighbors]
        return edges

    for node, color in zip(G, colors):
        if tips is True and G.degree(node) == 1:
            edge = get_edges(G, node)[0]

            node0 = np.array(pos[edge[0]])
            node1 = np.array(pos[edge[1]])

            vec = node0-node1
            vec_perp = np.array([-vec[1], vec[0]])
            vec_perp = vec_perp/np.linalg.norm(vec_perp)

            vec_pos = node0 + vec_perp*width[edge[0]]
            vec_neg = node0 - vec_perp*width[edge[0]]

            stack = np.vstack((vec_pos, node0+vec, vec_neg, vec_pos))

            polygon = Polygon(stack, facecolor=color, alpha=1)
            p = PatchCollection([polygon], match_original=True)
            ax.add_collection(p)

        if G.degree(node) == 2:
            edges = get_edges(G, node)

            points = []
            for edge in edges:
                points.append(get_points(edge))

            intersects = []
            intersects.append(get_intersect(
                points[0][0], points[0][2], points[1][1], points[1][3]))
            intersects.append(get_intersect(
                points[0][1], points[0][3], points[1][0], points[1][2]))

            stack = np.vstack((points[0][3], intersects[1], points[1][2],
                               points[1][3], intersects[0], points[0][2]))

            polygon = Polygon(stack, facecolor=color, alpha=1)
            p = PatchCollection([polygon], match_original=True)
            ax.add_collection(p)

        if G.degree(node) == 3:
            edges = get_edges(G, node)

            points = []
            for edge in edges:
                points.append(get_points(edge))

            intersects = []
            intersects.append(get_intersect(
                points[0][1], points[0][3], points[1][0], points[1][2]))
            intersects.append(get_intersect(
                points[1][1], points[1][3], points[2][0], points[2][2]))
            intersects.append(get_intersect(
                points[0][0], points[0][2], points[2][1], points[2][3]))

            stack = np.vstack((points[0][3], intersects[0], points[1][2],
                               points[1][3], intersects[1], points[2][2],
                               points[2][3], intersects[2], points[0][2]))

            polygon = Polygon(stack, facecolor=color, alpha=1)
            p = PatchCollection([polygon], match_original=True)
            ax.add_collection(p)

        if G.degree(node) == 4:
            edges = get_edges(G, node)

            points = []
            for edge in edges:
                points.append(get_points(edge))

            intersects = []
            intersects.append(get_intersect(
                points[0][1], points[0][3], points[1][0], points[1][2]))
            intersects.append(get_intersect(
                points[1][1], points[1][3], points[2][0], points[2][2]))
            intersects.append(get_intersect(
                points[2][1], points[2][3], points[3][0], points[3][2]))
            intersects.append(get_intersect(
                points[0][0], points[0][2], points[3][1], points[3][3]))

            stack = np.vstack((points[0][3], intersects[0], points[1][2],
                               points[1][3], intersects[1], points[2][2],
                               points[2][3], intersects[2], points[3][2],
                               points[3][3], intersects[3], points[0][2]))

            polygon = Polygon(stack, facecolor=color, alpha=1)
            p = PatchCollection([polygon], match_original=True)
            ax.add_collection(p)

        if G.degree(node) == 5:
            edges = get_edges(G, node)

            points = []
            for edge in edges:
                points.append(get_points(edge))

            intersects = []
            intersects.append(get_intersect(
                points[0][1], points[0][3], points[1][0], points[1][2]))
            intersects.append(get_intersect(
                points[1][1], points[1][3], points[2][0], points[2][2]))
            intersects.append(get_intersect(
                points[2][1], points[2][3], points[3][0], points[3][2]))
            intersects.append(get_intersect(
                points[3][1], points[3][3], points[4][0], points[4][2]))
            intersects.append(get_intersect(
                points[0][0], points[0][2], points[4][1], points[4][3]))

            stack = np.vstack((points[0][3], intersects[0], points[1][2],
                               points[1][3], intersects[1], points[2][2],
                               points[2][3], intersects[2], points[3][2],
                               points[3][3], intersects[3], points[4][2],
                               points[4][3], intersects[4], points[0][2]))

            polygon = Polygon(stack, facecolor=color, alpha=1)
            p = PatchCollection([polygon], match_original=True)
            ax.add_collection(p)

    # ax.axis('equal')
    return ax






