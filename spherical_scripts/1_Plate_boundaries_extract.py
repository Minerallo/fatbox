#%%
# Example: Global Plate Boundaries Network Extraction from a Numerical Model

# Load the necessary Python packages:
import os
import sys
sys.path.append('/Users/ponsm/Desktop/software/fatbox/fatbox')

import re
import numpy as np
import networkx as nx
import pickle
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance_matrix
from scipy.interpolate import griddata
from skimage import measure
from joblib import Parallel, delayed
from tqdm import tqdm
from ipywidgets import Layout, interactive, widgets

import cartopy.crs as ccrs
import cartopy.util as cutil
from cartopy.util import add_cyclic_point
from itertools import cycle

import pyvista as pv
import cmcrameri.cm as cmc
import geovista as gv
from geovista.common import to_cartesian
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from fatbox.preprocessing import *
from fatbox.metrics import *
from fatbox.edits import *
from fatbox.plots import *
from fatbox.utils import *

from spherical import *

import pickle
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import timeit

# from tqdm import tqdm
plt.close("all")
# Set up parameters
num_cores = -1
plot_figures = 'false'
threshold_value = 5e-16  # Strain rate to track
threshold_longitude = 5  # Number of pixels to connect faults when reaching +180/-180
pixels_interval_to_connect = 2  # Number of pixels to connect faults

# Define output directory
# output_directory = '/Users/ponsm/Desktop/software/fatbox/plate_boundaries/'
output_directory = '/Users/ponsm/Nextcloud/group_monitoring_earth_evolution_through_time/Research/Michael_Pons/models/Global_model_3D/V06c_R01f_Rodinia_2GPa_llsvps_ps_1x50My_init_2Myint_rhoc3160/plate_boundaries/'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Fault Extraction: Load strain rate map data at the surface of the model
folder_path = '/Volumes/Jerry/global_models_3d/V06a_R01f_Rodinia_2GPa_llsvps_ps_1x50My_init_2Myint/'
#solution_directory_path = os.path.join(folder_path, 'solution/')
solution_directory_path = folder_path

# List and filter files in the 'solution' directory that match 'solution_surface' with a .vtu extension
solution_surface_files = [f for f in os.listdir(solution_directory_path) if f.startswith('solution_surface') and f.endswith('.vtu')]

# Sort the files based on the numerical part
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else float('inf')

solution_surface_files = sorted(solution_surface_files, key=extract_number)

# Count the number of matching files
num_solution_surface_files = len(solution_surface_files)
print(f"Number of 'solution_surface' VTU files: {num_solution_surface_files}")

new_image_directory = output_directory + 'img_networks/'
os.makedirs(new_image_directory, exist_ok=True)

new_pickle_directory = output_directory + 'pickle_networks/'
os.makedirs(new_pickle_directory, exist_ok=True)

folder_path_statistics = folder_path + 'statistics'

start = timeit.default_timer()

times = get_times_solution_surface(folder_path_statistics)

 # Calculate the time intervals (dt) between consecutive times
dt = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]

files = solution_surface_files
#%%

def extract_faults(file_name):
# for i, file_name in enumerate(solution_surface_files):
#     # Build the full file path
#     file_path = os.path.join(solution_directory_path, file_name)
    
#     # Read the VTU file into a pyvista dataset
#     print(f"Processing Timestep {i+1}/{len(solution_surface_files)}: {file_name}")
    number_files = extract_output_number_from_filename(file_name)
    number_files_for_time= int(number_files)
    # print(number_files)
    current_time=times[number_files_for_time]

    # Build the full file path
    file_path = os.path.join(solution_directory_path, file_name)
    
    # Read the VTU file into a pyvista dataset
    data = pv.read(file_path)
    
    # Access the strain rate tensor array
    strain_rate_tensor = data.point_data['surface_strain_rate_tensor']

    # Calculate the magnitude (Frobenius norm) of the strain rate tensor
    strain_rate_magnitude = np.linalg.norm(strain_rate_tensor, axis=1)

    # Calculate the log10 of the strain rate magnitude
    strain_rate_magnitude_log10 = np.log10(strain_rate_magnitude)

    # Add the log10 magnitude array to the dataset for visualization or further analysis
    data.point_data['strain_rate_magnitude'] = strain_rate_magnitude_log10

    # Calculate radius and topography
    radius = np.sqrt(data.points[:, 0]**2 + data.points[:, 1]**2 + data.points[:, 2]**2)
    topography = radius - 6371000  # Assuming Earth's radius in meters

    # Convert coordinates to degrees
    longitude = np.arctan2(data.points[:, 1], data.points[:, 0]) * 180 / np.pi
    latitude = np.arcsin(data.points[:, 2] / radius) * 180 / np.pi

    # Add topography, longitude, and latitude as scalar arrays to the mesh
    data.point_data['Topography'] = topography
    data.point_data['Longitude'] = longitude
    data.point_data['Latitude'] = latitude

    # Optional plotting
    if plot_figures.lower() == 'true':
        plotter = pv.Plotter()
        plotter.add_mesh(data, scalars='strain_rate_magnitude', cmap='viridis', clim=[-16, -14])
        plotter.show()

    # Define grid of longitude and latitude for interpolation
    print("Interpolating strain rate magnitude on a grid...")
    lon_grid, lat_grid = np.meshgrid(np.linspace(-180, 180, 360), np.linspace(-90, 90, 180))
    lon_grid_shape = lon_grid.shape

    # Function to perform griddata interpolation
    def interpolate_strain_rate(method):
        return griddata((longitude, latitude), strain_rate_magnitude_log10, (lon_grid, lat_grid), method=method)

    # Parallelize the interpolation step
    strain_rate_magnitude_interp, strain_rate_magnitude_nearest = Parallel(n_jobs=num_cores)(
        delayed(interpolate_strain_rate)(method) for method in ['linear', 'nearest']
    )

    # Handle missing values in the linear interpolation by filling with nearest interpolation
    strain_rate_magnitude_interp = np.where(np.isnan(strain_rate_magnitude_interp), strain_rate_magnitude_nearest, strain_rate_magnitude_interp)

    # Apply a threshold to the interpolated strain rate magnitude
    print("Applying threshold and skeletonizing the data...")
    threshold = simple_threshold_binary(10**strain_rate_magnitude_interp, threshold_value)
    skeleton = skeleton_guo_hall(threshold)

    # Label connected components and filter small objects
    labels = measure.label(skeleton, connectivity=2)
    props = measure.regionprops(labels)

    min_area = 4  # Set this according to your needs
    skeleton_cleaned = np.zeros_like(skeleton, dtype=bool)
    for prop in props:
        if prop.area >= min_area:
            skeleton_cleaned[labels == prop.label] = 1

    # Flip the skeleton vertically (flip latitude)
    skeleton_cleaned_flipped = np.flipud(skeleton_cleaned)

    # Optional plotting
    if plot_figures.lower() == 'true':
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(skeleton_cleaned_flipped, vmin=0, vmax=1)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
        fig.add_axes(cax)
        fig.colorbar(im, cax=cax, orientation="horizontal")
        plt.show()

    # Convert the binary image to uint8 for OpenCV
    # print(f"Extracting and labeling components for Timestep {i+1}...")
    print(f"Extracting and labeling components ...")

    skeleton_cleaned_flipped_uint8 = (skeleton_cleaned_flipped * 255).astype(np.uint8)
    ret, markers = cv2.connectedComponents(skeleton_cleaned_flipped_uint8, connectivity=8)

    # Optional plotting
    if plot_figures.lower() == 'true':
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(markers, vmin=0)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
        fig.add_axes(cax)
        fig.colorbar(im, cax=cax, orientation="horizontal")
        plt.show()

    
    # Initialize an empty graph and a node counter
    G = nx.Graph()
    node = 0

    # Get the height of the image to invert the y-coordinate
    image_height = skeleton_cleaned_flipped.shape[0]

    print("Starting to add nodes to the graph based on the connected components...")

    # Loop through each connected component in the image
    for comp in range(1, ret):
        points = np.transpose(np.vstack((np.where(markers == comp))))

        for point in points:
            G.add_node(node)
            
            # Invert the y-coordinate to match graph plotting
            y_coordinate = image_height - point[0]
            
            G.nodes[node]['pos'] = (point[1], y_coordinate)  # (x, y) = (col, row)
            G.nodes[node]['component'] = comp
            node += 1

    print(f"Finished adding nodes. Total nodes added: {node}")

    # Optional: Plot the graph
    if plot_figures.lower() == 'true':
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.set_title('Network')

        # Draw the NetworkX graph on the plot
        nx.draw(G, 
                pos=nx.get_node_attributes(G, 'pos'), 
                node_size=1,
                ax=ax)
        ax.axis('equal')
        plt.show()

    # Convert positions to Cartesian coordinates
    print("Converting node positions to Cartesian coordinates...")
    cartesian_positions = {}
    points = []

    for node in G.nodes:
        lon, lat = G.nodes[node]['pos']
        lon = (lon / skeleton_cleaned_flipped.shape[1]) * 360 - 180
        lat = (lat / skeleton_cleaned_flipped.shape[0]) * 180 - 90
        
        x, y, z = latlon_to_cartesian(lat, lon)
        points.append([x, y, z])
        cartesian_positions[node] = (x, y, z)

    # Optional: Visualize the nodes and edges on a 3D plot
    if plot_figures.lower() == 'true':
        print("Visualizing the graph in 3D...")
        plotter = pv.Plotter()

        # Add a sphere to represent the Earth
        sphere = pv.Sphere(radius=6371000, theta_resolution=360, phi_resolution=180)
        plotter.add_mesh(sphere, color='white', opacity=0.6, style='wireframe')

        # Add nodes to the plotter
        points = np.array(points)
        point_cloud = pv.PolyData(points)
        plotter.add_mesh(point_cloud, color='red', point_size=10, render_points_as_spheres=True)

        # Add edges to the plotter
        for edge in G.edges:
            pos1 = cartesian_positions[edge[0]]
            pos2 = cartesian_positions[edge[1]]
            line = pv.Line(pos1, pos2)
            plotter.add_mesh(line, color='blue', line_width=0.5)

        # Show the plot
        plotter.show()

    # Calculate distance matrix and connect nearby nodes
    print("Calculating distance matrix and connecting nearby nodes...")

    for comp in range(1, ret):
        points = [G.nodes[node]['pos'] for node in G if G.nodes[node]['component'] == comp]
        nodes = [node for node in G if G.nodes[node]['component'] == comp]

        # Check if there are enough points to form a matrix
        if len(points) > 1:
            dm = distance_matrix(points, points)

            for n in range(len(points)):
                for m in range(len(points)):
                    if dm[n, m] < pixels_interval_to_connect and n != m:
                        G.add_edge(nodes[n], nodes[m])
        else:
            print(f"Component {comp} does not have enough points for distance matrix calculation.")

    # Convert positions to Cartesian coordinates again after connecting nodes
    print("Re-converting positions to Cartesian coordinates after connecting nodes...")
    cartesian_positions = {}
    points = []

    for node in G.nodes:
        lon, lat = G.nodes[node]['pos']
        lon = (lon / skeleton_cleaned_flipped.shape[1]) * 360 - 180
        lat = (lat / skeleton_cleaned_flipped.shape[0]) * 180 - 90
        
        x, y, z = latlon_to_cartesian(lat, lon)
        points.append([x, y, z])
        cartesian_positions[node] = (x, y, z)

    # Optional: Visualize the connected components
    if plot_figures.lower() == 'true':
        print("Visualizing the connected components in 3D...")
        plotter = pv.Plotter()

        # Add a sphere to represent the Earth
        sphere = pv.Sphere(radius=6371000, theta_resolution=360, phi_resolution=180)
        plotter.add_mesh(sphere, color='white', opacity=0.6, style='wireframe')

        # Add nodes to the plotter
        points = np.array(points)
        point_cloud = pv.PolyData(points)
        plot_cloud['component'] = [G.nodes[node]['component'] for node in G.nodes]
        plotter.add_mesh(point_cloud, scalars='component', render_points_as_spheres=True, point_size=10)

        # Parallelize the creation of lines
        lines = Parallel(n_jobs=num_cores)(delayed(create_line)(edge, cartesian_positions) for edge in G.edges)

        # Combine all lines into a single mesh and add it to the plotter
        multi_line = pv.MultiBlock(lines)
        plotter.add_mesh(multi_line, color='blue', line_width=3)

        # Show the plot
        plotter.show()

    # Simplify the graph (optional)
    print("Simplifying the graph...")
    G2 = simplify(G, 2)

    # Recompute Cartesian coordinates and points list after simplification
    cartesian_positions = {}
    points = []

    for node in G2.nodes:
        lon, lat = G2.nodes[node]['pos']
        lon = (lon / skeleton_cleaned_flipped.shape[1]) * 360 - 180
        lat = (lat / skeleton_cleaned_flipped.shape[0]) * 180 - 90

        x, y, z = latlon_to_cartesian(lat, lon)
        points.append([x, y, z])
        cartesian_positions[node] = (x, y, z)

    # Optional: Visualize the simplified graph
    if plot_figures.lower() == 'true':
        print("Visualizing the simplified graph in 3D...")
        plotter = pv.Plotter()

        # Add a sphere to represent the Earth
        sphere = pv.Sphere(radius=6371000, theta_resolution=360, phi_resolution=180)
        plotter.add_mesh(sphere, color='white', opacity=0.6, style='wireframe')

        # Add nodes to the plotter with colors based on their component
        points = np.array(points)
        point_cloud = pv.PolyData(points)
        point_cloud['component'] = [G2.nodes[node]['component'] for node in G2.nodes]
        plotter.add_mesh(point_cloud, scalars='component', render_points_as_spheres=True, point_size=10)

        # Parallelize the creation of lines
        lines = Parallel(n_jobs=num_cores)(delayed(create_line)(edge, cartesian_positions) for edge in G2.edges)

        # Combine all lines into a single mesh and add it to the plotter
        multi_line = pv.MultiBlock(lines)
        plotter.add_mesh(multi_line, color='blue', line_width=0.5)

        # Show the plot
        plotter.show()

    cdata_placeholder = np.zeros((skeleton_cleaned_flipped.shape[0], skeleton_cleaned_flipped.shape[1]))

    # Optional: 2D plot for verification
    if plot_figures.lower() == 'true':
        print("Creating a 2D plot for verification...")

        # Plot the components on a 2D plane for verification
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.imshow(cdata_placeholder, 'gray_r', vmin=0)
        plot_components(G2, node_size=1, ax=ax)
        plt.show()

    # Split triple junctions
    print("Splitting triple junctions in the graph...")
    G1 = split_triple_junctions(G2, 25, split='all')

    # Generate a list of colors for plotting
    colors = cycle(plt.cm.tab20.colors)  # Use a colormap with many distinct colors

    # Get the position of the nodes (assuming they have 'pos' attributes)
    pos = nx.get_node_attributes(G1, 'pos')

    # Optional: Visualize the connected components after splitting triple junctions
    
    # Extract the time step from the file name
    print("Visualizing the graph after splitting triple junctions...")
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    # Placeholder for background, if needed
    ax.imshow(cdata_placeholder, 'gray_r', vmin=0)
    title_str = f"Plate Boundaries\nTime: {current_time / 1e6:.2f} My"
    ax.set_title(title_str, fontsize=18)
    # Iterate over each connected component and plot it with a different color
    for component in nx.connected_components(G1):
        component_subgraph = G1.subgraph(component)
        color = next(colors)
        
        nx.draw(component_subgraph, pos, node_color=color, edge_color=color, node_size=1, ax=ax)

    # Display the plot
    if plot_figures.lower() == 'true':
        plt.show()

    # Save the plot to the specified directory with the time step in the filename
    file_name_with_number = f"network_plot_{number_files}_{current_time/1e6:.2f}.png"
    file_path_ouput = os.path.join(new_image_directory, file_name_with_number)
    plt.savefig(file_path_ouput, dpi=200)

    # Close the plot to free up memory
    plt.close(fig)

    print(f"Plot saved to {file_path_ouput}")
   

    # Add the strain rate to each node in the graph G1
    print("Adding strain rate to each node based on interpolated data...")
    cdata, clon2d, clat2d = cutil.add_cyclic(strain_rate_magnitude_interp, lon_grid, lat_grid)

    # Verify if node 0 exists
    if 0 in G1.nodes:
        print("Node 0 exists:", G1.nodes[0])
    else:
        print("Node 0 does not exist in the graph.")

    # Find and print details of a valid node to set the strain rate
    valid_node = list(G1.nodes)[0]  # Get the first valid node in the graph
    print(f"Using node {valid_node} to set the strain rate:")
    print(G1.nodes[valid_node])

    # Set the strain rate for each node, with bounds checking
    for node in G1.nodes:
        y_index = min(max(int(G1.nodes[node]['pos'][1]), 0), cdata.shape[0] - 1)
        x_index = min(max(int(G1.nodes[node]['pos'][0]), 0), cdata.shape[1] - 1)
        G1.nodes[node]['strain_rate'] = cdata[y_index, x_index]

    # Verify the strain rates for a few nodes
    print("Strain rate for a few nodes:")
    for node in list(G1.nodes)[:5]:  # Print the strain rates for the first 5 nodes
        print(f"Node {node}, Strain Rate: {G1.nodes[node]['strain_rate']}")

    # Optional: Visualize the nodes with their strain rates in a 3D plot
    if plot_figures.lower() == 'true':
        print("Visualizing the fault network with strain rates in 3D...")
        plotter = pv.Plotter()

        # Add a sphere to represent the Earth
        sphere = pv.Sphere(radius=6371000, theta_resolution=360, phi_resolution=180)
        plotter.add_mesh(sphere, color='white', opacity=0.6, style='wireframe')

        # Add nodes to the plotter with colors based on their strain rate
        points = np.array(points)
        point_cloud = pv.PolyData(points)
        point_cloud['strain_rate'] = [G1.nodes[node]['strain_rate'] for node in G1.nodes]
        plotter.add_mesh(point_cloud, scalars='strain_rate', render_points_as_spheres=True, point_size=10, cmap='viridis')

        # Parallelize the creation of lines
        lines = Parallel(n_jobs=num_cores)(delayed(create_line)(edge, cartesian_positions) for edge in G1.edges)

        # Combine all lines into a single mesh and add it to the plotter
        multi_line = pv.MultiBlock(lines)
        plotter.add_mesh(multi_line, color='blue', line_width=0.5)

        # Show the plot
        plotter.show()

    # Optional: 2D plot of the fault network with strain rate
    if plot_figures.lower() == 'true':
        print("Creating a 2D plot of the fault network with strain rate...")
        fig, ax = plt.subplots(figsize=(20, 12))

        ax.set_title('Fault network with strain rate')
        nx.draw(G1, 
                pos=nx.get_node_attributes(G1, 'pos'),
                node_color=np.array([G1.nodes[node]['strain_rate'] for node in G1.nodes]), 
                node_size=1,
                ax=ax)
        ax.axis('equal')
        plt.show()

    # Print out the graph edges and their details
    print("Listing the first 5 edges in the graph:")
    for jj, edge in enumerate(list(G1.edges)[:5]):  
        print(f"Edge {jj}: {edge}")

    # Calculate the length of each edge and find the maximum length
    print("Calculating the length of each edge in the graph...")
    max_length = 0
    max_length_edge = None

    for edge in G1.edges:
        if edge[0] in G1.nodes and edge[1] in G1.nodes:
            pos1 = G1.nodes[edge[0]]['pos']
            pos2 = G1.nodes[edge[1]]['pos']

            lon1, lat1 = pos1
            lon2, lat2 = pos2

            # Adjust for wrap-around if the difference in longitude exceeds 180 degrees
            if abs(lon2 - lon1) > 180:
                if lon1 < 0:
                    lon1 += 360
                else:
                    lon2 += 360

            # Calculate the distance using the Haversine function
            length = haversine_distance(lat1, lon1, lat2, lon2)
            G1.edges[edge]['length'] = length
            
            # Update max_length if the current length is greater
            if length > max_length:
                max_length = length
                max_length_edge = edge
        else:
            print(f"One or both of the nodes {edge} do not exist in the graph.")
            
    # Print edge lengths for a few edges
    print("Edge lengths for a few edges:")
    for edge in list(G1.edges)[:5]:
        print(f"Edge {edge}, Length: {G1.edges[edge]['length']}")

    # Print the maximum length and the corresponding edge
    print(f"Maximum edge length is {max_length} km for edge {max_length_edge}")

    # Optional: Visualize the edge lengths in 2D
    print("Creating a 2D plot of the fault network with edge lengths...")

    # Extract edge lengths
    edge_lengths = np.array([G1.edges[edge]['length'] for edge in G1.edges])

    if plot_figures.lower() == 'true':
        # Normalize edge lengths for colormap
        norm = mcolors.Normalize(vmin=edge_lengths.min(), vmax=edge_lengths.max())
        cmap = cm.viridis

        # Generate colors based on lengths
        edge_colors = [cmap(norm(length)) for length in edge_lengths]

        fig, ax = plt.subplots(figsize=(20, 12))

        ax.set_title('Fault network with edge lengths in kilometers')
        nx.draw(G1, 
                pos=nx.get_node_attributes(G1, 'pos'),
                edge_color=edge_colors, 
                node_size=0.001,
                ax=ax,
                with_labels=False,
                edge_cmap=cmap,
                edge_vmin=edge_lengths.min(),
                edge_vmax=edge_lengths.max())
        ax.axis('equal')

        # Add colorbar for edge lengths
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Edge Length (km)')

        plt.show()

    if plot_figures.lower() == 'true':
        # Optional: Visualize the network with edge lengths in 3D
        print("Visualizing the fault network with edge lengths in 3D...")
        plotter = pv.Plotter()

        # Add a sphere to represent the Earth
        sphere = pv.Sphere(radius=6371000, theta_resolution=360, phi_resolution=180)
        plotter.add_mesh(sphere, color='white', opacity=0.6, style='wireframe')

        # Add nodes to the plotter with colors based on their strain rate
        points = np.array([cartesian_positions[node] for node in G1.nodes])
        point_cloud = pv.PolyData(points)
        point_cloud['strain_rate'] = np.array([G1.nodes[node]['strain_rate'] for node in G1.nodes])
        plotter.add_mesh(point_cloud, scalars='strain_rate', render_points_as_spheres=True, point_size=10, cmap='viridis')

        # Parallelize the creation of lines for edges, including the length as scalar data
        lines = Parallel(n_jobs=num_cores)(
            delayed(create_line_with_length)(edge, cartesian_positions, G1.edges[edge]['length']) for edge in G1.edges
        )

        # Combine all lines into a single MultiBlock and add them to the plotter
        multi_line = pv.MultiBlock(lines)
        plotter.add_mesh(multi_line, scalars='length', cmap='inferno', line_width=3.0)

        # Show the plot
        plotter.show()

    # Calculate and print the strike for each edge
    print("Calculating the strike for each edge in the graph...")
    strikes = []
    for edge in G1.edges:
        pos1 = cartesian_positions[edge[0]]
        pos2 = cartesian_positions[edge[1]]
        strike = calculate_strike(pos1, pos2)
        strikes.append(strike)
        G1.edges[edge]['strike'] = strike

    # Print the maximum and minimum strike and the corresponding edge
    max_strike = max(strikes)
    max_strike_edge = list(G1.edges)[np.argmax(strikes)]
    print(f"Maximum strike is {max_strike} degrees for edge {max_strike_edge}")
    min_strike = min(strikes)
    min_strike_edge = list(G1.edges)[np.argmin(strikes)]
    print(f"Minimum strike is {min_strike} degrees for edge {min_strike_edge}")

    if plot_figures.lower() == 'true':
        # Optional: Visualize the network with strikes in 3D
        print("Visualizing the fault network with strikes in 3D...")
        plotter = pv.Plotter()

        # Add a sphere to represent the Earth
        sphere = pv.Sphere(radius=6371000, theta_resolution=360, phi_resolution=180)
        plotter.add_mesh(sphere, color='white', opacity=0.6, style='wireframe')

        # Add nodes to the plotter
        points = np.array([cartesian_positions[node] for node in G1.nodes])
        point_cloud = pv.PolyData(points)
        plotter.add_mesh(point_cloud, color='red', point_size=0.1, render_points_as_spheres=True)

        # Add edges to the plotter with colors based on their strikes
        for jj, edge in enumerate(G1.edges):
            pos1 = cartesian_positions[edge[0]]
            pos2 = cartesian_positions[edge[1]]
            line = pv.Line(pos1, pos2)
            plotter.add_mesh(line, color=edge_colors[jj], line_width=5.0)

        # Add a scalar bar for strikes
        dummy_scalar_array = np.linspace(min(strikes), max(strikes), len(G1.edges))
        dummy_points = np.random.rand(len(dummy_scalar_array), 3)  # Dummy points just for the scalar bar
        plotter.add_mesh(pv.PolyData(dummy_points), scalars=dummy_scalar_array, cmap='viridis', show_scalar_bar=True)

        # Show the plot
        plotter.show()

    # Re-load the graph as G (optional)
    G = G1

    # Check the number of connected components in the graph
    if nx.is_directed(G):
        num_components = nx.number_strongly_connected_components(G)
        components = list(nx.strongly_connected_components(G))
    else:
        num_components = nx.number_connected_components(G)
        components = list(nx.connected_components(G))

    print(f"Number of connected components: {num_components}")
    
    if plot_figures.lower() == 'true':
        cartesian_positions = {}
        points = []

        for node in G2.nodes:
            lon, lat = G2.nodes[node]['pos']
            lon = (lon / skeleton_cleaned_flipped.shape[1]) * 360 - 180
            lat = (lat / skeleton_cleaned_flipped.shape[0]) * 180 - 90

            x, y, z = latlon_to_cartesian(lat, lon)
            points.append([x, y, z])
            cartesian_positions[node] = (x, y, z)
        # Plot the edge attribute 'strike' in 3D
        plot_edge_attribute_3d(G, 'strike',cartesian_positions)

    # Plot the edge attribute 'strike' in 2D
    fig, ax = plt.subplots(figsize=(15, 15))
    plot_edge_attribute(G, 'strike', current_time, lon_grid_shape, ax=ax)
    if plot_figures.lower() == 'true':
            plt.show()

    # Save the plot to the specified directory with the time step in the filename
    file_name_with_number = f"Strikes_plot_{number_files}_{current_time/1e6:.2f}.png"
    file_path_ouput = os.path.join(new_image_directory, file_name_with_number)
    plt.savefig(file_path_ouput, dpi=200)

    # Plot a rose diagram of the strikes
    strikes = [G.edges[edge]['strike'] for edge in G.edges]
    if plot_figures.lower() == 'true':
        plot_rose(strikes)

    # Calculate mean strikes and lengths for faults (connected components)
    print("Calculating mean strikes and lengths for faults...")
    fault_strikes = []
    fault_lengths = []
    count = 0
    for cc in nx.connected_components(G):
        count += 1
        edges = G.edges(cc)
        edge_strikes = [G.edges[edge]['strike'] for edge in edges]
        edge_lengths = [G.edges[edge]['length'] for edge in edges]
        fault_strikes.extend(edge_strikes)
        fault_lengths.extend(edge_lengths)
        print(f"Processed component {count}")

    # Convert fault strikes and lengths to floats
    fault_strikes_float = [float(item) for item in fault_strikes]
    fault_lengths_float = [float(item) for item in fault_lengths]

    # Print the fault strikes and lengths
    # print("Fault strikes:", fault_strikes_float)
    # print("Fault lengths:", fault_lengths_float)

    # Plot the Rose diagram with fault strikes and lengths
    print("Plotting the Rose diagram for fault strikes and lengths...")
    if plot_figures.lower() == 'true':
        plot_rose(fault_strikes_float, fault_lengths_float)
        plt.show()
    
    # Save the current graph G1 to a file
    print("Saving the current graph to a file...")
    # filename_output = f'G_{str(i+1).zfill(5)}.p'
    # file_path_out = os.path.join(output_directory, filename_output)
    # with open(file_path_out, 'wb') as p:
    #     pickle.dump(G1, p)

    filename_output = f'G_{number_files}.p'
    file_path_out = os.path.join(new_pickle_directory, filename_output)
    
    # Save the graph as a pickle file
    with open(file_path_out, 'wb') as p:
        pickle.dump(G1, p)

    # print(f"Object G for timestep {i+1} has been saved to {file_path_out}")
    print(f"Object G has been saved to {file_path_out}")
    
# Parallel processing of the files
results = Parallel(n_jobs= num_cores)(delayed(extract_faults)(file) for file in files)

# Stop the timer
stop = timeit.default_timer()

# Print the time taken
print('Extract time: ', stop - start)


