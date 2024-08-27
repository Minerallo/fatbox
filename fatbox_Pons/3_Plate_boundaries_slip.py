#%%
# %reset
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
# , Normalize

from fatbox.preprocessing import *
from fatbox.metrics import *
from fatbox.edits import *
from fatbox.plots import *
from fatbox.utils import *

from spherical import *

import pickle
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from fatbox.plots import plot_components
import timeit
import math
import seaborn as sns


plt.close("all")
# Set up parameters
num_cores = -1
plot_figures = 'false'
pickup_factor = 5 #how many times dx dy do you want to pick the velocity values from the segment
folder_path = '/Volumes/Jerry/global_models_3d/V06a_R01f_Rodinia_2GPa_llsvps_ps_1x50My_init_2Myint/'

# Define output directory
output_directory = '/Users/ponsm/Desktop/software/fatbox/plate_boundaries/'


# solution_directory_path = os.path.join(folder_path, 'solution/')
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

#%%
new_image_directory = output_directory + 'img_slip/'
os.makedirs(new_image_directory, exist_ok=True)

new_pickle_directory = output_directory + 'pickle_slip/'
os.makedirs(new_pickle_directory, exist_ok=True)

folder_path_statistics = folder_path + 'statistics'

#%%

times_all = get_times_solution_surface(folder_path_statistics)

 # Calculate the time intervals (dt) between consecutive times
dt = [t2 - t1 for t1, t2 in zip(times_all[:-1], times_all[1:])]
times =times_all[1:]
# Print the time intervals
print("Time intervals (dt) between consecutive steps:", dt)
#%%   
# vistimes = get_times(folder_path_statistics)

# Start the timer
start = timeit.default_timer()

# Define the directory where the pickle files are stored
pickle_directory_networks = '../plate_boundaries/pickle_correlation/'

# List all files in the directory
# all_files = os.listdir(pickle_directory)

# Filter and sort out only the .p (pickle) files
pickle_files = [f for f in os.listdir(pickle_directory_networks) if f.startswith('G_') and f.endswith('.p')]
pickle_files_sorted = sorted(pickle_files, key=extract_number)

# Count the number of pickle files
num_pickle_files = len(pickle_files_sorted)
print(f"Number of pickle files found: {num_pickle_files}")

#%%
def slip(file_name,dt):

    print(file_name) 

    # %%

    # Print the current file number for tracking
    print(f"Processing file number: {file_name}")

    number_files = extract_output_number_from_pickle_file(file_name)
    print(number_files)
    number_files_out = int(number_files)
    current_dt = dt[number_files_out]
    #the current time is actually the time +1 in this case
    current_time = times[number_files_out]

    # Construct the correct filename based on the padded format
    # filename = f'G_{str(file).zfill(5)}.p'  # Adjusted to match the filename format

    # # Construct the full file path
    # file_path = os.path.join('../plate_boundaries/pickle', filename)
    
    # # Load the graph from the pickle file
    # G = pickle.load(open(file_path, 'rb'))

    #%%
    # Construct the full file path
    file_path = os.path.join(pickle_directory_networks, file_name)

    #for test uncomment bellow and run the cell
    # file_path = os.path.join(pickle_directory_networks,'G_00008.p')
    # Load the graph from the pickle file
    G = pickle.load(open(file_path, 'rb'))
    
    # Ensure the graph is a NetworkX Graph object
    G = nx.Graph(G)

    # You can now process G as needed
    print(f"Loaded graph from {file_path}")

    # Print all graph-level attributes
    print("Graph attributes:")
    for attr, value in G.graph.items():
        print(f"{attr}: {value}")

    # Print all node attributes
    print("\nNode attributes:")
    for node in G.nodes(data=True):
        print(f"Node {node[0]}: {node[1]}")

    # Print all edge attributes
    print("\nEdge attributes:")
    for edge in G.edges(data=True):
        print(f"Edge {edge[0]}-{edge[1]}: {edge[2]}")

    # Additionally, to get specific attributes
    print("\nSpecific node attributes:")
    for node, attr_dict in G.nodes(data=True):
        for attr, value in attr_dict.items():
            print(f"Node {node} has {attr}: {value}")

    print("\nSpecific edge attributes:")
    for (u, v, attr_dict) in G.edges(data=True):
        for attr, value in attr_dict.items():
            print(f"Edge {u}-{v} has {attr}: {value}")
    #%%

    # Iterate over all edges and check for 'strike' attribute
    for edge in G.edges(data=True):
        u, v, attr = edge
        if 'strike' in attr:
            print(f"Edge {u}-{v} has 'strike' attribute with value: {attr['strike']}")
        else:
            print(f"Edge {u}-{v} does not have 'strike' attribute.")

    # Extract fault labels from all nodes
    # fault_labels = nx.get_node_attributes(G, 'fault')

    # # Find the maximum fault label
    # max_fault_label = max(fault_labels.values())

    # print(f"The maximum fault label is: {max_fault_label}")

    # %%
    # Extract fault labels
    fault_labels = np.array(list(nx.get_node_attributes(G, 'fault').values()))

    # Print unique fault labels and their count
    unique_faults, counts = np.unique(fault_labels, return_counts=True)
    print(f"Unique fault labels: {unique_faults}")
    print(f"Counts for each fault label: {counts}")

    if plot_figures.lower() == 'true':
        # Visualize the distribution of fault labels
        plt.figure(figsize=(10, 6))
        plt.hist(fault_labels, bins=len(unique_faults), color='blue', alpha=0.7)
        plt.xlabel('Fault Labels')
        plt.ylabel('Number of Nodes')
        plt.title('Distribution of Fault Labels in the Graph')
        plt.show()


    # Find the corresponding solution_surface file
    matching_file = None
    for surface_file in solution_surface_files:
        if number_files in surface_file:
            matching_file = surface_file
            break

    if not matching_file:
        print(f"No matching solution_surface file found for {file_name}")
        return

    # Build the full file path for the matching solution_surface file
    file_path = os.path.join(solution_directory_path, matching_file)
    
    # Read the VTU file into a pyvista dataset
    data = pv.read(file_path)
    
    # Process the data as needed
    print(f"Loaded VTU file: {matching_file} for displacement file: {file_name}")

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

    # Check if the 'strike' attribute is present for the edges
    edge_strikes_present = all('strike' in G.edges[edge] for edge in G.edges)
    print("All edges have 'strike' attribute:", edge_strikes_present)


    # %%
    # Ensure the surface_velocity is separated into components
    surface_velocity = data.point_data['surface_velocity']
    v_x = surface_velocity[:, 0]
    v_y = surface_velocity[:, 1]
    v_z = surface_velocity[:, 2]

    # Calculate the velocity magnitude
    velocity_magnitude = np.sqrt(v_x**2 + v_y**2 + v_z**2)
    data.point_data['velocity_magnitude'] = velocity_magnitude
    data.point_data['v_x'] = v_x
    data.point_data['v_y'] = v_y
    data.point_data['v_z'] = v_z

    # Retrieve strike values from edges
    strikes = np.array([G.edges[edge]['strike'] for edge in G.edges])

    geographic_positions = {}
    cartesian_positions = {}
    points = []
    
    lon_grid, lat_grid = np.meshgrid(np.linspace(-180, 180, 360), np.linspace(-90, 90, 180))

    lon_grid_shape = lon_grid.shape

    for node in G.nodes:
        lon, lat = G.nodes[node]['pos']
        # Transform the (col, row) coordinates to lat, lon assuming the image covers the entire globe
        lon = (lon / lon_grid_shape[1]) * 360 - 180
        lat = (lat / lon_grid_shape[0]) * 180 - 90

        geographic_positions[node] = (lon, lat)
        
        x, y, z = latlon_to_cartesian(lat, lon)
        points.append([x, y, z])
        cartesian_positions[node] = (x, y, z)


    if plot_figures.lower() == 'true':
        # Normalize strike values for colormap
        norm = plt.Normalize(vmin=strikes.min(), vmax=strikes.max())
        cmap = plt.cm.viridis

        # Create a PyVista plotter
        plotter = pv.Plotter()

        # Add the surface velocity field
        plotter.add_mesh(data, scalars='velocity_magnitude', cmap='inferno', clim=[-0.05, 0.05])

        # Add nodes to the plotter
        points = np.array([cartesian_positions[node] for node in G.nodes])
        point_cloud = pv.PolyData(points)
        plotter.add_mesh(point_cloud, color='red', point_size=0.1, render_points_as_spheres=True)

        # Prepare all edges in one MultiBlock with corresponding strike values
        lines = []
        for i, edge in enumerate(G.edges):
            pos1 = cartesian_positions[edge[0]]
            pos2 = cartesian_positions[edge[1]]
            line = pv.Line(pos1, pos2)
            # Assign the strike value as a scalar
            line['strike'] = np.array([strikes[i], strikes[i]])
            lines.append(line)

        # Combine all lines into a single MultiBlock
        multi_line = pv.MultiBlock(lines)

        # Add each block to the plotter
        for block in multi_line:
            plotter.add_mesh(block, scalars='strike', cmap='viridis', line_width=5.0)

        # Add a scalar bar for strikes
        # plotter.add_scalar_bar(title="Strike", cmap='viridis', n_labels=5, width=0.08, height=0.6, vertical=True)

        # Show the plot
        plotter.show()

    # %%
        
    ##Lets paralelised griddata
    # Function to perform griddata interpolation
    def interpolate_velocity(method):
        return griddata((longitude, latitude), velocity_magnitude, (lon_grid, lat_grid), method=method)

    # Parallelize the interpolation steps
    v_mag_interp, v_mag_nearest = Parallel(n_jobs=num_cores)(
        delayed(interpolate_velocity)(method) for method in ['linear', 'nearest']
    )

    # Handle missing values in the linear interpolation by filling with nearest interpolation
    v_mag_interp = np.where(np.isnan(v_mag_interp), v_mag_nearest, v_mag_interp)

    # Function to perform griddata interpolation for a specific velocity component
    def interpolate_velocity_component(velocity_component, method):
        return griddata((longitude, latitude), velocity_component, (lon_grid, lat_grid), method=method)

    # Parallelize the interpolation for each component and method
    v_x_interp, v_x_nearest = Parallel(n_jobs=num_cores)(
        delayed(interpolate_velocity_component)(v_x, method) for method in ['linear', 'nearest']
    )

    v_y_interp, v_y_nearest = Parallel(n_jobs=num_cores)(
        delayed(interpolate_velocity_component)(v_y, method) for method in ['linear', 'nearest']
    )

    v_z_interp, v_z_nearest = Parallel(n_jobs=num_cores)(
        delayed(interpolate_velocity_component)(v_z, method) for method in ['linear', 'nearest']
    )

    # Handle missing values in the linear interpolation by filling with nearest interpolation
    v_x_interp = np.where(np.isnan(v_x_interp), v_x_nearest, v_x_interp)
    v_y_interp = np.where(np.isnan(v_y_interp), v_y_nearest, v_y_interp)
    v_z_interp = np.where(np.isnan(v_z_interp), v_z_nearest, v_z_interp)

    # %%

    # Print the geographic positions to verify
    # for node, pos in geographic_positions.items():
    #     print(f"Node {node}: Longitude = {pos[0]}, Latitude = {pos[1]}")

    # Set the extent of the plot to match the grid boundaries
    lon_min, lon_max = lon_grid.min(), lon_grid.max()
    lat_min, lat_max = lat_grid.min(), lat_grid.max()

    # Plot the interpolated velocity magnitude
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the interpolated velocity magnitude on the map
    c_scheme = ax.imshow(v_mag_interp, extent=(lon_min, lon_max, lat_min, lat_max), 
                        transform=ccrs.PlateCarree(), cmap='inferno', origin='lower')

    # Add a colorbar to the plot
    cb = plt.colorbar(c_scheme, ax=ax, orientation='horizontal', pad=0.05)
    cb.set_label('Velocity Magnitude')

    # Overlay the edges of the graph on the plot
    for edge in G.edges:
        pos1 = geographic_positions[edge[0]]
        pos2 = geographic_positions[edge[1]]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], transform=ccrs.Geodetic(), 
                color='blue', linewidth=1.5)

    # Overlay the nodes of the graph with larger size and different color
    for node, pos in geographic_positions.items():
        ax.scatter(pos[0], pos[1], c='yellow', edgecolor='black', s=100, transform=ccrs.PlateCarree(), zorder=5)
        ax.text(pos[0], pos[1], str(node), fontsize=9, ha='right', transform=ccrs.PlateCarree(), zorder=6, color='black')

    # Set the title of the plot
    plt.title('Interpolated Velocity Magnitude with Fault Network Edges and Nodes')
    if plot_figures.lower() == 'true':
        # Show the plot
        plt.show()

    # ## Extract velocities
    # Now we want to pick up the velocity left and right of each fault, but to do this we first need to calculate the direction of the fault:

    # %%
    # Use the precalculated geographic positions
    for node, pos in geographic_positions.items():
        G.nodes[node]['pos'] = pos

    # Example usage
    G4 = calculate_direction(G, 3, geographic_positions)
    H = calculate_pickup_points(G4, pickup_factor,geographic_positions)

    # Prepare edge data in parallel
    edge_lines = Parallel(n_jobs=num_cores)(delayed(plot_edge)(H,edge) for edge in H.edges)

    # Prepare node data in parallel
    node_data = Parallel(n_jobs=num_cores)(delayed(plot_node)(H,node, pos) for node, pos in H.nodes(data='pos'))

    if plot_figures.lower() == 'true':
        # Plot the interpolated velocity magnitude
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot the interpolated velocity magnitude on the map
        c_scheme = ax.imshow(v_mag_interp, extent=(lon_min, lon_max, lat_min, lat_max), 
                            transform=ccrs.PlateCarree(), cmap='inferno', origin='lower')

        # Add coastlines for context
        # ax.coastlines()

        # Add a colorbar to the plot
        cb = plt.colorbar(c_scheme, ax=ax, orientation='horizontal', pad=0.05)
        cb.set_label('Velocity Magnitude')

        # Plot edges
        for x_coords, y_coords in edge_lines:
            ax.plot(x_coords, y_coords, transform=ccrs.Geodetic(), color='green', linewidth=5, zorder=3)
        # Plot nodes
        for x, y, color, label in node_data:
            ax.scatter(x, y, c=color, edgecolor='black', s=10, transform=ccrs.PlateCarree(), zorder=5)
            ax.text(x, y, label, fontsize=0, ha='right', transform=ccrs.PlateCarree(), zorder=6, color='black')
        # Set the title of the plot
        plt.title('Interpolated Velocity Magnitude with Pickup Points and Edges (Graph H)')
        # Show the plot
        if plot_figures.lower() == 'true':
            plt.show()
        image_file_path = os.path.join(new_image_directory, f'Pickup_points_{number_files}_{current_time / 1e6:.2f}.png')
        plt.savefig(image_file_path, dpi=200)
        plt.close("all")

    # %%

    # Define the zoom boundaries
    zoom_lon_min = -30  # Set your desired minimum longitude for zoom
    zoom_lon_max = 30   # Set your desired maximum longitude for zoom
    zoom_lat_min = -20  # Set your desired minimum latitude for zoom
    zoom_lat_max = 20   # Set your desired maximum latitude for zoom

    # Plot the interpolated velocity magnitude with zoom
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set the extent of the plot to the zoom boundaries
    ax.set_extent([zoom_lon_min, zoom_lon_max, zoom_lat_min, zoom_lat_max], crs=ccrs.PlateCarree())

    # Plot the interpolated velocity magnitude on the map
    c_scheme = ax.imshow(v_mag_interp, extent=(lon_min, lon_max, lat_min, lat_max), 
                        transform=ccrs.PlateCarree(), cmap='inferno', origin='lower')

    # Add coastlines for context
    # ax.coastlines()

    # Add a colorbar to the plot
    cb = plt.colorbar(c_scheme, ax=ax, orientation='horizontal', pad=0.05)
    cb.set_label('Velocity Magnitude')

    # Plot edges
    for x_coords, y_coords in edge_lines:
        ax.plot(x_coords, y_coords, transform=ccrs.Geodetic(), color='green', linewidth=1.5, zorder=3)

    # Plot nodes
    for x, y, color, label in node_data:
        ax.scatter(x, y, c=color, edgecolor='black', s=3, transform=ccrs.PlateCarree(), zorder=5)
        # ax.text(x, y, label, fontsize=3, ha='right', transform=ccrs.PlateCarree(), zorder=6, color='black')

    # Set the title of the plot
    plt.title('Interpolated Velocity Magnitude with Pickup Points and Edges (Graph H) - Zoomed In')

    if plot_figures.lower() == 'true':
        plt.show()
    image_file_path = os.path.join(new_image_directory, f'Pickup_zoom_points_{number_files}_{current_time / 1e6:.2f}.png')
    plt.savefig(image_file_path, dpi=200)
    plt.close("all")

    # %%
    # Example usage:
    lon_min, lon_max = lon_grid.min(), lon_grid.max()
    lat_min, lat_max = lat_grid.min(), lat_grid.max()

    # Assuming lon_min, lon_max, lat_min, lat_max are already defined

    # Extract v_x, v_y, v_z attributes using the interpolated grids
    H_extract = extract_attribute(H, v_x_interp, 'v_x', lon_min, lon_max, lat_min, lat_max)
    H_extract = extract_attribute(H, v_y_interp, 'v_y', lon_min, lon_max, lat_min, lat_max)
    H_extract = extract_attribute(H, v_z_interp, 'v_z', lon_min, lon_max, lat_min, lat_max)



    # %%
    # Apply the filtering
    H_extract2 = filter_pickup_points_sphere(G4, H_extract)


    # %%
    # Example usage:
    G_slip_rate = calculate_slip_rate_sphere(G4, H_extract2, dim=3) 
    # %%
    # Example usage:
    G_slip = calculate_slip_sphere(G_slip_rate, H_extract2, dt=current_dt, dim=3)

    # Save the processed graph in the new pickle directory
    pickle_file_path = os.path.join(new_pickle_directory, f'G_{number_files}.p')
    pickle.dump(G_slip, open(pickle_file_path, "wb"))
    
    # Check if slip and slip_rate have been computed
    for node in list(G_slip.nodes(data=True))[:5]:  # Check the first 5 nodes
        print(f"Node {node[0]} attributes after slip calculation: {node[1]}")

    # %%
    # Plot the slip rate in G_slip
    fig, ax = plt.subplots(figsize=(16, 4))

    # Ensure geographic positions are correctly used for visualization
    for node, pos in geographic_positions.items():
        G_slip_rate.nodes[node]['pos'] = pos

    plot_attribute_zero(G_slip_rate, 'slip_rate', ax)

    # Overlay the interpolated velocity component
    ax.imshow(v_x_interp, extent=(lon_min, lon_max, lat_min, lat_max), cmap='gray', origin='lower', alpha=0.6)

    # Set limits if needed, based on specific coordinates or data
    plt.xlim([lon_min, lon_max])
    plt.ylim([lat_min, lat_max])

    if plot_figures.lower() == 'true':
        plt.show()
    image_file_path = os.path.join(new_image_directory, f'slip_rate_{number_files}_{current_time / 1e6:.2f}.png')
    plt.savefig(image_file_path, dpi=200)
    plt.close("all")


    # Plot the slip in G
    fig, ax = plt.subplots(figsize=(16, 4))

    # Ensure geographic positions are correctly used for visualization
    for node, pos in geographic_positions.items():
        G_slip.nodes[node]['pos'] = pos

    plot_attribute_zero(G_slip, 'slip', ax)

    # Overlay the interpolated velocity component
    ax.imshow(v_x_interp, extent=(lon_min, lon_max, lat_min, lat_max), cmap='gray', origin='lower', alpha=0.6)

    # Set limits if needed, based on specific coordinates or data
    plt.xlim([lon_min, lon_max])
    plt.ylim([lat_min, lat_max])

    if plot_figures.lower() == 'true':
        plt.show()
    image_file_path = os.path.join(new_image_directory, f'slip_{number_files}_{current_time / 1e6:.2f}.png')
    plt.savefig(image_file_path, dpi=200)
    plt.close("all")


    #%% Length slip plot
    # Get all unique fault labels
    fault_labels = get_fault_labels_for_sphere(G_slip)

    # Initialize dictionaries to store the total length and maximum slip for each fault
    fault_lengths = {}
    fault_max_slips = {}

    # Iterate over all fault labels to calculate the total length and maximum slip for each fault
    for fault_label in fault_labels:
        # Use the get_fault_for_sphere function to get the fault subgraph, total length, and max slip
        fault_subgraph, total_length, max_slip = get_fault_for_sphere(G_slip, fault_label)
        
        # Store the results in the dictionaries
        fault_lengths[fault_label] = total_length
        fault_max_slips[fault_label] = max_slip

        # Print the total length and maximum slip for the current fault
        print(f"Fault Label: {fault_label}, Total Length: {total_length:.2f} km, Maximum Slip: {max_slip:.2f} meters")

    # Convert the dictionaries to lists for plotting
    lengths = list(fault_lengths.values())
    slips = list(fault_max_slips.values())

    # Create the scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    scatter = ax.scatter(lengths, slips, c=slips, s=100, cmap='seismic', vmin=0)
    ax.set_xlabel('Length (km)')
    ax.set_ylabel('Slip (meters)')
    ax.axis('equal')
    ax.set_xlim(0, 5000)  # Set limit for the lengths between 0 and 5,000 km
    ax.set_ylim(0, 100000)  # Ensure slip values start from 0 on the y-axis

    # Add colorbar using the scatter object
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label("Slip (meters)")

    # Optionally show the plot if desired
    if plot_figures.lower() == 'true':
        plt.show()

    # Save the plot to a file
    image_file_path = os.path.join(new_image_directory, f'length_slip_{number_files}_{current_time / 1e6:.2f}.png')
    fig.savefig(image_file_path, dpi=200)
    plt.close("all")

    print(f"Saved image to {image_file_path}")
    # print(f"Saved graph to {pickle_file_path}")

# Parallel processing of the files
results = Parallel(n_jobs=num_cores)(delayed(slip)(file, dt) for file in pickle_files_sorted)

# Stop the timer
stop = timeit.default_timer()

# Print the time taken
print('Extract time: ', stop - start)