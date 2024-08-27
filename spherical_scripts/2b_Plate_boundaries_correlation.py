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



# %%
plt.close("all")
# Set up parameters
num_cores = -1
plot_figures = 'false'

# Define output directory
output_directory = '/Users/ponsm/Desktop/software/fatbox/plate_boundaries/'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

folder_path = '/Volumes/Jerry/global_models_3d/V06a_R01f_Rodinia_2GPa_llsvps_ps_1x50My_init_2Myint/'

folder_path_statistics = folder_path + 'statistics'

start = timeit.default_timer()

times = get_times_solution_surface(folder_path_statistics)

# Fault Extraction: Load strain rate map data at the surface of the model
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

new_image_directory =  output_directory + 'img_correlation/'
os.makedirs(new_image_directory, exist_ok=True)

new_pickle_directory = output_directory + 'pickle_correlation/'
os.makedirs(new_pickle_directory, exist_ok=True)

# Define the directory where the pickle files are stored
pickle_directory_networks = output_directory + 'pickle_networks/'

# List all files in the directory
# all_files = os.listdir(pickle_directory)

# Filter and sort out only the .p (pickle) files
pickle_files = [f for f in os.listdir(pickle_directory_networks) if f.startswith('G_') and f.endswith('.p')]
pickle_files_sorted = sorted(pickle_files, key=extract_number)

# Count the number of pickle files
num_pickle_files = len(pickle_files_sorted)
print(f"Number of pickle files found: {num_pickle_files}")

# Calculate the time intervals (dt) between consecutive times
dt = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]

# Initialize the list to store loaded graphs
Gs = []

# Function to load a single pickle file
def load_graph_with_index(index, file_name):
    file_pickle_path = os.path.join(pickle_directory_networks, file_name)
    G = pickle.load(open(file_pickle_path, 'rb'))
    print(f"Loaded graph from {file_pickle_path}")
    return index, G

# Use Parallel to load the graphs with their indices, 
#we should avoid using other than 1 node here or it will mess with the sorting of G
results = Parallel(n_jobs=1)(delayed(load_graph_with_index)(index, file_name) for index, file_name in enumerate(pickle_files_sorted))

# Sort the results by index to maintain order
results_sorted = sorted(results, key=lambda x: x[0])

# Extract the loaded graphs in the correct order
Gs = [result[1] for result in results_sorted]

# Now Gs contains all the loaded graphs in the correct order
print("All graphs have been loaded and sorted successfully.")

#%% The more recent way it to import everything and store all data in GS 

# Initialize the list to store loaded graphs
# Gs = []

# # Define the directory where the pickle files are stored
# directory_pickle_path = '/Users/ponsm/Desktop/software/fatbox/plate_boundaries/pickle_networks'

# # Loop to load the graphs
# for n in range(1, 5):  # Adjusted the range to include all four files
#     # Construct the full file path for each pickle file
#     file_pickle_path = directory_pickle_path + 'G_' + str(n).zfill(5) + '.p'
    
#     # Load the graph from the pickle file and append it to the list
#     Gs.append(pickle.load(open(file_pickle_path, 'rb')))
    
#     # Print a message indicating which file was loaded
#     print(f"Loaded graph from {file_pickle_path}")

# # Now Gs contains all the loaded graphs
# print("All graphs have been loaded successfully.")

# # Loop to plot all graphs
# for jj, G in enumerate(Gs):
#     fig, ax = plt.subplots(figsize=(16, 4))
#     plot_components(G, node_size=1, ax=ax)
#     ax.set_title(f"Graph {jj+1}")
#     plt.show()
#     # Optionally, you can save the plot instead of or in addition to showing it:
#     # plt.savefig(f"graph_{jj+1}.png")

# %%
max_comp = 0

for time in tqdm(range(len(Gs)-1)):

    number_files = extract_output_number_from_pickle_file(pickle_files_sorted[time])
    
    number_files_for_time= int(number_files)
    # print(number_files)
    current_time=times[number_files_for_time]


    G_0 = Gs[time]
    G_1 = Gs[time+1]

# #%%
# G_0 = Gs[0]
# G_1 = Gs[1]

    # Apply this to your graphs
    assign_fault_labels(G_0)
    assign_fault_labels(G_1)

    # Example usage for G_0 and G_1
    num_faults_G0 = count_faults(G_0)
    num_faults_G1 = count_faults(G_1)

    print(f"Number of faults in G_0: {num_faults_G0}")
    print(f"Number of faults in G_1: {num_faults_G1}")

    # %%
    R = 5
    correlations, smf, smb = correlation_slow(G_0, G_1, R=R)

    # %%
    fig, axs = plt.subplots(1, 2, figsize=(16,8))
    # title_str = f"Time: {current_time / 1e6:.2f} My"
    # ax.set_title(title_str, fontsize=18)
    axs[0].set_title('Similarity (forward)')
    im_0 = axs[0].imshow(smf, cmap='Blues_r')
    axs[0].set_yticks(range(smf.shape[0]))
    axs[0].set_yticklabels(get_fault_labels(G_0), fontsize=8)
    axs[0].set_xticks(range(smf.shape[1]))
    axs[0].set_xticklabels(get_fault_labels(G_1), fontsize=8)

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = fig.colorbar(im_0, cax=cax, orientation='vertical')
    cbar.set_label('Similarity')

    cbar.ax.plot([0, 800], [R]*2, 'r')

    for x in range(smf.shape[0]):
        for y in range(smf.shape[1]):
            if smf[x,y] < R:
                axs[0].text(y-0.25,x+0.25, int(smf[x,y]), color='red', fontsize=8)
                rect = patches.Rectangle((y-0.5, x-0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
                axs[0].add_patch(rect)
            else:
                pass


    axs[1].set_title('Similarity (backward)')
    im_1 = axs[1].imshow(np.transpose(smb), cmap='Blues_r')
    axs[1].set_yticks(range(smf.shape[0]))
    axs[1].set_yticklabels(get_fault_labels(G_0), fontsize=8)
    axs[1].set_xticks(range(smf.shape[1]))
    axs[1].set_xticklabels(get_fault_labels(G_1), fontsize=8)

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = fig.colorbar(im_1, cax=cax, orientation='vertical')
    cbar.set_label('Similarity')

    cbar.ax.plot([0, 800], [R]*2, 'r')

    for x in range(smb.shape[0]):
        for y in range(smb.shape[1]):
            if smb[x,y] < R:
                axs[1].text(x-0.25,y+0.25, int(smb[x,y]), color='red', fontsize=8)
                rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
                axs[1].add_patch(rect)
            else:
                pass

    if plot_figures.lower() == 'true':
        plt.show()
    # Save the plot to the specified directory with the time step in the filename
    file_name_with_number = f"Corr_Mat_{number_files}_{current_time / 1e6:.2f}.png"
    file_path_ouput = os.path.join(new_image_directory, file_name_with_number)
    plt.savefig(file_path_ouput, dpi=200)

    # %%
    # Define the colormap for correlation
    cmap_rb = mcolors.ListedColormap(['red', 'blue'])  # Blue for correlated, Red for anticorrelated


    # Assign correlation status to nodes in G_0
    for node in G_0:
        if G_0.nodes[node]['fault'] in [corr[0] for corr in correlations]:  # Check if fault is correlated
            G_0.nodes[node]['correlated'] = 1  # Correlated
        else:
            G_0.nodes[node]['correlated'] = 0  # Anticorrelated

    # Assign correlation status to nodes in G_1
    for node in G_1:
        if G_1.nodes[node]['fault'] in [corr[1] for corr in correlations]:  # Check if fault is correlated
            G_1.nodes[node]['correlated'] = 1  # Correlated
        else:
            G_1.nodes[node]['correlated'] = 0  # Anticorrelated

    # Plot for G_0
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_title('Time 0')
    nodes = plot_attribute_one(G_0, 'correlated', cmap=cmap_rb, vmin=0, vmax=1, ax=ax)
    fig.colorbar(nodes, ax=ax)
    if plot_figures.lower() == 'true':
            plt.show()
    # Save the plot to the specified directory with the time step in the filename
    file_name_with_number = f"G_0_corr_{number_files}_{current_time / 1e6:.2f}.png"
    file_path_ouput = os.path.join(new_image_directory, file_name_with_number)
    plt.savefig(file_path_ouput, dpi=200)

    # Plot for G_1
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_title('Time 1')
    nodes = plot_attribute_one(G_1, 'correlated', cmap=cmap_rb, vmin=0, vmax=1, ax=ax)
    fig.colorbar(nodes, ax=ax)
    if plot_figures.lower() == 'true':
        plt.show()
    # Save the plot to the specified directory with the time step in the filename
    file_name_with_number = f"G_1_corr_{number_files}_{current_time / 1e6:.2f}.png"
    file_path_ouput = os.path.join(new_image_directory, file_name_with_number)
    plt.savefig(file_path_ouput, dpi=200)


    # %%
    # Assuming G_0 is the graph you want to inspect

    # Print all graph-level attributes
    print("Graph attributes:")
    for attr, value in G_0.graph.items():
        print(f"{attr}: {value}")

    # Print all node attributes
    print("\nNode attributes:")
    for node in G_0.nodes(data=True):
        print(f"Node {node[0]}: {node[1]}")

    # Print all edge attributes
    print("\nEdge attributes:")
    for edge in G_0.edges(data=True):
        print(f"Edge {edge[0]}-{edge[1]}: {edge[2]}")

    # Additionally, to get specific attributes
    print("\nSpecific node attributes:")
    for node, attr_dict in G_0.nodes(data=True):
        for attr, value in attr_dict.items():
            print(f"Node {node} has {attr}: {value}")

    print("\nSpecific edge attributes:")
    for (u, v, attr_dict) in G_0.edges(data=True):
        for attr, value in attr_dict.items():
            print(f"Edge {u}-{v} has {attr}: {value}")

    # Iterate over all edges and check for 'strike' attribute
    for edge in G_0.edges(data=True):
        u, v, attr = edge
        if 'strike' in attr:
            print(f"Edge {u}-{v} has 'strike' attribute with value: {attr['strike']}")
        else:
            print(f"Edge {u}-{v} does not have 'strike' attribute.")

    # Check if the 'strike' attribute is present for all edges
    edge_strikes_present = all('strike' in G_0.edges[edge] for edge in G_0.edges)
    print("All edges have 'strike' attribute:", edge_strikes_present)

    filename_output = f'G_{number_files}.p'
    file_path_out = os.path.join(new_pickle_directory, filename_output)

    # Save the graph as a pickle file
    with open(file_path_out, 'wb') as p:
        pickle.dump(G_1, p)

    # print(f"Object G for timestep {i+1} has been saved to {file_path_out}")
    print(f"Object G has been saved to {file_path_out}")



