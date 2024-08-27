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

# Define output directory
output_directory = '/Users/ponsm/Desktop/software/fatbox/plate_boundaries/'

folder_path = '/Volumes/Jerry/global_models_3d/V06a_R01f_Rodinia_2GPa_llsvps_ps_1x50My_init_2Myint/'

folder_path_statistics = folder_path + 'statistics'

start = timeit.default_timer()

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

new_image_directory = output_directory + 'img_displacement/'
os.makedirs(new_image_directory, exist_ok=True)

new_pickle_directory = output_directory + 'pickle_displacement/'
os.makedirs(new_pickle_directory, exist_ok=True)

# Define the directory where the pickle files are stored
pickle_directory_slip= output_directory + 'pickle_slip/'

# List all files in the directory
# all_files = os.listdir(pickle_directory)

# Filter and sort out only the .p (pickle) files
pickle_files = [f for f in os.listdir(pickle_directory_slip) if f.startswith('G_') and f.endswith('.p')]
pickle_files_sorted = sorted(pickle_files, key=extract_number)

# Count the number of pickle files
num_pickle_files = len(pickle_files_sorted)
print(f"Number of pickle files found: {num_pickle_files}")

times_all = get_times_solution_surface(folder_path_statistics)

 # Calculate the time intervals (dt) between consecutive times
dt = [t2 - t1 for t1, t2 in zip(times_all[:-1], times_all[1:])]
times =times_all[1:]
# Initialize the list to store loaded graphs
Gs = []

# Function to load a single pickle file
def load_graph_with_index(index, file_name):
    file_pickle_path = os.path.join(pickle_directory_slip, file_name)
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

times_all = get_times_solution_surface(folder_path_statistics)

 # Calculate the time intervals (dt) between consecutive times
dt = [t2 - t1 for t1, t2 in zip(times_all[:-1], times_all[1:])]
times =times_all[1:]
# Print the time intervals
print("Time intervals (dt) between consecutive steps:", dt)

#%%
max_comp = 0

# def process_time_step(time, Gs, pickle_files_sorted, times):
# We want to make sure the labels never exceed a.
fault_ages = [0] * 50000  # Initialize fault ages array
fault_slip = [0] * 50000  # Initialize fault slip array
active_limit = 1e-4  # 0.1 mm/yr

# Loop over each time step
for time in tqdm(range(len(Gs)-1)):
                 
    number_files = extract_output_number_from_pickle_file(pickle_files_sorted[time])
    
    number_files_for_time = int(number_files)
    current_time = times[number_files_for_time]

    G_0 = Gs[time]
    G_1 = Gs[time+1]

    if time == 0:
        G_0 = write_slip_to_displacement(G_0, dim=2)  
        Gs[time] = G_0
    
    G_1 = write_slip_to_displacement(G_1, dim=2)

    cf = common_faults_sphere(G_0, G_1)
    print(cf)

    for fault in cf:
        fault_graph_0, _, _ = get_fault_for_sphere(G_0, fault)
        fault_graph_1, _, _ = get_fault_for_sphere(G_1, fault)
        
        points_0 = get_displacement_sphere(fault_graph_0, dim=3)
        points_1 = get_displacement_sphere(fault_graph_1, dim=3)
        
        for n in range(points_1.shape[0]):
            index = closest_node(points_1[n, 1:3], points_0[:, 1:3])
            points_1[n, 3] += points_0[index][3]
            points_1[n, 4] += points_0[index][4]
            points_1[n, 5] += points_0[index][5]
        
        G_1 = assign_displacement(G_1, points_1, dim=3)

    # Update fault ages
    labels_0 = get_fault_labels(G_0)
    labels_1 = get_fault_labels(G_1)

    # Assume fault was active during the entire previous timestep
    dt_current = dt[number_files_for_time]

    for label in labels_0:
        fault = get_fault(G_0, label)
        fault_slip[label] = compute_node_values(fault, 'slip_rate', 'max')
        
        if fault_slip[label] >= active_limit and label in labels_1:
            fault_ages[label] += dt_current

    # Assign fault ages and activity status to nodes in G_1
    for node in G_1.nodes:
        lab = G_1.nodes[node]['fault']
        G_1.nodes[node]['fault_age'] = fault_ages[lab]
        G_1.nodes[node]['fault_active'] = 1 if fault_slip[lab] >= active_limit else 0

    # Save the graph with updated fault ages
    processed_pickle_file_name = f"G_{number_files}.p"
    processed_pickle_file_path = os.path.join(new_pickle_directory, processed_pickle_file_name)
    with open(processed_pickle_file_path, 'wb') as pfile:
        pickle.dump(G_1, pfile)

    # Plot displacement for G_0
    fig_0 = plot_displacement_sphere(G_0, f'Displacement at time {current_time / 1e6:.2f} My for G_0', plot_figures)
    file_name_with_number = f"G_0_{number_files}_{current_time / 1e6:.2f}.png"
    file_path_output = os.path.join(new_image_directory, file_name_with_number)
    fig_0.savefig(file_path_output, dpi=200)
    plt.close(fig_0)  # Close the figure

    # Plot displacement for G_1
    fig_1 = plot_displacement_sphere(G_1, f'Displacement at time {current_time / 1e6:.2f} My for G_1', plot_figures)
    file_name_with_number = f"G_1_{number_files}_{current_time / 1e6:.2f}.png"
    file_path_output = os.path.join(new_image_directory, file_name_with_number)
    fig_1.savefig(file_path_output, dpi=200)
    plt.close(fig_1)  # Close the figure

    # Plot fault ages
    fault_labels_plot = [i for i, age in enumerate(fault_ages) if age > 0]  # Only include faults with non-zero ages
    fault_ages_filtered = [age for age in fault_ages if age > 0]

    plt.figure(figsize=(12, 8))
    plt.scatter(fault_labels_plot, fault_ages_filtered, c=fault_ages_filtered, cmap='viridis', s=100)
    plt.colorbar(label='Fault Age (years)')
    plt.xlabel('Fault Label')
    plt.ylabel('Fault Age (years)')
    plt.title(f'Fault Ages at Time {current_time / 1e6:.2f} My')
    plt.grid(True)

    # Save the fault age plot to a file
    fault_age_image_file_path = os.path.join(new_image_directory, f'fault_ages_{number_files}_{current_time / 1e6:.2f}.png')
    plt.savefig(fault_age_image_file_path, dpi=200)
    plt.close()

    print(f"Saved fault age plot to {fault_age_image_file_path}")

    # Plot fault ages on a map
    fig_age = plot_fault_age_map(G_1, f'Fault Ages at Time {current_time / 1e6:.2f} My', plot_figures, fault_ages)
    file_name_with_number = f"fault_ages_map_{number_files}_{current_time / 1e6:.2f}.png"
    file_path_output = os.path.join(new_image_directory, file_name_with_number)
    fig_age.savefig(file_path_output, dpi=200)
    plt.close(fig_age)  # Close the figure

# # Parallel processing of the files
# Parallel(n_jobs=num_cores)(
#     delayed(process_time_step)(time, Gs, pickle_files_sorted, times) for time in tqdm(range(len(Gs) - 1))
# )
# Stop the timer
stop = timeit.default_timer()

# Print the time taken
print('Extract time: ', stop - start)