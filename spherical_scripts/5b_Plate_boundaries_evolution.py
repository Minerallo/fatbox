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
# from matplotlib.patches import Polygon


plt.close("all")

# Set up parameters
num_cores = -1
plot_figures = 'false'

# Define output directory
output_directory = '/Users/ponsm/Desktop/software/fatbox/plate_boundaries/'

folder_path = '/Volumes/Jerry/global_models_3d/V06a_R01f_Rodinia_2GPa_llsvps_ps_1x50My_init_2Myint/'
folder_path_statistics = folder_path + 'statistics'

start = timeit.default_timer()

# Load and sort the solution surface files
# solution_directory_path = os.path.join(folder_path, 'solution/')
solution_directory_path = folder_path

solution_surface_files = [f for f in os.listdir(solution_directory_path) if f.startswith('solution_surface') and f.endswith('.vtu')]

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else float('inf')

solution_surface_files = sorted(solution_surface_files, key=extract_number)

new_image_directory = output_directory + 'img_evolution/'
os.makedirs(new_image_directory, exist_ok=True)

new_pickle_directory = output_directory + 'pickle_evolution/'
os.makedirs(new_pickle_directory, exist_ok=True)

# Load and sort pickle files
pickle_directory_slip = output_directory + 'pickle_slip/'
pickle_files = [f for f in os.listdir(pickle_directory_slip) if f.startswith('G_') and f.endswith('.p')]
pickle_files_sorted = sorted(pickle_files, key=extract_number)

times_all = get_times_solution_surface(folder_path_statistics)
dt = [t2 - t1 for t1, t2 in zip(times_all[:-1], times_all[1:])]
times = times_all[1:]

# Load graphs in parallel
def load_graph_with_index(index, file_name):
    file_pickle_path = os.path.join(pickle_directory_slip, file_name)
    G = pickle.load(open(file_pickle_path, 'rb'))
    print(f"Loaded graph from {file_pickle_path}")
    return index, G

results = Parallel(n_jobs=1)(delayed(load_graph_with_index)(index, file_name) for index, file_name in enumerate(pickle_files_sorted))
results_sorted = sorted(results, key=lambda x: x[0])
Gs = [result[1] for result in results_sorted]

print("All graphs have been loaded and sorted successfully.")

def bar_plot(attribute, faults, steps=[], ax=[]):
    colors = get_colors() 
    if ax == []:
        fig, ax = plt.subplots()  

    if steps == []:
        steps = range(attribute.shape[1])
    
    for n, step in enumerate(steps):
        bottom = 0
        for m, fault in enumerate(faults[:, step]):
            if np.isfinite(fault):
                a = attribute[m, step]
                ax.bar(n, a, 1, bottom=bottom, alpha=0.75, edgecolor='white', color=colors[int(fault),:])
                bottom += a
            else:
                break

# Initialize fault tracking arrays
max_comp = 45
faults = np.zeros((max_comp, len(Gs)))
faults[:, :] = np.nan

lengths = np.zeros((max_comp, len(Gs)))
lengths[:, :] = np.nan

displacements = np.zeros((max_comp, len(Gs)))
displacements[:, :] = np.nan

# Process each graph
for time in range(len(Gs)):
    G = Gs[time]
    labels = get_fault_labels_for_sphere(G)

    # Resize arrays if necessary
    if len(labels) > max_comp:
        new_faults = np.zeros((len(labels), len(Gs)))
        new_faults[:, :] = np.nan
        new_faults[:faults.shape[0], :faults.shape[1]] = faults

        new_lengths = np.zeros((len(labels), len(Gs)))
        new_lengths[:, :] = np.nan
        new_lengths[:lengths.shape[0], :lengths.shape[1]] = lengths

        new_displacements = np.zeros((len(labels), len(Gs)))
        new_displacements[:, :] = np.nan
        new_displacements[:displacements.shape[0], :displacements.shape[1]] = displacements

        faults = new_faults
        lengths = new_lengths
        displacements = new_displacements
        max_comp = len(labels)

    for n, label in enumerate(labels):
        fault, total_length, max_displacement = get_fault_for_sphere(G, label)
        faults[n, time] = label    
        lengths[n, time] = total_length
        displacements[n, time] = max_displacement

    lengths[:, time] = np.sort(lengths[:, time])
    displacements[:, time] = np.sort(displacements[:, time])

# Plot fault evolution over time
fig, ax = plt.subplots(figsize=(16, 5))
bar_plot(np.ones_like(faults), faults, steps=range(len(Gs)), ax=ax)
ax.set_xlim([-0.5, len(Gs) - 0.5])
ax.set_xlabel('Time Step')
ax.set_ylabel('Faults')
ax.set_title('Fault Evolution Over Time')

# Save the figure
bar_plot_filename = os.path.join(new_image_directory, 'fault_evolution_over_time.png')
fig.savefig(bar_plot_filename, dpi=200)
if plot_figures.lower() == 'true':
    plt.show()
plt.close()

def stack_plot(attribute, faults, steps, ax=[]):
    colors = get_colors()
    
    if ax == []:
        fig, ax = plt.subplots()  

    max_fault = int(np.nanmax(faults))

    x = np.arange(len(steps))
    y = np.zeros((max_fault, len(steps)))
        
    for N in range(max_fault):
        for n in steps:
            row = faults[:, n]
            if N in faults[:, n]:
                index = np.where(row == N)[0][0]
                y[N, n] = attribute[index, n]
            
    ax.stackplot(x, y, colors=colors[:max_fault, :], alpha=0.75, edgecolor='white', linewidth=0.5)  # Use 'colors' instead of 'fc'

# Plot faults weighted by length
steps = range(len(Gs))
fig, ax = plt.subplots(figsize=(16, 5))
stack_plot(lengths, faults, steps, ax)
ax.set_xlim([0, len(Gs)-1])
ax.set_xlabel('Time')
ax.set_ylabel('Faults weighted by length')

# Save the figure
length_plot_filename = os.path.join(new_image_directory, 'faults_weighted_by_length.png')
fig.savefig(length_plot_filename, dpi=200)
if plot_figures.lower() == 'true':
    plt.show()
plt.close()


# Plot faults weighted by displacement
fig, ax = plt.subplots(figsize=(16, 5))
stack_plot(displacements / 1000, faults, steps, ax)
ax.set_xlim([0, len(Gs)-1])
ax.set_xlabel('Time')
ax.set_ylabel('Faults weighted by displacement (km)')

# Save the figure
displacement_plot_filename = os.path.join(new_image_directory, 'faults_weighted_by_displacement.png')
fig.savefig(displacement_plot_filename, dpi=200)
if plot_figures.lower() == 'true':
    plt.show()
plt.close()

# # Initialize a new graph H to track faults over time
# H = nx.Graph()

# # Plot limited faults
# limited_faults = 25
# fig_limited = plot_faults_evolution(Gs, limited_faults=limited_faults, file_suffix=f"_limited_{limited_faults}")

# plot_filename = os.path.join(new_image_directory, f'fault_evolution_graph_limited_{limited_faults}.png')
# fig_limited.savefig(plot_filename, dpi=200)
# print(f"Figure saved as {plot_filename}")
# if plot_figures.lower() == 'true':

#     fig_limited.show()

# # Plot all faults (no limit)
# fig_all = plot_faults_evolution(Gs, limited_faults=None, file_suffix="_all_faults")

# plot_filename = os.path.join(new_image_directory, f'fault_evolution_graph_all_faults.png')
# fig_all.savefig(plot_filename, dpi=200)
# print(f"Figure saved as {plot_filename}")
# if plot_figures.lower() == 'true':
#     fig_all.show()

# #%%
# fault_number = 10

# H_sub = nx.subgraph(H, [node for node in H if H.nodes[node]['fault']==fault_number])

# fig, ax = plt.subplots(figsize=(20,10))
# nx.draw(H_sub,
#         pos = nx.get_node_attributes(H_sub, 'pos'),
#         labels=nx.get_node_attributes(H_sub, 'time'),
#         with_labels=True,
#         node_color = get_node_colors(H_sub, 'fault'),
#         ax=ax)
# plt.show()

# # Initialize two graphs: one for all faults and one for a limited number of faults
# H = nx.Graph()
# H_limited = nx.Graph()

# # Number of faults to display in the limited graph
# limited_faults = 25

# # Process each graph and update both H (all faults) and H_limited (limited faults) with nodes and edges
# for time in range(len(Gs) - 1):
#     print(f"Processing time step: {time}")

#     G_0 = Gs[time]
#     G_1 = Gs[time + 1]

#     # Add nodes for all faults
#     if time == 0:
#         H = H_add_nodes(H, G_0, time, limited_faults=None)
#     H = H_add_nodes(H, G_1, time + 1, limited_faults=None)

#     # Add nodes for limited faults
#     if time == 0:
#         H_limited = H_add_nodes(H_limited, G_0, time, limited_faults=limited_faults)
#     H_limited = H_add_nodes(H_limited, G_1, time + 1, limited_faults=limited_faults)

#     # Get dictionaries for fault labels and nodes
#     dic_0 = get_dictionary_sphere(G_0)
#     dic_1 = get_dictionary_sphere(G_1)
#     faults_0 = dic_0.keys()
#     faults_1 = dic_1.keys()

#     # Get common faults
#     faults = list(set(faults_0).intersection(set(faults_1)))

#     # Add edges based on the fault relationships for both full and limited graphs
#     for fault in faults:
#         starts = dic_0[fault]
#         ends = dic_1[fault]

#         if len(starts) == 1 and len(ends) == 1:
#             H.add_edge((time, starts[0]), (time + 1, ends[0]))
#             if fault in list(faults)[:limited_faults]:
#                 H_limited.add_edge((time, starts[0]), (time + 1, ends[0]))

#         elif len(starts) == 1 and len(ends) > 1:
#             for end in ends:
#                 H.add_edge((time, starts[0]), (time + 1, end))
#                 if fault in list(faults)[:limited_faults]:
#                     H_limited.add_edge((time, starts[0]), (time + 1, end))

#         elif len(starts) > 1 and len(ends) == 1:
#             for start in starts:
#                 H.add_edge((time, start), (time + 1, ends[0]))
#                 if fault in list(faults)[:limited_faults]:
#                     H_limited.add_edge((time, start), (time + 1, ends[0]))

#         elif len(starts) > 1 and len(ends) > 1:
#             if len(starts) == len(ends):
#                 for start, end in zip(starts, ends):
#                     H.add_edge((time, start), (time + 1, end))
#                     if fault in list(faults)[:limited_faults]:
#                         H_limited.add_edge((time, start), (time + 1, end))
#             elif len(starts) < len(ends):
#                 minimum = min(len(starts), len(ends))
#                 difference = len(ends) - len(starts)
#                 for n in range(minimum):
#                     H.add_edge((time, starts[n]), (time + 1, ends[n]))
#                     if fault in list(faults)[:limited_faults]:
#                         H_limited.add_edge((time, starts[n]), (time + 1, ends[n]))
#                 for n in range(difference):
#                     H.add_edge((time, starts[minimum-1]), (time + 1, ends[minimum-1+n+1]))
#                     if fault in list(faults)[:limited_faults]:
#                         H_limited.add_edge((time, starts[minimum-1]), (time + 1, ends[minimum-1+n+1]))
#             elif len(starts) > len(ends):
#                 minimum = min(len(starts), len(ends))
#                 difference = len(starts) - len(ends)
#                 for n in range(minimum):
#                     H.add_edge((time, starts[n]), (time + 1, ends[n]))
#                     if fault in list(faults)[:limited_faults]:
#                         H_limited.add_edge((time, starts[n]), (time + 1, ends[n]))
#                 for n in range(difference):
#                     H.add_edge((time, starts[minimum-1+n+1]), (time + 1, ends[minimum-1]))
#                     if fault in list(faults)[:limited_faults]:
#                         H_limited.add_edge((time, starts[minimum-1+n+1]), (time + 1, ends[minimum-1]))

# # Plot and save the graph with all faults
# fig = plot_faults_evolution(H, file_suffix="")
# plot_filename = os.path.join(new_image_directory, 'fault_evolution_graph.png')
# # fig.savefig(plot_filename, dpi=200)
# plt.show()

# # Plot and save the graph with limited faults
# fig_limited = plot_faults_evolution(H_limited, file_suffix=f"_limited_{limited_faults}")
# plot_filename_limited = os.path.join(new_image_directory, f'fault_evolution_graph_limited_{limited_faults}.png')
# # fig_limited.savefig(plot_filename_limited, dpi=200)
# plt.show()

# #%%
# # Process each graph and update H with nodes and edges
# H = nx.Graph()  # Initialize H outside the loop

# for time in range(len(Gs) - 1):
#     print(time)

#     G_0 = Gs[time]
#     G_1 = Gs[time + 1]

#     if time == 0:
#         H = H_add_nodes(H, G_0, time, limited_faults=None)  # Add nodes from G_0
#     H = H_add_nodes(H, G_1, time + 1, limited_faults=None)  # Add nodes from G_1

#     # Get dictionaries for fault labels and nodes
#     dic_0 = get_dictionary_sphere(G_0)
#     dic_1 = get_dictionary_sphere(G_1)
#     faults_0 = dic_0.keys()
#     faults_1 = dic_1.keys()

#     # Get common faults
#     faults = list(set(faults_0).intersection(set(faults_1)))

#     # Add edges based on the fault relationships
#     for fault in faults:
#         starts = dic_0[fault]
#         ends = dic_1[fault]

#         if len(starts) == 1 and len(ends) == 1:
#             H.add_edge((time, starts[0]), (time+1, ends[0]))
#         elif len(starts) == 1 and len(ends) > 1:
#             for end in ends:
#                 H.add_edge((time, starts[0]), (time+1, end))
#         elif len(starts) > 1 and len(ends) == 1:
#             for start in starts:
#                 H.add_edge((time, start), (time+1, ends[0]))
#         elif len(starts) > 1 and len(ends) > 1:
#             if len(starts) == len(ends):
#                 for start, end in zip(starts, ends):
#                     H.add_edge((time, start), (time+1, end))
#             elif len(starts) < len(ends):
#                 minimum = min(len(starts), len(ends))
#                 difference = len(ends) - len(starts)
#                 for n in range(minimum):
#                     H.add_edge((time, starts[n]), (time+1, ends[n]))
#                 for n in range(difference):
#                     H.add_edge((time, starts[minimum-1]), (time+1, ends[minimum-1+n+1]))
#             elif len(starts) > len(ends):
#                 minimum = min(len(starts), len(ends))
#                 difference = len(starts) - len(ends)
#                 for n in range(minimum):
#                     H.add_edge((time, starts[n]), (time+1, ends[n]))                
#                 for n in range(difference):
#                     H.add_edge((time, starts[minimum-1+n+1]), (time+1, ends[minimum-1]))

# # Plot and save the graph
# fig = plot_faults_evolution(H, file_suffix="")
# plot_filename = os.path.join(new_image_directory, 'fault_evolution_graph.png')
# fig.savefig(plot_filename, dpi=200)
# plt.show()


# Initialize the graph H to track faults over time
H = nx.Graph()

# Process each graph and update H with nodes and edges
for time in range(len(Gs) - 1):
    print(time)

    G_0 = Gs[time]
    G_1 = Gs[time + 1]

    # Add nodes for G_0 and G_1
    if time == 0:
        H = H_add_nodes_sphere(H, G_0, time)
    H = H_add_nodes_sphere(H, G_1, time + 1)

    # Loop over faults in G_0 and G_1 to create edges
    for label_0 in get_fault_labels_for_sphere(G_0):
        for label_1 in get_fault_labels_for_sphere(G_1):
            if label_0 == label_1:
                H.add_edge((time, label_0), (time + 1, label_1))

plt.figure(figsize=(20,10))
nx.draw(H,
        pos = nx.get_node_attributes(H, 'pos'),
        labels=nx.get_node_attributes(H, 'fault'),
        with_labels=True,
        node_color = get_node_colors(H, 'fault'))
if plot_figures.lower() == 'true':
    plt.show()
plot_filename = os.path.join(new_image_directory, 'fault_evolution_basic.png')
plt.savefig(plot_filename, dpi=200)
plt.close()

# Plot the graph with ordered nodes
plt.figure(figsize=(20,10))
nx.draw(H,
        pos = nx.get_node_attributes(H, 'pos'),
        labels=nx.get_node_attributes(H, 'fault'),
        with_labels=True,
        node_color = get_node_colors(H, 'fault'),
        node_size = [H.nodes[node]['displacement']*0.005 for node in H])
if plot_figures.lower() == 'true':
    plt.show()
plot_filename = os.path.join(new_image_directory, 'fault_evolution_scaled.png')
plt.savefig(plot_filename, dpi=200)
plt.close()

# Define the factor for scaling the displacement values
factor = 0.0001

# Create the width dictionary, mapping nodes to their scaled displacement values
width = dict(zip([node for node in H], [H.nodes[node]['displacement'] * factor for node in H]))

# Create the plot
fig, ax = plt.subplots(figsize=(20, 10))

# Plot the graph with the specified widths
ax = plot_width_sphere(H, ax, width, tips=False)

# Set the x-axis limits to range between 0 and the length of the times array
ax.set_xlim(0, len(times))

# Set the y-axis to correspond to the labels
y_labels = sorted(set(node[1] for node in H.nodes))
ax.set_ylim(min(y_labels) - 1, max(y_labels) + 1)

# Set y-axis labels to be the fault labels
ax.set_yticks(y_labels)
ax.set_yticklabels(y_labels)
if plot_figures.lower() == 'true':
    # Display the plot
    plt.show()
plot_filename = os.path.join(new_image_directory, 'fault_evolution_edge_widths.png')
plt.savefig(plot_filename, dpi=200)
plt.close()

#%%
# Stop the timer
stop = timeit.default_timer()

# Print the time taken
print('Extract time: ', stop - start)
