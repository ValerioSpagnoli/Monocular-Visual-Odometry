from src.geometry_utils import *

import plotly.graph_objects as go   
import matplotlib.pyplot as plt
import numpy as np


def plot_points(fig, points, name='points', mode='markers', color='blue', size=2, width=2):     

    x_coords = [position[0] for position in points]
    y_coords = [position[1] for position in points]
    z_coords = [position[2] for position in points]

    if mode == 'lines': fig.add_trace(go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode='lines', name=name, line=dict(color=color, width=width)))
    elif mode == 'markers': fig.add_trace(go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode='markers', name=name, marker=dict(color=color, size=size)))

def plot_matches(fig, set_1, set_2, name='Matches', color='blue', width=2):

    for i in range(len(set_1)):
        x_coord_1 = set_1[i][0]
        y_coord_1 = set_1[i][1]
        z_coord_1 = set_1[i][2]

        x_coord_2 = set_2[i][0]
        y_coord_2 = set_2[i][1]
        z_coord_2 = set_2[i][2]

        x_coords = [x_coord_1, x_coord_2]
        y_coords = [y_coord_1, y_coord_2]
        z_coords = [z_coord_1, z_coord_2]

        fig.add_trace(go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode='lines', name=name, legendgroup=name, showlegend=(i==0), line=dict(color=color, width=width)))

def plot_icp_frame(set_1, set_2, save_path, title='ICP Iteration', set_1_title='Set 1', set_1_color='green', set_1_marker='o', set_2_title='Set 2', set_2_color='red', set_2_marker='x'):

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=200)
    ax[0].imshow(np.ones((480, 640, 3)))
    ax[0].scatter([point[0] for point in set_1], [point[1] for point in set_1], color=set_1_color, marker=set_1_marker)
    ax[0].set_xticks(np.arange(0, 641, 80))
    ax[0].set_yticks(np.arange(0, 481, 80))
    ax[0].grid()
    ax[0].set_title(set_1_title)

    ax[1].imshow(np.ones((480, 640, 3)))
    ax[1].scatter([point[0] for point in set_2], [point[1] for point in set_2], color=set_2_color, marker=set_2_marker)
    ax[1].set_xticks(np.arange(0, 641, 80))
    ax[1].set_yticks(np.arange(0, 481, 80))
    ax[1].grid()
    ax[1].set_title(set_2_title)

    ax[2].imshow(np.ones((480, 640, 3)))
    ax[2].scatter([point[0] for point in set_1], [point[1] for point in set_1], color=set_1_color, marker=set_1_marker)
    ax[2].scatter([point[0] for point in set_2], [point[1] for point in set_2], color=set_2_color, marker=set_2_marker)

    for i in range(len(set_1)):
        x_coords = [set_1[i][0], set_2[i][0]]
        y_coords = [set_1[i][1], set_2[i][1]]
        ax[2].plot(x_coords, y_coords, color='violet', linewidth=1)

    ax[2].set_xticks(np.arange(0, 641, 80))
    ax[2].set_yticks(np.arange(0, 481, 80))
    ax[2].grid()
    ax[2].set_title(f'{set_1_title} and {set_2_title}')

    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(f'{save_path}.png')
    plt.close(fig)

def plot_icp_iterations_results(iterations_results, save_path):
    T = iterations_results['T']
    error = iterations_results['error']
    num_inliers = iterations_results['num_inliers']
    kernel_threshold = iterations_results['kernel_threshold']
    dumping_factor = iterations_results['dumping_factor']

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0, 0].plot(error)
    ax[0, 0].set_title('Error')
    ax[0, 0].set_xlabel('Iteration')
    ax[0, 0].set_ylabel('Error')
    ax[0, 0].set_xticks(np.arange(0, len(error), 1 if len(error) < 10 else 5))
    ax[0, 0].grid()

    ax[0, 1].plot(num_inliers)
    ax[0, 1].set_title('Number of Inliers')
    ax[0, 1].set_xlabel('Iteration')
    ax[0, 1].set_ylabel('Number of Inliers')
    ax[0, 1].set_xticks(np.arange(0, len(num_inliers), 1 if len(num_inliers) < 10 else 5))
    ax[0, 1].grid()

    ax[1, 0].plot(kernel_threshold)
    ax[1, 0].set_title('Kernel Threshold')
    ax[1, 0].set_xlabel('Iteration')
    ax[1, 0].set_ylabel('Kernel Threshold')
    ax[1, 0].set_xticks(np.arange(0, len(kernel_threshold), 1 if len(kernel_threshold) < 10 else 5))
    ax[1, 0].grid()

    ax[1, 1].plot(dumping_factor)
    ax[1, 1].set_title('Dumping Factor')
    ax[1, 1].set_xlabel('Iteration')
    ax[1, 1].set_ylabel('Dumping Factor')
    ax[1, 1].set_xticks(np.arange(0, len(dumping_factor), 1 if len(dumping_factor) < 10 else 5))
    ax[1, 1].grid()

    plt.tight_layout()
    plt.savefig(f'{save_path}.png')
    plt.close(fig)