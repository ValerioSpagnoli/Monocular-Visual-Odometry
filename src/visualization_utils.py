import plotly.graph_objects as go   
import matplotlib.pyplot as plt
import numpy as np
from src.utils import *


def plot_trajectory(fig, trajectory, C=np.eye(4), scale=1, name='Trajectory', color='blue', width=4):     
    trajectory_positions = []
    for i in range(len(trajectory)):
        pose = C @ trajectory[i]
        position = pose[:3, 3]
        trajectory_positions.append(position)

    x_coords = [position[0]*scale for position in trajectory_positions]
    y_coords = [position[1]*scale for position in trajectory_positions]
    z_coords = [position[2]*scale for position in trajectory_positions]

    fig.add_trace(go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode='lines', name=name, line=dict(color=color, width=width)))

def plot_point(fig, point, pose=False, C=np.eye(4), scale=1, name='Point', color='blue', size=2):
    if not pose: point = v2T([point[0], point[1], point[2], 0, 0, 0])
    point = C @ point
    point_position = point[:3, 3]

    x_coords = [point_position[0]*scale]
    y_coords = [point_position[1]*scale]
    z_coords = [point_position[2]*scale]

    fig.add_trace(go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode='markers', name=name, marker=dict(size=size, color=color)))

def plot_points(fig, points, pose=False, C=np.eye(4), scale=1, name='Points', color='blue', size=2):
    points_positions = []
    for i in range(len(points)):
        point = points[i]
        if not pose: point = v2T([point[0], point[1], point[2], 0, 0, 0])
        point = C @ point
        point_position = point[:3, 3]
        points_positions.append(point_position)

    x_coords = [point[0]*scale for point in points_positions]
    y_coords = [point[1]*scale for point in points_positions]
    z_coords = [point[2]*scale for point in points_positions]

    fig.add_trace(go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode='markers', name=name, marker=dict(size=size, color=color)))
