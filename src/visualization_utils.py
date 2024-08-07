import plotly.graph_objects as go   
import matplotlib.pyplot as plt
import numpy as np
from src.utils import *


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

        fig.add_trace(go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode='lines', name=name, line=dict(color=color, width=width)))
