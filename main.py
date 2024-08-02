from src.Camera import Camera
from src.Data import Data
from src.VisualOdometry import VisualOdometry
from src.utils import *
from src.data_association import *
import numpy as np
import plotly.graph_objects as go   
import matplotlib.pyplot as plt

vo = VisualOdometry()
vo.initialize()

NUM_FRAMES = 100
for i in range(1,NUM_FRAMES): 
    vo.update(i)


# | Plot trajectory in 3D
estimated_trajectory = vo.get_trajectory()
gt_trajectory = vo.get_data().get_trajectory_data()

C = vo.get_camera().get_camera_transform()
 
estimated_trajectory_in_world = [] 

for i in range(len(estimated_trajectory)):
    pose = estimated_trajectory[i]
    pose_in_world = C @ pose
    estimated_trajectory_in_world.append(pose_in_world)

estimated_positions = []
estimated_positions_in_world = [] 
for i in range(len(estimated_trajectory)):
    estimated_positions.append(estimated_trajectory[i][:3, 3])
    estimated_positions_in_world.append(estimated_trajectory_in_world[i][:3, 3])

gt_poses = []
gt_positions = []
for i in range(1,NUM_FRAMES+1):
    x_gt, y_gt, theta_gt = gt_trajectory[i]
    gt_poses.append(np.array([x_gt, y_gt, 0, 0, 0, theta_gt]))
    gt_positions.append(np.array([x_gt, y_gt, 0]))
    
fig = go.Figure()
scale = 0.208
gt_x_coords = [position[0] for position in gt_positions]
gt_y_coords = [position[1] for position in gt_positions]
gt_z_coords = [position[2] for position in gt_positions]
fig.add_trace(go.Scatter3d(x=gt_x_coords, y=gt_y_coords, z=gt_z_coords, mode='lines', name='GT trajectory', line=dict(color='green')))

estimated_x_coords = [position[0]*scale for position in estimated_positions]
estimated_y_coords = [position[1]*scale for position in estimated_positions]
estimated_z_coords = [position[2]*scale for position in estimated_positions]
fig.add_trace(go.Scatter3d(x=estimated_x_coords, y=estimated_y_coords, z=estimated_z_coords, mode='lines', name='Estimated trajectory', line=dict(color='red')))

estimated_x_coords_in_world = [position[0]*scale for position in estimated_positions_in_world]
estimated_y_coords_in_world = [position[1]*scale for position in estimated_positions_in_world]
estimated_z_coords_in_world = [position[2]*scale for position in estimated_positions_in_world]
fig.add_trace(go.Scatter3d(x=estimated_x_coords_in_world, y=estimated_y_coords_in_world, z=estimated_z_coords_in_world, mode='lines', name='Estimated trajectory in world frame', line=dict(color='blue')))


fig.update_layout(scene=dict(aspectmode='data'))
fig.show()


# | Plot world points in 3D
world_points = vo.get_map()['position']
print('Number of world points: ', len(world_points))

x_coords = [point[0] for point in world_points]
y_coords = [point[1] for point in world_points]
z_coords = [point[2] for point in world_points]

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode='markers', marker=dict(size=2)))
fig.show()