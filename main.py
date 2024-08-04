from src.Camera import Camera
from src.Data import Data
from src.VisualOdometry import VisualOdometry
from src.utils import *
from src.data_association import *
import numpy as np
import plotly.graph_objects as go   
import matplotlib.pyplot as plt
import time

start = time.time()
mean_time_per_frame = 0

initial_frame = 0
final_frame = 30

vo = VisualOdometry()
vo.initialize(initial_frame=initial_frame)

for i in range(initial_frame+1,final_frame): 
    start_frame = time.time()
    vo.update(i)
    end_frame = time.time()
    mean_time_per_frame += end_frame-start_frame

mean_time_per_frame /= (final_frame-initial_frame)
end = time.time()
print(f'Time elapsed: {end-start} [s] - {(end-start)/60} [min]')
print(f'Mean time per frame: {mean_time_per_frame} [s]')

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
initial_frame_pose = np.array([gt_trajectory[initial_frame][0], gt_trajectory[initial_frame][1], 0, 0, 0, gt_trajectory[initial_frame][2]])
for i in range(0, 120):
    pose = np.array([gt_trajectory[i][0], gt_trajectory[i][1], 0, 0, 0, gt_trajectory[i][2]]) - initial_frame_pose
    gt_poses.append(pose)
    gt_positions.append(np.array([pose[0], pose[1], pose[2]]))

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

gt_x_coords = [gt_positions[final_frame-1][0]]
gt_y_coords = [gt_positions[final_frame-1][1]]
gt_z_coords = [gt_positions[final_frame-1][2]]
fig.add_trace(go.Scatter3d(x=gt_x_coords, y=gt_y_coords, z=gt_z_coords, mode='markers', name='GT point in final frame', marker=dict(size=2, color='red')))

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