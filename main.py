from src.Camera import Camera
from src.Data import Data
from src.VisualOdometry import VisualOdometry
from src.utils import *
from src.data_association import *
from src.visualization_utils import *
import numpy as np
import plotly.graph_objects as go   
import matplotlib.pyplot as plt
import time

start = time.time()
mean_time_per_frame = 0

initial_frame = 0
final_frame = 40

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


C = vo.get_camera().get_camera_transform()

estimated_trajectory = vo.get_trajectory()
gt_trajectory = vo.get_data().get_trajectory_data_poses()

estimated_world_points = vo.get_map()['position']
gt_world_points = vo.get_data().get_world_data()['position']
matches = data_association_on_appearance(vo.get_map(), vo.get_data().get_world_data())
gt_world_points_matched = matches['points_2']

print('Number of world points: ', len(estimated_world_points))

scale = 0.208


fig = go.Figure()

plot_trajectory(fig, estimated_trajectory, scale=scale, name='Estimated trajectory', color='red', width=4)
plot_trajectory(fig, estimated_trajectory, C=C, scale=scale, name='Estimated trajectory in world', color='blue', width=4)
plot_trajectory(fig, gt_trajectory, name='Ground Truth trajectory', color='green', width=4)
plot_point(fig, gt_trajectory[final_frame-1], pose=True, name='Final GT pose', color='red', size=3)
plot_points(fig, estimated_world_points, pose=False, C=C, name='Map in world', scale=scale, color='orange', size=2)
plot_points(fig, gt_world_points, pose=False, name='Ground Truth map', color='green', size=2)
plot_points(fig, gt_world_points_matched, pose=False, name='Ground Truth map matched', color='blue', size=2)
 
fig.update_layout(scene=dict(aspectmode='data'))
fig.show()
