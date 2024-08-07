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

C = vo.get_camera().get_camera_transform()

estimated_trajectory = vo.get_trajectory()
estimated_trajectory_in_world = trajectory2world(estimated_trajectory, C)
gt_trajectory = vo.get_data().get_trajectory_data_poses()

estimated_world_points = vo.get_map()['position']
gt_world_points = vo.get_data().get_world_data()['position']
matches = data_association_on_appearance(vo.get_map(), vo.get_data().get_world_data())
estimated_world_points_matched = matches['points_1']
gt_world_points_matched = matches['points_2']


#scale = 0.208

delta_poses_estimated_trajectory = []
for i in range(len(estimated_trajectory_in_world)-1):
    rel_pose = np.linalg.inv(estimated_trajectory_in_world[i]) @ estimated_trajectory_in_world[i+1]
    delta_poses_estimated_trajectory.append(rel_pose)

delta_poses_gt_trajectory = []
for i in range(len(gt_trajectory)-1):
    rel_pose = np.linalg.inv(gt_trajectory[i]) @ gt_trajectory[i+1]
    delta_poses_gt_trajectory.append(rel_pose)

errors_T = []
for i in range(len(delta_poses_estimated_trajectory)):
    error_T = np.linalg.inv(delta_poses_estimated_trajectory[i]) @ delta_poses_gt_trajectory[i]
    errors_T.append(error_T)

rotation_errors = []
for i in range(len(errors_T)):
    rotation_error = np.trace(np.eye(3)-errors_T[i][:3,:3])
    rotation_errors.append(rotation_error)

translation_errors = []
for i in range(len(delta_poses_estimated_trajectory)):
    rel_pose = delta_poses_estimated_trajectory[i]
    rel_pose_gt = delta_poses_gt_trajectory[i]
    translation_error = np.linalg.norm(rel_pose[:3,3])/np.linalg.norm(rel_pose_gt[:3,3])
    translation_errors.append(translation_error)

scale = 1/np.mean(translation_errors)
scaled_estimated_world_points_matched = [scale*np.array(point) for point in estimated_world_points_matched]
rmse_world_map = np.sqrt(np.mean(np.linalg.norm(np.array(scaled_estimated_world_points_matched)-np.array(gt_world_points_matched), axis=1)**2))

print(f'Number of frames:       {final_frame-initial_frame}')
print(f'Time elapsed:           {end-start} [s] - {(end-start)/60} [min]')
print(f'Mean time per frame:    {mean_time_per_frame} [s]')
print()
print(f'Number of world points: {len(estimated_world_points)}')
print(f'RMSE world map:         {rmse_world_map}')
print()
print(f'Max rotation error:     {np.max(rotation_errors)}')
print(f'Min rotation error:     {np.min(rotation_errors)}')
print(f'Mean rotation error:    {np.mean(rotation_errors)}')
print()
print(f'Max translation error:  {np.max(translation_errors)}')
print(f'Min translation error:  {np.min(translation_errors)}')
print(f'Mean translation error: {np.mean(translation_errors)}')
print(f'scale:                  {1/np.mean(translation_errors)}')


fig = go.Figure()
plot_trajectory(fig, estimated_trajectory, scale=scale, name='Estimated trajectory', color='red', width=4)
plot_trajectory(fig, estimated_trajectory_in_world, scale=scale, name='Estimated trajectory in world', color='blue', width=4)
plot_trajectory(fig, gt_trajectory, name='Ground Truth trajectory', color='green', width=4)
plot_point(fig, gt_trajectory[final_frame-1], pose=True, name='Final GT pose', color='red', size=3)
plot_points(fig, estimated_world_points, pose=False, C=C, name='Map in world', scale=scale, color='orange', size=2)
plot_points(fig, gt_world_points, pose=False, name='Ground Truth map', color='green', size=2)
plot_points(fig, gt_world_points_matched, pose=False, name='Ground Truth map matched', color='blue', size=2)
 
fig.update_layout(scene=dict(aspectmode='data'))
fig.show()