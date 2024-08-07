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
final_frame = 45

vo = VisualOdometry()
vo.initialize(initial_frame=initial_frame)
for i in range(initial_frame+1,final_frame): 
    start_frame = time.time()
    vo.update(i)
    end_frame = time.time()
    mean_time_per_frame += end_frame-start_frame
mean_time_per_frame /= (final_frame-initial_frame)
end = time.time()


gt_trajectory = vo.get_data().get_trajectory_data_poses()
gt_world_points = vo.get_data().get_world_data()['position']

estimated_trajectory = vo.get_trajectory()
estimated_world_points_position = vo.get_map()['position']
estimated_world_points_appearance = vo.get_map()['appearance']

C = vo.get_camera().get_camera_transform()

T_1 = gt_trajectory[initial_frame]
T_2 = np.eye(4)
T_2[:3,:3] = C[:3,:3]
T = T_1 @ T_2
scale = 0.208

estimated_trajectory_in_world = translate(estimated_trajectory, T, scale=scale)
estimated_world_points_in_world = translate(estimated_world_points_position, T, are_points=True, scale=scale)

matches = data_association_on_appearance({'position':estimated_world_points_in_world, 'appearance':estimated_world_points_appearance}, vo.get_data().get_world_data())
estimated_world_points_in_world_matched = matches['points_1']
gt_world_points_matched = matches['points_2']


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

#scale = 1/np.mean(translation_errors)
scaled_estimated_world_points_in_world_matched = [scale*np.array(point) for point in estimated_world_points_in_world_matched]
rmse_world_map = np.sqrt(np.mean(np.linalg.norm(np.array(scaled_estimated_world_points_in_world_matched)-np.array(gt_world_points_matched), axis=1)**2))

print(f'Number of frames:       {final_frame-initial_frame}')
print(f'Time elapsed:           {end-start} [s] - {(end-start)/60} [min]')
print(f'Mean time per frame:    {mean_time_per_frame} [s]')
print()
print(f'Number of world points: {len(estimated_world_points_position)}')
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
plot_points(fig, poses2positions([gt_trajectory[initial_frame]]), name='Initial GT pose', mode='markers', color='blue', size=3)
plot_points(fig, poses2positions([gt_trajectory[final_frame]]), name='Final GT pose', mode='markers', color='blue', size=3)

plot_points(fig, poses2positions(gt_trajectory), name='Ground Truth trajectory', mode='lines', color='blue', width=3)
plot_points(fig, poses2positions(estimated_trajectory_in_world), name='Estimated trajectory in world', mode='lines', color='red', width=5)

plot_points(fig, estimated_world_points_in_world, name='Map in world', mode='markers', color='orange', size=2)
plot_points(fig, gt_world_points, name='Ground Truth map', mode='markers', color='green', size=2)
 
plot_matches(fig, estimated_world_points_in_world_matched, gt_world_points_matched, name='Map matches', color='blue', width=2)

fig.update_layout(scene=dict(aspectmode='data'))
fig.show()