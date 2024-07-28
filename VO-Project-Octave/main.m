close all
clear all
clc

source "./utils/data_utils.m"
source "./utils/visualization_utils.m"

global num_poses = 120;
global num_landmarks = 1000;

[camera, world_landmarks, trajectory, measurements] = load_data(num_poses, num_landmarks);

plot_world(world_landmarks);
plot_trajectory(trajectory);
plot_ground_truth_data(world_landmarks, trajectory);

pause;



