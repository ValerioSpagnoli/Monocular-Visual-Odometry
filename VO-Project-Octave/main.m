close all
clear all
clc

source "./utils/data_utils.m"
source "./utils/visualization_utils.m"

global num_poses = 120;
global num_landmarks = 1000;

[camera, world_landmarks, trajectory, measurements] = load_data(num_poses, num_landmarks);

# plot_world(world_landmarks);
# plot_trajectory(trajectory);
# plot_ground_truth_data(world_landmarks, trajectory);

## initialization

# takes the initial two measurements and computes the initial guess for the relative transformation
# T_0 identity matrix, T_1 meas 1. 

image_points_1 = measurements(1).image_points;
appearance_1 = measurements(1).appearances;
image_points_2 = measurements(2).image_points;
appearance_2 = measurements(2).appearances;

matches = struct("points1", [], "points2", [], "appearances", []);
for i = 1:size(image_points_1)(2)
    for j = 1:size(image_points_2)(2)
        if appearance_1(:,i) == appearance_2(:,j)
            matches.points1 = [matches.points1, image_points_1(:,i)];
            matches.points2 = [matches.points2, image_points_2(:,j)];
            matches.appearances = [matches.appearances, appearance_1(:,i)];
        end
    end
end 

% disp(matches.points1(:,1));
% disp(matches.points2(:,1));
% disp(matches.appearances(:,1));



# pause;