trajectory = dlmread('data/trajectory.dat');
pose_id = trajectory(:,1);
odometry_pose = trajectory(:,2:4);
ground_truth_pose = trajectory(:,5:7);

disp('Odometry pose')   
disp(odometry_pose(1:10,:))

disp('Ground truth pose')
disp(ground_truth_pose(1:10,:))

# plot the trajectory
figure(1)
p1 = plot3(odometry_pose(:,1), odometry_pose(:,2), odometry_pose(:,3), 'r');
hold on
p2 = plot3(ground_truth_pose(:,1), ground_truth_pose(:,2), ground_truth_pose(:,3), 'b');
hold on
xlabel('x')
ylabel('y')
zlabel('z')
grid on
waitfor(p1)
waitfor(p2)
