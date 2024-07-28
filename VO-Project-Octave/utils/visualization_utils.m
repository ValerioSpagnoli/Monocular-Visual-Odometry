1;

function plot_world(world_landmarks, traj)
    figure(1);
    position = world_landmarks.position;
    plot3(position(1,:), position(2,:), position(3,:), 'r*');
    grid on;
    xlabel('x');
    ylabel('y');
    zlabel('z');
    title('World landmarks');
end

function plot_trajectory(trajectory)
    figure(2);
    x = trajectory(1,:);  
    y = trajectory(2,:);
    z = zeros(size(x));
    plot3(x, y, z, 'b-');
    grid on;
    xlabel('x');
    ylabel('y');
    zlabel('z');
    title('Trajectory');
end

function plot_ground_truth_data(world_landmarks, trajectory)
    figure(3);

    % World landmarks
    position = world_landmarks.position;
    plot3(position(1,:), position(2,:), position(3,:), 'r*');
    hold on;

    % Trajectory
    x = trajectory(1,:);
    y = trajectory(2,:);
    z = zeros(size(x));
    plot3(x, y, z, 'b-');

    grid on;
    xlabel('x');
    ylabel('y');
    zlabel('z');
    title('World landmarks and Trajectory');
end
