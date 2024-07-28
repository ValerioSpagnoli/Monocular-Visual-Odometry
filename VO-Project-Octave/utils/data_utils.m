1;

function [camera, world_landmarks, trajectory, measurements] = load_data(num_poses, num_landmarks)
    camera = loadCameraInfos();
    world_landmarks = load_world_data(num_landmarks);
    trajectory = loadTrajectories(num_poses);
    measurements = loadMeasurements(num_poses, num_landmarks);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%* Camera struct format:
%* camera.K = [3 x 3]
%* camera.T = [4 x 4]
%* camera.z_near = scalar
%* camera.z_far = scalar
%* camera.width = scalar
%* camera.height = scalar
%* To access to the camera parameters:
%*  - K = camera.K
%*  - T = camera.T
%*  - z_near = camera.z_near
%*  - z_far = camera.z_far
%*  - width = camera.width
%*  - height = camera.height

function camera = loadCameraInfos()

	camera_file = fopen("data/camera.dat", 'r');
    if camera_file == -1
        disp(['Cannot open file: ', filepath]);
        return
    end

    camera = struct('K', [], 'T', [], 'z_near', 0, 'z_far', 0, 'width', 0, 'height', 0);
    
    while true
		line = fgetl(camera_file);
		if line == -1
            fclose(camera_file);
			break;
		end

        line = strtrim(line);
		elements = strsplit(line,':');

        switch(elements{1})
            case 'camera matrix'
                for j = 1:3
                    l = strtrim(fgetl(camera_file));
                    elements = strsplit(l,' ');
                    camera.K(j, :) = arrayfun(@(x) str2double(x), elements);
                end
            case 'cam_transform'
                for j = 1:4
                    l = strtrim(fgetl(camera_file));
                    elements = strsplit(l,' ');
                    camera.T(j, :) = arrayfun(@(x) str2double(x), elements);
                end
            case 'z_near'
                camera.z_near = str2double(elements{2});
            case 'z_far'
                camera.z_far = str2double(elements{2});
            case 'width'
                camera.width = str2double(elements{2});
            case 'height'
                camera.height = str2double(elements{2});
            otherwise
                if not(strcmp(elements{1}, ""))
                    disp(['[' filepath '] Error in reading element ', elements{1}]);
                end
                continue;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%* World landmarks struct format:
%* world_landmarks.position = [3 x num_landmarks]
%* world_landmarks.appearance = [10 x num_landmarks]
%* To access to the world landmarks parameters:
%*  - position = world_landmarks.position
%*    - To access to the i-th landmark position: position(:, i)
%*  - appearance = world_landmarks.appearance
%*    - To access to the i-th landmark appearance: appearance(:, i)

function world_landmarks = load_world_data(num_landmarks)
    world_data = load("data/world.dat");
    world_landmarks = struct('position', zeros(3, num_landmarks), 'appearance', zeros(10, num_landmarks));
    world_landmarks.position = world_data(:, 2:4)';
    world_landmarks.appearance = world_data(:, 5:14)';
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%* Trajectory struct format:
%* trajectory = [num_poses x 3]
%* To access to the i-th pose:
%*  - pose = trajectory(:, i)

function trajectory = loadTrajectories(num_poses)
    trajectory_data = load("data/trajectory.dat");
    trajectory = trajectory_data(:, 5:7)';
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%* Measuremets struct format:
%* measurements(i).image_points = [2 x num_landmarks]
%* measurements(i).appearances = [10 x num_landmarks]
%* To access to the i-th pose measurements: 
%*  - image_points = measurements(i).image_points
%*    - To access to the i-th image points of the i-th pose: image_points(:, i)
%*  - appearances = measurements(i).appearances
%*    - To access to the i-th appearance of the i-th pose: appearances(:, i)

function measurements = loadMeasurements(num_poses, num_landmarks)    
    measurement = struct('image_points', zeros(2, num_landmarks), 'appearances', zeros(10, num_landmarks));
    measurements = repmat(measurement, 1, num_poses);

    for i = 1:num_poses
        filepath = ["data/meas-", num2str(i-1, '%05.f'), ".dat"];    
        [num_of_meas, image_points, appearances] = loadMeasFile(filepath, num_landmarks);
        measurements(i).image_points = image_points;
        measurements(i).appearances = appearances;
    end
end

function [num_of_meas, image_points, appearances] = loadMeasFile(filepath, num_landmarks)
    num_of_meas = 1;
    image_points = zeros(2, num_landmarks);
    appearances = zeros(10, num_landmarks);

	measurement_file = fopen(filepath, 'r');
    if measurement_file == -1
        disp(['Cannot open file: ', filepath]);
        return
    end

    while true
		line = fgetl(measurement_file);

		if line == -1
            fclose(measurement_file);
			break;
		end

        line = strtrim(line);
		elements = strsplit(line,' ');

        switch(elements{1})
            case 'point'
                p = arrayfun(@(x) str2double(x), elements)(4:15);
                image_points(:, num_of_meas) = p(1:2);
                appearances(:, num_of_meas) = p(3:12);
                num_of_meas += 1;
            case {'seq:', 'gt_pose:', 'odom_pose:'}
                continue;
            otherwise
                if not(strcmp(elements{1}, ""))
                    disp(['[' filepath '] Error in reading element ', elements{1}]);
                end
                continue;
        end
    end

    num_of_meas = num_of_meas-1;
    image_points = image_points(:, 1:num_of_meas);
    appearances = appearances(:, 1:num_of_meas);
end