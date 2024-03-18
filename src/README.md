# Utils

## Camera.py
This class define the Camera object.

### Attributes:
- **camera_parameters_file** (protected): input file with the description of the camera
- **intrinsic_matrix** (protected): 3x3 matrix with the intrinsc camera parameters (focal lengths and principal point coordinates)
- **extrinsic_matrix** (protected): 4x4 matrix with the extrinsic camera parameters (rotation matrix, translation vector)
- **camera_range** (protected): 1x2 vector [z_near, z_far], i.e. how close/far the camera can percive objects
- **camera_resolution** (protected): 1x2 vector [width, height]

### Methods:
- **\__init__(camera_parameters_file)** (private): constructor, takes in input the file with the description of the camera parameters in the following format:
    ```
    camera matrix:
    [fx  0 cx]
    [0  fy cy]
    [0   0  1]
    cam_transform:
    [R | t]
    [0 | 1]
    z_near: <z_near>
    z_far:  <z_far>
    width: <width>
    height: <height>
    ```

- **__load_camera_parameters()** (private): reads the file 'camera_parameters_file' and saves the camera parameters into the private attribute of the class.

- **get_intrinsic_matrix()**: returns the intrinsic matrix (camera matrix) of the camera.

- **get_extrinsic_matrix()**: returns the extrinsic matrix (camera transform) of the camera.

- **get_camera_range()**: returns the camera range [z_near, z_far].

- **get_camera_resolution()**: returns the camera resolution [width, height].

- **pixel_to_camera(image_point)***: takes in input a 2D point in the image plane and return the corresponding 3D point in the camera frame.

- **camera_to_world(camera_point)**: takes a 3D point in the camera frame and return the corresponding 3D point in the robot frame.


---

## Data.py
This class define the Data object, which contain all the data.

### Attributes

- **folder_path** (private): folder path of the directory where are stored the data.
- **trajectory** (private): trajectory data
- **world** (private): world data
- **measurements** (private): measurements data

### Methods

**Trajectory data**:

The Trajectory data are ground truth data about the motion of the robot. Each row contains:
- ```pose_id```: tells at which sequence the data has been acquired. Concides with ```sequence_id``` of the Measurement data.
- ```odometry_pose```: tells which was the estimated pose of the robot in that frame (sequence, pose_id). Coincides with ```odometry_pose``` specified in the measurement data with ```sequence_id=pose_id```.
- ```ground_truth_pose```: tells which was the real pose of the robot in that frame (sequence, pose_id).Coincides with ```ground_truth_pose``` specified in the measurement data with ```sequence_id=pose_id```.

This set of data must be used **only for the evaluation**.


The methods related to the Trajectory data are:

- **\__init__(folder_path='data/')** (private): constuctor, takes in input the folder path of the directory where are stored the data.
  
- **__load_trajectory_data** (private): reads the file 'trajectory.dat' and saves the trajectory_data into the respective private attribute of the class. The 'trajectory.dat' must be in the following format:
    ```
    <pose_id> <odom_x odom_y odom_z> <gt_x gt_y gt_z>
    ...
    ```
    
- **get_trajectory(pose_id=None)**: return the trajectory data for a specific pose_id. If no pose_id is given, returns the entire trajectory data.

- **print_trajectory(pose_id=None)**: prints the trajectory data for a specific pose_id. If no pose_id is given, returns the entire trajectory data.
  

**World data**:

The World data specify the map of the environment where the robot moves. Each row contains:
- ```landmark_id```: this is the id that identify the landmark. Coincides with ```actual_point_id``` field of the ```point``` data in the measurements data.
- ```position```: tells the real position of the landmark in the world (global position).
- ```appearance```: is the descriptor of the features of the landmark. Coincides with ```appearance``` field of the ```point``` data in the measurements data.

This set of data must be **used only for the evaluation**.

The methods releated to the World data are:

- **__load_world__data** (private): reads the file 'world.dat' and saves the world_data into the respective private attribute of the class. The 'world.dat' file must be in the following format:
    ```
    <landmark_id> <x y z> <appearance_1> ... <appearance_10>
    ...
    ```
    where <appearance_i> is the descriptor of a landmark.


- **get_world(landmark_id=None)**: return the world data for a specific landmark_id. If no landmark_id is given, returns the entire world data.

- **print_world(landmark_id=None)**: prints the world data for a specific landmark_id. If no landmark_id is given, returns the entire world data.
  

**Measurements data**:

The Measurements data specify which are the measurement done by the robot in a specific squence. Each file contains:
- ```squence_id```: tells at which sequence the data has been acquired. Concides with ```pose_id``` of the Trajectory data.
- ```ground_truth_pose```: tells which was the real pose of the robot in that sequence. Coincides with ```ground_truth_pose``` specified in the measurement data with ```pose_id=sequence_id```.
- ```odometry_pose```: tells which was the estimated pose of the robot in that sequence. Coincides with ```odometry_pose``` specified in the measurement data with ```pose_id=sequence_id```. 
- Set of points (landmarks) seen by the robot in that sequence:
  - ```point_id```: is the id of the landmark for that computation squence of the robot. Indeed, is a number that starts always from 0 at each measurements, and grows incrementally.
    - ```actual_point_id```: is the real id  of the landmark. Coincides with ```landmark_id``` of the world data.
    - ```image_point```: represent the pair [row, col] where the landmark was observed in the camera in that sequence.
    - ```appearance```: is the description of the features of the landmark. Coincides with ```appearance``` field of the World data, where ```landmark_id=actual_point_id```.

The methods releated to the Measurements data are: 

- **__load_measurements_data** (private): reads all the files 'meas-xxxxx.dat' and saves the measurements_data into the respective private attribute of the class. Each 'meas-xxxxx.dat' file must be in the following format:
    ```
    seq: <squence number>
    gt_pose: <x y z>
    odom_pose: <x y z>
    point <point_id_current_measurement> <actual_point_id> <image_point> <appearance_1> ... <appearance_10>
    ...
    ```

    where:
    - <image_point> represent the pair [col, row] where the landmark is observed in the image;
    - <appearance_i> is the descriptor of a landmark.

- **get_measurements(sequence_id=None)**: return the measurements data for a specific sequence_id. If no sequence_id is given, returns the entire measurements data.

- **print_measurements(sequence_id=None)**: prints the measurements data for a specific sequence_id. If no sequence_id is given, returns the entire measurements data.

