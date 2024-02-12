# Utils

## Camera.py
This class define the Camera object.

### Attributes:
- **camera_parameters_file** (private): input file with the description of the camera
- **intrinsic_matrix** (private): 3x4 matrix with the intrinsc camera parameters (focal lengths and principal point coordinates)
- **extrinsic_matrix** (private): 4x4 matrix with the extrinsic camera parameters (rotation matrix, translation vector)
- **camera_range** (private): 1x2 vector [z_near, z_far], i.e. how close/far the camera can percive objects
- **camera_resolution** (private): 1x2 vector [width, height]

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

- **get_intrinsic_matrix(camera_matrix=False)**: returns the intrinsic matrix of the camera. If the parameter camera_matrix is set to True return the 3x3 camera matrix.

- **get_extrinsic_matrix()**: returns the extrinsic matrix (camera transform) of the camera.

- **get_camera_range()**: returns the camera range [z_near, z_far]

- **get_camera_resolution()**: returns the camera resolution [width, height]


---

## Data.py
This class define the Data object, which contain all the data.

### Attributes:
- **folder_path** (private): folder path of the directory where are stored the data.
- **trajectory** (private): trajectory data
- **world** (private): world data
- **measurements** (private): measurements data

### Methods
- **\__init__(folder_path='data/')** (private): constuctor, takes in input the folder path of the directory where are stored the data.
  
- **__load_trajectory_data** (private): reads the file 'trajectory.dat' and saves the trajectory_data into the respective private attribute of the class. The 'trajectory.dat' must be in the following format:
    ```
    <pose_id> <odom_x odom_y odom_z> <gt_x gt_y gt_z>
    ...
    ```
    
- **get_trajectory(pose_id=None)**: return the trajectory data for a specific pose_id. If no pose_id is given, returns the entire trajectory data.

- **print_trajectory(pose_id=None)**: prints the trajectory data for a specific pose_id. If no pose_id is given, returns the entire trajectory data.
  
- **__load_world__data** (private): reads the file 'world.dat' and saves the world_data into the respective private attribute of the class. The 'world.dat' file must be in the following format:
    ```
    <landmark_id> <x y z> <appearance_1> ... <appearance_10>
    ...
    ```
    where <appearance_i> is the descriptor of a landmark.


- **get_world(landmark_id=None)**: return the world data for a specific landmark_id. If no landmark_id is given, returns the entire world data.

- **print_world(landmark_id=None)**: prints the world data for a specific landmark_id. If no landmark_id is given, returns the entire world data.
  
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

