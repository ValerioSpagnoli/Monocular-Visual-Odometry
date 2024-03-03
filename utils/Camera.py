import time
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.handlers.clear()
logger.addHandler(handler)

import numpy as np

class Camera:
    def __init__(self, camera_parameters_file):

        self.__camera_parameters_file = camera_parameters_file
        self.camera_matrix, self._intrinsic_matrix, self.camera_transform, self._extrinsic_matrix, self._camera_range, self._camera_resolution = self.__load_camera_parameters()

    def __load_camera_parameters(self):
        logger.info(f'Loading camera parameters from {self.__camera_parameters_file}')
        start = time.time()

        camera_matrix = np.zeros((3, 3), dtype=float)   
        intrinsic_matrix = np.zeros((3, 4), dtype=float)
        camera_transform = np.zeros((4, 4), dtype=float)
        extrinsic_matrix = np.zeros((4, 4), dtype=float)
        camera_range = np.zeros(2, dtype=float)
        camera_resolution = np.zeros(2, dtype=int)

        try:
            with open(self.__camera_parameters_file, 'r') as file:
                data = file.readlines()
        except FileNotFoundError:
            assert False, f'Camera parameters file not found at {self.__camera_parameters_file}'

        for line in data:
            if line.startswith('camera matrix:'):
                for i in range(3):
                    values = data[data.index(line) + i + 1].split()
                    camera_matrix[i] = [float(value) for value in values]
                intrinsic_matrix[:3, :3] = camera_matrix

            elif line.startswith('cam_transform:'):
                camera_transform = np.zeros((4, 4))
                for i in range(4):
                    values = data[data.index(line) + i + 1].split()
                    camera_transform[i] = [float(value) for value in values]

                extrinsic_matrix[:3, :3] = np.array(camera_transform[:3, :3]).T
                extrinsic_matrix[:3, 3] = -np.dot(np.array(camera_transform[:3, :3]).T, np.array(camera_transform[:3, 3]))
                extrinsic_matrix[3, 3] = 1

            elif line.startswith('z_near'):
                camera_range[0] = float(line.split()[1])
            elif line.startswith('z_far'):
                camera_range[1] = float(line.split()[1])

            elif line.startswith('width'):
                camera_resolution[0] = int(line.split()[1])
            elif line.startswith('height'):
                camera_resolution[1] = int(line.split()[1])

        logger.info(f'{(time.time() - start):.2f}s - Camera parameters loaded successfully.')
        return camera_matrix, intrinsic_matrix, camera_transform, extrinsic_matrix, camera_range, camera_resolution


    #* GETTERS
    #* ------------------------------------------------------------------------------------- #

    def get_camera_matrix(self):
            
        '''
        Returns the camera matrix of the camera:

        [ fx  0 cx ]\\
        [  0 fy cy ]\\
        [  0  0  1 ]

        where: 
        - fx, fy: focal lengths in x and y directions
        - cx, cy: principal point coordinates
        '''

        return self.camera_matrix

    def get_intrinsic_matrix(self):
        
        '''
        Returns the intrinsic matrix of the camera:

        [ fx  0 cx 0 ]\\
        [  0 fy cy 0 ]\\
        [  0  0  1 0 ]

        where: 
        - fx, fy: focal lengths in x and y directions
        - cx, cy: principal point coordinates
        '''

        return self._intrinsic_matrix
    

    def get_camera_transform(self):
            
        '''
        Returns the camera transform of the camera:

        [ R | t ]\\
        [ 0 | 1 ]

        where:
        - R: rotation matrix (3x3)
        - t: translation vector (3x1)
        '''

        return self.camera_transform

    def get_extrinsic_matrix(self):
        
        '''
        Returns the extrinsic matrix (camera transform) of the camera:
            
        [ R | -R*t ]\\
        [ 0 |    1 ]
    
        where:
        - R: rotation matrix (3x3)
        - t: translation vector (3x1)
        '''

        return self._extrinsic_matrix


    def get_camera_range(self):

        '''
        Returns the camera range [z_near, z_far], i.e. how close/far the camera can perceive objects
        '''

        return self._camera_range


    def get_camera_resolution(self):

        '''
        Returns the camera resolution [width, height]
        '''

        return self._camera_resolution
    

    def pixel_to_camera(self, image_point):
        '''
        Returns the camera coordinates of a point in the image plane.

        Parameters:
        - image_point (np.array): 2D pixel coordinates
        
        Returns:
        - camera_point (np.array): 3D camera coordinates
        '''
        
        x_ndc = (image_point[0] / self._camera_resolution[0]) * 2 - 1
        y_ndc = 1 - (image_point[1] / self._camera_resolution[1]) * 2
        ndc = np.array([x_ndc, y_ndc, 1])

        inv_intrinsic = np.linalg.inv(self._intrinsic_matrix)

        return np.dot(inv_intrinsic, ndc)

        
    def camera_to_world(self, camera_point):
        '''
        Returns the world coordinates of a point in the camera frame.
        # compute the relative motion between meas_0 and meas_1
        gt_pose_0, odom_pose_0, points_0 = compute_points(meas_0)
        gt_pose_1, odom_pose_1, points_1 = compute_points(meas_1)

        matches = data_association(points_0, points_1)
        set1 = np.array([matches[key][0] for key in matches])
        set2 = np.array([matches[key][1] for key in matches])

        R, t = compute_R_t(set1, set2)
        print('R:', R)
        print('t:', t)
        Returns:
        - world_point (np.array): 3D world coordinates
        '''
        camera_point = [camera_point[0], camera_point[1], camera_point[2], 1]
        return np.dot(self._extrinsic_matrix, camera_point)[:3]


    def world_to_pixel_projection(self, world_point):
        '''
        Returns the pixel coordinates of a point in the world frame.

        Parameters:
        - world_point (np.array): 3D world coordinates

        Returns:
        - pixel_point (np.array): 2D pixel coordinates
        '''

        world_point_hom = np.array([world_point[0], world_point[1], world_point[2], 1])
        camera_point = np.dot(self._extrinsic_matrix, world_point_hom)
        image_point = np.dot(self._intrinsic_matrix, camera_point)
        u = image_point[0] / image_point[2]
        v = image_point[1] / image_point[2]
        return [u, v]