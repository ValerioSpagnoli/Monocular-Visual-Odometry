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
        self._intrinsic_matrix, self._extrinsic_matrix, self._camera_range, self._camera_resolution = self.__load_camera_parameters()

    def __load_camera_parameters(self):
        logger.info(f'Loading camera parameters from {self.__camera_parameters_file}')
        start = time.time()

        intrinsic_matrix = np.zeros((3, 3), dtype=float)
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
                camera_matrix = np.zeros((3, 3))
                for i in range(3):
                    values = data[data.index(line) + i + 1].split()
                    intrinsic_matrix[i] = [float(value) for value in values]

            elif line.startswith('cam_transform:'):
                for i in range(4):
                    values = data[data.index(line) + i + 1].split()
                    extrinsic_matrix[i] = [float(value) for value in values]

            elif line.startswith('z_near'):
                camera_range[0] = float(line.split()[1])
            elif line.startswith('z_far'):
                camera_range[1] = float(line.split()[1])

            elif line.startswith('width'):
                camera_resolution[0] = int(line.split()[1])
            elif line.startswith('height'):
                camera_resolution[1] = int(line.split()[1])

        logger.info(f'{(time.time() - start):.2f}s - Camera parameters loaded successfully.')
        return intrinsic_matrix, extrinsic_matrix, camera_range, camera_resolution


    #* GETTERS
    #* ------------------------------------------------------------------------------------- #

    def get_intrinsic_matrix(self):
        
        '''
        Returns the intrinsic matrix of the camera:

        [ fx  0 cx ]\\
        [  0 fy cy ]\\
        [  0  0  1 ]

        where: 
        - fx, fy: focal lengths in x and y directions
        - cx, cy: principal point coordinates
        '''

        return self._intrinsic_matrix
    

    def get_extrinsic_matrix(self):
        
        '''
        Returns the extrinsic matrix (camera transform) of the camera:
            
        [ R | t ]\\
        [ 0 | 1 ]
    
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
        
        #image_point = [image_point[0], image_point[1], 1]
        #return np.dot(np.linalg.inv(self._intrinsic_matrix), image_point)

        x_ndc = (image_point[0] / self._camera_resolution[0]) * 2 - 1
        y_ndc = 1 - (image_point[1] / self._camera_resolution[1]) * 2
        ndc = np.array([x_ndc, y_ndc, 1])

        inv_intrinsic = np.linalg.inv(self._intrinsic_matrix)

        return np.dot(inv_intrinsic, ndc)

        
    def camera_to_world(self, camera_point):
        '''
        Returns the world coordinates of a point in the camera frame.

        Parameters:
        - camera_point (np.array): 3D camera coordinates

        Returns:
        - world_point (np.array): 3D world coordinates
        '''
        camera_point = [camera_point[0], camera_point[1], camera_point[2], 1]
        return np.dot(self._extrinsic_matrix, camera_point)[:3]