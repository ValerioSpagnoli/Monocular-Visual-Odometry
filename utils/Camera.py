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
        self.__intrinsic_matrix, self.__extrinsic_matrix, self.__camera_range, self.__camera_resolution = self.__load_camera_parameters()

    def __load_camera_parameters(self):
        logger.info(f'Loading camera parameters from {self.__camera_parameters_file}')
        start = time.time()

        intrinsic_matrix = np.zeros((3, 4), dtype=float)
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
                    camera_matrix[i] = [float(value) for value in values]
                intrinsic_matrix[:, :3] = camera_matrix

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

    def get_intrinsic_matrix(self, camera_matrix=False):
        
        '''
        Returns the intrinsic matrix of the camera:

        [ fx  0 cx 0 ]\\
        [  0 fy cy 0 ]\\
        [  0  0  1 0 ]

        where: 
        - fx, fy: focal lengths in x and y directions
        - cx, cy: principal point coordinates

        Parameters:
        - camera_matrix (bool): if True, returns the 3x3 camera matrix, otherwise returns the 3x4 intrinsic matrix
        '''

        if camera_matrix:
            return self.__intrinsic_matrix[:, :3]

        return self.__intrinsic_matrix
    

    def get_extrinsic_matrix(self):
        
        '''
        Returns the extrinsic matrix (camera transform) of the camera:
            
        [ R | t ]\\
        [ 0 | 1 ]
    
        where:
        - R: rotation matrix (3x3)
        - t: translation vector (3x1)
        '''

        return self.__extrinsic_matrix


    def get_camera_range(self):

        '''
        Returns the camera range [z_near, z_far], i.e. how close/far the camera can perceive objects
        '''

        return self.__camera_range


    def get_camera_resolution(self):

        '''
        Returns the camera resolution [width, height]
        '''

        return self.__camera_resolution
    

