import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.handlers.clear()
logger.addHandler(handler)

import numpy as np
from . import utils


class Camera:
    def __init__(self, camera_parameters_file):

        self.__camera_parameters_file = camera_parameters_file
        self._camera_matrix, self._intrinsic_matrix, self._camera_transform, self._extrinsic_matrix, self._camera_range, self._camera_resolution = self.__load_camera_parameters()

    def __load_camera_parameters(self):
        logger.info(f'Loading camera parameters from {self.__camera_parameters_file}')
        start = utils.get_time()

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

        logger.info(f'{(utils.get_time() - start):.2f} [ms] - Camera parameters loaded successfully.')
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

        return self._camera_matrix

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

        return self._camera_transform

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
    
    
    def project(self, world_point):
        """
        Projects a 3D world point onto the camera image plane.

        Args:
            world_point (numpy.ndarray): The 3D world point to be projected.

        Returns:
            numpy.ndarray: The normalized image coordinates of the projected point, or [-1, -1] if the point is outside the camera's range or image resolution.
        """

        world_point_hom = np.append(world_point, 1)
        camera_point_hom = np.dot(self._extrinsic_matrix, world_point_hom)
        camera_point = (camera_point_hom / camera_point_hom[3])[:3]

        [z_near, z_far] = self._camera_range
        if camera_point[2] < z_near or camera_point[2] > z_far:
            return [-1, -1]
        
        image_point_hom = np.dot(self._camera_matrix, camera_point)
        image_point = (image_point_hom / image_point_hom[2])[:2]
        
        [width, height] = self._camera_resolution
        if image_point[0] < 0 or image_point[0] > width or image_point[1] < 0 or image_point[1] > height:
            return [-1, -1]

        return image_point
    

    def pixel_to_world_projection(self, image_point, depth, pose):
        
        t = self._camera_transform[:3, 3]
        R = self._camera_transform[:3, :3]
        robot_t = pose[:3, 3]
        robot_R = pose[:3, :3]
        
        image_point_hom = np.array([image_point[0], image_point[1], 1])
        camera_point = depth * np.dot(np.linalg.inv(self._camera_matrix), image_point_hom)
        world_point = t + np.dot(R, camera_point)
        estimated_world_point = robot_t + np.dot(robot_R, world_point)
        return estimated_world_point
    
