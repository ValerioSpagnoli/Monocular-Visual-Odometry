import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.handlers.clear()
logger.addHandler(handler)

import numpy as np
import cv2
from . import utils

class VisualOdometry:
    def __init__(self, camera, data):
        self.camera = camera
        self.data = data
        self.map = {'position':[], 'appearance':[]}
        self.trajectory = {'poses':[], 'world_points':[]}

    
    #* ###################################################################################################################### *#
    #* Protected methods
    
    def _add_to_map(self, world_points):
        """
        Add world points to the map.

        Args:
            world_points (dict): A dictionary containing the world points.
                It should have two keys: 'position' and 'appearance'.
                'position' should be a list of positions, and 'appearance'
                should be a list of appearances.

        Returns:
            None
        """
        positions = world_points['position']
        appearances = world_points['appearance']
        for i in range(len(positions)):
            self.map['position'].append(positions[i])
            self.map['appearance'].append(appearances[i])


    def _add_to_trajectory(self, T, world_points):
        """
        Add a pose and corresponding world points to the trajectory.

        Args:
            T (numpy.ndarray): The pose to be added to the trajectory.
            world_points (numpy.ndarray): The corresponding world points.

        Returns:
            None
        """
        self.trajectory['poses'].append(T)
        self.trajectory['world_points'].append(world_points)

    
    #* ###################################################################################################################### *#
    #* Public methods

    def data_association(self, set_1, set_2):
        """
        Perform data association between two sets of points using the appearances.

        Args:
            set_1 (dict): The first set of points and appearances.
                It should have the following keys:
                - 'points': A numpy array of shape (N, 2) representing the 2D points.
                - 'appearance': A numpy array of shape (N, M) representing the appearance features.

            set_2 (dict): The second set of points and appearances.
                It should have the following keys:
                - 'points': A numpy array of shape (N, 2) representing the 2D points.
                - 'appearance': A numpy array of shape (N, M) representing the appearance features.

        Returns:
            dict: A dictionary containing the matched points and appearances.
                It has the following keys:
                - 'points_1': A numpy array of shape (K, 2) representing the matched points from set_1.
                - 'points_2': A numpy array of shape (K, 2) representing the matched points from set_2.
                - 'appearance': A numpy array of shape (K, M) representing the matched appearance features.
        """

        logger.info('Computing data association')
        start = utils.get_time()
        
        points_1 = set_1['points']
        appearances_1 = set_1['appearance']
        
        points_2 = set_2['points']
        appearances_2 = set_2['appearance']

        matches_points_1 = []
        matches_points_2 = []
        matches_appearance = []

        for i in range(len(points_1)):
            point_1 = points_1[i]
            appearance_1 = appearances_1[i]
            
            for j in range(len(points_2)):
                point_2 = points_2[j]
                appearance_2 = appearances_2[j]

                if appearance_1 == appearance_2:
                    matches_points_1.append(point_1)
                    matches_points_2.append(point_2)
                    matches_appearance.append(appearance_1)
                    break   

        matches = {'points_1':np.array(matches_points_1), 'points_2':np.array(matches_points_2), 'appearance':np.array(matches_appearance)}
        
        logger.info(f'{(utils.get_time() - start):.2f} [ms] - Data association done.')

        return matches

 
    def initialize(self):
        """
        Initializes the visual odometry algorithm.

        Returns:
            tuple: A tuple containing the camera pose in frame 1 w.r.t. the world frame (T_1) and the world points
            with their positions and appearances.
        """
        
        logger.info('Initializing visual odometry')
        start = utils.get_time()

        #* Pose of the camera w.r.t. the robot
        C = self.camera.get_extrinsic_matrix()

        #* DATA ASSOCIATION

        #* Image points in frame 0 and frame 1
        points_0 = self.data.get_measurement_points(0)
        points_1 = self.data.get_measurement_points(1)

        #* Data association between the points in frame 0 and frame 1
        matches = self.data_association(points_0, points_1)
        set_0 = matches['points_1']
        set_1 = matches['points_2']

        #* RECOVER POSE

        #* Pose of the camera in frame 0 w.r.t. the world frame
        R_0 = np.eye(3)
        t_0 = np.zeros((3, 1))
        T_0 = utils.Rt2T(R_0, t_0)

        #* Pose of the camera in frame 1 w.r.t. camera in frame 0
        E, _ = cv2.findEssentialMat(set_0, set_1, self.camera.get_camera_matrix(), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R_0_1, t_0_1, _ = cv2.recoverPose(E, set_0, set_1, self.camera.get_camera_matrix())
        T_0_1 = utils.Rt2T(R_0_1, t_0_1)

        #* Pose of the camera in frame 1 w.r.t. the world frame
        T_1 = np.dot(-np.linalg.inv(C), T_0_1)

        #* TRIANGULATE POINTS

        #* Projection matrices of the camera in frame 0 and frame 1
        P1 = np.dot(self.camera.get_intrinsic_matrix(), np.eye(4))
        P2 = np.dot(self.camera.get_intrinsic_matrix(), T_0_1)

        #* world points w.r.t. camera in frame 0
        world_points_hom = cv2.triangulatePoints(P1, P2, set_0.T, set_1.T)

        #* world points w.r.t. the world frame
        world_points_hom = np.dot(np.linalg.inv(C), world_points_hom)

        #* Normalize the world points
        world_points = (world_points_hom / world_points_hom[3])[:3].T

        world_points = {'position': world_points, 'appearance': matches['appearance']}

        logger.info(f'{(utils.get_time() - start):.2f} [ms] - Visual odometry initialized.')

        return T_1, world_points
    

    def error_and_jacobian(self, world_point, image_point):
        """
        Compute the error and Jacobian of the transformation between a world point and its corresponding image point.

        Parameters:
            world_point (numpy.ndarray): The 3D coordinates of the world point.
            image_point (numpy.ndarray): The 2D coordinates of the image point.

        Returns:
            error (numpy.ndarray): The difference between the predicted image point and the actual image point.
            jacobian (numpy.ndarray): The Jacobian matrix representing the partial derivatives of the error with respect to the transformation parameters.
        """

        logger.info('Computing error and Jacobian')
        start = utils.get_time()    

        #* Compute the prediction
        predicted_image_point = self.camera.project(world_point)
        if predicted_image_point[0] == -1 and predicted_image_point[1] == -1:
            return None, None

        #* Compute the error
        error = predicted_image_point - image_point

        #* Compute the Jacobian of the transformation
        world_point_hom = np.append(world_point, 1)
        camera_point = np.dot(self.camera.get_extrinsic_matrix(), world_point_hom)
        camera_point = (camera_point / camera_point[3])[:3]
        Jr = np.zeros((3, 6))
        Jr[:3, :3] = np.eye(3)
        Jr[:3, 3:] = utils.skew(-camera_point)

        #* Compute the Jacobian of the projection
        image_point_hom = np.dot(self.camera.get_camera_matrix(), camera_point)
        iz = 1.0 / image_point_hom[2]
        iz2 = iz * iz

        Jp = np.array([
            [iz, 0, -image_point_hom[0]*iz2],
            [0, iz, -image_point_hom[1]*iz2]
        ])

        #* Compute the Jacobian
        jacobian = np.dot(Jp, np.dot(self.camera.get_camera_matrix(), Jr))

        logger.info(f'{(utils.get_time() - start):.2f} [ms] - Error and Jacobian computed.')

        return error, jacobian





    def update_state(self, T, world_points):
        """
        Updates the state of the visual odometry system.

        Args:
            T (numpy.ndarray): Transformation matrix representing the camera pose.
            world_points (list): List of 3D world points observed by the camera.

        Returns:
            None
        """
        self._add_to_map(world_points)
        self._add_to_trajectory(T, world_points)


    def get_map(self):
            """
            Returns the map associated with the VisualOdometry object.

            Returns:
                The map associated with the VisualOdometry object.
            """
            return self.map
    

    def get_trajectory(self, only_poses=False):
            """
            Returns the trajectory of the visual odometry.

            Parameters:
                only_poses (bool): If True, returns only the poses of the trajectory. 
                                   If False, returns the trajectory and the world points associated at each pose.

            Returns:
                dict or list: The trajectory of the visual odometry. If only_poses is True, 
                              returns a list of poses. If only_poses is False, returns the 
                              entire trajectory as a dictionary.
            """
            if only_poses: return self.trajectory['poses']
            return self.trajectory
