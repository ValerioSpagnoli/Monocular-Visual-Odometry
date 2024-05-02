import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.handlers.clear()
logger.addHandler(handler)

import numpy as np
import cv2
from scipy.optimize import least_squares
from . import utils

class VisualOdometry:
    def __init__(self, camera, data, kernel_threshold=1000, damping_factor=1, min_number_of_inliers=0):
        self.camera = camera
        self.data = data
        self.current_pose = np.eye(4)
        self.map = {'points':[], 'appearances':[]}
        for i in range(len(self.data.get_world())):
            landmark_position = self.data.get_world()[i]['landmark_position']
            landmark_appearances = self.data.get_world()[i]['landmark_appearance']
            self.map['points'].append(landmark_position)
            self.map['appearances'].append(landmark_appearances)

        self.trajectory = {'poses':[], 'world_points':[]}

        self._kernel_threshold = kernel_threshold
        self._damping_factor = damping_factor
        self._min_number_of_inliers = min_number_of_inliers

    
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
        positions = world_points['points']
        appearances = world_points['appearances']

        for i in range(len(positions)):
            position = positions[i]
            appearance = appearances[i]
            if appearance in self.map['appearances']: continue
            self.map['points'].append(positions[i])
            self.map['appearances'].append(appearances[i])


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
        appearances_1 = set_1['appearances']
        
        points_2 = set_2['points']
        appearances_2 = set_2['appearances']

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

        matches = {'points_1':matches_points_1, 'points_2':matches_points_2, 'appearances':matches_appearance}
        
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
        set_0 = np.array(matches['points_1'])
        set_1 = np.array(matches['points_2'])
        appearances = matches['appearances']

        #* RECOVER POSE

        #* Pose of the camera in frame 0 w.r.t. the world frame
        R_0 = np.eye(3)
        t_0 = np.zeros((3, 1))
        T_0 = utils.Rt2T(R_0, t_0)
        self.update_state(T_0, {'points':[], 'appearances':[]})

        #* Pose of the camera in frame 1 w.r.t. camera in frame 0
        E, _ = cv2.findEssentialMat(set_0, set_1, self.camera.get_camera_matrix(), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R_0_1, t_0_1, _ = cv2.recoverPose(E, set_0, set_1, self.camera.get_camera_matrix())
        T_0_1 = utils.Rt2T(R_0_1, t_0_1)

        #* Pose of the camera in frame 1 w.r.t. the world frame
        T_1 = np.dot(T_0, T_0_1)

        #* TRIANGULATE POINTS

        # #* Projection matrices of the camera in frame 0 and frame 1
        # P1 = np.dot(self.camera.get_intrinsic_matrix(), np.eye(4))
        # P2 = np.dot(self.camera.get_intrinsic_matrix(), T_0_1)

        # #* world points w.r.t. camera in frame 0
        # world_points_hom = cv2.triangulatePoints(P1, P2, set_0.T, set_1.T)

        # #* world points w.r.t. the world frame
        # world_points_hom = np.dot(np.linalg.inv(C), world_points_hom)

        # #* Normalize the world points
        # world_points = (world_points_hom / world_points_hom[3])[:3].T

        # scale = 0.204
        # world_points = world_points * scale

        # world_points = {'points': world_points, 'appearances': appearances}

        #* Update the state
        self.update_state(T_1, {'points':[], 'appearances':[]})

        logger.info(f'{(utils.get_time() - start):.2f} [ms] - Visual odometry initialized.')


    def one_step(self, sequence_id):
        measurements = self.data.get_measurement_points(sequence_id)
        matches = self.data_association(measurements, self.get_map())
        image_points = matches['points_1']
        world_points = matches['points_2']
        appearances = matches['appearances']
                                        
        T_0 = self.current_pose
        T_0_1, chi_stats, num_inliers = self.linearize(image_points, world_points)
        T_1 = np.dot(T_0, T_0_1)

        self.update_state(T_1, {'points':[], 'appearances':[]})


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
        robot_point = np.dot(np.linalg.inv(self.current_pose), np.append(world_point, 1))[:3]
        predicted_image_point = self.camera.project(robot_point)
        if predicted_image_point[0] == -1 and predicted_image_point[1] == -1:
            logger.warning('Point is behind the camera.')
            return None, None

        #* Compute the error
        error = predicted_image_point - image_point

        #* Compute the Jacobian of the transformation
        world_point_hom = np.append(world_point, 1)
        camera_point_hom = np.dot(self.camera.get_extrinsic_matrix(), world_point_hom)
        camera_point = (camera_point_hom / camera_point_hom[3])[:3]
        J_icp = np.zeros((3, 6))
        J_icp[:3, :3] = np.eye(3)
        J_icp[:3, 3:] = utils.skew(-np.floor(camera_point))

        #* Compute the Jacobian of the projection
        image_point_hom = np.dot(self.camera.get_camera_matrix(), camera_point)
        z_inv = 1.0 / image_point_hom[2]
        z_inv_square = z_inv * z_inv

        J_proj = np.array([ [z_inv, 0, -image_point_hom[0]*z_inv_square],
                            [0, z_inv, -image_point_hom[1]*z_inv_square] ])

        #* Compute the Jacobian
        jacobian = np.dot(J_proj, np.dot(self.camera.get_camera_matrix(), J_icp))

        logger.info(f'{(utils.get_time() - start):.2f} [ms] - Error and Jacobian computed.')

        return error, jacobian
    
    
    def linearize(self, image_points, world_points):
        
        logger.info('Linearizing')
        start = utils.get_time()
        
        H = np.zeros((6, 6))
        b = np.zeros(6)
        
        num_inliers = 0
        chi_stats = 0

        for i in range(len(world_points)):
            world_point = world_points[i]
            image_point = image_points[i]

            e, J = self.error_and_jacobian(world_point, image_point)
            if e is None or J is None: continue

            chi = np.dot(e.T, e)    
            if chi > self._kernel_threshold:
                e *= np.sqrt(self._kernel_threshold / chi)
                chi = self._kernel_threshold
            else:
                num_inliers += 1

            chi_stats += chi
            H += np.dot(J.T, J)
            b += np.dot(J.T, e)

        H += self._damping_factor * np.eye(6)
        dx = np.linalg.lstsq(H, -b, rcond=None)[0]
        T = utils.v2T(dx)

        logger.info(f'{(utils.get_time() - start):.2f} [ms] - Linearization done.')

        if num_inliers < self._min_number_of_inliers:
            return np.eye(4), chi_stats, num_inliers

        return T, chi_stats, num_inliers


    def update_state(self, T, world_points):
        """
        Updates the state of the visual odometry system.

        Args:
            T (numpy.ndarray): Transformation matrix representing the camera pose.
            world_points (list): List of 3D world points observed by the camera.

        Returns:
            None
        """
        self.current_pose = T
        #self._add_to_map(world_points)
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
