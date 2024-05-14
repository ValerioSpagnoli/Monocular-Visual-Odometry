import logging
logger = logging.getLogger()
logger.setLevel(logging.FATAL)

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
        self.initial_pose = np.eye(4)

        self.map = {'points':[], 'appearances':[]}
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

        counter = 0
        for i in range(len(positions)):
            position = positions[i]
            appearance = appearances[i]
            if appearance in self.map['appearances']: 
                counter += 1
                continue
            self.map['points'].append(position)
            self.map['appearances'].append(appearance)

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
                if np.equal(appearance_1, appearance_2).all():
                    matches_points_1.append(point_1)
                    matches_points_2.append(point_2)
                    matches_appearance.append(appearance_1)
                    break   

        matches = {'points_1':matches_points_1, 'points_2':matches_points_2, 'appearances':matches_appearance}
        
        logger.info(f'{(utils.get_time() - start):.2f} [ms] - Data association done.')

        return matches
    
    def normalize_points(self, points, camera_matrix):
        inv_cam_matrix = np.linalg.inv(camera_matrix)
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack([points, ones])
        points_normalized = np.dot(inv_cam_matrix, points_homogeneous.T).T
        return points_normalized[:, :2]
    

    def construct_matrix_A(self, points_0, points_1):
        """ Construct the matrix A used in the 8-point algorithm. """
        A = np.zeros((len(points_0), 9))
        for i, (p1, p2) in enumerate(zip(points_0, points_1)):
            x1, y1 = p1
            x2, y2 = p2
            A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
        return A
    
    def eight_point_algorithm(self, points_0, points_1, camera_matrix):
        """ The 8-point algorithm for estimating the essential matrix from point correspondences. """
        
        #* Normalize the points
        points_0_normalized = self.normalize_points(points_0, camera_matrix)
        points_1_normalized = self.normalize_points(points_1, camera_matrix)

        #* Construct matrix A from point correspondences
        A = self.construct_matrix_A(points_0_normalized, points_1_normalized)

        #* Solve for the essential matrix using SVD
        U, S, Vt = np.linalg.svd(A)
        F = Vt[-1].reshape(3, 3)

        #* Enforce rank-2 constraint on the essential matrix
        U, S, Vt = np.linalg.svd(F)
        S[2] = 0  # Set the smallest singular value to 0
        E = np.dot(U, np.dot(np.diag(S), Vt))

        return E
    
    def decompose_essential_matrix(self, E):
        """ Decompose the essential matrix into possible rotations and translation. """
        U, S, Vt = np.linalg.svd(E)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        t = U[:, 2]
        R1 = np.dot(U, np.dot(W, Vt))
        R2 = np.dot(U, np.dot(W.T, Vt))

        if np.linalg.det(R1) < 0: R1 = -R1
        if np.linalg.det(R2) < 0: R2 = -R2

        return R1, R2, t


    def triangulate_points(self, points_0, points_1, T_0, T_1):
        """ Triangulate points from two views. """

        intrinsic_matrix = self.camera.get_intrinsic_matrix()

        #* Projection matrices
        P_0 = intrinsic_matrix @ np.linalg.inv(T_0)
        P_1 = intrinsic_matrix @ np.linalg.inv(T_1)

        #* Triangulate points 
        points_homogeneous = cv2.triangulatePoints(P_0, P_1, points_0.T, points_1.T)
        points = (points_homogeneous[:3] / points_homogeneous[3]).T 

        triangulated_points_local = []
        triangulated_points_hom_local = []
        for point in points:
            if np.isfinite(point).all(): 
                triangulated_points_local.append(point)
                triangulated_points_hom_local.append(np.append(point, 1))
        triangulated_points_local = np.array(triangulated_points_local)
        triangulated_points_hom_local = np.array(triangulated_points_hom_local)

        triangulated_points_global = (T_0 @ triangulated_points_hom_local.T)[:3].T

        return triangulated_points_local, triangulated_points_global

 
    def initialize(self):
        """
        Initializes the visual odometry algorithm.

        Returns:
            tuple: A tuple containing the camera pose in frame 1 w.r.t. the world frame (T_1) and the world points
            with their positions and appearances.
        """
        
        logger.info('Initializing visual odometry')
        start = utils.get_time()

        #* Pose of the camera in frame 0 w.r.t. the world frame
        T_0 = np.eye(4)
        self.update_state(T_0, {'points':[], 'appearances':[]})
        self.initial_pose = T_0

        #* Image points in frame 0 and frame 1
        points_0 = self.data.get_measurement_points(0)
        points_1 = self.data.get_measurement_points(1)

        #* Data association between the points in frame 0 and frame 1
        matches = self.data_association(points_0, points_1)
        set_0 = np.array(matches['points_1'])
        set_1 = np.array(matches['points_2'])
        appearances = matches['appearances']

        #* Find the essential matrix
        K = self.camera.get_camera_matrix() 
        E, mask = cv2.findEssentialMat(set_0, set_1, K, method=cv2.RANSAC, prob=0.999, threshold=0.1)
        retval , R, t, mask = cv2.recoverPose(E, set_0, set_1, K)
        T_0_1 = utils.Rt2T(R, -t)
        T_1 = T_0 @ T_0_1

        #* Triangulate points
        triangulated_points_local, triangulated_points_global = self.triangulate_points(set_0, set_1, T_0, T_1)
        triangulated_points = {'points': triangulated_points_global, 'appearances': appearances}
        
        self.update_state(T_1, triangulated_points)

        logger.info(f'{(utils.get_time() - start):.2f} [ms] - Visual odometry initialized.')



    def one_step(self, sequence_id):
        prev_measurements = self.data.get_measurement_points(sequence_id - 1)
        measurements = self.data.get_measurement_points(sequence_id)
        map = self.get_map()
        K = self.camera.get_camera_matrix()
        T_0 = self.current_pose

        prev_appearance = prev_measurements['appearances']
        appearance = measurements['appearances']

        matches_3D = self.data_association(measurements, map)
        image_points_3D = np.array(matches_3D['points_1'])
        world_points_3D = np.array(matches_3D['points_2'])
        appearances_3D = matches_3D['appearances']

        matches_2D = self.data_association(prev_measurements, measurements)
        images_points_1_2D = np.array(matches_2D['points_1'])
        images_points_2_2D = np.array(matches_2D['points_2'])
        appearances_2D = matches_2D['appearances']
        
        #** Projective ICP (3D->2D)
        # w_T_c0 = T_0
        # T_1, chi_stats, num_inliers = self.linearize(image_points_3D, world_points_3D, w_T_c0)
        #T_1 = np.dot(w_T_c0, T_0_1)

        #** 2D->2D
        E, mask = cv2.findEssentialMat(images_points_1_2D, images_points_2_2D, K, method=cv2.RANSAC, prob=0.999, threshold=0.1)
        retval , R, t, mask = cv2.recoverPose(E, images_points_1_2D, images_points_2_2D, K)
        T_0_1 = utils.Rt2T(R, -t)
        T_1 = T_0 @ T_0_1

        number_of_world_points_1 = len(self.get_map()['points'])    

        triangulated_points_local, triangulated_points_global = self.triangulate_points(images_points_1_2D, images_points_2_2D, T_0, T_1)
        triangulated_points = {'points': triangulated_points_global, 'appearances': appearances_2D}

        self.update_state(T_1, triangulated_points)

        number_of_world_points_2 = len(self.get_map()['points'])

        number_of_points_in_meas_0_and_not_in_meas_1 = len(appearance) - len(matches_2D['points_1'])
        number_of_points_in_meas_1_and_not_in_meas_0 = len(prev_appearance) - len(matches_2D['points_1'])


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

        K = self.camera.get_camera_matrix()

        #* Compute the prediction
        proj_image_point_hom, proj_image_point = self.camera.project_point(world_point, self.current_pose)
        if proj_image_point_hom is None or proj_image_point is None:
            logger.warning('Point is behind the camera.')
            return None, None

        #* Compute the error
        error = proj_image_point - image_point

        #* Compute the Jacobian of the transformation
        J_icp = np.zeros((3, 6))
        J_icp[:3, :3] = np.eye(3)
        J_icp[:3, 3:] = utils.skew(-np.floor(world_point))

        #* Compute the Jacobian of the projection
        z_inv = 1.0 / proj_image_point_hom[2]
        z_inv_square = z_inv * z_inv

        J_proj = np.array([ [z_inv, 0, -proj_image_point_hom[0]*z_inv_square],
                            [0, z_inv, -proj_image_point_hom[1]*z_inv_square] ])
        
        #* Compute the Jacobian
        jacobian = J_proj @ K @ J_icp

        logger.info(f'{(utils.get_time() - start):.2f} [ms] - Error and Jacobian computed.')

        return error, jacobian
    
    
    def linearize(self, image_points, world_points, T):
        
        logger.info('Linearizing')
        start = utils.get_time()

        T_0 = T
        chi_stats = []
        num_inliers = []

        MAX_ITER = 1
        iteration = 0
        error = np.inf
        while (iteration < MAX_ITER) and (error > 1e-2):
        
            H = np.zeros((6, 6))
            b = np.zeros(6)
            
            num_inliers_ = 0
            chi_stats_ = 0

            for i in range(len(world_points)):
                world_point = world_points[i]
                image_point = image_points[i]

                e, J = self.error_and_jacobian(world_point, image_point)
                if e is None or J is None: continue
                error = np.linalg.norm(e)

                chi = np.dot(e.T, e)    
                if chi > self._kernel_threshold:
                    e *= np.sqrt(self._kernel_threshold / chi)
                    chi = self._kernel_threshold
                else:
                    num_inliers_ += 1

                chi_stats_ += chi
                H += np.dot(J.T, J)
                b += np.dot(J.T, e)

            chi_stats.append(chi_stats_)
            num_inliers.append(num_inliers_)

            if num_inliers_ >= self._min_number_of_inliers:
                H += self._damping_factor * np.eye(6)
                dx = np.linalg.lstsq(H, -b, rcond=None)[0]
                T = utils.v2T(dx) * T_0

            iteration += 1

        logger.info(f'{(utils.get_time() - start):.2f} [ms] - Linearization done.')

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
        self.current_to_initial_transform = np.linalg.inv(T)
        self._add_to_map(world_points)
        self._add_to_trajectory(self.current_pose, world_points)


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
