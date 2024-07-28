from src.Camera import Camera
from src.Data import Data
from src.data_association import *
from src.utils import *

import numpy as np
import cv2

class VisualOdometry:
    def __init__(self, kernel_threshold=10000, dumping_factor=1, min_inliners=8, num_iterations=30):

        #** Projective ICP parameters
        self.__kernel_threshold = kernel_threshold
        self.__dumping_factor = dumping_factor
        self.__min_inliners = min_inliners
        self.__num_iterations = num_iterations

        #** Camera and Data
        self.__camera = Camera()
        self.__data = Data()

        #** Trajectory:
        #* list of poses T (4x4 homogeneous matrix)
        self.__trajectory = []

        #** World Map (3D) in init camera coordinated
        #* Position: 3D (x, y, z)
        #* Appearance: 1x10 vectory
        self.__map = {'position':[], 'appearance':[]}


        #** Initial pose of the camera in global coordinates 
        self.__initial_pose = np.eye(4)

        #** Current pose of the camera in global coordinates
        self.__current_pose = np.eye(4)

    
    def triangulate_points(self, points_0, points_1, w_T_c0, w_T_c1):

        K = self.__camera.get_camera_matrix()

        T = np.linalg.inv(w_T_c1) @ w_T_c0
        R = T[:3, :3] 
        t = T[:3, 3].reshape(-1, 1)   

        #** Projection matrices
        P_0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P_1 = K @ np.hstack((R, t))        

        #** Triangulate points
        points_4D = cv2.triangulatePoints(P_0, P_1, points_0.T, points_1.T)
        points_4D = w_T_c0 @ points_4D
        points_3D = points_4D[:3] / points_4D[3]

        return points_3D.T


    def initialize(self):
        measurement_0 = self.__data.get_measurements_data_points(0)
        measurement_1 = self.__data.get_measurements_data_points(1)
 
        matches = data_association_on_appearance(measurement_0, measurement_1)
        points_0 = np.array(matches['points_1'])
        points_1 = np.array(matches['points_2'])

        w_T_c0 = np.eye(4)
        self.__set_initial_pose(w_T_c0)
        self.__update_state(w_T_c0, {'position':[], 'appearance':[]})

        #** Estimate the relative pose between the two frames
        K = self.__camera.get_camera_matrix()
        E, mask = cv2.findEssentialMat(points_0, points_1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, points_0, points_1, K, mask=mask)
        c0_T_c1 = Rt2T(R, -t)
        w_T_1 = w_T_c0 @ c0_T_c1

        #** Triangulate points
        points_3D = self.triangulate_points(points_0, points_1, w_T_c0, w_T_1)
        
        #** Update the state
        map = {'position':points_3D, 'appearance':matches['appearance']}
        self.__update_state(w_T_1, map)

    def update(self, index):
        current_measurement = self.__data.get_measurements_data_points(index)
        next_measurement = self.__data.get_measurements_data_points(index+1)

        #** Projective ICP 
        w_T_c1 = self.projective_ICP(next_measurement)
        print(f'Frame: {index}, w_T_c1: {np.round(w_T_c1, 2)}')
        
        #** Triangulate points
        matches = data_association_on_appearance(current_measurement, next_measurement)
        points_0 = np.array(matches['points_1'])
        points_1 = np.array(matches['points_2'])
        points_3D = self.triangulate_points(points_0, points_1, self.get_current_pose(), w_T_c1)

        #** Update the state
        map = {'position':points_3D, 'appearance':matches['appearance']}
        self.__update_state(w_T_c1, map)

    def projective_ICP(self, image_points):
        w_T_c0 = self.get_current_pose()
     
        for i in range(self.__num_iterations):
            #matches = data_association_2Dto3D(image_points, self.get_map(), self.__camera)
            matches = data_association_on_appearance(image_points, self.get_map())
            reference_image_points = np.array(matches['points_1'])
            current_world_points = np.array(matches['points_2'])

            w_T_c1, stop = self.one_step(reference_image_points, current_world_points, w_T_c0)
            if w_T_c1 is None: return w_T_c0

            w_T_c0 = w_T_c1
            self.__camera.set_c_T_w(np.linalg.inv(w_T_c0))

            if stop: break

        return w_T_c0

    def one_step(self, reference_image_points, current_world_points, w_T_c0):

        if (len(current_world_points) == 0): return w_T_c0, True

        H, b, num_inliers, chi_inliers, chi_outliers, error = self.linearize(reference_image_points, current_world_points)
        if num_inliers < self.__min_inliners: return None, None

        H += np.eye(6) * self.__dumping_factor
        dx = np.linalg.solve(H, -b)
        print(np.round(dx, 2))
        # print(f'total_error: {error}, dx: {np.linalg.norm(dx)}')   

        initial_guess = T2v(w_T_c0)
        updated_guess = initial_guess + dx
        w_T_c1 = v2T(updated_guess)

        return w_T_c1, error<0.1 or np.linalg.norm(dx)<1e-5

    def linearize(self, reference_image_points, currrent_world_points):
        H = np.zeros((6, 6))
        b = np.zeros(6)
        num_inliers = 0
        chi_inliers = 0
        chi_outliers = 0

        error = 0

        for i in range(len(reference_image_points)):
            reference_image_point = reference_image_points[i]
            current_world_point = currrent_world_points[i]

            e, jacobian = self.error_and_jacobian(reference_image_point, current_world_point)
            if e is None or jacobian is None: continue
    
            chi = e.T @ e 
            
            lambda_ = 1.0
            is_inlier = True
            if chi > self.__kernel_threshold:
                lambda_ = np.sqrt(self.__kernel_threshold / chi)
                is_inlier = False
                chi_outliers += chi
            else:
                num_inliers += 1
                chi_inliers += chi

            if is_inlier:
                H += jacobian.T @ jacobian * lambda_
                b += jacobian.T @ e * lambda_

            error += np.linalg.norm(e)
        
        error /= len(reference_image_points)

        return H, b, num_inliers, chi_inliers, chi_outliers, error

    def error_and_jacobian(self, reference_image_point, current_world_point):

        is_inside, projected_image_point = self.__camera.project_point(current_world_point)    
        if not is_inside: return None, None

        e = reference_image_point - projected_image_point

        p_hat_hom = np.linalg.inv(self.get_current_pose()) @ np.append(current_world_point, 1)
        p_hat = p_hat_hom[:3] / p_hat_hom[3]
        p_hat_cam = self.__camera.get_camera_matrix() @ p_hat

        J_icp = np.zeros((3, 6))
        J_icp[:3, :3] = np.eye(3)
        J_icp[:3, 3:] = -skew(p_hat)
        
        z_inv = 1.0 / p_hat_cam[2]
        z_inv2 = z_inv * z_inv

        J_proj = np.array([[z_inv, 0, -p_hat_cam[0] * z_inv2],
                           [0, z_inv, -p_hat_cam[1] * z_inv2]])
        
        J = J_proj @ self.__camera.get_camera_matrix() @ J_icp

        return e, J


    def __update_state(self, pose, map):
        self.__add_to_trajectory(pose)
        self.__add_to_global_map(map)
        self.set_current_pose(pose)
        self.__camera.set_c_T_w(np.linalg.inv(pose))

    def __add_to_trajectory(self, pose):
        self.__trajectory.append(pose)

    def __add_to_global_map(self, map):
        for i in range(len(map['position'])):
            position = map['position'][i]
            appearance = map['appearance'][i]
            if appearance not in self.__map['appearance']:
                self.__map['position'].append(position)
                self.__map['appearance'].append(appearance)

    def __set_initial_pose(self, pose): self.__initial_pose = pose
    def set_current_pose(self, pose): self.__current_pose = pose
    def get_initial_pose(self): return self.__initial_pose
    def get_current_pose(self): return self.__current_pose
    def get_trajectory(self): return self.__trajectory
    def get_map(self): return self.__map
    def get_data(self): return self.__data
    def get_camera(self): return self.__camera
    
    

    