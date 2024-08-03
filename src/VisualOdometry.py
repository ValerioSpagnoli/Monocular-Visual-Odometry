from src.Camera import Camera
from src.Data import Data
from src.data_association import *
from src.utils import *

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class VisualOdometry:
    def __init__(self, kernel_threshold=200, dumping_factor=1000, min_inliners=10, num_iterations=300):

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


    def initialize(self, initial_frame=0):
        measurement_0 = self.__data.get_measurements_data_points(initial_frame)
        measurement_1 = self.__data.get_measurements_data_points(initial_frame+1)
 
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
        if os.path.exists(f'outputs/frame_{index}'): os.system(f'rm -r outputs/frame_{index}')
        os.makedirs(f'outputs/frame_{index}', exist_ok=True)
        os.makedirs(f'outputs/frame_{index}/icp', exist_ok=True)
        os.makedirs(f'outputs/frame_{index}/plots', exist_ok=True)

        current_measurement = self.__data.get_measurements_data_points(index)
        next_measurement = self.__data.get_measurements_data_points(index+1)

        #** Projective ICP 
        w_T_c1, results = self.projective_ICP(next_measurement, index)
        
        #** Triangulate points
        matches = data_association_on_appearance(current_measurement, next_measurement)
        points_0 = np.array(matches['points_1'])
        points_1 = np.array(matches['points_2'])
        points_3D = self.triangulate_points(points_0, points_1, self.get_current_pose(), w_T_c1)

        #** Update the state
        map = {'position':points_3D, 'appearance':matches['appearance']}
        self.__update_state(w_T_c1, map)

    def projective_ICP(self, image_points, index):
        w_T_c0 = self.get_current_pose()

        i = 0
        stop = False
        
        counter_early_stopping = 0
        counter_error_stuck = 0
        counter_error_flickering = 0
        counter_data_association_on_appearance = 0
        counter_data_association_3Dto2D = 0
        previous_error = np.Inf

        kernel_threshold = self.__kernel_threshold
        dumping_factor = self.__dumping_factor

        num_inliers_history = []
        chi_inliers_history = []
        chi_outliers_history = []
        error_history = []
        dumping_factor_history = []
        kernel_threshold_history = []

        limit = 10
        error_slope_value_ring_buffer = np.ones(limit)
        mean_error_slope_value = 1
        sigma_error_slope_value = 1

        use_data_association_on_appearance = False

        while not stop:
            if i == self.__num_iterations+1: break

            matches_appearance = data_association_on_appearance(image_points, self.get_map(), projection=2, camera=self.__camera)
            reference_image_points_appearance = np.array(matches_appearance['points_1'])
            current_world_points_appearance = np.array(matches_appearance['points_2'])
            current_projected_world_points_appearance = np.array(matches_appearance['projected_points_2'])
            distance_matches_appearance = np.mean(np.linalg.norm(reference_image_points_appearance - current_projected_world_points_appearance, axis=1))

            matches_2Dto3D = data_association_2Dto3D(image_points, self.get_map(), self.__camera)
            reference_image_points_2Dto3D = np.array(matches_2Dto3D['points_1'])
            current_world_points_2Dto3D = np.array(matches_2Dto3D['points_2'])
            current_reprojected_world_points_2Dto3D = self.__camera.project_points(current_world_points_2Dto3D)
            distance_matches_2Dto3D = np.mean(np.linalg.norm(reference_image_points_2Dto3D - current_reprojected_world_points_2Dto3D, axis=1))
  
            if use_data_association_on_appearance: 
                reference_image_points = reference_image_points_appearance
                current_world_points = current_world_points_appearance
            else:
                reference_image_points = reference_image_points_2Dto3D
                current_world_points = current_world_points_2Dto3D

            if False:
                projected_world_points = self.__camera.project_points(current_world_points)
                fig, ax = plt.subplots()
                ax.imshow(np.ones((480, 640, 3)))
                ax.scatter([point[0] for point in reference_image_points], [point[1] for point in reference_image_points], color='green', marker='o')
                ax.scatter([point[0] for point in projected_world_points], [point[1] for point in projected_world_points], color='red', marker='x')
                plt.grid()
                plt.savefig(f'outputs/frame_{index}/icp/iteration_{i}_icp.png')
                plt.close(fig)

            w_T_c1, results, computation_done = self.one_step(reference_image_points, current_world_points, w_T_c0, kernel_threshold, dumping_factor)
            
            num_inliers = results['num_inliers']
            chi_inliers = results['chi_inliers']    
            chi_outliers = results['chi_outliers']
            kernel_threshold = results['kernel_threshold']

            w_T_c0 = w_T_c1
            self.__camera.set_c_T_w(np.linalg.inv(w_T_c0))

            if i>1:
                error_slope_value = np.abs(previous_error - chi_inliers)
                error_slope_value_ring_buffer[i % limit] = error_slope_value
                mean_error_slope_value = np.mean(error_slope_value_ring_buffer)
                sigma_error_slope_value = np.std(error_slope_value_ring_buffer)

                if not use_data_association_on_appearance:
                    if chi_inliers < previous_error and (sigma_error_slope_value > 1 or mean_error_slope_value < 1e-3) and dumping_factor < 1e8: dumping_factor *= 2
                    # elif sigma_error_slope_value < 1 and dumping_factor > self.__dumping_factor:  dumping_factor /= 2
                else:
                    dumping_factor = 80000

                if mean_error_slope_value < 1e-1: counter_error_stuck += 1
                else: counter_error_stuck = 0
                if sigma_error_slope_value > 1e-1: counter_error_flickering += 1
                else: counter_error_flickering = 0

                if use_data_association_on_appearance: counter_data_association_on_appearance += 1
                else: counter_data_association_3Dto2D += 1

                if computation_done and not use_data_association_on_appearance and counter_error_stuck >= limit: 
                    use_data_association_on_appearance = True
                    counter_error_stuck = 0
                    counter_error_flickering = 0
                
                if computation_done and use_data_association_on_appearance and (counter_error_stuck >= limit or counter_error_flickering >= limit): 
                    use_data_association_on_appearance = False
                    counter_error_stuck = 0
                    counter_error_flickering = 0
                    dumping_factor = self.__dumping_factor
                

                
                if computation_done and chi_inliers < 5 and (mean_error_slope_value < 1e-2 or sigma_error_slope_value < 1e-1): counter_early_stopping += 1
                else: counter_early_stopping = 0
                if (computation_done and chi_inliers < 1) or counter_early_stopping >= limit: stop = True
            
            print('Frame: ', index, ' - PICP Iteration: ', i)
            print('computation_done: ', computation_done)   
            print('num_inliers: ', num_inliers)
            print('kernel_threshold: ', kernel_threshold)
            print('dumping_factor: ', dumping_factor)
            print('previous_error: ', previous_error)   
            print('error: ', chi_inliers)
            print('mean_error_slope_value: ', mean_error_slope_value)
            print('sigma_error_slope_value: ', sigma_error_slope_value)
            print('counter early stopping: ', counter_early_stopping)
            print('counter error stuck: ', counter_error_stuck)
            print('counter error flickering: ', counter_error_flickering)
            print('distance_matches_appearance: ', distance_matches_appearance) 
            print('distance_matches_2Dto3D: ', distance_matches_2Dto3D)
            print('use_data_association_on_appearance: ', use_data_association_on_appearance)
            print('counter_data_association_on_appearance: ', counter_data_association_on_appearance)
            print('counter_data_association_3Dto2D: ', counter_data_association_3Dto2D)
            print('stop: ', stop)
            print('------------------------------- \n')

            num_inliers_history.append(num_inliers)
            chi_inliers_history.append(chi_inliers)
            chi_outliers_history.append(chi_outliers)
            error_history.append(chi_inliers)
            dumping_factor_history.append(dumping_factor)
            kernel_threshold_history.append(kernel_threshold)

            previous_error = chi_inliers
            i += 1

        if False:
            fig, ax = plt.subplots()
            ax.plot(num_inliers_history)
            ax.set_title('Number of inliers')
            ax.grid()
            plt.savefig(f'outputs/frame_{index}/plots/num_inliers.png')
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(chi_inliers_history)
            ax.set_title('Chi inliers')
            ax.grid()
            plt.savefig(f'outputs/frame_{index}/plots/chi_inliers.png')
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(chi_outliers_history)
            ax.set_title('Chi outliers')
            ax.grid()
            plt.savefig(f'outputs/frame_{index}/plots/chi_outliers.png')
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(error_history)
            ax.set_title('Error')
            ax.grid()
            plt.savefig(f'outputs/frame_{index}/plots/error.png')
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(dumping_factor_history)
            ax.set_title('Dumping factor')
            ax.grid()
            plt.savefig(f'outputs/frame_{index}/plots/dumping_factor.png')
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(kernel_threshold_history)
            ax.set_title('Kernel threshold')
            ax.grid()
            plt.savefig(f'outputs/frame_{index}/plots/kernel_threshold.png') 
            plt.close(fig)
                
        return w_T_c0, results

    def one_step(self, reference_image_points, current_world_points, w_T_c0, kernel_threshold, dumping_factor):

        if (len(current_world_points) == 0): return w_T_c0, None, False

        H, b, num_inliers, chi_inliers, chi_outliers, outlier_mask = self.linearize(reference_image_points, current_world_points, kernel_threshold)
        results = {'num_inliers': num_inliers, 'chi_inliers': chi_inliers, 'chi_outliers': chi_outliers, 'outlier_mask': outlier_mask, 'kernel_threshold': kernel_threshold}

        if num_inliers < self.__min_inliners: 
            kernel_threshold += 50
            results['kernel_threshold'] = kernel_threshold
            return w_T_c0, results, False
        
        if kernel_threshold > self.__kernel_threshold and chi_inliers < 5: 
            kernel_threshold -= 50
            results['kernel_threshold'] = kernel_threshold

        H += np.eye(6) * dumping_factor
        dx = np.linalg.solve(H, -b)
        w_T_c1 = v2T(dx) @ w_T_c0

        return w_T_c1, results, True

    def linearize(self, reference_image_points, currrent_world_points, kernel_threshold):
        H = np.zeros((6, 6))
        b = np.zeros(6)
        num_inliers = 0
        chi_inliers = 0
        chi_outliers = 0
        outlier_mask = np.zeros(len(reference_image_points))

        for i in range(len(reference_image_points)):
            reference_image_point = reference_image_points[i]
            current_world_point = currrent_world_points[i]
            
            e, jacobian = self.error_and_jacobian(reference_image_point, current_world_point)
            if e is None or jacobian is None: continue
                
            chi = e.T @ e 

            lambda_ = 1.0
            is_inlier = True
            if chi > kernel_threshold:
                lambda_ = np.sqrt(kernel_threshold / chi)
                is_inlier = False
                chi_outliers += chi
                outlier_mask[i] = 1
            else:
                num_inliers += 1
                chi_inliers += chi

            if is_inlier:
                H += jacobian.T @ jacobian * lambda_
                b += jacobian.T @ e * lambda_
            
        if chi_inliers > 0: chi_inliers /= num_inliers
        if chi_outliers > 0: chi_outliers /= (len(reference_image_points) - num_inliers)

        return H, b, num_inliers, chi_inliers, chi_outliers, outlier_mask
    
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
    
    

    