from src.Camera import Camera
from src.Data import Data
from src.data_association import *
from src.utils import *

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class VisualOdometry:
    def __init__(self, kernel_threshold=1500, dumping_factor=1000, min_inliners=12, num_iterations=150):

        #** Projective ICP parameters
        self.__kernel_threshold = kernel_threshold
        self.__min_kernel_threshold = 200
        self.__max_kernel_threshold = 1e5

        self.__dumping_factor = dumping_factor
        self.__min_dumping_factor = 100
        self.__max_dumping_factor = 1e6

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
        w_T_c1 = self.projective_ICP(next_measurement, index)
        
        #** Triangulate points
        matches = data_association_on_appearance(current_measurement, next_measurement)
        points_0 = np.array(matches['points_1'])
        points_1 = np.array(matches['points_2'])
        points_3D = self.triangulate_points(points_0, points_1, self.get_current_pose(), w_T_c1)

        #** Update the state
        map = {'position':points_3D, 'appearance':matches['appearance']}
        self.__update_state(w_T_c1, map)

    def projective_ICP(self, image_points, frame_index):
        w_T_c0 = self.get_current_pose()
        w_T_c0_ = w_T_c0.copy()

        kernel_threshold = self.__kernel_threshold
        dumping_factor = self.__dumping_factor
        
        limit = 10

        error_prev = np.inf
        error_slope_ring_buffer = np.zeros(limit)
        error_slope = 0
        error_mean_slope = 0
        error_sigma_slope = 0
        
        stuck_counter = 0
        flickering_counter = 0

        transforms = {'T':[], 'error':[]}

        stop = False
        icp_iteration = 0
        while not stop:
            if icp_iteration == self.__num_iterations: break
            icp_iteration += 1

            matches = data_association_on_appearance(image_points, self.get_map(), projection=2, camera=self.__camera)
            reference_image_points = np.array(matches['points_1'])
            current_world_points = np.array(matches['points_2'])
            projected_world_points = np.array(matches['projected_points_2'])

            if False:
                projected_world_points = self.__camera.project_points(current_world_points)

                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(np.ones((480, 640, 3)))
                ax[0].scatter([point[0] for point in reference_image_points], [point[1] for point in reference_image_points], color='green', marker='o')
                ax[0].set_xticks(np.arange(0, 640, 40))
                ax[0].set_yticks(np.arange(0, 480, 40))
                ax[0].grid()
                ax[0].set_title('Reference Image Points')

                ax[1].imshow(np.ones((480, 640, 3)))
                ax[1].scatter([point[0] for point in projected_world_points], [point[1] for point in projected_world_points], color='red', marker='x')
                ax[1].set_xticks(np.arange(0, 640, 40))
                ax[1].set_yticks(np.arange(0, 480, 40))
                ax[1].grid()
                ax[1].set_title('Projected World Points')

                ax[2].imshow(np.ones((480, 640, 3)))
                ax[2].scatter([point[0] for point in reference_image_points], [point[1] for point in reference_image_points], color='green', marker='o')
                ax[2].scatter([point[0] for point in projected_world_points], [point[1] for point in projected_world_points], color='red', marker='x')
                ax[2].set_xticks(np.arange(0, 640, 40))
                ax[2].set_yticks(np.arange(0, 480, 40))
                ax[2].grid()
                ax[2].set_title('Reference Image Points and Projected World Points')

                fig.suptitle(f'Frame: {frame_index}, Iteration: {icp_iteration}')

                plt.savefig(f'outputs/frame_{frame_index}/icp/iteration_{icp_iteration}_icp_subplots.png')
                plt.close(fig)


            w_T_c1, results, computation_done = self.one_step(reference_image_points, current_world_points, w_T_c0, kernel_threshold, dumping_factor)
            
            num_inliers = results['num_inliers']
            error = results['error']    
            kernel_threshold = kernel_threshold = self.__min_kernel_threshold if num_inliers == len(reference_image_points) else results['kernel_threshold']

            if icp_iteration > 1: 
                error_slope = np.abs(error_prev - error)
                error_slope_ring_buffer[icp_iteration % len(error_slope_ring_buffer)] = error_slope
                error_mean_slope = np.mean(error_slope_ring_buffer)
                error_sigma_slope = np.std(error_slope_ring_buffer)
            
            w_T_c0 = w_T_c1
            self.__camera.set_c_T_w(np.linalg.inv(w_T_c0))
            transforms['T'].append(w_T_c0)
            transforms['error'].append(error)
            error_prev = error

            if computation_done and error_mean_slope < 1e-2 and error_sigma_slope < 1e-2: stuck_counter += 1
            else: stuck_counter = 0
            if computation_done and error_mean_slope > 1 and error_sigma_slope > 1: flickering_counter += 1
            else: flickering_counter = 0

            if (dumping_factor/2) > self.__min_dumping_factor and (stuck_counter > limit or (stuck_counter == 0 and flickering_counter == 0)): dumping_factor /= 2
            if (dumping_factor*2) < self.__max_dumping_factor and flickering_counter > limit: dumping_factor *= 2

            print(f'Frame: {frame_index}, Iteration: {icp_iteration}')
            print(f'Num of reference image points: {len(reference_image_points)}')
            print(f'Num of current world points: {len(current_world_points)}')
            print(f'Num of projected world points: {len(projected_world_points)}')
            print(f'Num inliers: {num_inliers}')
            print(f'Error: {error}')
            print(f'Kernel threshold: {kernel_threshold}')
            print(f'Dumping factor: {dumping_factor}')
            print(f'Chi inliers slope: {error_slope}')
            print(f'Chi inliers mean slope: {error_mean_slope}')
            print(f'Chi inliers sigma slope: {error_sigma_slope}')
            print(f'Stuck counter: {stuck_counter}')
            print(f'Flickering counter: {flickering_counter}')
            print('-----------------------------------\n')

            if computation_done and error < 1.5: stop = True

        best_error = np.inf
        best_transform = None
        for i in range(len(transforms['T'])):
            if transforms['error'][i] < best_error:
                best_error = transforms['error'][i]
                best_transform = transforms['T'][i]
        w_T_c0 = best_transform
        self.__camera.set_c_T_w(np.linalg.inv(w_T_c0))

        print(f'Best transformation error: {best_error} (index: {transforms["error"].index(best_error)})')
        print('##################################\n')

        return w_T_c0

    def one_step(self, reference_image_points, current_world_points, w_T_c0, kernel_threshold, dumping_factor):

        if (len(current_world_points) == 0): return w_T_c0, None, False

        H, b, num_inliers, error = self.linearize(reference_image_points, current_world_points, kernel_threshold)
        results = {'num_inliers': num_inliers, 'error': error, 'kernel_threshold': kernel_threshold}

        if num_inliers < self.__min_inliners: 
            kernel_threshold += 50
            results['kernel_threshold'] = kernel_threshold
            return w_T_c0, results, False
        
        if kernel_threshold > self.__min_kernel_threshold and error < 5: 
            kernel_threshold -= 50
            results['kernel_threshold'] = kernel_threshold

        H += np.eye(6) * dumping_factor
        dx = np.linalg.solve(H, -b)
        w_T_c1 = v2T(dx) @ w_T_c0

        return w_T_c1, results, True

    def linearize(self, reference_image_points, currrent_world_points, kernel_threshold):
        H = np.zeros((6, 6))
        b = np.zeros(6)

        chi_inliers = []
        errors = []
        jacobians = []

        for i in range(len(reference_image_points)):
            reference_image_point = reference_image_points[i]
            current_world_point = currrent_world_points[i]
            
            error, jacobian = self.error_and_jacobian(reference_image_point, current_world_point)
            if error is None or jacobian is None: continue
                
            chi = error.T @ error 
            if chi <= kernel_threshold:
                chi_inliers.append(chi)
                errors.append(error)    
                jacobians.append(jacobian)
                 
        chi_inliers_mean = np.mean(chi_inliers)
        chi_inliers_mask = np.array(chi_inliers) < chi_inliers_mean

        chi_inliers = np.array(chi_inliers)[chi_inliers_mask]
        errors = np.array(errors)[chi_inliers_mask]
        jacobians = np.array(jacobians)[chi_inliers_mask]

        for i in range(len(errors)):
            error = errors[i]
            jacobian = jacobians[i]
            H += jacobian.T @ jacobian 
            b += jacobian.T @ error

        num_inliers = len(errors)
        error = np.mean(chi_inliers) if num_inliers > 0 else np.inf 

        return H, b, num_inliers, error
    
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
    
    

    