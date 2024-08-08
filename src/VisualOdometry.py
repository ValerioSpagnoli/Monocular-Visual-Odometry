from src.Camera import Camera
from src.Data import Data
from src.data_association import *
from src.utils import *
from src.visualization_utils import *

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import time

class VisualOdometry:
    def __init__(self, initial_frame=0, final_frame=120, verbose=False, save_plots=False, save_plots_indices=[]):

        self.__initial_frame = initial_frame
        self.__final_frame = final_frame
        self.__verbose = verbose    

        #** Save plots
        self.__save_plots = save_plots
        self.__save_plots_indices = save_plots_indices

        #** Projective ICP parameters
        self.__kernel_threshold = 1500
        self.__min_kernel_threshold = 200
        self.__max_kernel_threshold = 1e5

        self.__dumping_factor = 1000
        self.__min_dumping_factor = 1e3
        self.__max_dumping_factor = 1e7

        self.__min_inliners = 12
        self.__num_iterations = 150

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

        #** Current pose of the camera in global coordinates
        self.__current_pose = np.eye(4)
    

    #| --------------- Private methods --------------- |#

    def __triangulate_points(self, points_0, points_1, w_T_c0, w_T_c1):

        K = self.__camera.get_camera_matrix()

        T = np.linalg.inv(w_T_c1) @ w_T_c0
        R = T[:3, :3] 
        t = T[:3, 3].reshape(-1, 1)   

        #** Projection matrices
        P_0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P_1 = K @ np.hstack((R, t))        

        #** Triangulate points
        points_4D = cv2.triangulatePoints(P_0, P_1, points_0.T, points_1.T)

        points_3D = (points_4D[:3] / points_4D[3]).T

        points_3D_norms = np.linalg.norm(points_3D, axis=1)
        mask = points_3D_norms < 30
    
        points_3D_filtered = points_3D[mask]
        points_4D = np.hstack((points_3D_filtered, np.ones((points_3D_filtered.shape[0], 1))))
    
        points_4D = w_T_c0 @ points_4D.T
        points_3D = points_4D[:3] / points_4D[3]

        return points_3D.T, mask

    def __initialize(self):
        measurement_0 = self.__data.get_measurements_data_points(self.__initial_frame)
        measurement_1 = self.__data.get_measurements_data_points(self.__initial_frame+1)
 
        matches = data_association_on_appearance(measurement_0, measurement_1)
        points_0 = np.array(matches['points_1'])
        points_1 = np.array(matches['points_2'])
        appearances = np.array(matches['appearance'])

        w_T_c0 = np.eye(4)
        self.__update_state(w_T_c0, {'position':[], 'appearance':[]})

        #** Estimate the relative pose between the two frames
        K = self.__camera.get_camera_matrix()
        E, mask = cv2.findEssentialMat(points_0, points_1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, points_0, points_1, K, mask=mask)
        c0_T_c1 = Rt2T(R, -t)
        w_T_1 = w_T_c0 @ c0_T_c1

        #** Triangulate points
        points_3D, mask = self.__triangulate_points(points_0, points_1, w_T_c0, w_T_1)
        
        #** Update the state
        map = {'position':points_3D, 'appearance':appearances[mask].tolist()}
        self.__update_state(w_T_1, map)

    def __update(self, frame_index):
        if os.path.exists(f'outputs/frame_{frame_index}'): os.system(f'rm -r outputs/frame_{frame_index}')
        os.makedirs(f'outputs/frame_{frame_index}', exist_ok=True)
        os.makedirs(f'outputs/frame_{frame_index}/icp', exist_ok=True)
        os.makedirs(f'outputs/frame_{frame_index}/plots', exist_ok=True)

        current_measurement = self.__data.get_measurements_data_points(frame_index)
        next_measurement = self.__data.get_measurements_data_points(frame_index+1)
        
        #** Projective ICP 
        w_T_c1, is_valid, iterations_results = self.__projective_ICP(next_measurement, frame_index)
        if not is_valid: 
            self.__update_state(w_T_c1, {'position':[], 'appearance':[]})
            return
        
        #** Triangulate points
        matches = data_association_on_appearance(current_measurement, next_measurement)
        points_0 = np.array(matches['points_1'])
        points_1 = np.array(matches['points_2'])
        appearances = np.array(matches['appearance'])
        points_3D, mask = self.__triangulate_points(points_0, points_1, self.get_current_pose(), w_T_c1)

        #** Update the state
        map = {'position':points_3D, 'appearance':appearances[mask].tolist()}
        self.__update_state(w_T_c1, map)


        min_error_index = np.argmin(iterations_results['error'])
        max_error_index = np.argmax(iterations_results['error'])
        icp_iteration = len(iterations_results['T'])
        print(f'Frame: {frame_index}')
        print(f'  - Num iterations:                   {icp_iteration}\n')
        print(f'  - Error best iteration:             {np.round(iterations_results["error"][min_error_index], 5)} (index: {min_error_index})')
        print(f'  - Error worst iteration:            {np.round(iterations_results["error"][max_error_index], 5)} (index: {max_error_index})')    
        print(f'  - Mean error:                       {np.round(np.mean(iterations_results["error"]), 5)}\n')
         
        print(f'  - Num inliers best iteration:       {iterations_results["num_inliers"][min_error_index]}')
        print(f'  - Num inliers worst iteration:      {iterations_results["num_inliers"][max_error_index]}')
        print(f'  - Mean num inliers:                 {np.round(np.mean(iterations_results["num_inliers"]))}\n')
 
        print(f'  - Kernel threshold best iteration:  {iterations_results["kernel_threshold"][min_error_index]}')
        print(f'  - Kernel threshold worst iteration: {iterations_results["kernel_threshold"][max_error_index]}')
        print(f'  - Mean kernel threshold:            {np.round(np.mean(iterations_results["kernel_threshold"]))}\n')
         
        print(f'  - Dumping factor best iteration:    {iterations_results["dumping_factor"][min_error_index]}')
        print(f'  - Dumping factor worst iteration:   {iterations_results["dumping_factor"][max_error_index]}')
        print(f'  - Mean dumping factor:              {np.round(np.mean(iterations_results["dumping_factor"]))}\n')

        print(f'Applied transformation of index {min_error_index} to the camera')
        print('================================================================\n')

    def __projective_ICP(self, image_points, frame_index):
        w_T_c0 = self.get_current_pose()
        
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

        iterations_results = {'T':[], 'error':[], 'num_inliers':[], 'kernel_threshold':[], 'dumping_factor':[]}    

        stop = False
        icp_iteration = 0
        while not stop:
            if icp_iteration == self.__num_iterations: break
            icp_iteration += 1
            
            matches = data_association_on_appearance(image_points, self.get_map(), projection=2, camera=self.__camera)
            reference_image_points = np.array(matches['points_1'])
            current_world_points = np.array(matches['points_2'])
            projected_world_points = np.array(matches['projected_points_2'])
            
            if self.__save_plots and (len(self.__save_plots_indices) == 0 or frame_index in self.__save_plots_indices):
                projected_world_points = self.__camera.project_points(current_world_points)
                save_path = f'outputs/frame_{frame_index}/icp/iteration_{icp_iteration}_icp_subplots'
                plot_icp_frame(reference_image_points, projected_world_points, save_path, title=f'Frame: {frame_index}, Iteration: {icp_iteration}', set_1_title='Reference Image Points', set_2_title='Projected World Points')

            w_T_c1, results, computation_done = self.__one_step(reference_image_points, current_world_points, w_T_c0, kernel_threshold, dumping_factor)

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
            iterations_results['T'].append(w_T_c0)
            iterations_results['error'].append(error)
            iterations_results['num_inliers'].append(num_inliers)
            iterations_results['kernel_threshold'].append(kernel_threshold)
            iterations_results['dumping_factor'].append(dumping_factor)
            error_prev = error

            if computation_done and error_mean_slope < 1e-2 and error_sigma_slope < 1e-2: stuck_counter += 1
            else: stuck_counter = 0
            if computation_done and error_mean_slope > 1 and error_sigma_slope > 1: flickering_counter += 1
            else: flickering_counter = 0

            if (dumping_factor/2) > self.__min_dumping_factor and (stuck_counter > limit or (stuck_counter == 0 and flickering_counter == 0)): dumping_factor /= 2
            if (dumping_factor*2) < self.__max_dumping_factor and flickering_counter > limit: dumping_factor *= 2

            if self.__verbose:
                print(f'Frame: {frame_index}, Iteration: {icp_iteration}')
                print(f'  - Error:            {np.round(error, 5)}')
                print(f'  - Num inliers:      {num_inliers}')
                print(f'  - Kernel threshold: {kernel_threshold}')
                print(f'  - Dumping factor:   {np.round(dumping_factor, 5)}')
                print('------------------------------------------------------------\n')

            if computation_done and error < 0.5: stop = True

        max_error_index = np.argmax(iterations_results['error'])
        min_error_index = np.argmin(iterations_results['error'])

        T = iterations_results['T'][min_error_index]
        is_valid = True
        if iterations_results['error'][min_error_index] > 30: 
            T = self.get_current_pose()
            is_valid = False
        self.__camera.set_c_T_w(np.linalg.inv(T))
    
        return T, is_valid, iterations_results

    def __one_step(self, reference_image_points, current_world_points, w_T_c0, kernel_threshold, dumping_factor):

        if (len(current_world_points) == 0): return w_T_c0, {'num_inliers': 0, 'error': np.Inf, 'kernel_threshold': kernel_threshold}, False

        H, b, num_inliers, error = self.__linearize(reference_image_points, current_world_points, kernel_threshold)
        results = {'num_inliers': num_inliers, 'error': error, 'kernel_threshold': kernel_threshold}

        if num_inliers < self.__min_inliners and kernel_threshold < self.__max_kernel_threshold: 
            kernel_threshold += 50
            results['kernel_threshold'] = kernel_threshold
            return w_T_c0, results, False
        
        if kernel_threshold > self.__min_kernel_threshold and error < 5: 
            kernel_threshold -= 50
            results['kernel_threshold'] = kernel_threshold

        H += np.eye(6) * dumping_factor
        dx = np.linalg.lstsq(H, -b, rcond=None)[0]
        w_T_c1 = v2T(dx) @ w_T_c0

        return w_T_c1, results, True

    def __linearize(self, reference_image_points, currrent_world_points, kernel_threshold):
        H = np.zeros((6, 6))
        b = np.zeros(6)

        chi_inliers = []
        errors = []
        jacobians = []

        for i in range(len(reference_image_points)):
            reference_image_point = reference_image_points[i]
            current_world_point = currrent_world_points[i]
            
            error, jacobian = self.__error_and_jacobian(reference_image_point, current_world_point)
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
    
    def __error_and_jacobian(self, reference_image_point, current_world_point):

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
        self.__current_pose = pose
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
            else:
                index = self.__map['appearance'].index(appearance)
                self.__map['position'][index] = position




    #| --------------- Public methods --------------- |#

    def run(self):
        start = time.time()
        frame_times = []
        for i in range(self.__initial_frame, self.__final_frame):
            start_frame = time.time()

            if i == self.__initial_frame: self.__initialize()
            else: self.__update(i)

            end_frame = time.time()
            frame_times.append(end_frame-start_frame)
        
        end = time.time()
        mean_time_per_frame = np.mean(frame_times)
        total_time = end-start
        
        print(f'Mean time per frame: {mean_time_per_frame} [s]')
        print(f'Total time:          {total_time} [s]\n')

        return total_time, mean_time_per_frame

    def evaluate(self):
        gt_trajectory = self.get_data().get_trajectory_data_poses()
        gt_world_points = self.get_data().get_world_data()

        estimated_trajectory = self.get_trajectory()
        estimated_world_points = self.get_map()

        C = self.get_camera().get_camera_transform()
        T = gt_trajectory[self.__initial_frame] @ Rt2T(R=C[:3,:3], t=np.zeros(3)) if self.__initial_frame > 0 else C

        estimated_trajectory_in_world = transform(estimated_trajectory, T)
        estimated_world_points_in_world = transform(estimated_world_points['position'], T, are_points=True)

        delta_poses_estimated_trajectory = []
        for i in range(len(estimated_trajectory_in_world)-1):
            rel_pose = np.linalg.inv(estimated_trajectory_in_world[i]) @ estimated_trajectory_in_world[i+1]
            delta_poses_estimated_trajectory.append(rel_pose)

        delta_poses_gt_trajectory = []
        for i in range(len(gt_trajectory)-1):
            rel_pose = np.linalg.inv(gt_trajectory[i]) @ gt_trajectory[i+1]
            delta_poses_gt_trajectory.append(rel_pose)

        errors_T = []
        for i in range(len(delta_poses_estimated_trajectory)):
            error_T = np.linalg.inv(delta_poses_estimated_trajectory[i]) @ delta_poses_gt_trajectory[i]
            errors_T.append(error_T)

        rotation_errors = []
        for i in range(len(errors_T)):
            rotation_error = np.trace(np.eye(3)-errors_T[i][:3,:3])
            rotation_errors.append(rotation_error)

        translation_errors = []
        for i in range(len(delta_poses_estimated_trajectory)):
            rel_pose = delta_poses_estimated_trajectory[i]
            rel_pose_gt = delta_poses_gt_trajectory[i]
            translation_error = np.linalg.norm(rel_pose[:3,3])/np.linalg.norm(rel_pose_gt[:3,3])
            translation_errors.append(translation_error)
        
        max_rotation_error = np.max(rotation_errors)
        min_rotation_error = np.min(rotation_errors)
        mean_rotation_error = np.mean(rotation_errors)

        max_translation_error = np.max(translation_errors)
        min_translation_error = np.min(translation_errors)
        mean_translation_error = np.mean(translation_errors)

        scale = 1/mean_translation_error

        estimated_trajectory_in_world = transform(estimated_trajectory_in_world, scale=scale)
        estimated_world_points_in_world = transform(estimated_world_points_in_world, scale=scale, are_points=True)
        matches = data_association_on_appearance({'position':estimated_world_points_in_world, 'appearance':estimated_world_points['appearance']}, gt_world_points)
        estimated_world_points_in_world_matched = matches['points_1']
        gt_world_points_matched = matches['points_2']

        rmse_world_map = np.sqrt(np.mean(np.linalg.norm(np.array(estimated_world_points_in_world_matched)-np.array(gt_world_points_matched), axis=1)**2))
        num_world_points = len(estimated_world_points_in_world_matched)

        print(f'Number of world points: {num_world_points}')
        print(f'RMSE world map:         {rmse_world_map}\n')

        print(f'Max rotation error:     {max_rotation_error}')
        print(f'Min rotation error:     {min_rotation_error}')
        print(f'Mean rotation error:    {mean_rotation_error}\n')
        
        print(f'Max translation error:  {max_translation_error}')
        print(f'Min translation error:  {min_translation_error}')
        print(f'Mean translation error: {mean_translation_error}')  
        print(f'scale:                  {scale}')   

        fig = go.Figure()
        plot_points(fig, poses2positions([gt_trajectory[self.__initial_frame]]), name='Initial GT pose', mode='markers', color='deepskyblue', size=3)
        plot_points(fig, poses2positions([gt_trajectory[self.__final_frame]]), name='Final GT pose', mode='markers', color='deepskyblue', size=3)

        plot_points(fig, poses2positions(gt_trajectory), name='Ground Truth trajectory', mode='lines', color='blue', width=3)
        plot_points(fig, poses2positions(estimated_trajectory_in_world), name='Estimated trajectory', mode='lines', color='red', width=5)

        plot_points(fig, estimated_world_points_in_world, name='Estimated map', mode='markers', color='orange', size=2)
        plot_points(fig, gt_world_points['position'], name='Ground Truth map', mode='markers', color='green', size=2)
        
        plot_matches(fig, estimated_world_points_in_world_matched, gt_world_points_matched, name='Map matches', color='violet', width=2)

        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show()

    def get_current_pose(self): return self.__current_pose
    def get_trajectory(self): return self.__trajectory
    def get_map(self): return self.__map
    def get_data(self): return self.__data
    def get_camera(self): return self.__camera