from src.Camera import Camera
from src.Data import Data
from src.data_association import *
from src.visualization_utils import *
from src.ProjectiveICP import ProjectiveICP

import numpy as np
import time

class VisualOdometry:
    def __init__(self, initial_frame=0, final_frame=120, verbose=False, save_plots=False, save_plots_indices=[]):

        self.__initial_frame = initial_frame
        self.__final_frame = final_frame + 1
        self.__verbose = verbose    

        #** Save plots
        self.__save_plots = save_plots
        self.__save_plots_indices = save_plots_indices

        #** Projective ICP parameters
        self.__kernel_threshold = 1500
        self.__min_kernel_threshold = 100
        self.__max_kernel_threshold = 3*1e3

        self.__dumping_factor = 1e4
        self.__min_dumping_factor = 1e3
        self.__max_dumping_factor = 1e9

        self.__min_inliners = 10
        self.__num_iterations = 300

        #** Camera and Data
        self.__camera = Camera()
        self.__data = Data()
    
        #** Projective ICP
        self.ProjectiveICP = ProjectiveICP( self.__camera, 
                                            self.__data,
                                            self.__num_iterations,
                                            self.__min_inliners,
                                            self.__kernel_threshold,
                                            self.__min_kernel_threshold,
                                            self.__max_kernel_threshold,
                                            self.__dumping_factor,
                                            self.__min_dumping_factor,
                                            self.__max_dumping_factor,
                                            self.__verbose,
                                            self.__save_plots,
                                            self.__save_plots_indices)


    def run(self):
        start = time.time()
        frame_times = []
        for i in range(self.__initial_frame, self.__final_frame):
            start_frame = time.time()

            if i == self.__initial_frame: res = self.ProjectiveICP.initialize(i)
            else: res = self.ProjectiveICP.update(i)

            end_frame = time.time()
            frame_times.append(end_frame-start_frame)

            if not res: 
                print(f'Not valid transform found at frame {i}. Break.')
                break
        
        end = time.time()
        mean_time_per_frame = np.mean(frame_times)
        total_time = end-start
        
        print(f'Mean time per frame: {mean_time_per_frame} [s]')
        print(f'Total time:          {total_time} [s]\n')

        return total_time, mean_time_per_frame

    def evaluate(self):
        gt_trajectory = self.ProjectiveICP.get_data().get_trajectory_data_poses()
        gt_world_points = self.ProjectiveICP.get_data().get_world_data()

        estimated_trajectory = self.ProjectiveICP.get_trajectory()
        estimated_world_points = self.ProjectiveICP.get_map()

        C = self.ProjectiveICP.get_camera().get_camera_transform()
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

        rotation_errors = {'rad':[], 'deg':[]}
        for i in range(len(errors_T)):
            R_error = errors_T[i][:3,:3]
            rotation_error_rad = np.trace(np.eye(3)-R_error)
            rotation_error_deg = np.rad2deg(rotation_error_rad)
            rotation_errors['rad'].append(rotation_error_rad)
            rotation_errors['deg'].append(rotation_error_deg)

        translation_errors = {'ratio':[], 'norm':[]}
        for i in range(len(delta_poses_estimated_trajectory)):
            rel_pose = delta_poses_estimated_trajectory[i]
            rel_pose_gt = delta_poses_gt_trajectory[i]
            translation_error_ratio = np.linalg.norm(rel_pose[:3,3])/np.linalg.norm(rel_pose_gt[:3,3])
            translation_errors_norm = np.linalg.norm(rel_pose[:3,3]/np.linalg.norm(rel_pose[:3,3])-rel_pose_gt[:3,3]/np.linalg.norm(rel_pose_gt[:3,3]))
            translation_errors['ratio'].append(translation_error_ratio)
            translation_errors['norm'].append(translation_errors_norm)
        
        max_rotation_error_rad = np.max(rotation_errors['rad'])
        min_rotation_error_rad = np.min(rotation_errors['rad'])
        mean_rotation_error_rad = np.mean(rotation_errors['rad'])

        max_rotation_error_deg = np.max(rotation_errors['deg'])
        min_rotation_error_deg = np.min(rotation_errors['deg'])
        mean_rotation_error_deg = np.mean(rotation_errors['deg'])

        max_translation_error_ratio = np.max(translation_errors['ratio'])
        min_translation_error_ratio = np.min(translation_errors['ratio'])
        mean_translation_error_ratio = np.mean(translation_errors['ratio'])

        max_translation_error_norm = np.max(translation_errors['norm'])
        min_translation_error_norm = np.min(translation_errors['norm'])
        mean_translation_error_norm = np.mean(translation_errors['norm'])

        scale = 1/mean_translation_error_ratio

        estimated_trajectory_in_world = transform(estimated_trajectory_in_world, scale=scale)
        estimated_world_points_in_world = transform(estimated_world_points_in_world, scale=scale, are_points=True)
        matches = data_association_on_appearance({'position':estimated_world_points_in_world, 'appearance':estimated_world_points['appearance']}, gt_world_points)
        estimated_world_points_in_world_matched = matches['points_1']
        gt_world_points_matched = matches['points_2']

        rmse_world_map = np.sqrt(np.mean(np.linalg.norm(np.array(estimated_world_points_in_world_matched)-np.array(gt_world_points_matched), axis=1)**2))
        num_world_points = len(estimated_world_points_in_world_matched)

        print(f'Number of frames:                  {self.__final_frame-self.__initial_frame}')
        print(f'Number of world points:            {num_world_points}')
        print(f'RMSE world map [m]:                {np.round(rmse_world_map, 5)}\n')
        print(f'scale:                             {np.round(scale, 5)}\n')

        print(f'Rotation errors: \n')
        print(f'Max rotation error [rad]:          {np.round(max_rotation_error_rad, 5)}')
        print(f'Min rotation error [rad]:          {np.round(min_rotation_error_rad, 5)}')
        print(f'Mean rotation error [rad]:         {np.round(mean_rotation_error_rad, 5)}\n')
        
        print(f'Max rotation error [deg]:          {np.round(max_rotation_error_deg, 5)}')
        print(f'Min rotation error [deg]:          {np.round(min_rotation_error_deg, 5)}')
        print(f'Mean rotation error [deg]:         {np.round(mean_rotation_error_deg, 5)}\n')

        print(f'Translation errors: \n')
        print(f'Max translation error ratio:       {np.round(max_translation_error_ratio, 5)}')
        print(f'Min translation error ratio:       {np.round(min_translation_error_ratio, 5)}')
        print(f'Mean translation error ratio:      {np.round(mean_translation_error_ratio, 5)}\n')

        print(f'Max translation error norm:        {np.round(max_translation_error_norm, 5)}')
        print(f'Min translation error norm:        {np.round(min_translation_error_norm, 5)}')
        print(f'Mean translation error norm:       {np.round(mean_translation_error_norm, 5)}\n')

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
        fig.write_html("outputs/final_results/3D_plot.html")

        plot_final_results(rotation_errors, translation_errors, 'outputs/final_results/errors')