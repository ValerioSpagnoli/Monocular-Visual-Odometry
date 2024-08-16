from src.Camera import Camera
from src.Data import Data
from src.data_association import *
from src.visualization_utils import *
from src.ProjectiveICP import ProjectiveICP

import numpy as np
import time

class VisualOdometry:
    def __init__(self, initial_frame=0, 
                       final_frame=120, 
                       verbose=False, 
                       save_results=True, 
                       save_icp_plots=False, 
                       save_icp_plots_indices=[],
                       base_kernel_threshold=1500,
                       min_kernel_threshold=100,
                       max_kernel_threshold=3*1e3,
                       base_dumping_factor=1e4,
                       min_dumping_factor=1e3,
                       max_dumping_factor=1e9,
                       min_inliners=10,
                       num_iterations=300):


        #** General parameters
        self.__initial_frame = initial_frame
        self.__final_frame = final_frame
        self.__verbose = verbose    
        self.__save_results = save_results
        self.__save_icp_plots = save_icp_plots
        self.__save_icp_plots_indices = save_icp_plots_indices

        #** Projective ICP parameters
        self.__base_kernel_threshold = base_kernel_threshold
        self.__min_kernel_threshold = min_kernel_threshold
        self.__max_kernel_threshold = max_kernel_threshold

        self.__base_dumping_factor = base_dumping_factor
        self.__min_dumping_factor = min_dumping_factor
        self.__max_dumping_factor = max_dumping_factor

        self.__min_inliners = min_inliners
        self.__num_iterations = num_iterations

        self.__num_of_frames = 0

        #** Camera and Data
        self.__camera = Camera()
        self.__data = Data()
    
        #** Projective ICP
        self.ProjectiveICP = ProjectiveICP( self.__camera, 
                                            self.__data,
                                            self.__num_iterations,
                                            self.__min_inliners,
                                            self.__base_kernel_threshold,
                                            self.__min_kernel_threshold,
                                            self.__max_kernel_threshold,
                                            self.__base_dumping_factor,
                                            self.__min_dumping_factor,
                                            self.__max_dumping_factor,
                                            self.__verbose,
                                            self.__save_results,
                                            self.__save_icp_plots,
                                            self.__save_icp_plots_indices)


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

            self.__num_of_frames += 1
        
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

        rel_poses = {'estimated':[], 'gt':[]}
        errors = {'rotation':[], 'translation':[]}
        ratios = {'rotation':[], 'translation':[]}
        scales = []

        for i in range(len(estimated_trajectory_in_world)-1):
            rel_pose_est = np.linalg.inv(estimated_trajectory_in_world[i]) @ estimated_trajectory_in_world[i+1]
            rel_poses['estimated'].append(rel_pose_est)

            rel_pose_gt = np.linalg.inv(gt_trajectory[i]) @ gt_trajectory[i+1]
            rel_poses['gt'].append(rel_pose_gt)

            R_error = np.round(rel_pose_est[:3,:3],5)
            rotation_error = np.arccos((np.trace(R_error)-1)/2)
            errors['rotation'].append(rotation_error)

            rotation_ratio = np.linalg.norm(estimated_trajectory_in_world[i][:3,:3], 'fro')/np.linalg.norm(gt_trajectory[i][:3,:3], 'fro')
            translation_ratio = np.linalg.norm(estimated_trajectory_in_world[i+1][:3,3])/np.linalg.norm(gt_trajectory[i+1][:3,3])
            ratios['rotation'].append(rotation_ratio)
            ratios['translation'].append(translation_ratio)

            scale = 1/translation_ratio
            scales.append(scale)
            errors['translation'].append(np.linalg.norm(scale*estimated_trajectory_in_world[i][:3,3]-gt_trajectory[i][:3,3]))


        max_rotation_error = np.max(errors['rotation'])
        min_rotation_error = np.min(errors['rotation'])
        mean_rotation_error = np.mean(errors['rotation'])

        max_translation_error = np.max(errors['translation'])    
        min_translation_error = np.min(errors['translation'])
        mean_translation_error = np.mean(errors['translation'])

        max_rotation_ratio = np.max(ratios['rotation'])
        min_rotation_ratio = np.min(ratios['rotation'])
        mean_rotation_ratio = np.mean(ratios['rotation'])

        max_translation_ratio = np.max(ratios['translation'])
        min_translation_ratio = np.min(ratios['translation'])
        mean_translation_ratio = np.mean(ratios['translation'])

        scale = np.mean(scales)

        estimated_trajectory_in_world = transform(estimated_trajectory_in_world, scale=scale)
        estimated_world_points_in_world = transform(estimated_world_points_in_world, scale=scale, are_points=True)
        matches = data_association_on_appearance({'position':estimated_world_points_in_world, 'appearance':estimated_world_points['appearance']}, gt_world_points)
        estimated_world_points_in_world_matched = matches['points_1']
        gt_world_points_matched = matches['points_2']

        rmse_world_map = np.sqrt(np.mean(np.linalg.norm(np.array(estimated_world_points_in_world_matched)-np.array(gt_world_points_matched), axis=1)**2))
        num_world_points = len(estimated_world_points_in_world_matched)

        print(f'Number of frames:            {self.__num_of_frames}')
        print(f'Number of world points:      {num_world_points}')
        print(f'RMSE world map [m]:          {np.round(rmse_world_map, 3)}\n')
        print(f'scale:                       {np.round(scale, 3)}\n')

        print(f'Rotation errors: \n')
        print(f'Max rotation error [rad]:    {np.round(max_rotation_error, 3)}')
        print(f'Min rotation error [rad]:    {np.round(min_rotation_error, 3)}')
        print(f'Mean rotation error [rad]:   {np.round(mean_rotation_error, 3)}\n')
        
        print(f'Translation errors: \n')
        print(f'Max translation error:       {np.round(max_translation_error, 3)}')
        print(f'Min translation error:       {np.round(min_translation_error, 3)}')
        print(f'Mean translation error:      {np.round(mean_translation_error, 3)}\n')

        print(f'Ratios: \n')
        print(f'Max rotation ratio:          {np.round(max_rotation_ratio, 3)}')
        print(f'Min rotation ratio:          {np.round(min_rotation_ratio, 3)}')
        print(f'Mean rotation ratio:         {np.round(mean_rotation_ratio, 3)}\n')

        print(f'Max translation ratio:       {np.round(max_translation_ratio, 3)}')
        print(f'Min translation ratio:       {np.round(min_translation_ratio, 3)}')
        print(f'Mean translation ratio:      {np.round(mean_translation_ratio, 3)}\n')

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

        plot_final_results(errors, ratios, 'outputs/final_results/errors')