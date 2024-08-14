from omegaconf import OmegaConf
from src.VisualOdometry import VisualOdometry

config = OmegaConf.load("config/config.yaml")

initial_frame = config.general.initial_frame
final_frame = config.general.final_frame
verbose = config.general.verbose
save_results = config.general.save_results
save_icp_plots = config.general.save_icp_plots
save_icp_plots_indices = config.general.save_icp_plots_indices

base_kernel_threshold = config.picp.base_kernel_threshold
min_kernel_threshold = config.picp.min_kernel_threshold
max_kernel_threshold = config.picp.max_kernel_threshold

base_dumping_factor = config.picp.base_dumping_factor
min_dumping_factor = config.picp.min_dumping_factor
max_dumping_factor = config.picp.max_dumping_factor

min_inliners = config.picp.min_inliers
num_iterations = config.picp.num_iterations

vo = VisualOdometry(initial_frame=initial_frame, 
                    final_frame=final_frame, 
                    verbose=verbose, 
                    save_results=save_results,
                    save_icp_plots=save_icp_plots, 
                    save_icp_plots_indices=save_icp_plots_indices,
                    base_kernel_threshold=base_kernel_threshold,
                    min_kernel_threshold=min_kernel_threshold,
                    max_kernel_threshold=max_kernel_threshold,
                    base_dumping_factor=base_dumping_factor,
                    min_dumping_factor=min_dumping_factor,
                    max_dumping_factor=max_dumping_factor,
                    min_inliners=min_inliners,
                    num_iterations=num_iterations)

total_time, mean_time_per_frame = vo.run()
vo.evaluate()