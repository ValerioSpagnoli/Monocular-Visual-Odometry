from src.VisualOdometry import VisualOdometry

initial_frame = 0
final_frame = 50
verbose = False
save_plots = False
save_plots_indices = []

vo = VisualOdometry(initial_frame=initial_frame, 
                    final_frame=final_frame, 
                    verbose=verbose, 
                    save_plots=save_plots, 
                    save_plots_indices=save_plots_indices)

total_time, mean_time_per_frame = vo.run()
vo.evaluate()