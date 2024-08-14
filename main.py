from src.VisualOdometry import VisualOdometry

initial_frame = 0          #* frame of the data to start with
final_frame = 50           #* frame of the data to end with
verbose = False            #* print additional information for each iteration of PICP
save_plots = False         #* save plots of the PICP iterations (is very slow and takes a lot of memory)
save_plots_indices = []    #* save plots of the PICP iterations for the specified indices

vo = VisualOdometry(initial_frame=initial_frame, 
                    final_frame=final_frame, 
                    verbose=verbose, 
                    save_plots=save_plots, 
                    save_plots_indices=save_plots_indices)

total_time, mean_time_per_frame = vo.run()
vo.evaluate()