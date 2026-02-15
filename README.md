# Projective ICP Visual Odometry

<p align="center">
<img src="outputs/final_results//icp.gif" alt="First Estimate" width="800"/>
</p>

## Description
This project implements a projective ICP based visual odometry to estimate both the trajectory of a robot and the 3D map of the visual features.

## Explanation and visual/numerical results
The explanation and the visual and numerical results are published on [Github Pages - ValerioSpagnoli/VisualOdometry]([https://valeriospagnoli.github.io/VisualOdometry/](https://valeriospagnoli.github.io/Monocular-Visual-Odometry/)).

## Outputs
In the folder outputs there are the results of this project:
- ```outputs/final_results``` contains
  - ```3D_plots.html```: visual results;
  - ```errors.png```: the plot of the rotation and translation errors and translation ratio;
  - ```estimated_trajectory.dat```: id, x, y, z values
  - ```estimated_world_points.dat```: id, x, y, z, appearance_1, ..., appearance_10
  - ```errors.dat```: rotation_error, rotation_ratio, translation_error, translation_ratio.
- In ```outputs/frame_xxx``` there are the plots of the ```errors``` (mean of ```chi inliers``` of that frame), the ```number of inliers```, the ```kernel thresholds``` and the ```dumping factors``` of that frame (iteration by iteration).

## Usage
1. Install the requirements:
    ```
    pip3 install -r requirements.txt
    ```
2. Run the main
    ```
    python3 main.py
    ```
