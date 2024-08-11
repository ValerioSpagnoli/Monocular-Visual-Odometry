# Projective ICP Visual Odometry

<p align="center">
<img src="media/icp.gif" alt="First Estimate" width="800"/>
</p>

## Description
This project implements a projective ICP based visual odometry, to estimate both the trajectory of a robot and the 3D map of the visual features.
The robot is equioped with a monocular camera with known intrinsic and extrinsic parameters. 

### Data
Set of 120 measurements where each measurement contains a variable set of pairs (image_point, appearance), where:
- image point: `[r,c] 1x2` vector. 
- appearance: `[f_1, ..., f_{10}] 1x10` vector (feature descriptor)


### Algorithm

#### Initialization step
The initial pose (the one relative to the measurement 0) is set to the identity. Then the estimate of the first pose is computed with the following steps:
  
1. match the image points of the measurement 0 with the image points of the measurement 1 using the appearance;
2. estimate the essential matrix relative to the first two camera poses with the set of matched image points;
3. recover the relative position between the two camera poses using the essential matrix and the set of matched image points.

These steps provide a first estimate of the pose of the camera. 

Then a first estimated of the map is computed by triangulating the image points of the first two measurement, using the estimated pose.

#### Update step
The update steps takes one measurement at the time and perform the projective ICP algorithm between the current measurement and the current estimated 3D map, to recover the relative pose of the camera with these steps:
1. matches the image points of the current measurement with the 3D points of the map using the appearance;
2. performs one step of the projective ICP algorithm;
3. repeats until the maximum number of iterations is reached or a stopping criterion is met.

Using the new estimated pose of the camera from the projective ICP, are triangulated and added to the estimated map a new set of 3D points.

The update step is repeated for each measurement.

#### Projective ICP
A single step of the projective ICP is divided in two parts:
1. linearize the problem;
2. resolve the linerized problem with a least square approach.
   
The linearization part takes as input the reference image points (from the measurement) and the current world points from the estimated map, already matched, and the current pose of the camera w.r.t the world `w_T_c0`. Then, calculates the matrix *H* and the vector *b* by computing for each pair of points the error *e* and the jacobian *J* in this way:
- Projected world point: 
<p align="center">
<img src="media/equations/projected_world_point.png" alt="First Estimate" width="515"/>
</p>

 - Error: 
<p align="center">
<img src="media/equations/error.png" alt="First Estimate" width="105"/>
</p>

- Jacobian: 
<p align="center">
<img src="media/equations/jacobians.png" alt="First Estimate" width="240"/>
</p>

Then the error is used to compute the `chi = e^T e`:
- `if chi <= kernel_threshold: point is inlier`, 
- `else: point is outlier`.

The errors and jacobians from the inliers are used to compute *H* and *b* as:
<p align="center">
<img src="media/equations/H_b.png" alt="First Estimate" width="120"/>
</p>

Then a 6D vector describing the relative pose of the camera w.r.t the previous pose is calculated by solving 
<p align="center">
<img src="media/equations/system.png" alt="First Estimate" width="215"/>
</p>


## Results

### Visual results
<p align="center">
<iframe src="media/3D_plot.html" width="800" height="600"></iframe>
</p>

### Numerical results

The algorithm works well, but being without any correction the error increase with the iterations. Moreover, curves in general are very diffucult to handle. Indeed, as we can see in the plot below, the errors (both in rotation and translation) have a spike between the frames 15 and 24, which are the ones relative to the first curve of the trajectory.

<p align="center">
<img src="media/errors.png" alt="First Estimate" width="600"/>
</p>

| **General Parameters**            | **Value** | **Rotation Errors**               | **Value** | **Translation Errors**            | **Value** |
|-----------------------------------|-----------|-----------------------------------|-----------|-----------------------------------|-----------|
| Number of frames                  | 50        | Max rotation error [rad]          | 0.07941   | Max translation error ratio       | 5.11733   |
| Number of world points            | 314       | Min rotation error [rad]          | 0.00000   | Min translation error ratio       | 4.71346   |
| RMSE world map [m]                | 0.14540   | Mean rotation error [rad]         | 0.01111   | Mean translation error ratio      | 4.92998   |
| Scale                             | 0.20284   | Max rotation error [deg]          | 4.55003   | Max translation error norm        | 1.62579   |
|                                   |           | Min rotation error [deg]          | 0.00000   | Min translation error norm        | 1.38921   |
|                                   |           | Mean rotation error [deg]         | 0.63657   | Mean translation error norm       | 1.44381   |
