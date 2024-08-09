# Projective ICP Visual Odometry

## Description
This project implements a projective ICP based visual odometry, to estimate both the trajectory of a robot and the 3D map of the visual features.

### Data
Set of 120 measurements where each measurement contains a variable set of image points (1x2 vectors) and the associated feature descriptor (1x10 vectors, called appearance).

### Algorithm

#### Initialization
The initial pose (the one relative to the measurement 0) is set to the identity. Then the estimate of the first pose is computed with the following steps:
  
1. match the image points of the measurement 0 with the image points of the measurement 1 using the appearance;
2. estimate the essential matrix relative to the first two camera poses with the set of matched image points;
3. recover the relative position between the two camera poses using the essential matrix and the set of matched image points.

These steps provide a first estimate of the pose of the camera and an initial set of world points.

<p align="center">
<img src="outputs/Final Results/first_estimate.png" alt="First Estimate" width="600"/>
</p>

#### Update

#### Projective ICP

## Usage

## Results