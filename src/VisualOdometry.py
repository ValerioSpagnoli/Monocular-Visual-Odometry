import numpy as np
import cv2

from src import math

class VisualOdometry:
    def __init__(self, camera, data):
        self.camera = camera
        self.data = data
        self.map = {'position':[], 'appearance':[]}
        self.trajectory = {'poses':[], 'world_points':[]}


    def data_association(self, set_1, set_2):

        points_1 = set_1['points']
        appearances_1 = set_1['appearance']
        
        points_2 = set_2['points']
        appearances_2 = set_2['appearance']

        matches_points_1 = []
        matches_points_2 = []
        matches_appearance = []

        for i in range(len(points_1)):
            point_1 = points_1[i]
            appearance_1 = appearances_1[i]
            
            for j in range(len(points_2)):
                point_2 = points_2[j]
                appearance_2 = appearances_2[j]

                if appearance_1 == appearance_2:
                    matches_points_1.append(point_1)
                    matches_points_2.append(point_2)
                    matches_appearance.append(appearance_1)
                    break   
        
        return {'points_1':np.array(matches_points_1), 'points_2':np.array(matches_points_2), 'appearance':np.array(matches_appearance)}

 
    def initialize(self):

        #* Pose of the camera w.r.t. the robot
        C = self.camera.get_extrinsic_matrix()


        #* DATA ASSOCIATION

        #* Image points in frame 0 and frame 1
        points_0 = self.data.get_measurement_points(0)
        points_1 = self.data.get_measurement_points(1)
        
        #* Data association between the points in frame 0 and frame 1
        matches = self.data_association(points_0, points_1)
        set_0 = matches['points_1']  
        set_1 = matches['points_2']


        #* RECOVER POSE

        #* Pose of the camera in frame 0 w.r.t. the world frame
        R_0 = np.eye(3)
        t_0 = np.zeros((3, 1))
        T_0 = math.Rt2T(R_0, t_0)

        #* Pose of the camera in frame 1 w.r.t. camera in frame 0
        E, _ = cv2.findEssentialMat(set_0, set_1, self.camera.get_camera_matrix(), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R_0_1, t_0_1, _ = cv2.recoverPose(E, set_0, set_1, self.camera.get_camera_matrix())
        T_0_1 = math.Rt2T(R_0_1, t_0_1)

        #* Pose of the camera in frame 1 w.r.t. the world frame
        T_1 = np.dot(-np.linalg.inv(C), T_0_1)


        #* TRIANGULATE POINTS

        #* Projection matrices of the camera in frame 0 and frame 1
        P1 = np.dot(self.camera.get_intrinsic_matrix(), np.eye(4))
        P2 = np.dot(self.camera.get_intrinsic_matrix(), T_0_1)
        
        #* world points w.r.t. camera in frame 0
        world_points_hom = cv2.triangulatePoints(P1, P2, set_0.T, set_1.T)
        
        #* world points w.r.t. the world frame
        world_points_hom = np.dot(np.linalg.inv(C), world_points_hom)

        #* Normalize the world points
        world_points = (world_points_hom / world_points_hom[3])[:3].T

        world_points = {'position': world_points, 'appearance': matches['appearance']}
        
        return T_1, world_points
    

    def error_and_jacobian(self, world_point, image_point):
        
        #* Compute the prediction
        predicted_image_point = self.camera.project(world_point)
        if predicted_image_point[0] == -1 and predicted_image_point[1] == -1:
            return None, None

        #* Compute the error
        error = predicted_image_point - image_point

        #* Compute the Jacobian of the transformation
        world_point_hom = np.append(world_point, 1)
        camera_point = np.dot(self.camera.get_extrinsic_matrix(), world_point_hom)
        camera_point = (camera_point / camera_point[3])[:3]
        Jr = np.zeros((3, 6))
        Jr[:3, :3] = np.eye(3)
        Jr[:3, 3:] = math.skew(-camera_point)

        #* Compute the Jacobian of the projection
        image_point_hom = np.dot(self.camera.get_camera_matrix(), camera_point)
        iz = 1.0 / image_point_hom[2]
        iz2 = iz * iz

        Jp = np.array([
            [iz, 0, -image_point_hom[0]*iz2],
            [0, iz, -image_point_hom[1]*iz2]
        ])

        #* Compute the Jacobian
        jacobian = np.dot(Jp, np.dot(self.camera.get_camera_matrix(), Jr))

        return error, jacobian


    def add_to_map(self, world_points):
        positions = world_points['position']
        appearances = world_points['appearance']
        for i in range(len(positions)):
            self.map['position'].append(positions[i])
            self.map['appearance'].append(appearances[i])


    def add_to_trajectory(self, T, world_points):
        self.trajectory['poses'].append(T)
        self.trajectory['world_points'].append(world_points)

    def update_state(self, T, world_points):
        self.add_to_map(world_points)
        self.add_to_trajectory(T, world_points)

    def get_map(self):
        return self.map
    
    def get_trajectory(self, only_poses=False):
        if only_poses: return self.trajectory['poses']
        return self.trajectory
