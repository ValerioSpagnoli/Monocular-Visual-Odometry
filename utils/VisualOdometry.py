import numpy as np
import cv2

class VisualOdometry:
    def __init__(self, camera, data):
        self.camera = camera
        self.data = data
        self.map = None

    def data_association(self, set_1, set_2):

        matches = {'points_1':[], 'points_2':[], 'appearance':[]}

        points_1 = set_1['points']
        appearances_1 = set_1['appearance']
        
        points_2 = set_2['points']
        appearances_2 = set_2['appearance']
        
        for i in range(len(points_1)):
            point_1 = points_1[i]
            appearance_1 = appearances_1[i]
            
            for j in range(len(points_2)):
                point_2 = points_2[j]
                appearance_2 = appearances_2[j]

                if appearance_1 == appearance_2:
                    matches['points_1'].append(point_1)
                    matches['points_2'].append(point_2)
                    matches['appearance'].append(appearance_1)
                    break   

        return matches


    def Rt2T(self, R, t):
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t.T
        return T


    def initialize(self):

        #* Pose of the camera w.r.t. the robot
        C = self.camera.get_extrinsic_matrix()


        #* DATA ASSOCIATION

        #* Image points in frame 0 and frame 1
        points_0 = self.data.get_measurement_points(0)
        points_1 = self.data.get_measurement_points(1)
        
        #* Data association between the points in frame 0 and frame 1
        matches = self.data_association(points_0, points_1)
        set_0 = np.array(matches['points_1'])  
        set_1 = np.array(matches['points_2'])



        #* RECOVER POSE

        #* Pose of the camera in frame 0 w.r.t. the world frame
        R_0 = np.eye(3)
        t_0 = np.zeros((3, 1))
        T_0 = self.Rt2T(R_0, t_0)

        #* Pose of the camera in frame 1 w.r.t. camera in frame 0
        E, _ = cv2.findEssentialMat(set_0, set_1, self.camera.get_camera_matrix(), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R_0_1, t_0_1, _ = cv2.recoverPose(E, set_0, set_1, self.camera.get_camera_matrix())
        T_0_1 = self.Rt2T(R_0_1, t_0_1)

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
        
        return T_1, world_points