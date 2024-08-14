import numpy as np
import cv2

def Rx(theta):
    return np.array([[1, 0,              0            ], 
                     [0, np.cos(theta), -np.sin(theta)], 
                     [0, np.sin(theta),  np.cos(theta)]])

def Ry(theta):
    return np.array([[ np.cos(theta), 0, np.sin(theta)], 
                     [ 0,             1, 0            ], 
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], 
                     [np.sin(theta),  np.cos(theta), 0], 
                     [0,              0,             1]])

def euler2R(euler):
    Rot_x = Rx(euler[0])
    Rot_y = Ry(euler[1])
    Rot_z = Rz(euler[2])
    return Rot_x @ Rot_y @ Rot_z 

def R2euler(R):
    s = np.linalg.norm(np.diag(R))
    singular = s < 1e-6
    if singular:
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], s)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], s)
        z = 0
    return np.array([x, y, z])

def v2T(v):
    translation = np.array(v[:3])
    rotation_euler = np.array(v[3:])
    R = euler2R(rotation_euler.astype(np.float32))
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = translation.T
    return T

def T2v(T):
    v = np.zeros(6)
    v[:3] = T[:3,3]
    v[3:] = R2euler(T[:3,:3]).T[0]
    return v

def Rt2T(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t.T
    return T

def skew(w):
    return np.array([[    0, -w[2],  w[1]], 
                     [ w[2],     0, -w[0]], 
                     [-w[1],  w[0],    0]])

def triangulate_points(points_0, points_1, w_T_c0, w_T_c1, K, threshold=30):

    T = np.linalg.inv(w_T_c1) @ w_T_c0
    R = T[:3, :3] 
    t = T[:3, 3].reshape(-1, 1)   

    #** Projection matrices
    P_0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P_1 = K @ np.hstack((R, t))        

    #** Triangulate points
    points_4D = cv2.triangulatePoints(P_0, P_1, points_0.T, points_1.T)

    points_3D = (points_4D[:3] / points_4D[3]).T
    mask_depth = (points_3D[:, 2] > 0)# & (points_3D[:, 2] <= 5)

    points_3D_norms = np.linalg.norm(points_3D, axis=1)
    mask_norms = points_3D_norms < threshold 

    mask = mask_depth & mask_norms

    points_3D_filtered = points_3D[mask]
    points_4D = np.hstack((points_3D_filtered, np.ones((points_3D_filtered.shape[0], 1))))

    points_4D = w_T_c0 @ points_4D.T
    points_3D = points_4D[:3] / points_4D[3]

    return points_3D.T, mask

def transform(poses, T=np.eye(4), scale=1, are_points=False):
    translated = []
    for i in range(len(poses)):
        pose = poses[i]
        if are_points: pose = v2T([pose[0], pose[1], pose[2], 0, 0, 0])
        
        #* scale
        R = pose[:3,:3]
        t = pose[:3,3]
        pose = Rt2T(R=R, t=t*scale)

        #* translate
        pose = T @ pose

        if are_points: translated.append(pose[:3,3])
        else: translated.append(pose)
    return translated

def poses2positions(poses):
    positions = []
    for i in range(len(poses)):
        positions.append(poses[i][:3,3])
    return positions