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
    # return cv2.Rodrigues(euler)[0]
    Rot_x = Rx(euler[0])
    Rot_y = Ry(euler[1])
    Rot_z = Rz(euler[2])
    return Rot_x @ Rot_y @ Rot_z 

def R2euler(R):
    # rotation_vector = cv2.Rodrigues(R)[0]
    # roll = rotation_vector[0]
    # pitch = rotation_vector[1]
    # yaw = rotation_vector[2]

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