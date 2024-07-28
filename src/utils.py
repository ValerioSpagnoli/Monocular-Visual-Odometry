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
    return cv2.Rodrigues(euler)[0]

def R2euler(R):
    rotation_vector = cv2.Rodrigues(R)[0]
    roll = rotation_vector[0]
    pitch = rotation_vector[1]
    yaw = rotation_vector[2]
    return np.array([roll, pitch, yaw])

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