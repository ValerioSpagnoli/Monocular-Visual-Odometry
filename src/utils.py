import numpy as np

def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def Rt2T(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t.T
    return T
