import numpy as np
import time

def skew(v):
    """
    Computes the skew-symmetric matrix of a 3D vector.

    Parameters:
    v (numpy.ndarray): The 3D vector.

    Returns:
    numpy.ndarray: The skew-symmetric matrix.

    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def Rt2T(R, t):
    """
    Convert a rotation matrix R and a translation vector t into a homogeneous transformation matrix T.

    Parameters:
    R (numpy.ndarray): The rotation matrix of shape (3, 3).
    t (numpy.ndarray): The translation vector of shape (3,).

    Returns:
    numpy.ndarray: The homogeneous transformation matrix T of shape (4, 4).
    """
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t.T
    return T


def get_time(in_seconds=False):
    """
    Get the current time.

    Parameters:
        in_seconds (bool): If True, return the time in seconds. If False, return the time in milliseconds.

    Returns:
        float: The current time in seconds or milliseconds, depending on the value of `in_seconds`.
    """
    if in_seconds:
        return time.time()
    else:
        return time.time()*1000