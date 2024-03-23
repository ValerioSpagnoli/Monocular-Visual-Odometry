import numpy as np
import time


def Rx(theta):
    """
    Compute the rotation matrix around the x-axis.

    Parameters:
    theta (float): The rotation angle in radians.

    Returns:
    numpy.ndarray: The rotation matrix.
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def Ry(theta):
    """
    Compute the rotation matrix around the y-axis.

    Parameters:
    theta (float): The rotation angle in radians.

    Returns:
    numpy.ndarray: The rotation matrix.
    """
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def Rz(theta):
    """
    Compute the rotation matrix around the z-axis.

    Parameters:
    theta (float): The rotation angle in radians.

    Returns:
    numpy.ndarray: The rotation matrix.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

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

def v2T(v):
    """
    Convert a 6D vector into a homogeneous transformation matrix.

    Parameters:
    v (numpy.ndarray): The 6D vector.

    Returns:
    numpy.ndarray: The homogeneous transformation matrix T of shape (4, 4).
    """
    T = np.eye(4)
    R = np.dot(Rx(v[3]), np.dot(Ry(v[4]), Rz(v[5])))
    T[:3,:3] = R
    T[:3,3] = v[:3].T
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