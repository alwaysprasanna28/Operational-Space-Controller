"""Orientation error as an instantaneous angular rotation vector.

Sec. V of the paper (p.48) notes that the rotation/moment specification
matrices Sigma_r, Sigma_bar_r are only compatible with descriptions of
orientation error using *instantaneous angular rotations* -- not Euler
angles, direction cosines, or Euler parameters directly as configuration
coordinates. The standard way to get such an error vector between two
rotation matrices is the SO(3) log map (axis-angle "rotation vector"): the
constant angular velocity that rotates R into R_d over unit time. That is
what orientation_error implements.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def orientation_error(R: np.ndarray, R_d: np.ndarray) -> np.ndarray:
    """Axis-angle error vector rotating R onto R_d, expressed in the world frame.

    delta_phi = log(R_d R^T): the rotation vector such that a body currently
    at R, subject to a constant world-frame angular velocity delta_phi over
    unit time, would arrive at R_d.
    """
    R_err = R_d @ R.T
    return Rotation.from_matrix(R_err).as_rotvec()


def pose_error(p: np.ndarray, R: np.ndarray, p_d: np.ndarray, R_d: np.ndarray) -> np.ndarray:
    """6-vector pose error e = [p - p_d; e_rot], consistent with the paper's
    e = x - x_d convention (eq. 31): e_rot vanishes at R = R_d and points in
    the direction that -k_p e_rot must push R toward R_d.

    e_rot = log(R R_d^T) = -orientation_error(R, R_d). Derivation: writing
    R = Exp(-e_rot) R_d for small e_rot (so e_rot -> 0 as R -> R_d, mirroring
    "current - desired"), matching terms gives e_rot = log(R R_d^T).
    """
    e_pos = p - p_d
    e_rot = log_map(R @ R_d.T)
    return np.concatenate([e_pos, e_rot])


def log_map(R: np.ndarray) -> np.ndarray:
    """SO(3) log map: rotation matrix -> axis-angle rotation vector."""
    return Rotation.from_matrix(R).as_rotvec()


def exp_map(rotvec: np.ndarray) -> np.ndarray:
    """SO(3) exp map: axis-angle rotation vector -> rotation matrix."""
    return Rotation.from_rotvec(rotvec).as_matrix()
