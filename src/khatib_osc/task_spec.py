"""Generalized task specification matrices Omega, Omega_tilde (Sec. II, eq. 1-4).

Sigma_f (eq. 1) picks out the position directions left free for motion, in a
task frame R_f obtained from the reference frame by rotation S_f that aligns
z_f with the force direction f_d. Sigma_bar_f = I - Sigma_f (eq. 2) is its
complement: the force-controlled directions. Sigma_r, Sigma_bar_r (defined
"similarly" for moments, per the paper's text after eq. 2) do the same for
rotational freedom in a frame R_r aligned by S_r with the moment direction.

Omega, Omega_tilde (eq. 3-4) assemble both position and rotation blocks,
expressed back in the reference frame R_0 by conjugating with S_f/S_r, so
that a single controller can work entirely in R_0 without per-task frame
transforms elsewhere (this is exactly the motivation the paper gives for
introducing Omega/Omega_tilde in the first place).
"""

from __future__ import annotations

import numpy as np


def position_spec(free_x: bool, free_y: bool, free_z: bool) -> np.ndarray:
    """Sigma_f (eq. 1): diag of binary free-motion flags along task-frame x_f,y_f,z_f."""
    return np.diag([float(free_x), float(free_y), float(free_z)])


def complement(sigma: np.ndarray) -> np.ndarray:
    """Sigma_bar = I - Sigma (eq. 2)."""
    return np.eye(sigma.shape[0]) - sigma


def generalized_task_spec(S_f: np.ndarray, Sigma_f: np.ndarray, S_r: np.ndarray, Sigma_r: np.ndarray) -> np.ndarray:
    """Omega (eq. 3): block-diagonal [S_f^T Sigma_f S_f, S_r^T Sigma_r S_r], acting on R_0 vectors."""
    top = S_f.T @ Sigma_f @ S_f
    bottom = S_r.T @ Sigma_r @ S_r
    Omega = np.zeros((6, 6))
    Omega[:3, :3] = top
    Omega[3:, 3:] = bottom
    return Omega


def make_task_spec(
    S_f: np.ndarray, Sigma_f: np.ndarray, S_r: np.ndarray, Sigma_r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Both Omega (eq. 3, motion directions) and Omega_tilde (eq. 4, force directions)."""
    Sigma_bar_f = complement(Sigma_f)
    Sigma_bar_r = complement(Sigma_r)
    Omega = generalized_task_spec(S_f, Sigma_f, S_r, Sigma_r)
    Omega_tilde = generalized_task_spec(S_f, Sigma_bar_f, S_r, Sigma_bar_r)
    return Omega, Omega_tilde


def surface_wiping_spec(surface_normal_R0: np.ndarray = np.array([0.0, 0.0, 1.0])) -> tuple[np.ndarray, np.ndarray]:
    """Task spec for sliding on a surface while holding a normal contact force.

    Motion is free along the two directions tangent to the surface and in
    all 3 rotations (orientation is motion-controlled, held rigid by the
    tracking law); force is controlled along the surface normal, no moment
    control. S_f rotates R_0's z axis onto the given surface normal (identity
    for a flat horizontal table, i.e. normal = world z); S_r = I since no
    moment-control frame is needed (Sigma_r = I, Sigma_bar_r = 0 regardless
    of S_r).
    """
    z = surface_normal_R0 / np.linalg.norm(surface_normal_R0)
    if np.allclose(z, [0, 0, 1]):
        S_f = np.eye(3)
    else:
        # S_f maps R_0 vectors into R_f, where z_f is aligned with the surface
        # normal (per the paper's definition of R_f right before eq. 1); i.e.
        # S_f @ z == [0,0,1]. Build the inverse rotation (world z -> normal)
        # via Rodrigues' formula, then transpose.
        v = np.cross([0, 0, 1], z)
        c = np.dot([0, 0, 1], z)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_z_to_normal = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
        S_f = R_z_to_normal.T
    Sigma_f = position_spec(True, True, False)  # free (motion) in x_f, y_f; z_f is force-controlled
    S_r = np.eye(3)
    Sigma_r = np.eye(3)  # all rotations motion-controlled
    return make_task_spec(S_f, Sigma_f, S_r, Sigma_r)
