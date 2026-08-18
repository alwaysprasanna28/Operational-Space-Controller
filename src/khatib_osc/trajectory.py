"""Minimum-jerk Cartesian pose trajectories: x_d(t), xdot_d(t), xddot_d(t).

Feeds eq. (31)'s F*_m = I xddot_d - k_p(x-x_d) - k_v(xdot-xdot_d). The paper
writes x, xdot, xddot as if the whole m0-dimensional operational-space
vector lived in a single vector space; for a 6D pose task that is literally
true for the 3 position rows, and for the 3 rotation rows it is true in the
sense the paper itself requires (Sec. V): only *instantaneous angular
rotation* quantities (angular velocity/acceleration) are compatible with the
Sigma_r formulation, not integrated Euler angles. So the rotational channel
here is planned directly in (R, omega, alpha) -- never as an Euler-angle
trajectory that gets differentiated -- exactly the representation the paper
requires the F, Sigma matrices to operate on.

Position: quintic (minimum-jerk) interpolation p(t) = p0 + s(t)(p1-p0).
Orientation: slerp between R0 and R1 with the same time-scaling s(t):
    R(t) = R0 Exp(s(t) phi),   phi = log(R0^T R1)
so that the body-frame angular velocity is exactly sdot(t) phi (phi is
constant), giving closed-form
    omega_d(t) = R(t) (sdot(t) phi)
    alpha_d(t) = R(t) (sddot(t) phi)
(derived from Rdot = R [omega_local]_x and omega_local x omega_local = 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from khatib_osc.orientation import log_map


def quintic_scaling(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """s(tau), sdot(tau), sddot(tau) for tau in [0,1], with s(0)=sdot(0)=sddot(0)=0
    and s(1)=1, sdot(1)=sddot(1)=0 (minimum-jerk boundary conditions)."""
    tau = np.clip(tau, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sdot = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    sddot = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, sdot, sddot


@dataclass
class PoseWaypoint:
    """A single (position, orientation) target reached over duration T."""

    p1: np.ndarray
    R1: np.ndarray
    T: float


class PoseTrajectory:
    """Minimum-jerk point-to-point Cartesian pose trajectory, p0/R0 -> p1/R1 over [0, T]."""

    def __init__(self, p0: np.ndarray, R0: np.ndarray, p1: np.ndarray, R1: np.ndarray, T: float):
        self.p0 = np.asarray(p0, dtype=float)
        self.R0 = np.asarray(R0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)
        self.R1 = np.asarray(R1, dtype=float)
        self.T = float(T)
        self.phi = log_map(self.R0.T @ self.R1)  # constant local rotation vector

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (p, R, v, omega, a, alpha) at time t."""
        tau = np.clip(t / self.T, 0.0, 1.0) if self.T > 0 else 1.0
        s, sdot, sddot = quintic_scaling(np.array(tau))
        s, sdot, sddot = float(s), float(sdot / self.T), float(sddot / self.T**2)

        p = self.p0 + s * (self.p1 - self.p0)
        v = sdot * (self.p1 - self.p0)
        a = sddot * (self.p1 - self.p0)

        R = self.R0 @ Rotation.from_rotvec(s * self.phi).as_matrix()
        omega = R @ (sdot * self.phi)
        alpha = R @ (sddot * self.phi)

        return p, R, v, omega, a, alpha

    def evaluate_stacked(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (x, R, xdot, xddot) with x=p (3,), xdot/xddot = [v;omega]/[a;alpha] (6,)."""
        p, R, v, omega, a, alpha = self.evaluate(t)
        xdot = np.concatenate([v, omega])
        xddot = np.concatenate([a, alpha])
        return p, R, xdot, xddot
