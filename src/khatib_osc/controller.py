"""Operational-space control laws: unconstrained motion (Sec. IV) and hybrid
motion/force (Sec. V).

Unconstrained motion, eq. (29)-(31), (34):
    F*_m = xddot_d - k_p (x - x_d) - k_v (xdot - xdot_d)
    F    = Lambda(x) F*_m + mu(x,xdot) + p(x)
    Gamma = J^T(q) F                                          (eq. 28)

Hybrid motion/force, eq. (45)-(47):
    F_m   = Lambda(x) Omega F*_m
    F_a   = Omega_tilde F*_a + Lambda(x) Omega_tilde F*_s
    F_ccg = mu(x,xdot) + p(x)
    F     = F_m + F_a + F_ccg
    Gamma = J^T(q) F
with F*_a = k_f (F_d - F_meas) (force error feedback, per the k_f block in
Fig. 3) and F*_s = -k_vf * xdot (end-effector velocity damping "that acts in
the direction of force control", per the paper's text after eq. 47). Note:
eq. (48)'s F_a = Lambda(x) Omega_tilde F*_s is explicitly the *impact
transition* stage (free motion -> first contact); this implementation only
targets steady contact regulation, where F*_s is retained purely as a
damping term for stability, not as the impact-dissipation law of eq. (48).
That is a deliberate, documented simplification -- see README limitations.

Both laws use MuJoCo-computed A(q), b(q,qdot), g(q), J(q) (dynamics.py) --
"from scratch" here means the control law itself, not a reimplementation of
rigid-body dynamics (see README).
"""

from __future__ import annotations

import numpy as np


def motion_command(
    e: np.ndarray,
    edot: np.ndarray,
    xddot_d: np.ndarray,
    kp: np.ndarray,
    kv: np.ndarray,
) -> np.ndarray:
    """F*_m (eq. 31): decoupled end-effector command for unconstrained motion.

    e, edot are the pose error and its derivative, e = [p-p_d; e_rot] (see
    orientation.pose_error -- a plain x-x_d subtraction is only valid for the
    position rows; e_rot is the SO(3)-correct analogue). edot = xdot - xdot_d
    is a plain subtraction, valid componentwise since linear/angular
    velocities live in a vector space (no manifold subtlety there).
    """
    return xddot_d - kp @ e - kv @ edot


def unconstrained_torque(
    J: np.ndarray,
    Lambda: np.ndarray,
    mu: np.ndarray,
    p: np.ndarray,
    F_m_star: np.ndarray,
) -> np.ndarray:
    """Gamma (eq. 28, with F from eq. 29-30): full unconstrained-motion OSC torque."""
    F = Lambda @ F_m_star + mu + p
    return J.T @ F


def hybrid_torque(
    J: np.ndarray,
    Lambda: np.ndarray,
    mu: np.ndarray,
    p: np.ndarray,
    Omega: np.ndarray,
    Omega_tilde: np.ndarray,
    F_m_star: np.ndarray,
    F_a_star: np.ndarray,
    F_s_star: np.ndarray,
) -> np.ndarray:
    """Gamma (eq. 28, with F from eq. 45-46): hybrid motion/force OSC torque."""
    F_m = Lambda @ (Omega @ F_m_star)
    F_a = Omega_tilde @ F_a_star + Lambda @ (Omega_tilde @ F_s_star)
    F_ccg = mu + p
    F = F_m + F_a + F_ccg
    return J.T @ F


def force_command(F_d: np.ndarray, F_meas: np.ndarray, kf: np.ndarray) -> np.ndarray:
    """F*_a: force error feedback (k_f block, Fig. 3)."""
    return kf @ (F_d - F_meas)


def force_damping_command(xdot: np.ndarray, kvf: np.ndarray) -> np.ndarray:
    """F*_s: end-effector velocity damping acting in the force-control direction."""
    return -kvf @ xdot
