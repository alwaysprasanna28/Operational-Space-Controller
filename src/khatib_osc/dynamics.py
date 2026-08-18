"""End-effector (operational-space) dynamics, Khatib (1987) Sec. III.

Joint-space quantities A(q), b(q,qdot)+g(q) come straight from MuJoCo's own
rigid-body dynamics engine (mj_fullM, qfrc_bias) -- see README "Design notes"
for why that is the intended reading of "from scratch" here. Everything from
the Jacobian onward (eq. 18, 20-25, 42) is derived and implemented directly
from the paper, not borrowed from an existing OSC library.

Notation mirrors the paper:
    A(q), b(q,qdot), g(q)   joint-space mass, Coriolis/centrifugal, gravity  (eq. 17)
    J(q)                    basic Jacobian, [v;w] = J(q) qdot               (eq. 42)
    Lambda(x)                operational-space (Cartesian) mass matrix       (eq. 18)
    h(q,qdot) = Jdot(q) qdot                                                (eq. 21)
    mu(x,xdot)              operational centrifugal/Coriolis forces          (eq. 24)
    p(x)                    operational gravity forces                      (eq. 25)
"""

from __future__ import annotations

import mujoco
import numpy as np

from khatib_osc.robot import Robot


def mass_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """A(q), the n x n joint-space mass matrix (eq. 17)."""
    A = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, A)
    return A


def site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    """Basic Jacobian J(q), stacked [Jv; Jw], 6 x n, s.t. [v;w] = J(q) qdot (eq. 42)."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack([jacp, jacr])


def site_jacobian_dot(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    """Time derivative Jdot(q,qdot) of the basic Jacobian, 6 x n.

    Requires data to already hold valid velocities/accelerations (i.e. call
    after mj_forward or mj_step at the current qpos, qvel).
    """
    jacp_dot = np.zeros((3, model.nv))
    jacr_dot = np.zeros((3, model.nv))
    body_id = model.site_bodyid[site_id]
    point = data.site_xpos[site_id].copy()
    mujoco.mj_jacDot(model, data, jacp_dot, jacr_dot, point, body_id)
    return np.vstack([jacp_dot, jacr_dot])


def gravity(robot: Robot, qpos: np.ndarray) -> np.ndarray:
    """g(q): evaluate qfrc_bias at qvel = 0, where Coriolis/centrifugal vanish.

    Uses robot.scratch so the live simulation state (robot.data) is untouched.
    """
    d = robot.scratch
    d.qpos[:] = qpos
    d.qvel[:] = 0.0
    mujoco.mj_forward(robot.model, d)
    return d.qfrc_bias.copy()


def coriolis_centrifugal(robot: Robot, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """b(q,qdot): joint-space centrifugal/Coriolis forces (eq. 17), via qfrc_bias - g(q)."""
    d = robot.scratch
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    mujoco.mj_forward(robot.model, d)
    bias_total = d.qfrc_bias.copy()
    return bias_total - gravity(robot, qpos)


def lambda_matrix(A: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Lambda(x), the operational-space mass matrix, via (J A^-1 J^T)^-1.

    Algebraically identical to eq. (18)'s J^-T A J^-1 when J is square and
    nonsingular (our case: 6-DOF nonredundant arm, 6x6 J); this form is the
    numerically standard one since it avoids inverting J directly. Equation
    (18) is not "reimplemented" separately -- test_dynamics_identities.py
    checks the two forms agree, which is the point of that identity.
    """
    A_inv = np.linalg.inv(A)
    return np.linalg.inv(J @ A_inv @ J.T)


def lambda_matrix_eq18(A: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Lambda(x) = J^-T(q) A(q) J^-1(q), the literal form of eq. (18)."""
    J_inv = np.linalg.inv(J)
    return J_inv.T @ A @ J_inv


def h_vector(J_dot: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """h(q,qdot) = Jdot(q) qdot (eq. 21)."""
    return J_dot @ qvel


def mu_operational(J: np.ndarray, b: np.ndarray, Lambda: np.ndarray, h: np.ndarray) -> np.ndarray:
    """mu(x,xdot) = J^-T(q) b(q,qdot) - Lambda(x) h(q,qdot) (eq. 24)."""
    return np.linalg.solve(J.T, b) - Lambda @ h


def p_operational(J: np.ndarray, g: np.ndarray) -> np.ndarray:
    """p(x) = J^-T(q) g(q) (eq. 25)."""
    return np.linalg.solve(J.T, g)


def operational_space_dynamics(
    robot: Robot, qpos: np.ndarray, qvel: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience bundle: (J, Lambda, mu, p, xdot) at the given state.

    xdot = J(q) qdot is the operational (Cartesian) velocity, [v; w] (eq. 42).
    """
    model, data = robot.model, robot.data
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)

    A = mass_matrix(model, data)
    J = site_jacobian(model, data, robot.ee_site_id)
    J_dot = site_jacobian_dot(model, data, robot.ee_site_id)

    b = coriolis_centrifugal(robot, qpos, qvel)
    g = gravity(robot, qpos)

    Lambda = lambda_matrix(A, J)
    h = h_vector(J_dot, qvel)
    mu = mu_operational(J, b, Lambda, h)
    p = p_operational(J, g)

    xdot = J @ qvel
    return J, Lambda, mu, p, xdot
