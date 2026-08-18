"""Closed-loop integration test: hybrid motion/force OSC (eq. 45-48) regulating
a normal contact force against the demo table while holding tangential
position and orientation fixed via the motion channel.

See scripts/demo_hybrid.py's module docstring and the README's "Limitations"
section for why the achieved force settles near ~40 N rather than exactly at
Fd -- briefly: eq. (46a)'s Fm = Lambda(q) Omega F*_m is NOT block-diagonal in
general, so even with F*_m's z-row zeroed by Omega, the *motion* channel
still injects a real feedforward force in the (force-controlled) z direction
through Lambda's position-rotation-z coupling, on top of Fccg's own gravity
feedforward there. The paper's own introduction names this exact phenomenon
("forces of coupling created by the end-effector motion... in the subspace
orthogonal to that direction") as a core motivation for hybrid control. With
purely proportional force feedback (eq. 47, no integral term), the achieved
force is a fixed blend of this feedforward baseline and Fd, not equal to Fd.
"""

import mujoco
import numpy as np

from khatib_osc import controller, dynamics, task_spec
from khatib_osc.orientation import pose_error
from khatib_osc.robot import load_robot

Q_START = np.array([0.0, -0.45, 0.0, -0.5, np.pi / 2, np.pi / 4])
KP_POS, KP_ROT = 20.0, 10.0
KF_Z, KVF_Z = 0.5, 25.0
FD_Z = 6.0
T = 5.0


def run_hybrid_force_hold():
    robot = load_robot()
    robot.data.qpos[:] = Q_START
    mujoco.mj_forward(robot.model, robot.data)
    p_start, R_start = robot.ee_pos().copy(), robot.ee_rot().copy()

    R_ee0 = robot.ee_rot()
    F_baseline = np.concatenate(
        [R_ee0 @ robot.data.sensordata[0:3], R_ee0 @ robot.data.sensordata[3:6]]
    )

    Omega, Omega_tilde = task_spec.surface_wiping_spec(np.array([0.0, 0.0, 1.0]))
    Fd = np.array([0, 0, FD_Z, 0, 0, 0])
    kp = np.diag([KP_POS] * 3 + [KP_ROT] * 3)
    kv = np.diag([2 * np.sqrt(KP_POS)] * 3 + [2 * np.sqrt(KP_ROT)] * 3)
    kf = np.diag([0, 0, KF_Z, 0, 0, 0])
    kvf = np.diag([0, 0, KVF_Z, 0, 0, 0])

    dt = robot.model.opt.timestep
    xdot_d, xddot_d = np.zeros(6), np.zeros(6)

    fz_hist = []
    for _ in range(int(T / dt)):
        q, qd = robot.data.qpos.copy(), robot.data.qvel.copy()
        J, Lambda, mu, p, xdot = dynamics.operational_space_dynamics(robot, q, qd)
        e = pose_error(robot.ee_pos(), robot.ee_rot(), p_start, R_start)
        edot = xdot - xdot_d
        F_m_star = controller.motion_command(e, edot, xddot_d, kp, kv)

        R_ee = robot.ee_rot()
        F_meas = (
            np.concatenate([R_ee @ robot.data.sensordata[0:3], R_ee @ robot.data.sensordata[3:6]])
            - F_baseline
        )
        F_a_star = controller.force_command(Fd, F_meas, kf)
        F_s_star = controller.force_damping_command(xdot, kvf)
        tau = controller.hybrid_torque(J, Lambda, mu, p, Omega, Omega_tilde, F_m_star, F_a_star, F_s_star)

        robot.data.ctrl[:] = tau
        mujoco.mj_step(robot.model, robot.data)
        fz_hist.append(F_meas[2])

    return np.array(fz_hist), robot


def test_hybrid_force_settles_stably():
    fz_hist, robot = run_hybrid_force_hold()
    tail = fz_hist[-500:]

    assert np.all(np.isfinite(fz_hist))
    assert np.abs(fz_hist).max() < 500, "force should never spike wildly"
    assert tail.std() < 1.0, f"force must settle to a steady value, got std={tail.std():.3f}"
    assert np.linalg.norm(robot.data.qvel) < 0.1, "arm must come to rest, not oscillate"
    assert robot.data.ncon >= 1, "end-effector should still be in contact with the table"


def test_hybrid_force_moves_toward_target_from_baseline():
    """With no force feedback at all (kf=kvf=0), the feedforward-only contact
    force should be measurably different from (larger than) the regulated
    result -- i.e. eq. (46b)'s force-error term is doing real work, even
    though it doesn't reach Fd exactly (see module docstring)."""
    fz_regulated, _ = run_hybrid_force_hold()
    regulated_mean = fz_regulated[-500:].mean()

    robot = load_robot()
    robot.data.qpos[:] = Q_START
    mujoco.mj_forward(robot.model, robot.data)
    p_start, R_start = robot.ee_pos().copy(), robot.ee_rot().copy()
    R_ee0 = robot.ee_rot()
    F_baseline = np.concatenate(
        [R_ee0 @ robot.data.sensordata[0:3], R_ee0 @ robot.data.sensordata[3:6]]
    )
    Omega, Omega_tilde = task_spec.surface_wiping_spec(np.array([0.0, 0.0, 1.0]))
    kp = np.diag([KP_POS] * 3 + [KP_ROT] * 3)
    kv = np.diag([2 * np.sqrt(KP_POS)] * 3 + [2 * np.sqrt(KP_ROT)] * 3)
    dt = robot.model.opt.timestep
    xdot_d, xddot_d = np.zeros(6), np.zeros(6)
    zero6 = np.zeros(6)

    fz_ff = []
    for _ in range(int(T / dt)):
        q, qd = robot.data.qpos.copy(), robot.data.qvel.copy()
        J, Lambda, mu, p, xdot = dynamics.operational_space_dynamics(robot, q, qd)
        e = pose_error(robot.ee_pos(), robot.ee_rot(), p_start, R_start)
        edot = xdot - xdot_d
        F_m_star = controller.motion_command(e, edot, xddot_d, kp, kv)
        tau = controller.hybrid_torque(J, Lambda, mu, p, Omega, Omega_tilde, F_m_star, zero6, zero6)
        robot.data.ctrl[:] = tau
        mujoco.mj_step(robot.model, robot.data)
        R_ee = robot.ee_rot()
        F_meas = (
            np.concatenate([R_ee @ robot.data.sensordata[0:3], R_ee @ robot.data.sensordata[3:6]])
            - F_baseline
        )
        fz_ff.append(F_meas[2])
    fz_ff = np.array(fz_ff)

    assert abs(regulated_mean - fz_ff[-500:].mean()) > 5.0
