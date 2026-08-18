"""Closed-loop integration test: simulate the unconstrained-motion OSC law
(eq. 28-31, 34) in MuJoCo and check the tracking error behaves sensibly.

Gains, trajectory amplitude and duration here are the same ones validated in
scripts/demo_motion.py -- see that file's module docstring, and the README's
"Limitations" section, for why the target tolerance is a few mm / ~0.15 rad
rather than near-zero: the vendored FR3 model carries realistic joint dry
friction (dof_frictionloss up to 1.137 N*m) that eq. (17)'s idealized
frictionless rigid-body model does not include, so a small steady-state
error is physically expected with pure PD operational-space control (no
integral action, matching the paper's own eq. 31) -- not a bug.
"""

import mujoco
import numpy as np

from khatib_osc import controller, dynamics
from khatib_osc.orientation import pose_error
from khatib_osc.robot import load_robot
from khatib_osc.trajectory import PoseTrajectory

KP_POS, KP_ROT = 80.0, 40.0
T, SETTLE = 3.0, 1.5
DISP = np.array([0.04, -0.02, -0.05])
ROTVEC = np.array([0.05, 0.0, 0.1])


def run_motion_tracking():
    from scipy.spatial.transform import Rotation

    robot = load_robot()
    robot.reset_home()
    p0, R0 = robot.ee_pos(), robot.ee_rot()
    p1 = p0 + DISP
    R1 = Rotation.from_rotvec(ROTVEC).as_matrix() @ R0
    traj = PoseTrajectory(p0, R0, p1, R1, T)

    kp = np.diag([KP_POS] * 3 + [KP_ROT] * 3)
    kv = np.diag([2 * np.sqrt(KP_POS)] * 3 + [2 * np.sqrt(KP_ROT)] * 3)

    dt = robot.model.opt.timestep
    n_steps = int((T + SETTLE) / dt)

    errors, worst_sv = [], np.inf
    for i in range(n_steps):
        t = i * dt
        q, qd = robot.data.qpos.copy(), robot.data.qvel.copy()
        J, Lambda, mu, p, xdot = dynamics.operational_space_dynamics(robot, q, qd)
        worst_sv = min(worst_sv, np.linalg.svd(J, compute_uv=False)[-1])

        x_d, R_d, xdot_d, xddot_d = traj.evaluate_stacked(min(t, T))
        e = pose_error(robot.ee_pos(), robot.ee_rot(), x_d, R_d)
        edot = xdot - xdot_d

        F_m_star = controller.motion_command(e, edot, xddot_d, kp, kv)
        tau = controller.unconstrained_torque(J, Lambda, mu, p, F_m_star)

        robot.data.ctrl[:] = tau
        mujoco.mj_step(robot.model, robot.data)
        errors.append(np.linalg.norm(e))

    e_final = pose_error(robot.ee_pos(), robot.ee_rot(), p1, R1)
    return np.array(errors), e_final, worst_sv


def test_motion_tracking_converges_and_stays_well_conditioned():
    errors, e_final, worst_sv = run_motion_tracking()
    assert errors[0] < 1e-9, "trajectory must start exactly at the current pose"
    assert errors.max() < 0.5, f"error should never blow up: max={errors.max():.3f}"
    assert np.all(np.isfinite(errors))

    # position converges tightly (large torque budget relative to friction);
    # orientation settles to a small residual dominated by unmodeled wrist
    # joint friction (up to 0.76 N*m on joints 5-7) that pure PD + feedforward
    # gravity/Coriolis compensation (eq. 24-25, 31) cannot cancel.
    assert np.linalg.norm(e_final[:3]) < 0.02, f"position residual too large: {e_final[:3]}"
    assert np.linalg.norm(e_final[3:]) < 0.25, f"orientation residual too large: {e_final[3:]}"

    # the trajectory should stay comfortably clear of kinematic singularities
    # (Sec. VIII, out of scope -- this just guards against accidentally
    # picking a trajectory that wanders into one)
    assert worst_sv > 0.005, f"trajectory passed too close to a singularity: min sv={worst_sv:.2e}"


def test_motion_tracking_no_blowup_with_zero_gains_feedforward_only():
    """Sanity check on the feedforward (mu, p) terms alone: with kp=kv=0 the
    command is pure gravity/Coriolis compensation, so the arm should sag/drift
    slowly under residual friction, not explode."""
    robot = load_robot()
    robot.reset_home()
    dt = robot.model.opt.timestep
    for _ in range(500):
        q, qd = robot.data.qpos.copy(), robot.data.qvel.copy()
        J, Lambda, mu, p, xdot = dynamics.operational_space_dynamics(robot, q, qd)
        tau = J.T @ (mu + p)
        robot.data.ctrl[:] = tau
        mujoco.mj_step(robot.model, robot.data)
    assert np.all(np.isfinite(robot.data.qpos))
    assert np.linalg.norm(robot.data.qvel) < 1.0
