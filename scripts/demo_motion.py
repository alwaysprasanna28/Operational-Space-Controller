"""Demo: unconstrained operational-space motion control (Sec. III-IV, eq. 28-31, 34).

Tracks a minimum-jerk Cartesian pose trajectory from the FR3's ready pose.
Gains/amplitude/duration are the ones validated in
tests/test_controller_tracking.py; see that file's module docstring for why
the orientation channel settles to a small (~0.1-0.2 rad) residual rather
than exactly zero -- the vendored model's realistic joint dry friction
(dof_frictionloss up to 1.137 N*m), which eq. (17)'s idealized frictionless
rigid-body model does not include.

Produces:
    results/motion_tracking_error.png
    results/motion_joint_torques.png
    results/motion_demo.mp4
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from khatib_osc import controller, dynamics
from khatib_osc.orientation import pose_error
from khatib_osc.robot import load_robot
from khatib_osc.trajectory import PoseTrajectory

RESULTS = Path(__file__).resolve().parents[1] / "results"

KP_POS, KP_ROT = 80.0, 40.0
T, SETTLE = 3.0, 1.5
DISP = np.array([0.04, -0.02, -0.05])
ROTVEC = np.array([0.05, 0.0, 0.1])
VIDEO_FPS = 30


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
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

    ts, pos_err, rot_err, taus = [], [], [], []

    renderer = mujoco.Renderer(robot.model, height=480, width=640)
    frame_every = max(1, int(1.0 / VIDEO_FPS / dt))
    frames = []

    for i in range(n_steps):
        t = i * dt
        q, qd = robot.data.qpos.copy(), robot.data.qvel.copy()
        J, Lambda, mu, p, xdot = dynamics.operational_space_dynamics(robot, q, qd)

        x_d, R_d, xdot_d, xddot_d = traj.evaluate_stacked(min(t, T))
        e = pose_error(robot.ee_pos(), robot.ee_rot(), x_d, R_d)
        edot = xdot - xdot_d

        F_m_star = controller.motion_command(e, edot, xddot_d, kp, kv)
        tau = controller.unconstrained_torque(J, Lambda, mu, p, F_m_star)

        robot.data.ctrl[:] = tau
        mujoco.mj_step(robot.model, robot.data)

        ts.append(t)
        pos_err.append(np.linalg.norm(e[:3]))
        rot_err.append(np.linalg.norm(e[3:]))
        taus.append(tau.copy())

        if i % frame_every == 0:
            renderer.update_scene(robot.data)
            frames.append(renderer.render().copy())

    renderer.close()

    ts = np.array(ts)
    taus = np.array(taus)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ts, pos_err)
    axes[0].axvline(T, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("|e_pos| [m]")
    axes[0].set_title("Position tracking error")
    axes[1].plot(ts, rot_err)
    axes[1].axvline(T, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("|e_rot| [rad]")
    axes[1].set_title("Orientation tracking error")
    fig.tight_layout()
    fig.savefig(RESULTS / "motion_tracking_error.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for j, name in enumerate(["j1", "j2", "j3", "j5", "j6", "j7"]):
        ax.plot(ts, taus[:, j], label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("torque [N*m]")
    ax.set_title("Commanded joint torques, Gamma = J^T F (eq. 28)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "motion_joint_torques.png", dpi=150)
    plt.close(fig)

    imageio.mimsave(RESULTS / "motion_demo.mp4", frames, fps=VIDEO_FPS)

    print(f"final |e_pos| = {pos_err[-1]:.4f} m, |e_rot| = {rot_err[-1]:.4f} rad")
    print(f"wrote {RESULTS}/motion_tracking_error.png")
    print(f"wrote {RESULTS}/motion_joint_torques.png")
    print(f"wrote {RESULTS}/motion_demo.mp4")


if __name__ == "__main__":
    main()
