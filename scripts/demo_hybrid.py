"""Demo: hybrid motion/force operational-space control (Sec. II, V; eq. 45-48).

The end-effector starts just above the demo table with its wiping-tool tip
(see build_model.py) and is held at that pose by the motion channel while
the force channel regulates the normal (table-perpendicular) contact force
via a proportional force-error law (eq. 47) plus damping. Sigma_f/Sigma_bar_f
(task_spec.surface_wiping_spec) select x,y + all rotations as motion and z
(the surface normal) as force -- eq. (1)-(4).

See tests/test_hybrid_force.py's module docstring for why the regulated
force settles near ~40 N rather than exactly at Fd=6 N: eq. (46a)'s
Fm = Lambda(q) Omega F*_m is not block-diagonal, so the motion channel
(here: holding position/orientation against gravity) still injects a real
feedforward force into the force-controlled z direction through Lambda's
coupling -- precisely the "forces of coupling created by the end-effector
motion... in the subspace orthogonal to [the force] direction" the paper's
own introduction identifies as a core motivation for hybrid control. With
purely proportional force feedback (no integral term, matching eq. 47 as
given), the achieved force is a fixed blend of that feedforward baseline and
Fd, not equal to Fd.

Table contact uses a softened solref/solimp (see build_model.py) -- a
simulation-side choice, not a change to the control law -- because eq. (48)'s
impact-transition control (the paper's own answer to a stiff first-contact
transient) is explicitly out of scope here.

Produces:
    results/hybrid_force_tracking.png
    results/hybrid_demo.mp4
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from khatib_osc import controller, dynamics, task_spec
from khatib_osc.orientation import pose_error
from khatib_osc.robot import load_robot

RESULTS = Path(__file__).resolve().parents[1] / "results"

Q_START = np.array([0.0, -0.45, 0.0, -0.5, np.pi / 2, np.pi / 4])
KP_POS, KP_ROT = 20.0, 10.0
KF_Z, KVF_Z = 0.5, 25.0
FD_Z = 6.0
T = 5.0
VIDEO_FPS = 30


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    robot = load_robot()
    robot.data.qpos[:] = Q_START
    mujoco.mj_forward(robot.model, robot.data)
    p_start, R_start = robot.ee_pos().copy(), robot.ee_rot().copy()

    R_ee0 = robot.ee_rot()
    F_baseline = np.concatenate(
        [R_ee0 @ robot.data.sensordata[0:3], R_ee0 @ robot.data.sensordata[3:6]]
    )  # tare the wrist F/T sensor: it reads link7's own weight even with no contact

    Omega, Omega_tilde = task_spec.surface_wiping_spec(np.array([0.0, 0.0, 1.0]))
    Fd = np.array([0, 0, FD_Z, 0, 0, 0])
    kp = np.diag([KP_POS] * 3 + [KP_ROT] * 3)
    kv = np.diag([2 * np.sqrt(KP_POS)] * 3 + [2 * np.sqrt(KP_ROT)] * 3)
    kf = np.diag([0, 0, KF_Z, 0, 0, 0])
    kvf = np.diag([0, 0, KVF_Z, 0, 0, 0])

    dt = robot.model.opt.timestep
    n_steps = int(T / dt)
    xdot_d, xddot_d = np.zeros(6), np.zeros(6)

    ts, fz_hist, ee_z_hist = [], [], []

    renderer = mujoco.Renderer(robot.model, height=480, width=640)
    frame_every = max(1, int(1.0 / VIDEO_FPS / dt))
    frames = []

    for i in range(n_steps):
        t = i * dt
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

        ts.append(t)
        fz_hist.append(F_meas[2])
        ee_z_hist.append(robot.ee_pos()[2])

        if i % frame_every == 0:
            renderer.update_scene(robot.data)
            frames.append(renderer.render().copy())

    renderer.close()

    ts, fz_hist, ee_z_hist = np.array(ts), np.array(fz_hist), np.array(ee_z_hist)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ts, fz_hist)
    axes[0].axhline(FD_Z, color="gray", linestyle="--", linewidth=1, label=f"F_d = {FD_Z} N")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("F_z [N]")
    axes[0].set_title("Normal contact force (eq. 47 force-error feedback)")
    axes[0].legend()
    axes[1].plot(ts, ee_z_hist)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("z [m]")
    axes[1].set_title("End-effector height")
    fig.tight_layout()
    fig.savefig(RESULTS / "hybrid_force_tracking.png", dpi=150)
    plt.close(fig)

    imageio.mimsave(RESULTS / "hybrid_demo.mp4", frames, fps=VIDEO_FPS)

    print(f"final F_z = {fz_hist[-500:].mean():.2f} N (target {FD_Z} N), std={fz_hist[-500:].std():.3f}")
    print(f"wrote {RESULTS}/hybrid_force_tracking.png")
    print(f"wrote {RESULTS}/hybrid_demo.mp4")


if __name__ == "__main__":
    main()
