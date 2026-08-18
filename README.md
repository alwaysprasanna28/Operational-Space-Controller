# Operational-Space Control on a MuJoCo Franka FR3

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![MuJoCo](https://img.shields.io/badge/simulator-MuJoCo-orange)
![Tests](https://img.shields.io/badge/tests-pytest-green)

<p align="center">
  <img src="assets/franka_fr3/fr3.png" alt="Franka FR3 in MuJoCo" width="600">
</p>

A from-scratch implementation of

> O. Khatib, *A Unified Approach for Motion and Force Control of Robot
> Manipulators: The Operational Space Formulation*, IEEE Journal of Robotics
> and Automation, RA-3(1), Feb. 1987.

for a 6-DOF, torque-actuated Franka FR3 in MuJoCo.

The end-effector dynamics (Sec. III):

```
Lambda(x) xddot + mu(x,xdot) + p(x) = F        Lambda(x) = J^-T(q) A(q) J^-1(q)
Gamma = J^T(q) F
```

The unconstrained-motion control law (Sec. IV):

```
F*_m  = xddot_d - K_p (x - x_d) - K_v (xdot - xdot_d)
F     = Lambda(x) F*_m + mu(x,xdot) + p(x)
Gamma = J^T(q) F
```

The hybrid motion/force control law (Sec. II, V), with the generalized
task-specification matrices Omega (motion directions) and Omega_tilde (force
directions) built from Sigma_f (free-motion mask) and its complement
Sigma_bar_f = I - Sigma_f:

```
Omega       = blockdiag(S_f^T Sigma_f S_f,       S_r^T Sigma_r S_r)
Omega_tilde = blockdiag(S_f^T Sigma_bar_f S_f,   S_r^T Sigma_bar_r S_r)

F*_a  = K_f (F_d - F_meas)              # force error feedback
F*_s  = -K_vf xdot                      # velocity damping in the force direction
F     = Lambda(x) Omega F*_m + Omega_tilde F*_a + Lambda(x) Omega_tilde F*_s
        + mu(x,xdot) + p(x)
Gamma = J^T(q) F
```

Joint-space dynamics — the mass matrix A(q), bias forces b(q,xdot)+g(q), and
Jacobian J(q) — come from MuJoCo's own rigid-body engine; the operational-space
quantities above (Lambda, mu, p, Omega, Omega_tilde, and both torque laws) are
implemented directly from the paper's derivations, not from an existing OSC
library.

## Contents

- [Scope](#scope)
- [Quick start](#quick-start)
- [Results](#results)
- [Why the FR3 is reduced to 6 DOF](#why-the-fr3-is-reduced-to-6-dof)
- [Design choices made explicitly](#design-choices-made-explicitly-per-user-direction)
- [Layout](#layout)
- [License](#license)

## Scope

In scope: unconstrained operational-space motion control (Sec. III-IV) and
hybrid motion/force control (Sec. II, V). Out of scope: redundant-manipulator
null-space control (Sec. VI-VII) and singularity-robust control (Sec. VIII)
— see [Why the FR3 is reduced to 6 DOF](#why-the-fr3-is-reduced-to-6-dof) for
how that shaped the model.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
cd "paper implementation/khatib_osc_fr3"
uv sync --extra dev

# (re)generate the derived MJCF from the vendored Menagerie franka_fr3 model
uv run python scripts/build_model.py

uv run pytest tests/ -v

uv run python scripts/demo_motion.py    # -> results/motion_*.png, motion_demo.mp4
uv run python scripts/demo_hybrid.py    # -> results/hybrid_*.png, hybrid_demo.mp4

# interactive viewer (needs a display)
uv run python -c "
import mujoco, mujoco.viewer
from khatib_osc.robot import load_robot
r = load_robot()
mujoco.viewer.launch(r.model, r.data)
"
```

## Results

**Unconstrained motion control** (`scripts/demo_motion.py`) — a minimum-jerk
Cartesian pose trajectory x_d(t) tracked by
`F*_m = xddot_d - K_p(x-x_d) - K_v(xdot-xdot_d)`,
`F = Lambda(x) F*_m + mu(x,xdot) + p(x)`, `Gamma = J^T F`:

<p align="center">
  <img src="results/motion_tracking_error.png" alt="Motion tracking error" width="700">
</p>

**Hybrid motion/force control** (`scripts/demo_hybrid.py`) — position/orientation
held by the motion law above while a normal contact force against a table is
regulated by `F*_a = K_f(F_d - F_meas)` through the `Omega`/`Omega_tilde` split:

<p align="center">
  <img src="results/hybrid_force_tracking.png" alt="Hybrid force tracking" width="700">
</p>

Rendered MP4s of both demos are in [`results/`](results/)
(`motion_demo.mp4`, `hybrid_demo.mp4`).

## Why the FR3 is reduced to 6 DOF

Sections III-V derive the end-effector equations of motion assuming the
manipulator is *not* redundant: the number of task-space DOF m0 equals the
number of joints n. FR3 has 7 joints. Rather than leave the 7th joint's
null-space motion uncontrolled (which Sec. VI-VII's dynamically-consistent
null-space treatment would fix, but is explicitly out of scope),
`scripts/build_model.py` welds joint 4 (the elbow) at a fixed angle by
folding its rotation into the static body transform of `fr3_link4` and
deleting the joint — a kinematically *exact* reduction (verified to agree
with the original 7-DOF model to 1e-12; see `scripts/build_model.py`'s own
numerical check in its git history / the comments there). The remaining 6
joints become torque `motor` actuators (`ctrlrange` = each joint's real
`actuatorfrcrange`), so `Gamma = J^T F` is the complete, well-posed torque
solution exactly as the paper derives it for a nonredundant arm — no
null-space term needed or added.

The specific joint-4 angle and the "ready pose" used as the working
configuration were chosen empirically to avoid two *distinct* kinematic
singularities discovered while validating this reduction (both confirmed via
`numpy.linalg.svd` on the site Jacobian, not guessed):

- The Menagerie's own `home` keyframe (`q2 = 0`) leaves joint 3's axis
  exactly coincident with joint 1's, independent of joint 4 — `rank(J) = 5`.
- The standard Franka "ready" pose (`q5 = 0`) is *also* singular once joint 4
  is welded — confirmed by sweeping every other joint independently and
  finding `q5 = 0` alone reproduces `rank(J) = 5` regardless of the rest.

`build_model.py`'s keyframe therefore uses
`[0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]` with `q5 = -0.5`, giving a minimum
Jacobian singular value of ~0.0125 at rest and >0.0028 under ±0.35 rad
joint perturbations in every direction.

`trajectory.py`'s `PoseTrajectory` supplies x_d, xdot_d, xddot_d for the
motion command above: minimum-jerk position interpolation, and closed-form
slerp angular velocity/acceleration (derived in the module docstring,
cross-checked against finite differences in `tests/test_orientation.py`).
Orientation error is the SO(3) log map `e_rot = log(R R_d^T)` — an
instantaneous angular rotation vector, not Euler angles — see
`orientation.pose_error`, `orientation.log_map`.

## Design choices made explicitly, per user direction

- **Single-rate control loop**: every step recomputes A, b, g, J, Jdot,
  Lambda, mu, p fresh (no Fig. 3 two-rate dynamic-parameter/servo split —
  that was the paper's answer to 1980s real-time compute limits, not needed
  here).
- **Trajectory tracking**, not straight-line goal regulation: the motion
  demo drives `F*_m = xddot_d - K_p(x-x_d) - K_v(xdot-xdot_d)` with a
  time-varying x_d(t), xdot_d(t), xddot_d(t) (min-jerk), rather than
  regulating toward a fixed goal with a velocity-capped straight-line law.
- **6-DOF reduction over null-space damping**: see
  [above](#why-the-fr3-is-reduced-to-6-dof).

## Layout

```
assets/franka_fr3/        vendored MuJoCo Menagerie franka_fr3 (Apache-2.0, see LICENSE)
                           + fr3_osc.xml, fr3_osc_scene.xml (generated, see build_model.py)
src/khatib_osc/           dynamics.py, orientation.py, trajectory.py,
                           task_spec.py, controller.py, robot.py
scripts/                  build_model.py, demo_motion.py, demo_hybrid.py
tests/                    dynamic identities, orientation/trajectory, task spec,
                           closed-loop motion tracking, closed-loop hybrid force
results/                  demo plots + MP4s (generated)
```

## License

The `khatib_osc` source in this repository has no license file yet — treat
it as all-rights-reserved unless the author states otherwise. The vendored
robot model under [`assets/franka_fr3/`](assets/franka_fr3/) is MuJoCo
Menagerie's `franka_fr3`, licensed Apache-2.0 (see
[`assets/franka_fr3/LICENSE`](assets/franka_fr3/LICENSE)).
