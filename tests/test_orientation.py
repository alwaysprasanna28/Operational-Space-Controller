"""Correctness of the SO(3) error/log/exp machinery (orientation.py) and the
closed-form angular velocity/acceleration used by trajectory.py."""

import numpy as np
from scipy.spatial.transform import Rotation

from khatib_osc.orientation import exp_map, log_map, orientation_error, pose_error
from khatib_osc.trajectory import PoseTrajectory


def random_rotation(rng):
    return Rotation.from_rotvec(rng.uniform(-np.pi, np.pi, size=3)).as_matrix()


def test_log_exp_round_trip(rng):
    for _ in range(20):
        R = random_rotation(rng)
        rv = log_map(R)
        R2 = exp_map(rv)
        assert np.allclose(R, R2, atol=1e-8)


def test_orientation_error_zero_at_target(rng):
    for _ in range(10):
        R = random_rotation(rng)
        e = orientation_error(R, R)
        assert np.allclose(e, 0, atol=1e-10)


def test_orientation_error_small_angle_sign():
    """R_d is a small +theta rotation about z from R=I; error should point +z,
    matching the derivation in orientation.pose_error's docstring."""
    theta = 0.05
    R = np.eye(3)
    R_d = Rotation.from_rotvec([0, 0, theta]).as_matrix()
    e = orientation_error(R, R_d)  # log(R_d R^T)
    assert np.allclose(e, [0, 0, theta], atol=1e-8)

    e_pose = pose_error(np.zeros(3), R, np.zeros(3), R_d)[3:]
    assert np.allclose(e_pose, [0, 0, -theta], atol=1e-8)


def test_pose_trajectory_angular_velocity_matches_finite_difference(rng):
    """omega_d(t) from the closed form must match dR/dt via [omega]_x R,
    checked by finite-differencing R(t) itself."""
    p0, p1 = np.zeros(3), rng.uniform(-1, 1, size=3)
    R0 = random_rotation(rng)
    R1 = random_rotation(rng)
    T = 2.0
    traj = PoseTrajectory(p0, R0, p1, R1, T)

    dt = 1e-6
    for t in [0.3, 0.9, 1.5, 1.9]:
        _, R, _, omega, _, _ = traj.evaluate(t)
        _, R_plus, *_ = traj.evaluate(t + dt)
        R_dot_numeric = (R_plus - R) / dt
        omega_hat = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0],
        ])
        R_dot_analytic = omega_hat @ R
        assert np.allclose(R_dot_numeric, R_dot_analytic, atol=1e-4)


def test_pose_trajectory_boundary_conditions(rng):
    p0, p1 = np.zeros(3), rng.uniform(-1, 1, size=3)
    R0 = random_rotation(rng)
    R1 = random_rotation(rng)
    T = 1.3
    traj = PoseTrajectory(p0, R0, p1, R1, T)

    p, R, v, omega, a, alpha = traj.evaluate(0.0)
    assert np.allclose(p, p0, atol=1e-10)
    assert np.allclose(R, R0, atol=1e-10)
    assert np.allclose(v, 0, atol=1e-10)
    assert np.allclose(omega, 0, atol=1e-10)

    p, R, v, omega, a, alpha = traj.evaluate(T)
    assert np.allclose(p, p1, atol=1e-8)
    assert np.allclose(R, R1, atol=1e-8)
    assert np.allclose(v, 0, atol=1e-8)
    assert np.allclose(omega, 0, atol=1e-8)
