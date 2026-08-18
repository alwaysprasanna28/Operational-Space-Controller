"""Numerically verify the dynamic identities of Sec. III against MuJoCo's own A(q), J(q)."""

import mujoco
import numpy as np

from khatib_osc import dynamics


def random_qpos(robot, rng, spread=0.6):
    home = robot.home_qpos()
    lo = robot.model.jnt_range[:, 0]
    hi = robot.model.jnt_range[:, 1]
    q = home + spread * rng.uniform(-1, 1, size=home.shape)
    return np.clip(q, lo + 1e-3, hi - 1e-3)


def test_mass_matrix_spd(robot, rng):
    for _ in range(5):
        q = random_qpos(robot, rng)
        robot.data.qpos[:] = q
        mujoco.mj_forward(robot.model, robot.data)
        A = dynamics.mass_matrix(robot.model, robot.data)
        assert np.allclose(A, A.T, atol=1e-8)
        eigvals = np.linalg.eigvalsh(A)
        assert np.all(eigvals > 0)


def test_lambda_matches_eq18_literal_form(robot, rng):
    """Lambda(x) = J^-T(q) A(q) J^-1(q) (eq. 18) must agree with the numerically
    standard (J A^-1 J^T)^-1 form used in dynamics.lambda_matrix."""
    for _ in range(5):
        q = random_qpos(robot, rng)
        robot.data.qpos[:] = q
        mujoco.mj_forward(robot.model, robot.data)
        A = dynamics.mass_matrix(robot.model, robot.data)
        J = dynamics.site_jacobian(robot.model, robot.data, robot.ee_site_id)
        assert np.abs(np.linalg.det(J)) > 1e-4, "sampled a near-singular config, retry"

        Lambda_standard = dynamics.lambda_matrix(A, J)
        Lambda_eq18 = dynamics.lambda_matrix_eq18(A, J)
        assert np.allclose(Lambda_standard, Lambda_eq18, atol=1e-6)
        assert np.allclose(Lambda_standard, Lambda_standard.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(Lambda_standard) > 0)


def test_gravity_matches_zero_velocity_bias(robot, rng):
    """dynamics.gravity(q) must equal qfrc_bias evaluated in-place at qvel=0."""
    for _ in range(5):
        q = random_qpos(robot, rng)
        robot.data.qpos[:] = q
        robot.data.qvel[:] = 0.0
        mujoco.mj_forward(robot.model, robot.data)
        expected = robot.data.qfrc_bias.copy()
        g = dynamics.gravity(robot, q)
        assert np.allclose(g, expected, atol=1e-10)


def test_mu_p_decomposition_matches_combined_bias(robot, rng):
    """eq. (24)+(25): mu(x,xdot) + p(x) must equal J^-T qfrc_bias(q,qdot) - Lambda h,
    i.e. the mu/p split is a linear decomposition of the same total bias mapping."""
    for _ in range(5):
        q = random_qpos(robot, rng)
        qdot = rng.uniform(-0.5, 0.5, size=q.shape)

        robot.data.qpos[:] = q
        robot.data.qvel[:] = qdot
        mujoco.mj_forward(robot.model, robot.data)
        bias_total = robot.data.qfrc_bias.copy()

        A = dynamics.mass_matrix(robot.model, robot.data)
        J = dynamics.site_jacobian(robot.model, robot.data, robot.ee_site_id)
        J_dot = dynamics.site_jacobian_dot(robot.model, robot.data, robot.ee_site_id)
        assert np.abs(np.linalg.det(J)) > 1e-4

        Lambda = dynamics.lambda_matrix(A, J)
        h = dynamics.h_vector(J_dot, qdot)

        b = dynamics.coriolis_centrifugal(robot, q, qdot)
        g = dynamics.gravity(robot, q)
        mu = dynamics.mu_operational(J, b, Lambda, h)
        p = dynamics.p_operational(J, g)

        combined_expected = np.linalg.solve(J.T, bias_total) - Lambda @ h
        assert np.allclose(mu + p, combined_expected, atol=1e-6)


def test_operational_space_dynamics_bundle_consistent(robot, rng):
    q = random_qpos(robot, rng)
    qdot = rng.uniform(-0.3, 0.3, size=q.shape)
    J, Lambda, mu, p, xdot = dynamics.operational_space_dynamics(robot, q, qdot)
    assert J.shape == (6, robot.nv)
    assert Lambda.shape == (6, 6)
    assert mu.shape == (6,)
    assert p.shape == (6,)
    assert np.allclose(xdot, J @ qdot, atol=1e-10)
