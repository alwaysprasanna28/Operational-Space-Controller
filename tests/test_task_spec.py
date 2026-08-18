"""eq. (1)-(4): task specification matrices."""

import numpy as np

from khatib_osc.task_spec import (
    complement,
    make_task_spec,
    position_spec,
    surface_wiping_spec,
)


def test_complement_is_involution():
    Sigma = position_spec(True, False, True)
    Sigma_bar = complement(Sigma)
    assert np.allclose(Sigma + Sigma_bar, np.eye(3))
    assert np.allclose(complement(Sigma_bar), Sigma)


def test_omega_and_omega_tilde_partition_identity():
    rng = np.random.default_rng(1)
    for _ in range(10):
        S_f = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        S_r = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        Sigma_f = position_spec(True, False, True)
        Sigma_r = position_spec(False, True, True)
        Omega, Omega_tilde = make_task_spec(S_f, Sigma_f, S_r, Sigma_r)
        assert np.allclose(Omega + Omega_tilde, np.eye(6), atol=1e-8)


def test_surface_wiping_spec_flat_table():
    Omega, Omega_tilde = surface_wiping_spec(np.array([0.0, 0.0, 1.0]))
    assert np.allclose(Omega, np.diag([1, 1, 0, 1, 1, 1]))
    assert np.allclose(Omega_tilde, np.diag([0, 0, 1, 0, 0, 0]))


def test_surface_wiping_spec_tilted_table():
    normal = np.array([0.0, 0.3, 0.95])
    normal /= np.linalg.norm(normal)
    Omega, Omega_tilde = surface_wiping_spec(normal)
    assert np.allclose(Omega + Omega_tilde, np.eye(6), atol=1e-8)
    # force direction (row space of Omega_tilde's position block) must be the surface normal
    pos_block = Omega_tilde[:3, :3]
    eigvals, eigvecs = np.linalg.eigh(pos_block)
    force_dir = eigvecs[:, np.argmax(eigvals)]
    force_dir *= np.sign(force_dir @ normal)
    assert np.allclose(force_dir, normal, atol=1e-6)
