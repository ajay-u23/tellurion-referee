import numpy as np
import pytest

from referee import physics_fit

G, L, M = 9.81, 1.0, 1.0


def test_energy_conservation_no_damping():
    theta, omega = physics_fit.simulate(0.4, 0.0, dt=0.01, n_steps=2000, g=G, L=L, b=0.0, m=M)
    e = physics_fit.energy(theta, omega, G, L, M)
    rel_drift = np.max(np.abs(e - e[0])) / abs(e[0])
    assert rel_drift < 1e-6


def test_small_angle_period():
    dt = 0.001
    theta, omega = physics_fit.simulate(0.05, 0.0, dt=dt, n_steps=20000, g=G, L=L, b=0.0, m=M)
    zero_crossings = np.where(np.diff(np.sign(omega)))[0]
    half_period = (zero_crossings[1] - zero_crossings[0]) * dt
    period = 2 * half_period
    expected = 2 * np.pi * np.sqrt(L / G)
    assert abs(period - expected) / expected < 1e-3


def test_nominal_fit_recovers_clean_window():
    theta, omega = physics_fit.simulate(0.3, 0.0, dt=0.05, n_steps=40, g=G, L=L, b=0.0, m=M)
    mask = np.ones_like(theta, dtype=bool)
    err = physics_fit.nominal_fit_error(0.05, theta, omega, mask, G, L, 0.0, M)
    assert err < 1e-6


def test_refit_recovers_longer_string():
    true_L = 1.15
    theta, omega = physics_fit.simulate(0.3, 0.0, dt=0.05, n_steps=40, g=G, L=true_L, b=0.0, m=M)
    mask = np.ones_like(theta, dtype=bool)

    nominal_err = physics_fit.nominal_fit_error(0.05, theta, omega, mask, G, L, 0.0, M)
    refit_err, fitted = physics_fit.refit_fit_error(0.05, theta, omega, mask, g0=G, L0=L, b0=0.0)

    assert refit_err < nominal_err
    assert fitted["L"] == pytest.approx(true_L, rel=0.02)
    assert refit_err < 1e-4


def test_refit_does_not_explain_a_magnet():
    k, phi_m = 1.5, 1.2

    def tau_ext(t, th, om):
        return k * np.sin(phi_m - th) if t >= 2.0 else 0.0

    theta, omega = physics_fit.simulate(0.3, 0.0, dt=0.05, n_steps=100, g=G, L=L, b=0.0, m=M, tau_ext_fn=tau_ext)
    mask = np.ones_like(theta, dtype=bool)
    window = slice(70, 91)

    nominal_err = physics_fit.nominal_fit_error(0.05, theta[window], omega[window], mask[window], G, L, 0.0, M)
    refit_err, _ = physics_fit.refit_fit_error(0.05, theta[window], omega[window], mask[window], g0=G, L0=L, b0=0.0)

    assert nominal_err > 0.01
    assert refit_err > 0.01
