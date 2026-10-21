"""Reference pendulum physics: simulate, energy, nominal fit, refit.

Used by the referee's checks 3 and 4 (plan-referee.md S3.3.3-3.3.4).
Integrator is fixed-step RK4 -- good enough for v0; the reality engine's
real dataset is produced with solve_ivp(DOP853), not this.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def accel(theta: float, omega: float, g: float, L: float, b: float, m: float = 1.0, tau_ext: float = 0.0) -> float:
    return -(g / L) * np.sin(theta) - (b / (m * L**2)) * omega + tau_ext / (m * L**2)


def simulate(theta0: float, omega0: float, dt: float, n_steps: int,
             g: float, L: float, b: float, m: float = 1.0, tau_ext_fn=None):
    """RK4 rollout of the pendulum ODE. tau_ext_fn(t, theta, omega) -> external torque."""
    theta = np.empty(n_steps + 1)
    omega = np.empty(n_steps + 1)
    theta[0], omega[0] = theta0, omega0
    t = 0.0
    for i in range(n_steps):
        tau = tau_ext_fn(t, theta[i], omega[i]) if tau_ext_fn else 0.0

        def deriv(th, om):
            return om, accel(th, om, g, L, b, m, tau)

        k1 = deriv(theta[i], omega[i])
        k2 = deriv(theta[i] + 0.5 * dt * k1[0], omega[i] + 0.5 * dt * k1[1])
        k3 = deriv(theta[i] + 0.5 * dt * k2[0], omega[i] + 0.5 * dt * k2[1])
        k4 = deriv(theta[i] + dt * k3[0], omega[i] + dt * k3[1])
        theta[i + 1] = theta[i] + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        omega[i + 1] = omega[i] + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        t += dt
    return theta, omega


def energy(theta: np.ndarray, omega: np.ndarray, g: float, L: float, m: float = 1.0) -> np.ndarray:
    return 0.5 * m * L**2 * omega**2 + m * g * L * (1 - np.cos(theta))


def _first_valid(mask: np.ndarray) -> int:
    idx = np.argmax(mask.astype(bool))
    return int(idx)


def nominal_fit_error(dt: float, meas_theta: np.ndarray, meas_omega: np.ndarray, mask: np.ndarray,
                       g: float, L: float, b: float, m: float = 1.0) -> float:
    """Re-simulate the window from its first measured point with nominal params; RMSE vs measured."""
    mask = np.asarray(mask, dtype=bool)
    idx0 = _first_valid(mask)
    n_steps = len(meas_theta) - 1 - idx0
    if n_steps <= 0:
        return 0.0
    sim_theta, sim_omega = simulate(meas_theta[idx0], meas_omega[idx0], dt, n_steps, g, L, b, m)
    seg_mask = mask[idx0:]
    if not seg_mask.any():
        return 0.0
    err_theta = (sim_theta - meas_theta[idx0:])[seg_mask]
    err_omega = (sim_omega - meas_omega[idx0:])[seg_mask]
    return float(np.sqrt(np.mean(err_theta**2 + err_omega**2)))


def refit_fit_error(dt: float, meas_theta: np.ndarray, meas_omega: np.ndarray, mask: np.ndarray,
                     g0: float = 9.81, L0: float = 1.0, b0: float = 0.0):
    """Least-squares re-estimate of (L, b) over the window, g held fixed.

    g is fixed rather than jointly fit: for a single frictionless swing only
    the ratio g/L is observable (nondimensionalizing time by sqrt(L/g) removes
    g and L separately from the ODE), so fitting all three is under-determined
    and the optimizer can land anywhere along that degenerate valley -- which
    also lets it spuriously "explain away" an unseen force. Treating g as a
    known constant and refitting only (L, b) removes that degeneracy.
    Returns (fit_error, {g, L, b}).
    """
    mask = np.asarray(mask, dtype=bool)
    idx0 = _first_valid(mask)
    n_steps = len(meas_theta) - 1 - idx0
    if n_steps <= 0:
        return 0.0, {"g": g0, "L": L0, "b": b0}
    seg_mask = mask[idx0:]
    theta0, omega0 = meas_theta[idx0], meas_omega[idx0]
    meas_theta_seg = meas_theta[idx0:][seg_mask]
    meas_omega_seg = meas_omega[idx0:][seg_mask]

    def residuals(params):
        L, b = params
        sim_theta, sim_omega = simulate(theta0, omega0, dt, n_steps, g0, L, b)
        return np.concatenate([
            (sim_theta[seg_mask] - meas_theta_seg),
            (sim_omega[seg_mask] - meas_omega_seg),
        ])

    result = least_squares(residuals, x0=[L0, b0], bounds=([0.1, 0.0], [3.0, 2.0]))
    fit_error = float(np.sqrt(np.mean(result.fun**2))) if result.fun.size else 0.0
    fitted = {"g": float(g0), "L": float(result.x[0]), "b": float(result.x[1])}
    return fit_error, fitted
