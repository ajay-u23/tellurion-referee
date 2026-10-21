"""The referee's five checks (plan-referee.md S3.3), computed over one window."""
from __future__ import annotations

import numpy as np

from . import physics_fit


def forecast_residual(meas_theta: np.ndarray, meas_omega: np.ndarray,
                       forecast_theta: float, forecast_omega: float) -> float:
    dtheta = meas_theta[-1] - forecast_theta
    domega = meas_omega[-1] - forecast_omega
    return float(np.sqrt(dtheta**2 + domega**2))


def sensor_consistency(dt: float, meas_theta: np.ndarray, meas_omega: np.ndarray, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    idx = np.where(mask)[0]
    dropout_frac = float(1.0 - mask.mean())
    if len(idx) < 3:
        return {"error": 0.0, "stuck": False, "jump": False, "dropout_frac": dropout_frac}

    theta_v, omega_v, t_v = meas_theta[idx], meas_omega[idx], idx * dt
    dtheta_dt = np.diff(theta_v) / np.diff(t_v)
    omega_mid = 0.5 * (omega_v[:-1] + omega_v[1:])
    consistency_error = float(np.sqrt(np.mean((dtheta_dt - omega_mid) ** 2)))

    tail = theta_v[-min(10, len(theta_v)):]
    stuck = bool(np.std(tail) < 1e-6)
    jump = bool(np.max(np.abs(np.diff(theta_v))) > 1.0)  # > ~57 deg in one sample step is nonphysical here
    return {"error": consistency_error, "stuck": stuck, "jump": jump, "dropout_frac": dropout_frac}


def energy_drift(theta: np.ndarray, omega: np.ndarray, g: float, L: float, m: float = 1.0) -> float:
    e = physics_fit.energy(theta, omega, g, L, m)
    if abs(e[0]) < 1e-9:
        return 0.0
    return float(np.max(np.abs(e - e[0])) / abs(e[0]))


def window_features(history: dict) -> dict:
    """history keys: dt, measured_theta, measured_omega, measured_mask, nominal_params
    ({g, L, m, b}), forecast_theta, forecast_omega. Nothing else -- see referee.BANNED_KEYS."""
    dt = history["dt"]
    theta = np.asarray(history["measured_theta"], dtype=float)
    omega = np.asarray(history["measured_omega"], dtype=float)
    mask = np.asarray(history["measured_mask"], dtype=np.uint8)
    p = history["nominal_params"]
    g, L, b, m = p["g"], p["L"], p.get("b", 0.0), p.get("m", 1.0)

    residual = forecast_residual(theta, omega, history["forecast_theta"], history["forecast_omega"])
    consistency = sensor_consistency(dt, theta, omega, mask)
    nominal_err = physics_fit.nominal_fit_error(dt, theta, omega, mask, g, L, b, m)
    refit_err, refit_params = physics_fit.refit_fit_error(dt, theta, omega, mask, g0=g, L0=L, b0=b)
    drift = None if b > 1e-9 else energy_drift(theta, omega, g, L, m)

    return {
        "residual": residual,
        "sensor_inconsistency": consistency["error"],
        "sensor_stuck": consistency["stuck"],
        "sensor_jump": consistency["jump"],
        "dropout_frac": consistency["dropout_frac"],
        "nominal_fit_error": nominal_err,
        "refit_fit_error": refit_err,
        "refit_params": refit_params,
        "energy_drift": drift,
    }
