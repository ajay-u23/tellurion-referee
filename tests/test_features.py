import numpy as np

from referee import physics_fit
from referee.features import forecast_residual, sensor_consistency

G, L, M = 9.81, 1.0, 1.0


def test_sensor_consistency_clean():
    theta, omega = physics_fit.simulate(0.3, 0.0, dt=0.05, n_steps=40, g=G, L=L, b=0.0, m=M)
    mask = np.ones_like(theta, dtype=np.uint8)
    result = sensor_consistency(0.05, theta, omega, mask)
    assert result["error"] < 0.01
    assert not result["stuck"]
    assert not result["jump"]


def test_sensor_consistency_detects_stuck_sensor():
    theta, omega = physics_fit.simulate(0.3, 0.0, dt=0.05, n_steps=40, g=G, L=L, b=0.0, m=M)
    stuck_theta = theta.copy()
    stuck_theta[20:] = stuck_theta[20]
    mask = np.ones_like(theta, dtype=np.uint8)

    result = sensor_consistency(0.05, stuck_theta, omega, mask)
    assert result["stuck"]
    assert result["error"] > 0.01


def test_forecast_residual_zero_when_forecast_matches():
    assert forecast_residual(np.array([0.1, 0.2]), np.array([0.5, 0.4]), 0.2, 0.4) == 0.0


def test_forecast_residual_positive_when_forecast_wrong():
    r = forecast_residual(np.array([0.1, 0.2]), np.array([0.5, 0.4]), 0.0, 0.0)
    assert r > 0
