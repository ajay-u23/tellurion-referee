import numpy as np
import pytest

from referee import physics_fit
from referee.model import StubForecaster
from referee.referee import Referee, assert_no_leakage

DT = 0.05
NOMINAL = {"g": 9.81, "L": 1.0, "m": 1.0, "b": 0.0}
WINDOW = slice(70, 91)
ONSET = 40


def _history(theta, omega, mask, forecaster):
    theta_w, omega_w, mask_w = theta[WINDOW], omega[WINDOW], mask[WINDOW]
    f_theta, f_omega = forecaster.forecast_last_step(theta_w, omega_w, DT)
    return {
        "dt": DT,
        "measured_theta": theta_w,
        "measured_omega": omega_w,
        "measured_mask": mask_w,
        "nominal_params": NOMINAL,
        "forecast_theta": f_theta,
        "forecast_omega": f_omega,
    }


@pytest.fixture
def gentle_forecaster():
    return StubForecaster(**NOMINAL)


@pytest.fixture
def referee(gentle_forecaster):
    """Thresholds come from calibration on clean cal windows, not hand-picked
    numbers -- same rule the real referee follows (plan-referee.md S3.3)."""
    ref = Referee()
    cal_histories = []
    for theta0 in (0.1, 0.2, 0.3):
        theta, omega = physics_fit.simulate(theta0, 0.0, DT, 100, **NOMINAL)
        mask = np.ones_like(theta, dtype=np.uint8)
        cal_histories.append(_history(theta, omega, mask, gentle_forecaster))
    ref.fit_thresholds(cal_histories)
    return ref


def test_clean_gentle_run_is_trust(referee, gentle_forecaster):
    theta, omega = physics_fit.simulate(0.3, 0.0, DT, 100, **NOMINAL)
    mask = np.ones_like(theta, dtype=np.uint8)
    result = referee.step(_history(theta, omega, mask, gentle_forecaster))
    assert result["verdict"] == "trust"


def test_clean_large_swing_is_model_error(referee, gentle_forecaster):
    """The gentle-only forecaster breaks outside its training range, but
    nominal physics (same equations, any amplitude) still fits -> model_error."""
    theta, omega = physics_fit.simulate(2.0, 0.0, DT, 100, **NOMINAL)
    mask = np.ones_like(theta, dtype=np.uint8)
    result = referee.step(_history(theta, omega, mask, gentle_forecaster))
    assert result["verdict"] == "model_error"


def test_stuck_sensor_is_instrument_fault(referee, gentle_forecaster):
    theta, omega = physics_fit.simulate(0.3, 0.0, DT, 100, **NOMINAL)
    stuck_theta = theta.copy()
    stuck_theta[ONSET:] = stuck_theta[ONSET]
    mask = np.ones_like(theta, dtype=np.uint8)
    result = referee.step(_history(stuck_theta, omega, mask, gentle_forecaster))
    assert result["verdict"] == "instrument_fault"


def test_magnet_is_discovery(referee, gentle_forecaster):
    k, phi_m = 1.5, 1.2

    def tau_ext(t, th, om):
        return k * np.sin(phi_m - th) if t >= ONSET * DT else 0.0

    theta, omega = physics_fit.simulate(0.3, 0.0, DT, 100, **NOMINAL, tau_ext_fn=tau_ext)
    mask = np.ones_like(theta, dtype=np.uint8)
    result = referee.step(_history(theta, omega, mask, gentle_forecaster))
    assert result["verdict"] == "discovery"


def test_leakage_guard_rejects_forbidden_keys():
    history = {
        "dt": DT, "measured_theta": np.zeros(5), "measured_omega": np.zeros(5),
        "measured_mask": np.ones(5), "nominal_params": NOMINAL,
        "forecast_theta": 0.0, "forecast_omega": 0.0,
        "expected_verdict": "trust",  # forbidden
    }
    with pytest.raises(ValueError):
        assert_no_leakage(history)


def test_fit_thresholds_runs_on_clean_cal_windows(gentle_forecaster):
    ref = Referee()
    cal_histories = []
    for theta0 in (0.1, 0.2, 0.3):
        theta, omega = physics_fit.simulate(theta0, 0.0, DT, 100, **NOMINAL)
        mask = np.ones_like(theta, dtype=np.uint8)
        cal_histories.append(_history(theta, omega, mask, gentle_forecaster))

    thresholds = ref.fit_thresholds(cal_histories)
    assert thresholds.residual > 0
    assert thresholds.fit_error > 0
