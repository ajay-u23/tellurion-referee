"""Reads the stubbed data contract end to end through the referee.

This is the integration check: generate data shaped like the real
reality-engine handoff, run it through the same Referee used in unit
tests, and confirm the verdicts match the labels -- without the referee
ever touching the label fields directly (see assert_no_leakage).
"""
import numpy as np

from make_stub_data import NOMINAL, build_dataset
from referee.model import StubForecaster
from referee.referee import Referee

WINDOW = slice(70, 91)


def _history(dataset, i, forecaster, dt):
    theta = dataset["measured_states"][i, :, 0]
    omega = dataset["measured_states"][i, :, 1]
    mask = dataset["measured_mask"][i, :, 0]
    theta_w, omega_w, mask_w = theta[WINDOW], omega[WINDOW], mask[WINDOW]
    f_theta, f_omega = forecaster.forecast_last_step(theta_w, omega_w, dt)
    g, L, m, b = dataset["nominal_params"][i]
    return {
        "dt": dt, "measured_theta": theta_w, "measured_omega": omega_w,
        "measured_mask": mask_w,
        "nominal_params": {"g": float(g), "L": float(L), "m": float(m), "b": float(b)},
        "forecast_theta": f_theta, "forecast_omega": f_omega,
    }


def test_referee_matches_expected_verdict_on_every_test_trajectory():
    dataset = build_dataset()
    dt = dataset["dt"]
    forecaster = StubForecaster(**NOMINAL)

    cal_idx = [i for i, s in enumerate(dataset["split"]) if s == "cal"]
    test_idx = [i for i, s in enumerate(dataset["split"]) if s == "test"]
    assert cal_idx and test_idx

    referee = Referee()
    referee.fit_thresholds([_history(dataset, i, forecaster, dt) for i in cal_idx])

    for i in test_idx:
        history = _history(dataset, i, forecaster, dt)
        result = referee.step(history)
        expected = str(dataset["expected_verdict"][i])
        assert result["verdict"] == expected, (
            f"trajectory {i} ({dataset['perturbation_type'][i]}/{dataset['perturbation_subtype'][i]}): "
            f"expected {expected}, got {result['verdict']} ({result['evidence']})"
        )


def test_stub_dataset_matches_contract_shapes():
    dataset = build_dataset()
    n, t = dataset["trajectory_id"].shape[0], dataset["times"].shape[0]
    assert dataset["true_states"].shape == (n, t, 2)
    assert dataset["measured_states"].shape == (n, t, 2)
    assert dataset["measured_mask"].shape == (n, t, 2)
    assert dataset["nominal_params"].shape == (n, 4)
    assert dataset["actual_params"].shape == (n, 4)
    assert set(np.unique(dataset["split"])) <= {"train", "val", "cal", "test"}
    assert set(np.unique(dataset["expected_verdict"])) <= {
        "trust", "model_error", "instrument_fault", "discovery"
    }
