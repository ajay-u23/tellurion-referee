"""Stand-in for the reality engine's pendulum_v1 dataset.

Matches the data contract (plan-reality-engine.md S5.1) field-for-field so
this can be swapped for the real tellurion_sim output with no code changes
elsewhere. build_dataset() is pure (no file I/O) so tests can use it
in-memory; run this file directly to also write an .npz + metadata.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from referee import physics_fit  # noqa: E402

DT = 0.05
N_STEPS = 100          # 5 s
ONSET = 40
NOMINAL = {"g": 9.81, "L": 1.0, "m": 1.0, "b": 0.0}


def _base_arrays(theta, omega):
    mask = np.ones_like(theta, dtype=np.uint8)
    return theta.copy(), omega.copy(), mask


def _scenario_trust(theta0=0.3):
    """Clean gentle swing -> trust. Also used (with varying theta0) for cal."""
    theta, omega = physics_fit.simulate(theta0, 0.0, DT, N_STEPS, **NOMINAL)
    m_theta, m_omega, mask = _base_arrays(theta, omega)
    return dict(true_theta=theta, true_omega=omega, meas_theta=m_theta, meas_omega=m_omega,
                mask=mask, actual_params=dict(NOMINAL), regime="gentle",
                perturbation_type="none", perturbation_subtype="none", severity="none",
                onset_step=-1, expected_verdict="trust")


def _scenario_large_swing():
    """Clean, but outside the gentle training regime -> model_error."""
    theta, omega = physics_fit.simulate(2.0, 0.0, DT, N_STEPS, **NOMINAL)
    m_theta, m_omega, mask = _base_arrays(theta, omega)
    return dict(true_theta=theta, true_omega=omega, meas_theta=m_theta, meas_omega=m_omega,
                mask=mask, actual_params=dict(NOMINAL), regime="large",
                perturbation_type="none", perturbation_subtype="none", severity="none",
                onset_step=-1, expected_verdict="model_error")


def _scenario_stuck_sensor(theta0=0.3):
    """Gentle swing; the angle sensor freezes at onset -> instrument_fault.
    True motion is untouched (plan-reality-engine.md S4.2)."""
    theta, omega = physics_fit.simulate(theta0, 0.0, DT, N_STEPS, **NOMINAL)
    m_theta, m_omega, mask = _base_arrays(theta, omega)
    m_theta[ONSET:] = m_theta[ONSET]
    return dict(true_theta=theta, true_omega=omega, meas_theta=m_theta, meas_omega=m_omega,
                mask=mask, actual_params=dict(NOMINAL), regime="gentle",
                perturbation_type="sensor_fault", perturbation_subtype="stuck", severity="high",
                onset_step=ONSET, expected_verdict="instrument_fault")


def _scenario_magnet(theta0=0.3, k=1.5, phi_m=1.2):
    """Gentle swing; a magnet pulls on the bob from onset -> discovery."""
    def tau_ext(t, th, om):
        return k * np.sin(phi_m - th) if t >= ONSET * DT else 0.0

    theta, omega = physics_fit.simulate(theta0, 0.0, DT, N_STEPS, **NOMINAL, tau_ext_fn=tau_ext)
    m_theta, m_omega, mask = _base_arrays(theta, omega)
    return dict(true_theta=theta, true_omega=omega, meas_theta=m_theta, meas_omega=m_omega,
                mask=mask, actual_params=dict(NOMINAL), regime="gentle",
                perturbation_type="unseen_force", perturbation_subtype="magnet", severity="med",
                onset_step=ONSET, expected_verdict="discovery")


def build_dataset() -> dict:
    cal_runs = [_scenario_trust(theta0=t) for t in (0.1, 0.2, 0.3)]
    for r in cal_runs:
        r["split"] = "cal"
    test_runs = [_scenario_trust(theta0=0.3), _scenario_large_swing(),
                 _scenario_stuck_sensor(), _scenario_magnet()]
    for r in test_runs:
        r["split"] = "test"

    runs = cal_runs + test_runs
    n, t = len(runs), N_STEPS + 1
    times = np.arange(t) * DT

    true_states = np.zeros((n, t, 2), dtype=np.float32)
    measured_states = np.zeros((n, t, 2), dtype=np.float32)
    measured_mask = np.zeros((n, t, 2), dtype=np.uint8)
    true_energy = np.zeros((n, t), dtype=np.float32)
    nominal_params = np.zeros((n, 4), dtype=np.float32)
    actual_params = np.zeros((n, 4), dtype=np.float32)

    for i, r in enumerate(runs):
        true_states[i, :, 0], true_states[i, :, 1] = r["true_theta"], r["true_omega"]
        measured_states[i, :, 0], measured_states[i, :, 1] = r["meas_theta"], r["meas_omega"]
        measured_mask[i, :, 0] = r["mask"]
        measured_mask[i, :, 1] = r["mask"]
        true_energy[i] = physics_fit.energy(r["true_theta"], r["true_omega"], NOMINAL["g"], NOMINAL["L"], NOMINAL["m"])
        nominal_params[i] = [NOMINAL["g"], NOMINAL["L"], NOMINAL["m"], NOMINAL["b"]]
        ap = r["actual_params"]
        actual_params[i] = [ap["g"], ap["L"], ap["m"], ap["b"]]

    return {
        "times": times,
        "true_states": true_states,
        "measured_states": measured_states,
        "measured_mask": measured_mask,
        "true_energy": true_energy,
        "nominal_params": nominal_params,
        "actual_params": actual_params,
        "trajectory_id": np.arange(n, dtype=np.int64),
        "split": np.array([r["split"] for r in runs]),
        "regime": np.array([r["regime"] for r in runs]),
        "perturbation_type": np.array([r["perturbation_type"] for r in runs]),
        "perturbation_subtype": np.array([r["perturbation_subtype"] for r in runs]),
        "severity": np.array([r["severity"] for r in runs]),
        "onset_step": np.array([r["onset_step"] for r in runs], dtype=np.int32),
        "expected_verdict": np.array([r["expected_verdict"] for r in runs]),
        "dt": DT,
    }


def save_dataset(dataset: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in dataset.items() if k != "dt"}
    np.savez(out_dir / "stub_pendulum.npz", **arrays)
    metadata = {
        "system": "pendulum", "state_names": ["theta", "omega"],
        "param_names": ["g", "L", "m", "b"], "dt": dataset["dt"],
        "horizon": N_STEPS, "solver": "RK4 (stub; real data uses solve_ivp DOP853)",
        "note": "Stand-in data for referee v0 development -- NOT the real reality-engine dataset.",
    }
    (out_dir / "stub_pendulum.metadata.json").write_text(json.dumps(metadata, indent=2))


def load_stub_dataset(path: Path) -> dict:
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


if __name__ == "__main__":
    ds = build_dataset()
    out = Path(__file__).resolve().parents[1] / "data"
    save_dataset(ds, out)
    print(f"wrote {out / 'stub_pendulum.npz'} ({len(ds['trajectory_id'])} trajectories)")
