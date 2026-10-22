"""Contract checks against real simulator output, never the locked evaluation data."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tellurion_sim.contract import dataset_errors, validate_dataset
from tellurion_sim.dataset import (
    generate,
    load_config,
    load_dataset,
    metadata,
    save_dataset,
)


@pytest.fixture(scope="module")
def generated():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/pendulum.yaml")
    cfg.update(
        n_steps=12,
        splits={"train": 2, "val": 1, "cal": 40, "test": 0},
        onset={"min": 3, "max": 7},
    )
    return generate(cfg, seed=7, workers=1), metadata(cfg, seed=7)


def test_real_engine_matches_contract(generated):
    ds, meta = generated
    assert dataset_errors(ds, meta) == []
    assert set(ds["perturbation_type"]) == {
        "none",
        "param_shift",
        "sensor_fault",
        "unseen_force",
    }


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("trajectory_id", 0, "unique"),
        ("split", "unknown", "split"),
        ("regime", "unknown", "regime"),
        ("perturbation_type", "unknown", "perturbation_type"),
        ("perturbation_subtype", "magnet", "perturbation_subtype"),
        ("severity", "high", "severity"),
        ("onset_step", 50, "onset_step"),
        ("expected_verdict", "discovery", "expected_verdict"),
    ],
)
def test_rejects_invalid_labels(generated, field, value, message):
    ds, meta = generated
    broken = deepcopy(ds)
    # The first trajectory is clean training data.
    if field == "trajectory_id":
        broken[field][1] = value
    else:
        broken[field] = (
            broken[field].astype("U32")
            if broken[field].dtype.kind == "U"
            else broken[field]
        )
        broken[field][0] = value
    with pytest.raises(ValueError, match=message):
        validate_dataset(broken, meta)


def test_rejects_fault_in_training_split(generated):
    ds, meta = generated
    broken = deepcopy(ds)
    i = np.flatnonzero(broken["perturbation_type"] == "sensor_fault")[0]
    broken["split"][i] = "train"
    assert any("train/val" in error for error in dataset_errors(broken, meta))


@pytest.mark.parametrize("channel", [0, 1])
def test_rejects_sensor_corruption_outside_fault(generated, channel):
    ds, meta = generated
    broken = deepcopy(ds)
    broken["measured_states"][0, 0, channel] += 1
    assert any("measured" in error for error in dataset_errors(broken, meta))


def test_rejects_omega_fault_and_pre_onset_fault(generated):
    ds, meta = generated
    i = np.flatnonzero(ds["perturbation_type"] == "sensor_fault")[0]
    for step, channel, message in [(8, 1, "omega"), (0, 0, "onset")]:
        broken = deepcopy(ds)
        broken["measured_states"][i, step, channel] += 1
        assert any(message in error for error in dataset_errors(broken, meta))


@pytest.mark.parametrize("mask,value", [(2, 0.0), (0, 0.0), (1, np.nan), (1, np.inf)])
def test_rejects_invalid_measurements(generated, mask, value):
    ds, meta = generated
    broken = deepcopy(ds)
    broken["measured_mask"][0, 0, 0] = mask
    broken["measured_states"][0, 0, 0] = value
    assert dataset_errors(broken, meta)


@pytest.mark.parametrize("field", ["times", "true_states", "trajectory_id"])
def test_bad_shapes_report_errors_without_crashing(generated, field):
    ds, meta = generated
    broken = {**ds, field: np.array(0, dtype=ds[field].dtype)}
    assert dataset_errors(broken, meta)


def test_rejects_metadata_and_time_grid_mismatch(generated):
    ds, meta = generated
    assert dataset_errors(ds, {**meta, "dt": 0.1})
    assert dataset_errors(ds, {**meta, "state_names": ["theta"]})
    broken = deepcopy(ds)
    broken["times"][4] = broken["times"][3]
    assert dataset_errors(broken, meta)


def test_rejects_extra_fields_and_non_array_values(generated):
    ds, meta = generated
    assert dataset_errors({**ds, "dt": meta["dt"]}, meta)
    assert dataset_errors({**ds, "times": ds["times"].tolist()}, meta)


@pytest.mark.parametrize(
    "field", ["true_states", "true_energy", "nominal_params", "actual_params"]
)
def test_rejects_nonfinite_ground_truth(generated, field):
    ds, meta = generated
    broken = deepcopy(ds)
    broken[field].flat[0] = np.nan
    assert any(field in error for error in dataset_errors(broken, meta))


def test_writer_rejects_invalid_data_before_creating_files(generated, tmp_path):
    ds, meta = generated
    broken = deepcopy(ds)
    broken["trajectory_id"][1] = broken["trajectory_id"][0]
    with pytest.raises(ValueError, match="trajectory_id"):
        save_dataset(broken, meta, tmp_path / "invalid")
    assert not list(tmp_path.iterdir())


def test_loader_rejects_misfiled_test_rows(generated, tmp_path):
    ds, _ = generated
    broken = deepcopy(ds)
    broken["split"][0] = "test"
    # Deliberately corrupt a development archive, using synthetic rows only.
    np.savez(tmp_path / "invalid.npz", **broken)
    with pytest.raises(ValueError, match="test"):
        load_dataset(tmp_path / "invalid")


def test_round_trip_and_default_loading_never_opens_locked_file(
    generated, tmp_path, monkeypatch
):
    ds, meta = generated
    prefix = tmp_path / "pendulum"
    save_dataset(ds, meta, prefix)
    # Even an unreadable locked file must not affect development loading.
    (tmp_path / "pendulum.test.npz").write_bytes(b"locked: do not open")
    real_load = np.load
    opened = []

    def tracked_load(path, **kwargs):
        opened.append(Path(path).name)
        return real_load(path, **kwargs)

    monkeypatch.setattr(np, "load", tracked_load)
    actual = load_dataset(prefix, splits=("train", "val", "cal"))
    for key in ds:
        np.testing.assert_array_equal(ds[key], actual[key], err_msg=key)
    assert opened == ["pendulum.npz"]


def test_empty_split_archive_is_valid(generated, tmp_path):
    ds, meta = generated
    empty = {key: value if key == "times" else value[:0] for key, value in ds.items()}
    validate_dataset(empty, meta)
    save_dataset(empty, meta, tmp_path / "empty")
    actual = load_dataset(tmp_path / "empty")
    assert actual["true_states"].shape == (0, len(ds["times"]), 2)


def test_checker_handles_no_ideal_trajectories(generated, tmp_path):
    from tellurion_sim.check import check_dataset

    ds, meta = generated
    keep = ds["perturbation_type"] == "unseen_force"
    forced = {
        key: value if key == "times" else value[keep] for key, value in ds.items()
    }
    save_dataset(forced, meta, tmp_path / "forced")
    ok, lines = check_dataset(tmp_path / "forced")
    assert ok
    assert any(line.startswith("[SKIP] energy drift") for line in lines)


@pytest.mark.parametrize(
    "ptype,subtype", [("sensor_fault", "dropout"), ("param_shift", "length")]
)
def test_csv_matches_sensor_and_parameter_timing(generated, tmp_path, ptype, subtype):
    import csv

    from tellurion_sim.dataset import RunSpec, simulate_run
    from tellurion_sim.export import COLUMNS, write_run_csv
    from tellurion_sim.systems import Pendulum

    ds, _ = generated
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/pendulum.yaml")
    cfg["n_steps"] = len(ds["times"]) - 1
    onset = 4
    if ptype == "sensor_fault":
        # Force dropout to exercise missing CSV cells deterministically.
        cfg[ptype][subtype]["high"] = 1.0
    run = simulate_run(
        cfg,
        RunSpec(0, "scenario", "gentle", ptype, subtype, "high", onset),
        np.random.default_rng(0),
    )
    labels = {
        "perturbation_type": ptype,
        "perturbation_subtype": subtype,
        "severity": "high",
        "onset_step": onset,
        "expected_verdict": run["expected_verdict"],
    }
    path = write_run_csv(
        tmp_path / "scenario.csv", Pendulum(), ds["times"], run, labels
    )
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == COLUMNS
        rows = list(reader)
    assert len(rows) == len(ds["times"])
    for k, row in enumerate(rows):
        assert all(
            row[key] == ""
            for key in ("predicted_angle_rad", "confidence_0to1", "pred_x", "pred_y")
        )
        params = run["nominal_params"] if k < onset else run["actual_params"]
        assert float(row["true_x"]) == pytest.approx(
            params[1] * np.sin(run["true_states"][k, 0]), abs=1e-8
        )
        assert float(row["true_y"]) == pytest.approx(
            -params[1] * np.cos(run["true_states"][k, 0]), abs=1e-8
        )
        if ptype == "sensor_fault":
            assert row["measured_valid"] == ("1" if k < onset else "0")
            assert (row["measured_angle_rad"] == "") == (k >= onset)
        assert row["expected_verdict"] == run["expected_verdict"]


def test_real_calibration_data_can_be_consumed_by_referee(generated):
    from referee.model import StubForecaster
    from referee.referee import BANNED_KEYS, Referee

    ds, meta = generated
    i = np.flatnonzero((ds["split"] == "cal") & (ds["expected_verdict"] == "trust"))[0]
    theta, omega = ds["measured_states"][i].T
    params = dict(zip(meta["param_names"], map(float, ds["nominal_params"][i])))
    forecast_theta, forecast_omega = StubForecaster(**params).forecast_last_step(
        theta, omega, meta["dt"]
    )
    history = {
        "dt": meta["dt"],
        "measured_theta": theta,
        "measured_omega": omega,
        "measured_mask": ds["measured_mask"][i, :, 0],
        "nominal_params": params,
        "forecast_theta": forecast_theta,
        "forecast_omega": forecast_omega,
    }
    assert not BANNED_KEYS.intersection(history)
    referee = Referee()
    referee.fit_thresholds([history])
    assert referee.step(history)["verdict"] == "trust"
