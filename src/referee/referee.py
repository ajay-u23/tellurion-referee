"""Referee v0: rule-based verdict logic. No learned components.

Decision order matches plan-referee.md S3.3 / the hackathon guide's flowchart:
no disagreement -> trust; sensors disagree with each other -> instrument_fault;
physics (nominal or refit) explains it -> model_error; otherwise -> discovery.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import window_features

# plan-referee.md S0.4: the referee must never see these at decision time.
BANNED_KEYS = {
    "true_states", "actual_params", "perturbation_type",
    "perturbation_subtype", "severity", "onset_step", "expected_verdict",
}


def assert_no_leakage(history: dict) -> None:
    leaked = BANNED_KEYS & set(history.keys())
    if leaked:
        raise ValueError(f"Referee.step received forbidden keys: {sorted(leaked)}")


@dataclass
class Thresholds:
    residual: float = 0.05
    sensor_inconsistency: float = 0.05
    fit_error: float = 0.05


class Referee:
    def __init__(self, thresholds: Thresholds | None = None):
        self.thresholds = thresholds or Thresholds()

    def fit_thresholds(self, cal_histories: list[dict], margin: float = 3.0,
                        floors: tuple[float, float, float] = (0.01, 0.01, 0.01)) -> Thresholds:
        """cal_histories must be CLEAN (trust) windows only -- labels are allowed
        here because this runs offline on the cal split, not inside step()."""
        residuals, inconsistencies, fit_errors = [], [], []
        for h in cal_histories:
            assert_no_leakage(h)
            feats = window_features(h)
            residuals.append(feats["residual"])
            inconsistencies.append(feats["sensor_inconsistency"])
            fit_errors.append(min(feats["nominal_fit_error"], feats["refit_fit_error"]))

        def pick(values, floor):
            return max(float(np.percentile(values, 95)) * margin, floor) if values else floor

        self.thresholds = Thresholds(
            residual=pick(residuals, floors[0]),
            sensor_inconsistency=pick(inconsistencies, floors[1]),
            fit_error=pick(fit_errors, floors[2]),
        )
        return self.thresholds

    def step(self, history: dict) -> dict:
        assert_no_leakage(history)
        feats = window_features(history)
        t = self.thresholds

        if feats["residual"] <= t.residual:
            verdict = "trust"
            evidence = "forecast matches measurement within tolerance"
        elif feats["sensor_stuck"] or feats["sensor_jump"] or feats["sensor_inconsistency"] > t.sensor_inconsistency:
            verdict = "instrument_fault"
            evidence = self._sensor_evidence(feats)
        elif feats["nominal_fit_error"] <= t.fit_error:
            verdict = "model_error"
            evidence = "sensors agree; nominal physics explains the window; the model's forecast does not"
        elif feats["refit_fit_error"] <= t.fit_error:
            verdict = "model_error"
            evidence = f"sensors agree; physics fits after refitting params {feats['refit_params']}"
        else:
            verdict = "discovery"
            evidence = "sensors agree; neither nominal nor refit physics explains the window"

        return {"score": float(feats["residual"]), "verdict": verdict, "evidence": evidence, "features": feats}

    @staticmethod
    def _sensor_evidence(feats: dict) -> str:
        if feats["sensor_stuck"]:
            return "angle sensor appears stuck (near-zero variance) while the speed sensor still reports motion"
        if feats["sensor_jump"]:
            return "angle sensor jumped beyond a physically possible step"
        return "angle and speed sensors disagree with each other"
