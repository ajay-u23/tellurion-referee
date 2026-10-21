"""Stub forecaster standing in for the real 5-seed ensemble (Agent A's job).

It deliberately linearizes sin(theta) ~= theta, i.e. it behaves like a model
trained only on gentle swings: accurate for small angles, wrong for large
ones. Swap for the real ensemble via the same forecast_step() interface;
nothing else in the referee needs to change.
"""
from __future__ import annotations


class StubForecaster:
    def __init__(self, g: float = 9.81, L: float = 1.0, b: float = 0.0, m: float = 1.0):
        self.g, self.L, self.b, self.m = g, L, b, m

    def forecast_step(self, theta: float, omega: float, dt: float) -> tuple[float, float]:
        domega = -(self.g / self.L) * theta - (self.b / (self.m * self.L**2)) * omega
        next_omega = omega + domega * dt
        next_theta = theta + omega * dt
        return next_theta, next_omega

    def forecast_last_step(self, theta_hist, omega_hist, dt: float) -> tuple[float, float]:
        """Predict the final point in the window from the one before it."""
        return self.forecast_step(theta_hist[-2], omega_hist[-2], dt)
