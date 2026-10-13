"""Compatibility for the pinned Ultralytics metrics on newer NumPy releases."""
import numpy as np
from ultralytics.utils import metrics


class _MetricsNumpy:
    """Expose the renamed trapezoidal integrator only within Ultralytics metrics."""

    def __getattr__(self, name):
        if name == "trapz":
            return np.trapezoid
        return getattr(np, name)


def install_metrics_compatibility():
    """Retain Ultralytics' AP algorithm without altering the global NumPy module."""
    if not hasattr(metrics.np, "trapz"):
        metrics.np = _MetricsNumpy()
