"""Exercise nonzero AP so legacy NumPy calls cannot hide behind empty detections."""
import numpy as np
import pytest

import paddy_yolo
from ultralytics.utils import metrics
from paddy_yolo.compat import install_metrics_compatibility


def test_perfect_detections_have_nonzero_average_precision():
    ap, precision, recall = metrics.compute_ap(np.array([0.5, 1.0]), np.array([1.0, 1.0]))
    # YOLO's 101-point interpolation includes a zero-precision endpoint.
    assert ap == pytest.approx(0.995)
    assert precision.shape == recall.shape == (4,)


def test_metrics_compatibility_is_idempotent_and_local_to_ultralytics():
    original_numpy_has_trapz = hasattr(np, "trapz")
    install_metrics_compatibility()
    original_adapter = metrics.np
    install_metrics_compatibility()
    assert metrics.np is original_adapter
    assert hasattr(np, "trapz") == original_numpy_has_trapz
    assert metrics.np.array is np.array
