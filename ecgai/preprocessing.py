"""Prepare one model-ready ECG using the original inference-loader convention.

Preserves waveform amplitudes and channel sequence. No filtering, resampling,
unit conversion, lead derivation, normalization, padding or cropping is applied.
"""

from __future__ import annotations

import numpy as np


SAMPLE_RATE_HZ = 500
N_SAMPLES = 5000

def _numeric_array(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    if result.dtype.kind not in "iuf":
        raise ValueError("ECG must contain real numeric samples, not objects, strings, booleans or complex values.")
    if not np.isfinite(result).all():
        raise ValueError("ECG contains non-finite samples.")
    return result


def _float32(array: np.ndarray) -> np.ndarray:
    if np.max(np.abs(array), initial=0) > np.finfo(np.float32).max:
        raise ValueError("ECG amplitudes exceed the model's float32 range.")
    return np.ascontiguousarray(array, dtype=np.float32)


def prepare_model_array(array: np.ndarray) -> np.ndarray:
    """Validate an already prepared waveform without changing its scale/order.

    Accepts (5000, 12), (12, 5000), or either with a leading singleton dimension.
    Returns one (5000, 12) float32 ECG. The caller must already have verified
    that its sampling rate, amplitude scale and lead sequence match the study
    input convention. Shape alone cannot establish that compatibility.
    """
    result = _numeric_array(array)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if result.shape == (12, N_SAMPLES):
        result = result.T
    if result.shape != (N_SAMPLES, 12):
        raise ValueError("Prepared ECG must have shape (5000, 12) or (12, 5000), optionally with one leading singleton dimension.")
    return _float32(result)
