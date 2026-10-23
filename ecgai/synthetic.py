"""Mathematical smoke-test ECG; no patient-derived samples or parameters."""

import numpy as np

from .preprocessing import prepare_model_array


def make_synthetic_ecg() -> np.ndarray:
    """Return a 10-second, 500-Hz, (5000, 12) float32 test signal in microvolts.

    This is not a physiologically validated simulation. Its model score has no
    clinical meaning. Pulse shapes and amplitudes below are chosen constants.
    """
    time = np.arange(5000) / 500.0
    phase = time % 0.8
    pulse = (
        75 * np.exp(-((phase - 0.12) / 0.035) ** 2)
        - 100 * np.exp(-((phase - 0.23) / 0.012) ** 2)
        + 750 * np.exp(-((phase - 0.25) / 0.01) ** 2)
        - 180 * np.exp(-((phase - 0.28) / 0.014) ** 2)
        + 180 * np.exp(-((phase - 0.48) / 0.065) ** 2)
    )
    gains = np.array([0.7, 1.0, -0.3, 0.2, 0.6, 1.0, 0.8, 0.5])
    raw = pulse[:, None] * gains[None, :]
    # Construct twelve artificial channels directly. These relations describe
    # this mathematical example, not a vendor export conversion procedure.
    lead_i, lead_ii = raw[:, 0], raw[:, 1]
    twelve_leads = np.column_stack([
        lead_i, lead_ii, lead_ii - lead_i,
        -(lead_i + lead_ii) / 2.0,
        lead_i - lead_ii / 2.0, lead_ii - lead_i / 2.0,
        raw[:, 2:],
    ])
    return prepare_model_array(twelve_leads)
