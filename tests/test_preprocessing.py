"""Synthetic checks of model-input preservation and invalid-input rejection."""

import unittest

import numpy as np

from ecgai.preprocessing import prepare_model_array


class PreprocessingTests(unittest.TestCase):
    def test_study_format_preserves_amplitudes_and_channels(self):
        # Distinct channel offsets expose unintended reordering/normalization.
        source = np.arange(5000 * 12, dtype=np.float64).reshape(5000, 12) / 2 - 1200
        for candidate in (source, source.T, source[None], source.T[None]):
            actual = prepare_model_array(candidate)
            np.testing.assert_array_equal(actual, source.astype(np.float32))
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(actual.flags.c_contiguous)

    def test_model_input_rejects_missing_values_objects_and_batches(self):
        for invalid in (
            np.full((5000, 12), np.nan), np.full((5000, 12), np.inf),
            np.full((5000, 12), 1e300), np.zeros((5000, 12), dtype=object),
            np.zeros((5000, 12), dtype=complex), np.zeros((5000, 12), dtype=bool),
            np.zeros((2, 5000, 12)), np.zeros((2500, 12)), np.zeros((5000, 11)),
        ):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype), self.assertRaises(ValueError):
                prepare_model_array(invalid)


if __name__ == "__main__":
    unittest.main()
