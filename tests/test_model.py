"""Synthetic tests of released weights, architecture and inference behavior."""

import unittest

import numpy as np

from ecgai.model import PARAMETER_COUNT, load_model, predict_scores
from ecgai.preprocessing import prepare_model_array
from ecgai.synthetic import make_synthetic_ecg


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()
        cls.ecg = prepare_model_array(make_synthetic_ecg())

    def test_architecture_and_weights(self):
        self.assertEqual(self.model.input_shape, (None, 5000, 12))
        self.assertEqual(self.model.output_shape, (None, 1))
        self.assertEqual(self.model.count_params(), PARAMETER_COUNT)
        self.assertEqual(len(self.model.weights), 20)
        encoder = self.model.get_layer("encoder")
        self.assertEqual(encoder.get_layer("pool_2").output.shape[1:], (1250, 512))
        self.assertEqual(encoder.get_layer("weighted_features").output.shape[1:], (1250, 1024))

    def test_batch_and_repeat_inference_agree(self):
        single = predict_scores(self.model, self.ecg)
        repeated = predict_scores(self.model, self.ecg)
        batch = predict_scores(self.model, np.stack([self.ecg, self.ecg]), batch_size=2)
        np.testing.assert_allclose(single, repeated, rtol=0, atol=1e-6)
        np.testing.assert_allclose(batch, np.repeat(single, 2), rtol=0, atol=3e-5)
        self.assertTrue(np.all((single >= 0) & (single <= 1)))

    def test_rejects_invalid_inputs(self):
        invalid = [np.zeros((12, 5000)), np.zeros((0, 5000, 12)),
                   np.full((5000, 12), np.nan), np.zeros((5000, 12), dtype=complex),
                   np.full((5000, 12), 1e100)]
        for x in invalid:
            with self.subTest(shape=x.shape, dtype=x.dtype):
                with self.assertRaises(ValueError):
                    predict_scores(self.model, x)
        for size in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                predict_scores(self.model, self.ecg, batch_size=size)


if __name__ == "__main__":
    unittest.main()
