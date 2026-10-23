"""Original manuscript encoder-attention network, for inference.

The architecture is defined explicitly; only numerical layer weights are loaded.
The sigmoid output is an uncalibrated ECG-AI score, not an absolute risk estimate.
"""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

INPUT_SHAPE = (5000, 12)
PARAMETER_COUNT = 14_048_513
WEIGHTS_DIRECTORY = Path(__file__).resolve().parents[1] / "weights"


def build_model():
    """Construct the published neural architecture without compiling or training."""
    layers = tf.keras.layers
    inputs = layers.Input(shape=INPUT_SHAPE, name="ecg")
    x = inputs
    for index, (filters, kernel) in enumerate(((256, 5), (512, 11), (1024, 21)), 1):
        x = layers.Conv1D(filters, kernel, padding="same", name=f"conv_{index}")(x)
        x = layers.GroupNormalization(
            groups=32, axis=-1, epsilon=0.001, name=f"group_norm_{index}"
        )(x)
        x = layers.Activation("relu", name=f"relu_{index}")(x)
        x = layers.Dropout(0.4389385018557409, name=f"dropout_{index}")(x)
        if index < 3:
            x = layers.MaxPooling1D(pool_size=2, name=f"pool_{index}")(x)

    attention = layers.Conv1D(1024, 1, padding="same", name="attention_logits")(x)
    attention = layers.Softmax(axis=1, name="temporal_attention")(attention)
    x = layers.Multiply(name="weighted_features")([x, attention])
    x = layers.GlobalAveragePooling1D(name="global_average")(x)
    x = layers.Dense(512, name="embedding")(x)
    x = layers.GroupNormalization(
        groups=32, axis=-1, epsilon=0.001, name="embedding_group_norm"
    )(x)
    encoder = tf.keras.Model(inputs, x, name="encoder")
    classifier_input = layers.Input(shape=INPUT_SHAPE, name="waveform")
    output = layers.Dense(1, activation="sigmoid", name="ecgai_score")(
        encoder(classifier_input)
    )
    return tf.keras.Model(classifier_input, output, name="ecgai_cardiomyopathy")


def load_model():
    """Load the release weights after checking the supplied SHA-256 manifest."""
    manifest = json.loads((WEIGHTS_DIRECTORY / "manifest.json").read_text())
    path = WEIGHTS_DIRECTORY / "ecgai_weights.h5"
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Model weights failed SHA-256 verification; obtain the intact release file.")
    model = build_model()
    model.load_weights(str(path))
    if model.count_params() != PARAMETER_COUNT:
        raise ValueError("Unexpected model architecture.")
    return model


def predict_scores(model, ecgs, *, batch_size=1):
    """Score preprocessed ECGs; return one raw score per recording.

    Accepts (5000,12) or (N,5000,12). Uses inference mode and small batches
    so a GPU is optional. No calibration or clinical decision threshold is applied.
    """
    x = np.asarray(ecgs)
    if x.dtype.kind not in "iuf":
        raise ValueError("ECGs must be real numerical arrays.")
    if x.ndim == 2:
        x = x[np.newaxis, ...]
    if x.ndim != 3 or x.shape[1:] != INPUT_SHAPE or x.shape[0] == 0:
        raise ValueError("Expected (5000,12) or a nonempty (N,5000,12) array.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    with np.errstate(over="ignore", invalid="ignore"):
        x = x.astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("ECGs must contain finite float32 values.")
    predictions = [
        np.asarray(model(x[start:start + batch_size], training=False)).reshape(-1)
        for start in range(0, len(x), batch_size)
    ]
    scores = np.concatenate(predictions)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("The model returned invalid scores.")
    return scores
