"""
Lightweight tests: verify model builds, compiles, and produces correct shapes.
Uses synthetic data -- no dataset download required.
"""

import numpy as np
import tensorflow as tf

from src.model import build_model, compile_model
from src.data import build_augmentation


def test_model_output_shape():
    model = build_model()
    dummy = np.random.rand(4, 28, 28, 1).astype("float32")
    out = model(dummy)
    assert out.shape == (4, 10)


def test_model_compiles_and_trains_one_step():
    model = build_model()
    model = compile_model(model)

    x = np.random.rand(16, 28, 28, 1).astype("float32")
    y = np.random.randint(0, 10, size=(16,))

    history = model.fit(x, y, epochs=1, batch_size=8, verbose=0)
    assert "loss" in history.history


def test_probabilities_sum_to_one():
    model = build_model()
    dummy = np.random.rand(2, 28, 28, 1).astype("float32")
    out = model(dummy).numpy()
    sums = out.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(2), atol=1e-4)


def test_augmentation_preserves_shape():
    aug = build_augmentation()
    dummy = tf.random.uniform((4, 28, 28, 1))
    out = aug(dummy, training=True)
    assert out.shape == dummy.shape


if __name__ == "__main__":
    test_model_output_shape()
    test_model_compiles_and_trains_one_step()
    test_probabilities_sum_to_one()
    test_augmentation_preserves_shape()
    print("All tests passed.")
