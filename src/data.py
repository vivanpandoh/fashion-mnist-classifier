"""
Data loading, preprocessing, and augmentation pipeline for Fashion-MNIST.
"""

import tensorflow as tf
import numpy as np

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_data():
    """Loads the full Fashion-MNIST dataset (60k train / 10k test)."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return (x_train, y_train), (x_test, y_test)


def make_train_val_split(x_train, y_train, val_fraction=0.1, seed=42):
    """Splits off a validation set while preserving the full training pool."""
    rng = np.random.default_rng(seed)
    n = len(x_train)
    indices = rng.permutation(n)
    val_size = int(n * val_fraction)

    val_idx, train_idx = indices[:val_size], indices[val_size:]
    return (x_train[train_idx], y_train[train_idx]), (x_train[val_idx], y_train[val_idx])


def build_augmentation():
    """Light augmentation appropriate for clothing silhouettes.

    Avoids vertical flips (a shoe or bag flipped vertically is unrealistic)
    and keeps rotation/shift/zoom mild since garment shape is class-defining.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomFlip(mode="horizontal"),
    ], name="augmentation")


def make_datasets(x_train, y_train, x_val, y_val, x_test, y_test,
                   batch_size=64, augment=True):
    """Builds tf.data pipelines.

    Batch size defaults to 64 (rather than 128/256) specifically to increase
    the number of gradient/BN update steps per epoch on the full 60k-image
    training set -- this is what resolves the BN moving-statistics
    convergence issue seen with larger batches + too few epochs.
    """
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10_000, seed=42)

    if augment:
        aug = build_augmentation()
        train_ds = train_ds.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds
