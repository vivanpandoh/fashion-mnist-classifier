"""
CNN architecture for Fashion-MNIST classification.

Design notes
------------
- BatchNorm requires enough gradient update steps for its moving mean/variance
  to converge to stable population statistics. With a small dataset subset or
  very few epochs, train-time batch statistics diverge from what gets frozen
  for inference, causing a train/inference mismatch (high train accuracy,
  near-chance test accuracy). This model is trained on the FULL 60k training
  set with a moderate number of epochs and a smaller batch size specifically
  to accumulate enough update steps for BN statistics to stabilize.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_model(input_shape=(28, 28, 1), num_classes=10, l2_reg=1e-4):
    """Builds a compact CNN with BatchNorm + Dropout for Fashion-MNIST.

    Architecture: 3 conv blocks (increasing filters) -> GAP -> dense head.
    Each conv block: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout.
    """
    inputs = layers.Input(shape=input_shape)

    x = _conv_block(inputs, 32, l2_reg)
    x = _conv_block(x, 64, l2_reg)
    x = _conv_block(x, 128, l2_reg)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="fashion_mnist_cnn")


def _conv_block(x, filters, l2_reg, dropout_rate=0.25):
    x = layers.Conv2D(filters, 3, padding="same",
                       kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same",
                       kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def compile_model(model, learning_rate=1e-3):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
