"""
Training entry point.

Usage:
    python -m src.train --epochs 40 --batch-size 64
"""

import argparse
import os
import json

import tensorflow as tf

from src.data import load_data, make_train_val_split, make_datasets
from src.model import build_model, compile_model


def get_callbacks(checkpoint_path, log_dir):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True,
            save_weights_only=False, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.CSVLogger(os.path.join(log_dir, "training_log.csv")),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("Loading full Fashion-MNIST dataset (60k train / 10k test)...")
    (x_train, y_train), (x_test, y_test) = load_data()
    (x_train, y_train), (x_val, y_val) = make_train_val_split(
        x_train, y_train, val_fraction=args.val_fraction
    )
    print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")

    train_ds, val_ds, test_ds = make_datasets(
        x_train, y_train, x_val, y_val, x_test, y_test,
        batch_size=args.batch_size, augment=not args.no_augment,
    )

    steps_per_epoch = len(x_train) // args.batch_size
    print(f"Batch size: {args.batch_size} -> {steps_per_epoch} steps/epoch "
          f"(sufficient volume for BatchNorm statistics to converge).")

    model = build_model()
    model = compile_model(model, learning_rate=args.lr)
    model.summary()

    checkpoint_path = os.path.join(args.output_dir, "best_model.keras")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=get_callbacks(checkpoint_path, args.log_dir),
    )

    print("\nEvaluating on test set (inference mode, using frozen BN stats)...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    final_path = os.path.join(args.output_dir, "final_model.keras")
    model.save(final_path)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "epochs_trained": len(history.history["loss"]),
            "batch_size": args.batch_size,
        }, f, indent=2)

    print(f"Saved best model to {checkpoint_path}")
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
