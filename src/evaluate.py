"""
Standalone evaluation script: loads a saved model and reports detailed metrics.

Usage:
    python -m src.evaluate --model-path models/best_model.keras
"""

import argparse

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from src.data import load_data, CLASS_NAMES


def plot_confusion_matrix(cm, class_names, output_path="assets/confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Fashion-MNIST Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved confusion matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/best_model.keras")
    parser.add_argument("--save-plot", action="store_true")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)

    _, (x_test, y_test) = load_data()

    y_pred_probs = model.predict(x_test, batch_size=64)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)
    if args.save_plot:
        plot_confusion_matrix(cm, CLASS_NAMES)
    else:
        print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    main()
