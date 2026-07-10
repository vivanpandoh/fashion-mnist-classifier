"""
Run inference on a single image file (28x28 grayscale, or auto-resized).

Usage:
    python -m src.predict --model-path models/best_model.keras --image path/to/img.png
"""

import argparse

import numpy as np
import tensorflow as tf
from PIL import Image

from src.data import CLASS_NAMES


def preprocess_image(image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/best_model.keras")
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    x = preprocess_image(args.image)

    probs = model.predict(x)[0]
    top_idx = np.argsort(probs)[::-1][:3]

    print(f"\nPredictions for {args.image}:")
    for idx in top_idx:
        print(f"  {CLASS_NAMES[idx]:<15s} {probs[idx]*100:5.2f}%")


if __name__ == "__main__":
    main()
