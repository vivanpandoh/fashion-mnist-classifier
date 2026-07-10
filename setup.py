from setuptools import setup, find_packages

setup(
    name="fashion-mnist-classifier",
    version="1.0.0",
    description="A production-quality CNN for Fashion-MNIST image classification in TensorFlow",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.15,<2.22",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "scikit-learn>=1.3",
        "pillow>=10.0",
    ],
)
