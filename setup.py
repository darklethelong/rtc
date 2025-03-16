from setuptools import setup, find_packages

setup(
    name="complaint_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "tqdm>=4.62.3",
        "nltk>=3.6.3",
        "transformers>=4.30.0",
        "tensorboard>=2.12.0"
    ],
    python_requires=">=3.8",
) 