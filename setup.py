"""
Setup script for DecisionTree Strategy System.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="decisiontree-strategy",
    version="0.2.0",
    author="DecisionTree Team",
    description="A hybrid Decision Tree and Transformer-based stock trading strategy system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/birddograbbit/DecisionTree",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=5.0.0",
            "pytest-mock>=3.14.0",
            "pytest-benchmark>=5.0.0",
            "memory-profiler>=0.61.0",
        ],
        "transformer": [
            "torch>=2.0.0",
            "einops>=0.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "decision-tree-strategy=strategy_runner:main",
            "optimize-hyperparams=optimize_hyperparameters:main",
        ],
    },
)
