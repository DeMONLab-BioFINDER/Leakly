#!/usr/bin/env python3
"""Setuptools configuration for publishing Leakly on PyPI."""
from pathlib import Path
import re

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def read_readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def read_version() -> str:
    init_file = (ROOT / "leakly" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = ["\']([^"\']+)["\']', init_file, re.M)
    if not match:
        raise RuntimeError("Unable to find __version__ in leakly/__init__.py")
    return match.group(1)


setup(
    name="Leakly",
    version=read_version(),
    description=(
        "Leakage checks for machine-learning pipelines using permutation tests."
    ),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="DeMONLab-BioFINDER",
    license="MIT",
    url="https://github.com/DeMONLab-BioFINDER/Leakly",
    project_urls={
        "Source": "https://github.com/DeMONLab-BioFINDER/Leakly",
        "Issues": "https://github.com/DeMONLab-BioFINDER/Leakly/issues",
    },
    packages=find_packages(include=["leakly", "leakly.*"]),
    include_package_data=True,
    package_data={"leakly": ["*.yaml"]},
    python_requires=">=3.10",
    install_requires=[
        "joblib>=1.3",
        "matplotlib>=3.7",
        "numpy>=1.23",
        "pandas>=1.5",
        "PyYAML>=6.0",
        "scikit-learn>=1.6",
        "scipy>=1.9",
        "tqdm>=4.64",
    ],
    extras_require={
        "dev": [
            "build>=1.0",
            "pytest>=8.0",
            "pytest-cov>=5.0",
            "twine>=5.0",
            "wheel>=0.41",
        ],
        "notebook": [
            "ipykernel>=6.0",
            "ipywidgets>=8.0",
            "jupyter>=1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "data-leakage",
        "machine-learning",
        "permutation-test",
        "scikit-learn",
    ],
)
