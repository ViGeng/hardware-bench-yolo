#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hardware-ml-benchamrk",
    version="0.1.4",
    author="zihan.deng",
    author_email="zihan.deng0517@gmail.com",
    description="A deep learning model hardware benchmarking tool",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Zihan-D/hardware-bench-yolo",
    packages=find_packages(),
    keywords=["deep learning", "machine learning", "hardware", "benchmark", "pytorch", "performance", "AI", "python"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        #"License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=[
    "torch>=1.9.0",
    "torchvision>=0.10.0",
    "numpy>=1.19.0",
    "psutil>=5.8.0",
    "Pillow>=8.0.0",
    "tqdm>=4.60.0",
    "opencv-python",
    "timm>=0.6.0",
    ]
,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "full": [
            "ultralytics>=8.0.0",
            "timm",
            "segmentation-models-pytorch>=0.3.0",
            "nvidia-ml-py3>=7.352.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hardware-ml-benchmark=hardware_ml_benchmark.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)