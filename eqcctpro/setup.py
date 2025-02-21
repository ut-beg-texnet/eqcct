from setuptools import setup, find_packages
import os
import sys

# Ensure the user has the latest Python version
REQUIRED_PYTHON = (3, 10, 14)  # Change this to the minimum version you require

if sys.version_info < REQUIRED_PYTHON:
    sys.stderr.write(f"""
==========================
Unsupported Python version
==========================
This package requires Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}.{REQUIRED_PYTHON[2]} or higher.
You are using Python {sys.version_info.major}.{sys.version_info.minor}.
Please upgrade Python and try again.

Visit https://www.python.org/downloads/ to install the latest version.
""")
    sys.exit(1)

# Ensure TensorFlow and CUDA are initialized before running
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["NVIDIA_LOG_LEVEL"] = "ERROR"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="eqcctpro",
    version="0.4.2",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.3",
        "matplotlib==3.10.0",
        "obspy==1.4.1",
        "progress==1.6",
        "psutil==6.1.1",
        "ray==2.42.1",
        "schedule==1.2.2",
        "sdnotify==0.3.2",
        "tensorflow>=2.15,<2.19",  # Updated TensorFlow constraint
        "tensorflow-estimator>=2.15,<2.19",  # Updated TensorFlow Estimator constraint
        "tensorflow-io-gcs-filesystem==0.37.1",
        "tensorboard==2.15.2",
        "tensorboard-data-server==0.7.2",
        "silence-tensorflow==1.2.3",
        "scipy==1.15.1",
        "protobuf==4.25.6",
        "grpcio==1.70.0",
        "absl-py==2.1.0",
        "h5py==3.12.1",
        "pynvml==12.0.0",
    ],
    long_description=description,
    long_description_content_type="text/markdown"
)
