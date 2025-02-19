#!/usr/bin/env python
# -*- encoding: utf8 -*-
# import glob
# import inspect
# import io
# import os

from setuptools import setup
import numpy
long_description = """
Source code: https://github.com/chenyk1990/eqcct""".strip() 


def read(*names, **kwargs):
    return io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")).read()

# from numpy.distutils.core import setup 
setup(
    name="eqcct",
    version="0.0.0.1",
    license='MIT License',
    description="EQCCT package is a production-ready EarthQuake detection and phase-picking method using the Compact Convolutional Transformer",
    long_description=long_description,
    author="eqcct developing team",
    author_email="chenyk2016@gmail.com",
    url="https://github.com/chenyk1990/eqcct",
    packages=['eqcct'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    keywords=[
        "seismology", "earthquake seismology", "exploration seismology", "array seismology", "science", "engineering", "artificial intelligence", "deep learning"
    ],
    install_requires=[
        "numpy", 
        "scipy", 
        "matplotlib",
        "tqdm",
        "obspy",
        "tensorflow==2.8.0",
        "pandas==1.3.5",
        "protobuf==3.20.1",
        "scikit-learn==1.0.2"
    ],
    python_requires='==3.7.16',
    extras_require={
        "docs": ["sphinx", "ipython", "runipy"]
    }
)
