from setuptools import setup, find_packages

setup(
    name="predictor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "eqcctpro>=0.8.2",
        "numpy==1.26.4",
        "obspy>=1.4.1",
        "pandas>=2.0",
        "ray>=2.42",
        "silence-tensorflow>=1.2",
        "tensorflow>=2.20",
    ],
    author="Camilo Munoz",
    description="Predictor module for SCEQCCT",
    zip_safe=False,
)
