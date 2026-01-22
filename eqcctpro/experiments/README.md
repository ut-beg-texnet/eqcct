# Experiments Directory

This directory contains scripts for running the EQCCTPro pipeline and evaluating system performance.

## Subdirectories
- `main/`: Contains `run.py`, the primary entry point for executing seismic detection jobs in production.
- `workbench/`: Contains benchmarking and profiling tools:
    - `test_cpus_and_gpus.py`: Automated system evaluation to find optimal parallelization configurations.
    - `measure_model_memory_usage.py`: Profiles model memory footprints for accurate resource allocation.
