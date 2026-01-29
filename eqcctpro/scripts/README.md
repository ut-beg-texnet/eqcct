# Scripts Directory

Utility scripts for analysis and visualization of EQCCTPro outputs.

## Subdirectories

### `analysis/`
- **`analyze_trial_results_efficiency.py`**: A powerful analysis tool for benchmarking CSVs. It calculates throughput, analyzes memory utilization (requested vs. actual), and identifies performance bottlenecks. Supports single-file, batch, and ModelActor vs. Ripper comparison modes.

### `visualization/`
- **`visualize_trial_results.py`**: Generates interactive 3D Plotly visualizations of runtime, RAM, and VRAM usage across different hardware and concurrency configurations.
    - **`--model` Usage**: 
        - In **single-file mode**, this overrides the title of the plots.
        - In **`--compare` mode**, this is used as a search keyword to find the corresponding ModelActor and Ripper result directories (e.g., `--model phasenetlight` will match directories containing `phasenetlight_stead`).
    - **`--compare`**: Compares ModelActor vs. Ripper side-by-side. Requires `--model` and `--trial_type` (cpu/gpu).
- **`visualize_gpu_plotly.py`**: (Legacy) Generates interactive visualizations of GPU usage and detection results.
- **`gpu_usage_monitor.py`**: Tools for real-time monitoring of GPU resources during execution.
