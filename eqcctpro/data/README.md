# Data Directory

This directory contains the seismic waveform data used by EQCCTPro.

## Structure
- `230_stations_1_min_dt/`: A sample dataset containing 1-minute long mseed waveforms for 229 stations.
- `scripts/`: Contains `create_dataset.py`, which helps in downloading and organizing waveform data from FDSNWS sources into the format required by EQCCTPro.
- `archives/`: (Optional) Storage for compressed dataset files.

## Input Format
EQCCTPro expects waveforms to be organized by time-chunk subdirectories, each containing station-specific subdirectories with three-component mseed files.
