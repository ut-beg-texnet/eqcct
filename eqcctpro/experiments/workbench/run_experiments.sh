#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Prefer this repo's eqcctpro (must match options in test_cpus_and_gpus.py, e.g. conc_station_tasks_max_only).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Starting experiment suite..."

# --- Initial Workbench Tests ---
python3 test_cpus_and_gpus.py --mode gpu --arch ripper --model eqtransformer --cpu_start 0 --cpu_end 20

python3 test_cpus_and_gpus.py --mode gpu --arch ripper --model eqtransformer_non_conservative --cpu_start 0 --cpu_end 20

python3 test_cpus_and_gpus.py --mode gpu --arch modelactor --model eqtransformer_non_conservative --cpu_start 0 --cpu_end 20

# --- 41 and 46 trials - PhaseNetLight ---
python3 test_cpus_and_gpus.py --mode gpu --arch ripper --model phasenetlight --cpu_start 0 --cpu_end 41 --stations_cap 228 --single_cpu_block

python3 test_cpus_and_gpus.py --mode gpu --arch ripper --model phasenetlight --cpu_start 0 --cpu_end 46 --stations_cap 228 --single_cpu_block

python3 test_cpus_and_gpus.py --mode gpu --arch modelactor --model phasenetlight --cpu_start 0 --cpu_end 41 --stations_cap 228 --single_cpu_block

python3 test_cpus_and_gpus.py --mode gpu --arch modelactor --model phasenetlight --cpu_start 0 --cpu_end 46 --stations_cap 228 --single_cpu_block

# --- 41 and 46 trials - EQTransformer ---
python3 test_cpus_and_gpus.py --mode gpu --arch ripper --model eqtransformer --cpu_start 0 --cpu_end 41 --stations_cap 228 --single_cpu_block

python3 test_cpus_and_gpus.py --mode gpu --arch ripper --model eqtransformer --cpu_start 0 --cpu_end 46 --stations_cap 228 --single_cpu_block

python3 test_cpus_and_gpus.py --mode gpu --arch modelactor --model eqtransformer --cpu_start 0 --cpu_end 41 --stations_cap 228 --single_cpu_block

python3 test_cpus_and_gpus.py --mode gpu --arch modelactor --model eqtransformer --cpu_start 0 --cpu_end 46 --stations_cap 228 --single_cpu_block

# --- 41 and 46 trials - EQTransformer Non-Conservative ---
python3 test_cpus_and_gpus.py --mode gpu --arch ripper --model eqtransformer_non_conservative --cpu_start 0 --cpu_end 41 --stations_cap 228 --single_cpu_block

python3 test_cpus_and_gpus.py --mode gpu --arch ripper --model eqtransformer_non_conservative --cpu_start 0 --cpu_end 46 --stations_cap 228 --single_cpu_block

python3 test_cpus_and_gpus.py --mode gpu --arch modelactor --model eqtransformer_non_conservative --cpu_start 0 --cpu_end 41 --stations_cap 228 --single_cpu_block

python3 test_cpus_and_gpus.py --mode gpu --arch modelactor --model eqtransformer_non_conservative --cpu_start 0 --cpu_end 46 --stations_cap 228 --single_cpu_block

# # --- Coarse station-task concurrency (step 10) at 228 stations ---
# # Draft: supplemental Ripper search at 228 stations / 20 CPUs, concurrency stepped by 10,
# # folded with the primary 20% grid for reporting. This block extends the same coarse grid to
# # PhaseNet, PhaseNetLight, EQTransformer, EQTNC; CPU and GPU; Ripper and ModelActor; and
# # 20-, 41-, and 46-CPU single blocks (cpu_start 0, cpu_end N).
# for _mode in cpu gpu; do
#   for _cpu_end in 20 41 46; do
#     for _arch in ripper modelactor; do
#       for _model in eqcct phasenet phasenetlight eqtransformer eqtransformer_non_conservative; do
#         python3 test_cpus_and_gpus.py --mode "${_mode}" --arch "${_arch}" --model "${_model}" \
#           --cpu_start 0 --cpu_end "${_cpu_end}" --stations_cap 228 --single_cpu_block --conc_step 10
#       done
#     done
#   done
# done

# echo "All experiments completed successfully."