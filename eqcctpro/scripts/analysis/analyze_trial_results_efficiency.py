"""
Efficiency and Memory Analysis Script for EQCCTPro Trial Results
================================================================

This script analyzes computational efficiency and memory costs from EvaluateSystem
trial CSV files. It supports both ModelActor and Ripper execution modes, for
both CPU and GPU trials.

Features:
- Throughput and resource efficiency analysis
- Memory utilization (Requested vs Actual)
- Diminishing returns analysis
- Performance lever identification
- Concurrency vs Runtime scaling
- ModelActor vs Ripper comparison
- Batch analysis of multiple result directories

Usage:
------
# Single file analysis
python analyze_trial_results_efficiency.py <csv_path> [--output_dir <dir>]

# Batch analysis of all results
python analyze_trial_results_efficiency.py --batch --results_root results/csv/ --output_dir analysis_results/

# Compare ModelActor vs Ripper for a model
python analyze_trial_results_efficiency.py --compare --model eqcct --trial_type cpu --results_root results/csv/

Examples:
--------
python analyze_trial_results_efficiency.py csv/eval_gpu_eqcct_modelactor/gpu_test_results.csv --output_dir analysis/
python analyze_trial_results_efficiency.py --batch --results_root results/csv/ --output_dir batch_analysis/
python analyze_trial_results_efficiency.py --compare --model phasenet_original --trial_type gpu --results_root results/csv/
"""

import pandas as pd
import numpy as np
import ast
import os
import argparse
import glob
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

# Set non-interactive backend for matplotlib
plt.switch_backend('Agg')


def parse_gpu_list(gpu_val):
    """Parse GPU list from various formats and return the count."""
    if pd.isna(gpu_val):
        return 0
    try:
        parsed = ast.literal_eval(str(gpu_val))
        if isinstance(parsed, list):
            return len(parsed)
        return int(parsed)
    except:
        try:
            return int(float(gpu_val))
        except:
            return 0


def detect_execution_mode(df):
    """
    Detect if trials use ModelActor or Ripper execution mode.
    
    Returns: 'modelactor', 'ripper', or 'mixed'
    """
    if 'N ModelActors' not in df.columns:
        return 'unknown'
    
    # Check for Ripper mode: N ModelActors = 0 and Actual Ripper Concurrent Tasks > 0
    n_actors = df['N ModelActors'].fillna(0)
    
    has_modelactors = (n_actors > 0).any()
    has_ripper = (n_actors == 0).any()
    
    # Also check Comments column for explicit RIPPER MODE indication
    if 'Comments' in df.columns:
        ripper_comments = df['Comments'].astype(str).str.contains('RIPPER MODE', na=False).any()
        if ripper_comments:
            has_ripper = True
    
    if has_modelactors and has_ripper:
        return 'mixed'
    elif has_ripper:
        return 'ripper'
    else:
        return 'modelactor'


def get_concurrency_column(df, execution_mode):
    """
    Get the appropriate concurrency column based on execution mode.
    
    For ModelActor: Use 'N ModelActors'
    For Ripper: Use 'Actual Ripper Concurrent Tasks' or 'Number of Concurrent Station Tasks'
    """
    if execution_mode == 'ripper':
        if 'Actual Ripper Concurrent Tasks' in df.columns:
            ripper_col = df['Actual Ripper Concurrent Tasks'].fillna(0)
            if (ripper_col > 0).any():
                return 'Actual Ripper Concurrent Tasks'
        return 'Number of Concurrent Station Tasks'
    else:
        return 'N ModelActors'


def analyze_efficiency(csv_path, output_dir=None, verbose=True, desired_runtime=None):
    """
    Analyzes computational efficiency and memory costs from trial CSV files.
    Calculates diminishing returns, identifies performance levers, and saves summaries.
    
    Supports both ModelActor and Ripper execution modes, and both CPU and GPU trials.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return None

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use a buffer to capture summary text
    summary_buffer = StringIO()

    def log(message):
        if verbose:
            print(message)
        summary_buffer.write(message + "\n")

    log(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # =========================================================================
    # COLUMN DEFINITIONS (Updated for CANONICAL_CSV_HEADER)
    # =========================================================================
    cpu_col = 'Number of CPUs Allocated for Ray to Use'
    task_col = 'Number of Concurrent Station Tasks'
    actor_col = 'N ModelActors'
    ripper_task_col = 'Actual Ripper Concurrent Tasks'
    station_col = 'Number of Stations Used'
    timechunk_col = 'Total Number of Timechunks'
    concurrent_tc_col = 'Concurrent Timechunks Used'
    
    # Timing columns
    total_trial_time_col = 'Total Trial Time (s)'           # Entire trial: setup + actor creation + processing
    actor_creation_time_col = 'Actor Creation Time (s)'     # Time to spin up ModelActors (empty for Ripper)
    avg_model_load_time_col = 'Avg Model Load Time (s)'     # Average model load time per task (Ripper only)
    waveform_proc_time_col = 'Waveform Processing Time (s)' # Average time to load waveforms per task
    picker_runtime_col = 'Total Run time for Picker (s)'    # Total time for all task processing
    runtime_col = total_trial_time_col  # Default runtime for main plots uses total trial time
    
    # Memory columns (PID-isolated tracking)
    req_vram_actor_col = 'Requested VRAM per Actor (MB)'
    req_ram_actor_col = 'Requested RAM per Actor (MB)'
    total_req_vram_col = 'Total Requested VRAM (MB)'
    total_req_ram_col = 'Total Requested RAM (MB)'
    actual_vram_col = 'Process Tree VRAM (MB)'
    actual_ram_col = 'Process Tree RAM (MB)'
    peak_ram_col = 'Peak RAM (MB)'
    ram_growth_col = 'RAM Growth (MB)'
    vram_overhead_col = 'VRAM Overhead (MB)'
    ram_overhead_col = 'RAM Overhead (MB)'
    vram_util_col = 'VRAM Utilization (%)'
    ram_util_col = 'RAM Utilization (%)'
    num_workers_col = 'Num Worker Processes'

    # =========================================================================
    # DATA PREPROCESSING
    # =========================================================================
    # Automatic Model and Trial Type Detection
    model_name = df['Model Used'].iloc[0] if 'Model Used' in df.columns else "Unknown Model"
    
    df['GPU Count'] = df['GPUs Used'].apply(parse_gpu_list)
    is_gpu_trial = df['GPU Count'].max() > 0
    trial_type = "GPU-based" if is_gpu_trial else "CPU-based"
    
    # Detect execution mode (ModelActor vs Ripper) - MUST BE DONE BEFORE DISPLAY NAME DEFINITIONS
    execution_mode = detect_execution_mode(df)
    concurrency_col = get_concurrency_column(df, execution_mode)

    # Rename for display in correlation matrix and summary
    display_runtime_col = 'Total Trial Runtime (s)'
    display_cpu_col = 'Number of CPUs Used'
    display_gpu_col = 'Number of GPUs Used'
    display_picking_col = 'Total Picking Time (s)'
    if execution_mode == 'modelactor':
        display_concurrency_col = "Number of ModelActor's Created"
        display_setup_col = 'Actor Creation Time (s)'
    else:
        display_concurrency_col = 'Ripper Concurrent Tasks'
        display_setup_col = 'Avg Model Load Time (s)'
    
    display_waveform_col = 'Waveform Processing Time (s)'
    display_ram_col = 'Process Tree RAM (MB)'
    display_vram_col = 'Process Tree VRAM (MB)'
    display_stations_col = 'Number of Stations Used'
    display_throughput_col = 'Throughput (Stations/s)'
    
    log(f"\n{'='*70}")
    log(f"EQCCTPRO EFFICIENCY ANALYSIS")
    log(f"{'='*70}")
    log(f"Detected Model: {model_name}")
    log(f"Detected Trial Type: {trial_type}")
    log(f"Detected Execution Mode: {execution_mode.upper()}")
    log(f"Concurrency Column: {concurrency_col}")
    log(f"Total Trials in File: {len(df)}")

    # Add CSV description
    csv_name = f"efficiency_analysis_results_{execution_mode}.csv"
    log(f"\nNote: The file '{csv_name}' contains the full processed trial data.")
    log("It includes derived metrics such as Throughput (Stations/s), Resource Efficiency (Throughput per CPU/GPU),")
    log("and Memory Overhead (Process Tree RAM - Requested RAM), enabling granular analysis of each trial.")
    
    # Filter successful trials only
    df_success = df[df['Trial Success'] == 1.0].copy()
    log(f"Successful Trials: {len(df_success)}")
    
    if len(df_success) == 0:
        log("Error: No successful trials found. Exiting.")
        return None
    
    # Convert numeric columns
    numeric_cols = [cpu_col, task_col, actor_col, ripper_task_col, station_col,
                    total_trial_time_col, actor_creation_time_col, avg_model_load_time_col,
                    waveform_proc_time_col, picker_runtime_col,
                    total_req_vram_col, total_req_ram_col, actual_vram_col, actual_ram_col,
                    ram_growth_col, vram_util_col, ram_util_col, num_workers_col]
    
    for col in numeric_cols:
        if col in df_success.columns:
            df_success[col] = pd.to_numeric(df_success[col], errors='coerce')
    
    # Fill missing VRAM values with 0 for CPU trials
    if not is_gpu_trial:
        for col in [actual_vram_col, total_req_vram_col, vram_overhead_col]:
            if col in df_success.columns:
                df_success[col] = df_success[col].fillna(0)

    # Create unified concurrency column for analysis
    df_success['Effective Concurrency'] = df_success[concurrency_col].fillna(1)

    # =========================================================================
    # DERIVED METRICS
    # =========================================================================
    # 1. Throughput (Stations per second) - based on Total Trial Time
    df_success['Throughput (Stations/s)'] = df_success[station_col] / df_success[runtime_col].replace(0, np.nan)
    
    # 1b. Picker Throughput - based on Total Run time for Picker (pure processing time)
    if picker_runtime_col in df_success.columns:
        df_success['Picker Throughput (Stations/s)'] = df_success[station_col] / df_success[picker_runtime_col].replace(0, np.nan)
    
    # 2. Resource Efficiency
    df_success['Throughput/CPU'] = df_success['Throughput (Stations/s)'] / df_success[cpu_col].replace(0, np.nan)
    df_success['Throughput/Concurrency'] = df_success['Throughput (Stations/s)'] / df_success['Effective Concurrency'].replace(0, np.nan)
    
    if is_gpu_trial:
        df_success['Throughput/GPU'] = df_success.apply(
            lambda row: row['Throughput (Stations/s)'] / row['GPU Count'] if row['GPU Count'] > 0 else np.nan, axis=1
        )

    # 3. Memory Efficiency Metrics
    df_success['RAM per Station (MB)'] = df_success[actual_ram_col] / df_success[station_col].replace(0, np.nan)
    df_success['RAM per Concurrency Unit (MB)'] = df_success[actual_ram_col] / df_success['Effective Concurrency'].replace(0, np.nan)
    
    if is_gpu_trial:
        df_success['VRAM per Station (MB)'] = df_success[actual_vram_col] / df_success[station_col].replace(0, np.nan)
        df_success['VRAM per Concurrency Unit (MB)'] = df_success[actual_vram_col] / df_success['Effective Concurrency'].replace(0, np.nan)

    # 4. Memory Overhead Analysis
    if actual_ram_col in df_success.columns and total_req_ram_col in df_success.columns:
        df_success['Calculated RAM Overhead (MB)'] = df_success[actual_ram_col] - df_success[total_req_ram_col]
    
    if is_gpu_trial and actual_vram_col in df_success.columns and total_req_vram_col in df_success.columns:
        df_success['Calculated VRAM Overhead (MB)'] = df_success[actual_vram_col] - df_success[total_req_vram_col]

    # 5. Resource Cost Score
    gpu_weight = 10
    df_success['Resource Cost Score'] = df_success[cpu_col] + (df_success['GPU Count'] * gpu_weight)
    df_success['Throughput per Cost Unit'] = df_success['Throughput (Stations/s)'] / df_success['Resource Cost Score']

    # =========================================================================
    # ANALYSIS OUTPUTS
    # =========================================================================
    log("\n" + "="*70)
    log(f"NUMERICAL ANALYSIS SUMMARY: {model_name} ({trial_type}, {execution_mode.upper()})")
    log("="*70)

    log("\n--- Analysis Formulas ---")
    log(f"1. Throughput (Stations/s) = ['{station_col}'] / ['{display_runtime_col}']")
    log("2. Gain % (Diminishing Returns) = ((Current Throughput - Previous Throughput) / Previous Throughput) * 100")
    log(f"3. {display_concurrency_col} = ['{concurrency_col}']")
    log(f"   (Actual actors spawned for ModelActor mode, or actual/requested tasks for Ripper mode)")
    log("4. Resource Cost Score = CPUs + (GPUs * 10)")
    log("5. RAM Overhead = Process Tree RAM - Total Requested RAM")
    log("6. RAM Utilization = (Process Tree RAM / Total Requested RAM) * 100")

    log("\n--- Key Timing Metrics ---")
    log(f"Average {display_runtime_col}: {df_success[total_trial_time_col].mean():.2f} s")
    if picker_runtime_col in df_success.columns:
        log(f"Average Total Picking Time:  {df_success[picker_runtime_col].mean():.2f} s")
    if execution_mode == 'modelactor' and actor_creation_time_col in df_success.columns:
        log(f"Average Actor Creation Time: {df_success[actor_creation_time_col].mean():.2f} s")
    elif execution_mode == 'ripper' and avg_model_load_time_col in df_success.columns:
        log(f"Average Model Load Time:     {df_success[avg_model_load_time_col].mean():.4f} s")
    if waveform_proc_time_col in df_success.columns:
        log(f"Average Waveform Proc Time:  {df_success[waveform_proc_time_col].mean():.4f} s")

    log("\n--- Interpretation Guide ---")
    log(f"A. Correlation Matrix (correlation_matrix_{execution_mode}.png):")
    log(f"   - Values range from -1.0 to +1.0.")
    log(f"   - Positive Correlation (+): As the resource increases, runtime INCREASES (e.g., more stations = more work).")
    log(f"   - Negative Correlation (-): As the resource increases, runtime DECREASES (e.g., more CPUs = faster processing).")
    log(f"   - Values near +/- 1.0 indicate strong relationships; near 0.0 indicate no relationship.")

    # =========================================================================
    # MEMORY ANALYSIS: REQUESTED vs ACTUAL
    # =========================================================================
    log("\n" + "-"*70)
    log("MEMORY ANALYSIS: REQUESTED vs ACTUAL")
    log("-"*70)
    
    # RAM Analysis
    if actual_ram_col in df_success.columns and total_req_ram_col in df_success.columns:
        valid_ram = df_success[[total_req_ram_col, actual_ram_col, ram_util_col]].dropna()
        if len(valid_ram) > 0:
            avg_req_ram = valid_ram[total_req_ram_col].mean()
            avg_actual_ram = valid_ram[actual_ram_col].mean()
            avg_ram_util = valid_ram[ram_util_col].mean()
            
            log(f"\nRAM Usage Summary:")
            log(f"  Average Requested RAM:     {avg_req_ram:,.1f} MB")
            log(f"  Average Actual RAM:        {avg_actual_ram:,.1f} MB")
            log(f"  Average RAM Utilization:   {avg_ram_util:.1f}%")
            log(f"  Average RAM Overhead:      {avg_actual_ram - avg_req_ram:+,.1f} MB")
            
            if avg_ram_util < 100:
                log(f"  Interpretation: Actual usage is LOWER than requested (efficient memory sharing)")
            else:
                log(f"  Interpretation: Actual usage is HIGHER than requested (framework overhead)")
    
    # VRAM Analysis (GPU trials only)
    if is_gpu_trial and actual_vram_col in df_success.columns and total_req_vram_col in df_success.columns:
        valid_vram = df_success[[total_req_vram_col, actual_vram_col, vram_util_col]].dropna()
        valid_vram = valid_vram[valid_vram[actual_vram_col] > 0]
        if len(valid_vram) > 0:
            avg_req_vram = valid_vram[total_req_vram_col].mean()
            avg_actual_vram = valid_vram[actual_vram_col].mean()
            avg_vram_util = valid_vram[vram_util_col].mean()
            
            log(f"\nVRAM Usage Summary:")
            log(f"  Average Requested VRAM:    {avg_req_vram:,.1f} MB")
            log(f"  Average Actual VRAM:       {avg_actual_vram:,.1f} MB")
            log(f"  Average VRAM Utilization:  {avg_vram_util:.1f}%")
            log(f"  Average VRAM Overhead:     {avg_actual_vram - avg_req_vram:+,.1f} MB")

    # Memory Growth Analysis
    if ram_growth_col in df_success.columns:
        valid_growth = df_success[ram_growth_col].dropna()
        if len(valid_growth) > 0:
            log(f"\nRAM Growth Analysis (per trial):")
            log(f"  Mean RAM Growth:    {valid_growth.mean():+,.1f} MB")
            log(f"  Median RAM Growth:  {valid_growth.median():+,.1f} MB")
            log(f"  Max RAM Growth:     {valid_growth.max():+,.1f} MB")
            log(f"  Min RAM Growth:     {valid_growth.min():+,.1f} MB")

    # =========================================================================
    # PERFORMANCE LEVERS ANALYSIS
    # =========================================================================
    log("\n" + "-"*70)
    log(f"PERFORMANCE LEVERS ({display_runtime_col} Correlations)")
    log("-"*70)
    
    def calculate_impact(lever_col):
        if lever_col not in df_success.columns:
            return np.nan
        # Avoid correlation calculation if either column has zero variance
        if df_success[lever_col].nunique() <= 1 or df_success[runtime_col].nunique() <= 1:
            return np.nan
        return df_success[lever_col].corr(df_success[runtime_col])

    cpu_impact = calculate_impact(cpu_col)
    task_impact = calculate_impact(task_col)
    concurrency_impact = calculate_impact('Effective Concurrency')
    station_impact = calculate_impact(station_col)
    
    log(f"\nCorrelation with {display_runtime_col} (negative = faster with more):")
    log(f"  1. {display_stations_col}:           {station_impact:+.3f} (expected: positive)")
    log(f"  2. {display_cpu_col}:               {cpu_impact:+.3f}")
    log(f"  3. {display_concurrency_col}:        {concurrency_impact:+.3f}")
    
    # Optional mode-specific impacts
    if execution_mode == 'ripper':
        if ripper_task_col in df_success.columns:
            ripper_impact = calculate_impact(ripper_task_col)
            log(f"  4. Actual Ripper Tasks:          {ripper_impact:+.3f}")
    
    if is_gpu_trial:
        gpu_impact = calculate_impact('GPU Count')
        log(f"  5. {display_gpu_col}:                    {gpu_impact:+.3f}")
    
    log("\n  Interpretation:")
    log(f"  - Negative correlation: More of this resource = faster {display_runtime_col}")
    log("  - Positive correlation: More of this = slower (e.g., more stations = more work)")
    log("  - Close to 0: Weak relationship")

    # =========================================================================
    # CONCURRENCY VS RUNTIME ANALYSIS
    # =========================================================================
    log("\n" + "-"*70)
    log(f"CONCURRENCY VS RUNTIME ANALYSIS ({execution_mode.upper()} MODE)")
    log("-"*70)
    
    # Analyze how concurrency affects runtime
    concurrency_groups = df_success.groupby('Effective Concurrency').agg({
        runtime_col: ['mean', 'std', 'count'],
        station_col: 'mean',
        'Throughput (Stations/s)': 'mean'
    }).round(2)
    concurrency_groups.columns = [f'Avg {display_runtime_col}', 'Std Runtime', 'Count', 'Avg Stations', 'Avg Throughput']
    concurrency_groups = concurrency_groups.reset_index()
    
    log(f"\nPerformance by {display_concurrency_col}:")
    log(concurrency_groups.to_string(index=False))
    
    # Analyze runtime scaling with stations
    log(f"\n\n{display_runtime_col} Scaling by Workload Size:")
    station_quartiles = df_success[station_col].quantile([0.25, 0.5, 0.75]).values
    
    for q, label in zip([0.25, 0.5, 0.75], ['Small', 'Medium', 'Large']):
        threshold = df_success[station_col].quantile(q)
        subset = df_success[df_success[station_col] <= threshold]
        if len(subset) > 0:
            avg_runtime = subset[runtime_col].mean()
            avg_concurrency = subset['Effective Concurrency'].mean()
            avg_throughput = subset['Throughput (Stations/s)'].mean()
            log(f"  {label} workloads (≤{threshold:.0f} stations): Avg {display_runtime_col}={avg_runtime:.2f}s, Avg {display_concurrency_col}={avg_concurrency:.1f}, Throughput={avg_throughput:.2f} st/s")

    # =========================================================================
    # SWEET SPOT ANALYSIS
    # =========================================================================
    log("\n" + "-"*70)
    log("OPTIMAL CONFIGURATION ('SWEET SPOT')")
    log("-"*70)
    
    best_cost_efficiency = df_success.loc[df_success['Throughput per Cost Unit'].idxmax()]
    log(f"\nBest Throughput per Resource Cost:")
    log(f"  Configuration: {best_cost_efficiency[cpu_col]:.0f} {display_cpu_col}, {best_cost_efficiency['GPU Count']:.0f} {display_gpu_col}, {best_cost_efficiency[task_col]:.0f} Concurrent Tasks")
    log(f"  {display_concurrency_col}: {best_cost_efficiency['Effective Concurrency']:.0f}")
    log(f"  Stations Processed: {best_cost_efficiency[station_col]:.0f}")
    log(f"  Throughput: {best_cost_efficiency['Throughput (Stations/s)']:.3f} stations/s")
    log(f"  {display_runtime_col}: {best_cost_efficiency[runtime_col]:.2f}s")
    log(f"  {display_ram_col}: {best_cost_efficiency[actual_ram_col]:.1f} MB")
    if is_gpu_trial and actual_vram_col in df_success.columns:
        log(f"  {display_vram_col}: {best_cost_efficiency[actual_vram_col]:.1f} MB")

    # Fastest overall configuration
    fastest = df_success.loc[df_success[runtime_col].idxmin()]
    log(f"\nFastest Overall Configuration:")
    log(f"  Configuration: {fastest[cpu_col]:.0f} {display_cpu_col}, {fastest['GPU Count']:.0f} {display_gpu_col}, {fastest[task_col]:.0f} Concurrent Tasks")
    log(f"  {display_concurrency_col}: {fastest['Effective Concurrency']:.0f}")
    log(f"  Stations Processed: {fastest[station_col]:.0f}")
    log(f"  {display_runtime_col}: {fastest[runtime_col]:.2f}s")

    # =========================================================================
    # DIMINISHING RETURNS ANALYSIS
    # =========================================================================
    log("\n" + "-"*70)
    log("DIMINISHING RETURNS ANALYSIS")
    log("-"*70)
    
    # Analyze diminishing returns for concurrency scaling
    log(f"\n--- Scaling by {display_concurrency_col} ({execution_mode.upper()}) ---")
    median_stations = df_success[station_col].median()
    tolerance = median_stations * 0.2
    
    similar_workload = df_success[
        (df_success[station_col] >= median_stations - tolerance) &
        (df_success[station_col] <= median_stations + tolerance)
    ]
    
    if len(similar_workload) > 3:
        concurrency_scaling = similar_workload.groupby('Effective Concurrency')['Throughput (Stations/s)'].mean().sort_index()
        
        prev_val = None
        for n_conc, val in concurrency_scaling.items():
            if prev_val is not None:
                gain = (val - prev_val) / prev_val * 100 if prev_val > 0 else 0
                log(f"  {int(n_conc-1)} -> {int(n_conc)} units: {gain:+.1f}% throughput gain")
            prev_val = val
    else:
        log("  Insufficient data for concurrency scaling analysis at median workload")

    # CPU scaling analysis
    log("\n--- Scaling by CPU Allocation ---")
    base_gpu = 1 if is_gpu_trial else 0
    cpu_scaling = df_success[df_success['GPU Count'] == base_gpu].groupby(cpu_col)['Throughput (Stations/s)'].mean()
    
    prev_val = None
    for n_cpu, val in cpu_scaling.items():
        if prev_val is not None:
            gain = (val - prev_val) / prev_val * 100 if prev_val > 0 else 0
            log(f"  {int(n_cpu-1)} -> {int(n_cpu)} CPUs: {gain:+.1f}% throughput gain")
        prev_val = val

    # =========================================================================
    # CORRELATION ANALYSIS & VISUALIZATION
    # =========================================================================
    log("\n" + "-"*70)
    log("CORRELATION MATRIX")
    log("-"*70)
    
    # Prepare data for correlation with renamed columns for display and enforced order
    df_corr = df_success.copy()
    
    # Define mapping for renaming
    rename_map = {
        total_trial_time_col: display_runtime_col,
        'Effective Concurrency': display_concurrency_col,
        cpu_col: display_cpu_col,
        'GPU Count': display_gpu_col,
        picker_runtime_col: display_picking_col,
        waveform_proc_time_col: display_waveform_col,
        actual_ram_col: display_ram_col,
        actual_vram_col: display_vram_col,
        station_col: display_stations_col
    }
    
    # Add mode-specific setup timing to rename map
    if execution_mode == 'modelactor':
        rename_map[actor_creation_time_col] = display_setup_col
    else:
        rename_map[avg_model_load_time_col] = display_setup_col
        
    df_corr = df_corr.rename(columns=rename_map)
    
    # Determine if this is a GPU trial (has non-zero GPU usage)
    is_gpu_trial = trial_type.lower() == 'gpu' or (display_gpu_col in df_corr.columns and df_corr[display_gpu_col].sum() > 0)
    
    # Define desired order: Hardware -> Timing -> Memory -> Other
    # For CPU trials, exclude GPU count and VRAM (not relevant)
    ordered_cols = [
        # Hardware - always include CPUs, only include GPUs for GPU trials
        display_cpu_col,
    ]
    if is_gpu_trial:
        ordered_cols.append(display_gpu_col)
    ordered_cols.append(display_concurrency_col)
    
    # Timing
    ordered_cols.extend([
        display_runtime_col,
        display_picking_col,
        display_setup_col,
        display_waveform_col,
    ])
    
    # Memory - always include RAM, only include VRAM for GPU trials
    ordered_cols.append(display_ram_col)
    if is_gpu_trial:
        ordered_cols.append(display_vram_col)
    
    # Other
    ordered_cols.extend([
        display_stations_col,
        display_throughput_col
    ])
    
    # Filter to only include columns that exist in the renamed dataframe
    existing_cols = [c for c in ordered_cols if c in df_corr.columns]
    
    # Track columns with constant values (zero variance) - they will show NaN correlations
    constant_cols = [col for col in existing_cols if df_corr[col].nunique() <= 1]
    
    # Include all columns (even constant ones) - constant columns will show NaN correlations
    # This allows the user to see the full matrix structure even before all trial data is collected
    corr_cols = existing_cols
    
    # Log which columns have constant values
    if constant_cols:
        log(f"\nNOTE: The following columns currently have constant values (will show NaN correlations):")
        for col in constant_cols:
            unique_val = df_corr[col].iloc[0] if len(df_corr) > 0 else "N/A"
            log(f"  - {col}: all values = {unique_val}")
        log("  (Correlations will become meaningful once trial data has variation in these columns)")
    
    if not is_gpu_trial:
        log(f"\nNOTE: GPU-related columns (Number of GPUs Used, Process Tree VRAM) excluded for CPU trials.")
    
    if len(corr_cols) > 1:
        corr_matrix = df_corr[corr_cols].corr()
        
        log(f"\nKey correlations with {display_runtime_col}:")
        for col in corr_cols:
            if col != display_runtime_col and display_runtime_col in corr_matrix.columns:
                corr_val = corr_matrix.loc[col, display_runtime_col]
                if pd.notna(corr_val):
                    log(f"  {col}: {corr_val:+.3f}")
                else:
                    log(f"  {col}: NaN (constant values)")
    else:
        log("\nInsufficient columns to calculate correlation matrix.")
        corr_matrix = pd.DataFrame()

    # Save Correlation Matrix as Plot
    if not corr_matrix.empty:
        # Dynamic figure size based on number of columns
        fig_size = max(10, len(corr_cols) * 1.2)
        plt.figure(figsize=(fig_size, fig_size * 0.85))
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5,
                    center=0, vmin=-1, vmax=1)
        
        # Build title
        title = f"Correlation Matrix: {model_name} ({trial_type}, {execution_mode.upper()})\n(Ordered by Hardware, Timing, Memory)"
        if constant_cols:
            constant_names = ', '.join(constant_cols)
            title += f"\nConstant values (NaN): {constant_names}"
        
        plt.title(title, fontsize=10)
        plt.tight_layout()
        
        plot_name = f"correlation_matrix_{execution_mode}.png"
        plot_path = os.path.join(output_dir, plot_name) if output_dir else plot_name
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log(f"\nCorrelation plot saved to {plot_path}")
    else:
        log("\nSkipping correlation plot due to insufficient columns.")

    # =========================================================================
    # ADDITIONAL VISUALIZATIONS
    # =========================================================================
    
    # 1. Total Trial Time vs Stations by Concurrency
    plt.figure(figsize=(12, 8))
    
    # Use a scatter plot with a colorbar for rainbow scale
    scatter = plt.scatter(df_success[station_col], df_success[runtime_col], 
                         c=df_success['Effective Concurrency'], 
                         cmap='rainbow', alpha=0.8, s=60, edgecolors='k', linewidths=0.5)
    
    # Add colorbar (legend)
    conc_min = df_success['Effective Concurrency'].min()
    conc_max = df_success['Effective Concurrency'].max()
    cbar_ticks = np.arange(np.floor(conc_min/10)*10, np.ceil(conc_max/10)*10 + 11, 10)
    # Ensure at least min and max are somewhat represented if the range is small
    if len(cbar_ticks) <= 1:
        cbar_ticks = [conc_min, conc_max]
        
    cbar = plt.colorbar(scatter, ticks=cbar_ticks)
    conc_label = "N Model Actors Used" if execution_mode == "modelactor" else "Concurrent Tasks Used"
    cbar.set_label(conc_label, fontsize=12)
    
    # Add desired runtime horizontal line if provided
    if desired_runtime is not None:
        plt.axhline(y=desired_runtime, color='red', linestyle='--', linewidth=2, 
                    label=f'Desired Runtime ({desired_runtime}s)')
        plt.legend(loc='upper left')

    # Set labels
    plt.xlabel('Total Number of Stations to Process', fontsize=12)
    plt.ylabel('Total Trial Time (s)', fontsize=12)
    
    # X-axis ticks: step size of 10
    min_stations = int(df_success[station_col].min())
    max_stations = int(df_success[station_col].max())
    # Start from a multiple of 10 if possible, or just the min
    x_start = (min_stations // 10) * 10
    plt.xticks(np.arange(x_start, max_stations + 11, 10))
    
    # Y-axis ticks: step size of 5, from 0 to max rounded up to nearest 5
    max_runtime = df_success[runtime_col].max()
    if desired_runtime is not None:
        max_runtime = max(max_runtime, desired_runtime)
    y_max_rounded = int(np.ceil(max_runtime / 5.0) * 5)
    plt.yticks(np.arange(0, y_max_rounded + 6, 5))
    plt.ylim(0, y_max_rounded + 2) # Small buffer
    
    plt.title(f'Total Trial Time vs Workload Size by Concurrency\n{model_name} ({trial_type}, {execution_mode.upper()})', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plot_name = f"total_trial_time_vs_stations_by_concurrency_{execution_mode}.png"
    plot_path = os.path.join(output_dir, plot_name) if output_dir else plot_name
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log(f"Total Trial Time vs Stations plot saved to {plot_path}")

    # 1b. Picker Runtime vs Stations by Concurrency (pure processing time)
    if picker_runtime_col in df_success.columns and df_success[picker_runtime_col].notna().any():
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(df_success[station_col], df_success[picker_runtime_col], 
                             c=df_success['Effective Concurrency'], 
                             cmap='rainbow', alpha=0.8, s=60, edgecolors='k', linewidths=0.5)
        
        cbar = plt.colorbar(scatter, ticks=cbar_ticks)
        cbar.set_label(conc_label, fontsize=12)
        
        if desired_runtime is not None:
            plt.axhline(y=desired_runtime, color='red', linestyle='--', linewidth=2, 
                        label=f'Desired Runtime ({desired_runtime}s)')
            plt.legend(loc='upper left')

        plt.xlabel('Total Number of Stations to Process', fontsize=12)
        plt.ylabel('Picker Runtime (s)', fontsize=12)
        
        plt.xticks(np.arange(x_start, max_stations + 11, 10))
        
        max_picker_runtime = df_success[picker_runtime_col].max()
        if desired_runtime is not None:
            max_picker_runtime = max(max_picker_runtime, desired_runtime)
        y_max_rounded = int(np.ceil(max_picker_runtime / 5.0) * 5)
        plt.yticks(np.arange(0, y_max_rounded + 6, 5))
        plt.ylim(0, y_max_rounded + 2)
        
        plt.title(f'Picker Runtime vs Workload Size by Concurrency\n{model_name} ({trial_type}, {execution_mode.upper()})', fontsize=14)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plot_name = f"picker_runtime_vs_stations_by_concurrency_{execution_mode}.png"
        plot_path = os.path.join(output_dir, plot_name) if output_dir else plot_name
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log(f"Picker Runtime vs Stations plot saved to {plot_path}")

    # 2. Requested vs Actual RAM scatter
    if actual_ram_col in df_success.columns and total_req_ram_col in df_success.columns:
        plt.figure(figsize=(10, 8))
        valid_data = df_success[[total_req_ram_col, actual_ram_col, 'Effective Concurrency']].dropna()
        
        scatter = plt.scatter(valid_data[total_req_ram_col], valid_data[actual_ram_col], 
                             c=valid_data['Effective Concurrency'], cmap='rainbow', alpha=0.7, s=50,
                             edgecolors='k', linewidths=0.5)
        
        max_val = max(valid_data[total_req_ram_col].max(), valid_data[actual_ram_col].max())
        plt.plot([0, max_val], [0, max_val], 'r--', label='Estimated Prediction RAM Cost')
        
        conc_min = valid_data['Effective Concurrency'].min()
        conc_max = valid_data['Effective Concurrency'].max()
        cbar_ticks = np.arange(np.floor(conc_min/10)*10, np.ceil(conc_max/10)*10 + 11, 10)
        if len(cbar_ticks) <= 1:
            cbar_ticks = [conc_min, conc_max]
            
        conc_label = 'N ModelActors' if execution_mode == 'modelactor' else 'Concurrent Tasks'
        plt.colorbar(scatter, label=conc_label, ticks=cbar_ticks)
        plt.xlabel('Total Requested RAM (MB)')
        plt.ylabel('Process Tree RAM (MB)')
        plt.title(f'Requested vs Actual RAM\n{model_name} ({trial_type}, {execution_mode.upper()})')
        
        # Add desired runtime line if provided
        if desired_runtime is not None:
            plt.axhline(y=desired_runtime, color='red', linestyle='--', linewidth=2, 
                        label=f'Desired Runtime ({desired_runtime}s)')
            plt.legend(loc='upper left')
        else:
            plt.legend(loc='upper left')
            
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_name = f"requested_vs_actual_ram_{execution_mode}.png"
        plot_path = os.path.join(output_dir, plot_name) if output_dir else plot_name
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log(f"Requested vs Actual RAM plot saved to {plot_path}")

    # =========================================================================
    # AGGREGATED STATISTICS
    # =========================================================================
    log("\n" + "-"*70)
    log("AGGREGATED METRICS (Averages by Hardware Configuration)")
    log("-"*70)
    
    agg_cols = {
        runtime_col: 'mean',
        'Throughput (Stations/s)': 'mean',
        'Effective Concurrency': 'mean',
        actual_ram_col: 'mean',
    }
    if is_gpu_trial and actual_vram_col in df_success.columns:
        agg_cols[actual_vram_col] = 'mean'
    
    agg_stats = df_success.groupby([cpu_col, 'GPU Count']).agg(agg_cols).round(2)
    col_names = [display_cpu_col, display_gpu_col, f'Avg {display_runtime_col}', 'Avg Throughput', f'Avg {display_concurrency_col}', f'Avg {display_ram_col}']
    if is_gpu_trial:
        col_names.insert(6, f'Avg {display_vram_col}')
    
    # Re-order agg_stats columns to match col_names after reset_index
    agg_stats = agg_stats.reset_index()
    agg_stats.columns = col_names
    log(agg_stats.to_string(index=False))

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    csv_name = f"efficiency_analysis_results_{execution_mode}.csv"
    txt_name = f"efficiency_summary_{execution_mode}.txt"
    
    if output_dir:
        csv_path_out = os.path.join(output_dir, csv_name)
        txt_path_out = os.path.join(output_dir, txt_name)
    else:
        csv_path_out = csv_name
        txt_path_out = txt_name

    df_success.to_csv(csv_path_out, index=False)
    with open(txt_path_out, "w") as f:
        f.write(summary_buffer.getvalue())
    
    log(f"\nDetailed CSV saved to {csv_path_out}")
    log(f"Summary TXT saved to {txt_path_out}")
    log(f"\nAnalysis complete!")
    
    return df_success


def compare_modelactor_vs_ripper(modelactor_csv, ripper_csv, output_dir=None, desired_runtime=None):
    """
    Compare ModelActor and Ripper execution modes side by side.
    """
    if not os.path.exists(modelactor_csv):
        print(f"Error: ModelActor CSV not found: {modelactor_csv}")
        return
    if not os.path.exists(ripper_csv):
        print(f"Error: Ripper CSV not found: {ripper_csv}")
        return
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n{'='*70}")
    print("MODELACTOR VS RIPPER COMPARISON")
    print(f"{'='*70}")
    
    # Load data
    df_ma = pd.read_csv(modelactor_csv)
    df_rp = pd.read_csv(ripper_csv)
    
    # Filter successful trials
    df_ma = df_ma[df_ma['Trial Success'] == 1.0].copy()
    df_rp = df_rp[df_rp['Trial Success'] == 1.0].copy()
    
    # Get model info
    model_name = df_ma['Model Used'].iloc[0] if 'Model Used' in df_ma.columns else "Unknown"
    df_ma['GPU Count'] = df_ma['GPUs Used'].apply(parse_gpu_list)
    df_rp['GPU Count'] = df_rp['GPUs Used'].apply(parse_gpu_list)
    is_gpu = df_ma['GPU Count'].max() > 0
    trial_type = "GPU" if is_gpu else "CPU"
    
    print(f"Model: {model_name}")
    print(f"Trial Type: {trial_type}")
    print(f"ModelActor Trials: {len(df_ma)}")
    print(f"Ripper Trials: {len(df_rp)}")
    
    # Column definitions
    total_trial_time_col = 'Total Trial Time (s)'
    picker_runtime_col = 'Total Run time for Picker (s)'
    runtime_col = total_trial_time_col  # Use total trial time for main comparisons
    station_col = 'Number of Stations Used'
    actual_ram_col = 'Process Tree RAM (MB)'
    actual_vram_col = 'Process Tree VRAM (MB)'
    task_col = 'Number of Concurrent Station Tasks'
    actor_col = 'N ModelActors'
    actor_creation_time_col = 'Actor Creation Time (s)'
    avg_model_load_time_col = 'Avg Model Load Time (s)'
    waveform_proc_time_col = 'Waveform Processing Time (s)'
    
    # Calculate throughput (using total trial time)
    df_ma['Throughput (Stations/s)'] = df_ma[station_col] / df_ma[runtime_col].replace(0, np.nan)
    df_rp['Throughput (Stations/s)'] = df_rp[station_col] / df_rp[runtime_col].replace(0, np.nan)
    
    # Calculate picker throughput (pure processing time)
    if picker_runtime_col in df_ma.columns:
        df_ma['Picker Throughput (Stations/s)'] = df_ma[station_col] / df_ma[picker_runtime_col].replace(0, np.nan)
    if picker_runtime_col in df_rp.columns:
        df_rp['Picker Throughput (Stations/s)'] = df_rp[station_col] / df_rp[picker_runtime_col].replace(0, np.nan)
    
    # Summary statistics
    print(f"\n--- Overall Performance Summary ---")
    
    metrics = {
        'Mean Throughput (st/s)': ('Throughput (Stations/s)', 'mean'),
        'Median Throughput (st/s)': ('Throughput (Stations/s)', 'median'),
        'Mean Total Runtime (s)': (runtime_col, 'mean'),
        'Mean Picker Time (s)': (picker_runtime_col, 'mean'),
        'Mean Waveform Time (s)': (waveform_proc_time_col, 'mean'),
        'Mean Process RAM (MB)': (actual_ram_col, 'mean'),
    }
    
    if is_gpu:
        metrics['Mean Process VRAM (MB)'] = (actual_vram_col, 'mean')
    
    print(f"\n{'Metric':<30} {'ModelActor':>15} {'Ripper':>15} {'Difference':>15}")
    print("-" * 75)
    
    for name, (col, agg) in metrics.items():
        ma_val = getattr(df_ma[col], agg)() if col in df_ma.columns else np.nan
        rp_val = getattr(df_rp[col], agg)() if col in df_rp.columns else np.nan
        diff = ma_val - rp_val
        diff_pct = (diff / rp_val * 100) if rp_val != 0 else 0
        print(f"{name:<30} {ma_val:>15.2f} {rp_val:>15.2f} {diff:>+10.2f} ({diff_pct:+.1f}%)")
    
    # Scaling analysis
    print(f"\n--- Scaling by Concurrency Level ---")
    print("(Comparing similar configurations)")
    
    # Match by station count and concurrent tasks
    for n_tasks in sorted(df_ma[task_col].dropna().unique())[:5]:
        ma_subset = df_ma[df_ma[task_col] == n_tasks]
        rp_subset = df_rp[df_rp[task_col] == n_tasks]
        
        if len(ma_subset) > 0 and len(rp_subset) > 0:
            ma_throughput = ma_subset['Throughput (Stations/s)'].mean()
            rp_throughput = rp_subset['Throughput (Stations/s)'].mean()
            speedup = (ma_throughput / rp_throughput - 1) * 100 if rp_throughput > 0 else 0
            
            print(f"  {int(n_tasks)} concurrent tasks: MA={ma_throughput:.2f} st/s, RP={rp_throughput:.2f} st/s, Speedup: {speedup:+.1f}%")
    
    # Visualization: Throughput comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Throughput distribution
    ax = axes[0, 0]
    ax.hist(df_ma['Throughput (Stations/s)'], bins=20, alpha=0.7, label='ModelActor', color='blue')
    ax.hist(df_rp['Throughput (Stations/s)'], bins=20, alpha=0.7, label='Ripper', color='orange')
    ax.set_xlabel('Throughput (Stations/s)')
    ax.set_ylabel('Frequency')
    ax.set_title('Throughput Distribution Comparison')
    ax.legend()
    
    # 2. Runtime vs Stations
    ax = axes[0, 1]
    ax.scatter(df_ma[station_col], df_ma[runtime_col], alpha=0.5, label='ModelActor', color='blue')
    ax.scatter(df_rp[station_col], df_rp[runtime_col], alpha=0.5, label='Ripper', color='orange')
    
    # Add desired runtime line if provided
    if desired_runtime is not None:
        ax.axhline(y=desired_runtime, color='red', linestyle='--', linewidth=2, 
                   label=f'Desired Runtime ({desired_runtime}s)')
        ax.legend()
    else:
        ax.legend()
        
    ax.set_xlabel('Number of Stations')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime vs Workload Size')
    
    # 3. RAM usage comparison
    ax = axes[1, 0]
    ax.boxplot([df_ma[actual_ram_col].dropna(), df_rp[actual_ram_col].dropna()],
               labels=['ModelActor', 'Ripper'])
    ax.set_ylabel('Process Tree RAM (MB)')
    ax.set_title('RAM Usage Distribution')
    
    # 4. Throughput by concurrent tasks
    ax = axes[1, 1]
    ma_by_task = df_ma.groupby(task_col)['Throughput (Stations/s)'].mean()
    rp_by_task = df_rp.groupby(task_col)['Throughput (Stations/s)'].mean()
    
    x = np.arange(len(ma_by_task))
    width = 0.35
    ax.bar(x - width/2, ma_by_task.values, width, label='ModelActor', color='blue')
    ax.bar(x + width/2, rp_by_task.values[:len(x)], width, label='Ripper', color='orange')
    ax.set_xlabel('Concurrent Tasks')
    ax.set_ylabel('Mean Throughput (Stations/s)')
    ax.set_title('Throughput by Concurrency Level')
    ax.set_xticks(x)
    ax.set_xticklabels([int(t) for t in ma_by_task.index])
    ax.legend()
    
    # Sanitize model name for filenames
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    
    plt.suptitle(f'{model_name} - ModelActor vs Ripper ({trial_type})', fontsize=14)
    plt.tight_layout()
    
    plot_name = f"comparison_modelactor_vs_ripper_{safe_model_name}_{trial_type.lower()}.png"
    plot_path = os.path.join(output_dir, plot_name) if output_dir else plot_name
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nComparison plot saved to {plot_path}")


def batch_analyze(results_root, output_dir=None, desired_runtime=None):
    """
    Batch analyze all result directories in the results root.
    """
    if not os.path.exists(results_root):
        print(f"Error: Results root not found: {results_root}")
        return
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n{'='*70}")
    print("BATCH ANALYSIS OF ALL TRIAL RESULTS")
    print(f"{'='*70}")
    print(f"Results root: {results_root}")
    
    # Find all result directories
    result_dirs = []
    for item in os.listdir(results_root):
        item_path = os.path.join(results_root, item)
        if os.path.isdir(item_path) and item.startswith('eval_'):
            result_dirs.append(item)
    
    print(f"Found {len(result_dirs)} result directories")
    
    # Process each directory
    summary_data = []
    for result_dir in sorted(result_dirs):
        dir_path = os.path.join(results_root, result_dir)
        
        # Find CSV file
        csv_files = glob.glob(os.path.join(dir_path, '*_test_results.csv'))
        if not csv_files:
            print(f"  Skipping {result_dir}: No test results CSV found")
            continue
        
        csv_path = csv_files[0]
        
        # Create output directory for this analysis
        analysis_output = os.path.join(output_dir, result_dir) if output_dir else result_dir + '_analysis'
        
        print(f"\nAnalyzing: {result_dir}")
        df = analyze_efficiency(csv_path, analysis_output, verbose=False, desired_runtime=desired_runtime)
        
        if df is not None:
            # Collect summary info
            execution_mode = detect_execution_mode(df)
            model = df['Model Used'].iloc[0] if 'Model Used' in df.columns else 'Unknown'
            df['GPU Count'] = df['GPUs Used'].apply(parse_gpu_list)
            trial_type = 'GPU' if df['GPU Count'].max() > 0 else 'CPU'
            
            # Use Total Trial Time as the primary runtime metric
            total_trial_col = 'Total Trial Time (s)'
            picker_col = 'Total Run time for Picker (s)'
            
            if total_trial_col in df.columns and df[total_trial_col].notna().any():
                df['Throughput (Stations/s)'] = df['Number of Stations Used'] / df[total_trial_col]
                runtime_mean = df[total_trial_col].mean()
                runtime_min = df[total_trial_col].min()
            else:
                df['Throughput (Stations/s)'] = df['Number of Stations Used'] / df[picker_col]
                runtime_mean = df[picker_col].mean()
                runtime_min = df[picker_col].min()
            
            summary_data.append({
                'Directory': result_dir,
                'Model': model,
                'Trial Type': trial_type,
                'Execution Mode': execution_mode,
                'Total Trials': len(df),
                'Mean Throughput (st/s)': df['Throughput (Stations/s)'].mean(),
                'Mean Total Trial Time (s)': runtime_mean,
                'Min Total Trial Time (s)': runtime_min,
            })
    
    # Save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'batch_analysis_summary.csv') if output_dir else 'batch_analysis_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*70}")
        print("BATCH ANALYSIS SUMMARY")
        print(f"{'='*70}")
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to {summary_path}")


def find_comparison_files(results_root, model, trial_type):
    """
    Find ModelActor and Ripper CSV files for a given model and trial type.
    """
    trial_type = trial_type.lower()
    model = model.lower()
    
    # Pattern matching
    modelactor_pattern = f"eval_{trial_type}_{model}_modelactor"
    ripper_pattern = f"eval_{trial_type}_{model}_ripper"
    
    modelactor_csv = None
    ripper_csv = None
    
    for item in os.listdir(results_root):
        item_path = os.path.join(results_root, item)
        if os.path.isdir(item_path):
            if modelactor_pattern in item.lower():
                csv_files = glob.glob(os.path.join(item_path, '*_test_results.csv'))
                if csv_files:
                    modelactor_csv = csv_files[0]
            elif ripper_pattern in item.lower():
                csv_files = glob.glob(os.path.join(item_path, '*_test_results.csv'))
                if csv_files:
                    ripper_csv = csv_files[0]
    
    return modelactor_csv, ripper_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze EQCCTPro GPU/CPU Trial Efficiency and Memory Usage (ModelActor & Ripper)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file analysis
  python analyze_trial_results_efficiency.py csv/eval_gpu_eqcct_modelactor/gpu_test_results.csv
  python analyze_trial_results_efficiency.py csv/eval_cpu_eqcct_ripper/cpu_test_results.csv --output_dir analysis/

  # Batch analysis of all results
  python analyze_trial_results_efficiency.py --batch --results_root results/csv/ --output_dir batch_analysis/

  # Compare ModelActor vs Ripper
  python analyze_trial_results_efficiency.py --compare --model eqcct --trial_type cpu --results_root results/csv/
  python analyze_trial_results_efficiency.py --compare --model phasenet_original --trial_type gpu --results_root results/csv/
        """
    )
    
    # Single file mode
    parser.add_argument('csv_path', type=str, nargs='?', default=None,
                       help='Path to the results CSV file (for single file analysis)')
    parser.add_argument('--output_dir', type=str, default='analysis_results', 
                       help='Directory to save results (default: analysis_results/)')
    parser.add_argument('--desired_runtime', type=float, default=None,
                       help='Add a desired runtime horizontal line (s) to the plot')
    
    # Batch mode
    parser.add_argument('--batch', action='store_true',
                       help='Run batch analysis on all result directories')
    parser.add_argument('--results_root', type=str, default='results/csv/',
                       help='Root directory containing result subdirectories (for batch/compare modes)')
    
    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                       help='Compare ModelActor vs Ripper for a specific model')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name for comparison (e.g., eqcct, phasenet_original)')
    parser.add_argument('--trial_type', type=str, default=None, choices=['cpu', 'gpu'],
                       help='Trial type for comparison (cpu or gpu)')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch analysis mode
        batch_analyze(args.results_root, args.output_dir, desired_runtime=args.desired_runtime)
    elif args.compare:
        # Comparison mode
        if not args.model or not args.trial_type:
            print("Error: --compare requires --model and --trial_type arguments")
            parser.print_help()
        else:
            modelactor_csv, ripper_csv = find_comparison_files(args.results_root, args.model, args.trial_type)
            if modelactor_csv and ripper_csv:
                compare_modelactor_vs_ripper(modelactor_csv, ripper_csv, args.output_dir, desired_runtime=args.desired_runtime)
            else:
                print(f"Error: Could not find both ModelActor and Ripper results for {args.model} ({args.trial_type})")
                if not modelactor_csv:
                    print(f"  Missing: ModelActor CSV")
                if not ripper_csv:
                    print(f"  Missing: Ripper CSV")
    elif args.csv_path:
        # Single file analysis mode
        analyze_efficiency(args.csv_path, args.output_dir, desired_runtime=args.desired_runtime)
    else:
        parser.print_help()
