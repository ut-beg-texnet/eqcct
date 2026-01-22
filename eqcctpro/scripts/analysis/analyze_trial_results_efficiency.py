"""
Efficiency and Memory Analysis Script for EQCCTPro Trial Results
================================================================

This script analyzes computational efficiency and memory costs from EvaluateSystem
trial CSV files. It provides insights into:
- Throughput and resource efficiency
- Memory utilization (Requested vs Actual)
- Diminishing returns analysis
- Performance lever identification
- Concurrency vs Runtime scaling

Usage:
------
python analyze_trial_results_efficiency.py <csv_path> [--output_dir <dir>]

Example:
--------
python analyze_trial_results_efficiency.py csv/eval_gpu_eqcct/gpu_test_results.csv --output_dir analysis_results/
"""

import pandas as pd
import numpy as np
import ast
import os
import argparse
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


def analyze_efficiency(csv_path, output_dir=None):
    """
    Analyzes computational efficiency and memory costs from trial CSV files.
    Calculates diminishing returns, identifies performance levers, and saves summaries.
    
    Updated for the new CANONICAL_CSV_HEADER with PID-isolated memory tracking.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use a buffer to capture summary text
    summary_buffer = StringIO()

    def log(message):
        print(message)
        summary_buffer.write(message + "\n")

    log(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # =========================================================================
    # COLUMN DEFINITIONS (Updated for new CANONICAL_CSV_HEADER)
    # =========================================================================
    # Core configuration columns
    cpu_col = 'Number of CPUs Allocated for Ray to Use'
    task_col = 'Number of Concurrent Station Tasks'
    actor_col = 'N ModelActors'
    runtime_col = 'Total Run time for Picker (s)'
    station_col = 'Number of Stations Used'
    timechunk_col = 'Total Number of Timechunks'
    concurrent_tc_col = 'Concurrent Timechunks Used'
    
    # Memory columns (new PID-isolated tracking)
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
    
    log(f"\n{'='*60}")
    log(f"EQCCTPRO EFFICIENCY ANALYSIS")
    log(f"{'='*60}")
    log(f"Detected Model: {model_name}")
    log(f"Detected Trial Type: {trial_type}")
    log(f"Total Trials in File: {len(df)}")
    
    # Filter successful trials only
    df_success = df[df['Trial Success'] == 1.0].copy()
    log(f"Successful Trials: {len(df_success)}")
    
    if len(df_success) == 0:
        log("Error: No successful trials found. Exiting.")
        return
    
    # Convert numeric columns
    numeric_cols = [cpu_col, task_col, actor_col, runtime_col, station_col,
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

    # =========================================================================
    # DERIVED METRICS
    # =========================================================================
    # 1. Throughput (Stations per second)
    df_success['Throughput (Stations/s)'] = df_success[station_col] / df_success[runtime_col]
    
    # 2. Resource Efficiency
    df_success['Throughput/CPU'] = df_success['Throughput (Stations/s)'] / df_success[cpu_col]
    df_success['Throughput/Actor'] = df_success['Throughput (Stations/s)'] / df_success[actor_col].replace(0, np.nan)
    if is_gpu_trial:
        df_success['Throughput/GPU'] = df_success.apply(
            lambda row: row['Throughput (Stations/s)'] / row['GPU Count'] if row['GPU Count'] > 0 else np.nan, axis=1
        )

    # 3. Memory Efficiency Metrics
    df_success['RAM per Station (MB)'] = df_success[actual_ram_col] / df_success[station_col]
    df_success['RAM per Actor (MB)'] = df_success[actual_ram_col] / df_success[actor_col].replace(0, np.nan)
    if is_gpu_trial:
        df_success['VRAM per Station (MB)'] = df_success[actual_vram_col] / df_success[station_col]
        df_success['VRAM per Actor (MB)'] = df_success[actual_vram_col] / df_success[actor_col].replace(0, np.nan)

    # 4. Memory Overhead Analysis (Actual - Requested)
    # RAM Overhead is already in the CSV, but we can verify/recalculate
    if actual_ram_col in df_success.columns and total_req_ram_col in df_success.columns:
        df_success['Calculated RAM Overhead (MB)'] = df_success[actual_ram_col] - df_success[total_req_ram_col]
    
    if is_gpu_trial and actual_vram_col in df_success.columns and total_req_vram_col in df_success.columns:
        df_success['Calculated VRAM Overhead (MB)'] = df_success[actual_vram_col] - df_success[total_req_vram_col]

    # 5. Resource Cost (Hypothetical)
    # Assume 1 GPU is worth 10 CPUs in "cost"
    gpu_weight = 10
    df_success['Resource Cost Score'] = df_success[cpu_col] + (df_success['GPU Count'] * gpu_weight)
    df_success['Throughput per Cost Unit'] = df_success['Throughput (Stations/s)'] / df_success['Resource Cost Score']

    # =========================================================================
    # ANALYSIS OUTPUTS
    # =========================================================================
    log("\n" + "="*60)
    log(f"NUMERICAL ANALYSIS SUMMARY: {model_name} ({trial_type})")
    log("="*60)

    log("\n--- Analysis Formulas ---")
    log("1. Throughput (Stations/s) = Total Stations / Total Runtime")
    log("2. Gain % (Diminishing Returns) = ((Current - Previous) / Previous) * 100")
    log("3. Resource Cost Score = CPUs + (GPUs * 10)")
    log("4. RAM Overhead = Process Tree RAM - Total Requested RAM")
    log("5. RAM Utilization = (Process Tree RAM / Total Requested RAM) * 100")

    # =========================================================================
    # MEMORY ANALYSIS: REQUESTED vs ACTUAL
    # =========================================================================
    log("\n" + "-"*60)
    log("MEMORY ANALYSIS: REQUESTED vs ACTUAL")
    log("-"*60)
    
    # RAM Analysis (both CPU and GPU trials)
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
        # Filter out zero values (often from trials where GPU wasn't actually used)
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
    log("\n" + "-"*60)
    log("PERFORMANCE LEVERS (Runtime Correlations)")
    log("-"*60)
    
    def calculate_impact(lever_col):
        if lever_col not in df_success.columns:
            return np.nan
        return df_success[lever_col].corr(df_success[runtime_col])

    cpu_impact = calculate_impact(cpu_col)
    task_impact = calculate_impact(task_col)
    actor_impact = calculate_impact(actor_col)
    station_impact = calculate_impact(station_col)
    
    log(f"\nCorrelation with Runtime (negative = faster with more):")
    log(f"  1. Number of Stations:           {station_impact:+.3f} (expected: positive)")
    log(f"  2. Number of CPUs:               {cpu_impact:+.3f}")
    log(f"  3. Concurrent Tasks:             {task_impact:+.3f}")
    log(f"  4. N ModelActors:                {actor_impact:+.3f}")
    
    if is_gpu_trial:
        gpu_impact = calculate_impact('GPU Count')
        log(f"  5. GPU Count:                    {gpu_impact:+.3f}")
    
    log("\n  Interpretation:")
    log("  - Negative correlation: More of this resource = faster runtime")
    log("  - Positive correlation: More of this = slower (e.g., more stations = more work)")
    log("  - Close to 0: Weak relationship")

    # =========================================================================
    # CONCURRENCY VS RUNTIME ANALYSIS
    # =========================================================================
    log("\n" + "-"*60)
    log("CONCURRENCY VS RUNTIME ANALYSIS")
    log("-"*60)
    
    # Analyze how N ModelActors affects runtime for different workloads
    if actor_col in df_success.columns:
        actor_groups = df_success.groupby(actor_col).agg({
            runtime_col: ['mean', 'std', 'count'],
            station_col: 'mean',
            'Throughput (Stations/s)': 'mean'
        }).round(2)
        actor_groups.columns = ['Avg Runtime (s)', 'Std Runtime', 'Count', 'Avg Stations', 'Avg Throughput']
        actor_groups = actor_groups.reset_index()
        
        log(f"\nPerformance by N ModelActors:")
        log(actor_groups.to_string(index=False))
    
    # Analyze runtime scaling with stations for different concurrency levels
    log(f"\n\nRuntime Scaling by Workload Size:")
    station_quartiles = df_success[station_col].quantile([0.25, 0.5, 0.75]).values
    
    for q, label in zip([0.25, 0.5, 0.75], ['Small', 'Medium', 'Large']):
        threshold = df_success[station_col].quantile(q)
        subset = df_success[df_success[station_col] <= threshold]
        if len(subset) > 0:
            avg_runtime = subset[runtime_col].mean()
            avg_actors = subset[actor_col].mean() if actor_col in subset.columns else 0
            avg_throughput = subset['Throughput (Stations/s)'].mean()
            log(f"  {label} workloads (≤{threshold:.0f} stations): Avg Runtime={avg_runtime:.2f}s, Avg Actors={avg_actors:.1f}, Throughput={avg_throughput:.2f} st/s")

    # =========================================================================
    # SWEET SPOT ANALYSIS
    # =========================================================================
    log("\n" + "-"*60)
    log("OPTIMAL CONFIGURATION ('SWEET SPOT')")
    log("-"*60)
    
    best_cost_efficiency = df_success.loc[df_success['Throughput per Cost Unit'].idxmax()]
    log(f"\nBest Throughput per Resource Cost:")
    log(f"  Configuration: {best_cost_efficiency[cpu_col]:.0f} CPUs, {best_cost_efficiency['GPU Count']:.0f} GPUs, {best_cost_efficiency[task_col]:.0f} Concurrent Tasks")
    log(f"  N ModelActors: {best_cost_efficiency[actor_col]:.0f}")
    log(f"  Stations Processed: {best_cost_efficiency[station_col]:.0f}")
    log(f"  Throughput: {best_cost_efficiency['Throughput (Stations/s)']:.3f} stations/s")
    log(f"  Total Runtime: {best_cost_efficiency[runtime_col]:.2f}s")
    log(f"  Process Tree RAM: {best_cost_efficiency[actual_ram_col]:.1f} MB")
    if is_gpu_trial and actual_vram_col in df_success.columns:
        log(f"  Process Tree VRAM: {best_cost_efficiency[actual_vram_col]:.1f} MB")

    # Fastest overall configuration
    fastest = df_success.loc[df_success[runtime_col].idxmin()]
    log(f"\nFastest Overall Configuration:")
    log(f"  Configuration: {fastest[cpu_col]:.0f} CPUs, {fastest['GPU Count']:.0f} GPUs, {fastest[task_col]:.0f} Concurrent Tasks")
    log(f"  N ModelActors: {fastest[actor_col]:.0f}")
    log(f"  Stations Processed: {fastest[station_col]:.0f}")
    log(f"  Total Runtime: {fastest[runtime_col]:.2f}s")

    # =========================================================================
    # DIMINISHING RETURNS ANALYSIS
    # =========================================================================
    log("\n" + "-"*60)
    log("DIMINISHING RETURNS ANALYSIS")
    log("-"*60)
    
    # Analyze diminishing returns for actor count scaling
    if actor_col in df_success.columns:
        log("\n--- Scaling by N ModelActors ---")
        # For a fixed workload size (median stations), see how actors affect throughput
        median_stations = df_success[station_col].median()
        tolerance = median_stations * 0.2  # 20% tolerance
        
        similar_workload = df_success[
            (df_success[station_col] >= median_stations - tolerance) &
            (df_success[station_col] <= median_stations + tolerance)
        ]
        
        if len(similar_workload) > 3:
            actor_scaling = similar_workload.groupby(actor_col)['Throughput (Stations/s)'].mean().sort_index()
            
            prev_val = None
            for n_actors, val in actor_scaling.items():
                if prev_val is not None:
                    gain = (val - prev_val) / prev_val * 100 if prev_val > 0 else 0
                    log(f"  {int(n_actors-1)} -> {int(n_actors)} actors: {gain:+.1f}% throughput gain")
                prev_val = val
        else:
            log("  Insufficient data for actor scaling analysis at median workload")

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
    log("\n" + "-"*60)
    log("CORRELATION MATRIX")
    log("-"*60)
    
    # Select columns for correlation matrix
    corr_cols = [cpu_col, 'GPU Count', task_col, actor_col, station_col, runtime_col, 
                 'Throughput (Stations/s)', actual_ram_col]
    if is_gpu_trial and actual_vram_col in df_success.columns:
        corr_cols.append(actual_vram_col)
    
    # Filter to existing columns
    corr_cols = [c for c in corr_cols if c in df_success.columns]
    
    corr_matrix = df_success[corr_cols].corr()
    
    # Log key correlations
    log(f"\nKey correlations with {runtime_col}:")
    for col in corr_cols:
        if col != runtime_col:
            corr_val = corr_matrix.loc[col, runtime_col] if runtime_col in corr_matrix.columns else np.nan
            log(f"  {col}: {corr_val:+.3f}")

    # Save Correlation Matrix as Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5,
                center=0, vmin=-1, vmax=1)
    plt.title(f"Correlation Matrix: {model_name} ({trial_type})")
    plt.tight_layout()
    
    plot_name = "correlation_matrix.png"
    plot_path = os.path.join(output_dir, plot_name) if output_dir else plot_name
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log(f"\nCorrelation plot saved to {plot_path}")

    # =========================================================================
    # ADDITIONAL VISUALIZATIONS
    # =========================================================================
    
    # 1. Runtime vs Stations by Actor Count
    plt.figure(figsize=(10, 6))
    for n_actors in sorted(df_success[actor_col].dropna().unique()):
        subset = df_success[df_success[actor_col] == n_actors]
        plt.scatter(subset[station_col], subset[runtime_col], label=f'{int(n_actors)} actors', alpha=0.7)
    plt.xlabel('Number of Stations')
    plt.ylabel('Runtime (s)')
    plt.title(f'Runtime vs Workload Size by Actor Count\n{model_name} ({trial_type})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_name = "runtime_vs_stations_by_actors.png"
    plot_path = os.path.join(output_dir, plot_name) if output_dir else plot_name
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log(f"Runtime vs Stations plot saved to {plot_path}")

    # 2. Memory Utilization Distribution
    if ram_util_col in df_success.columns:
        plt.figure(figsize=(10, 6))
        valid_util = df_success[ram_util_col].dropna()
        plt.hist(valid_util, bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(x=100, color='red', linestyle='--', label='100% (Requested = Actual)')
        plt.xlabel('RAM Utilization (%)')
        plt.ylabel('Frequency')
        plt.title(f'RAM Utilization Distribution\n{model_name} ({trial_type})')
        plt.legend()
        plt.tight_layout()
        
        plot_name = "ram_utilization_distribution.png"
        plot_path = os.path.join(output_dir, plot_name) if output_dir else plot_name
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log(f"RAM utilization plot saved to {plot_path}")

    # 3. Requested vs Actual RAM scatter
    if actual_ram_col in df_success.columns and total_req_ram_col in df_success.columns:
        plt.figure(figsize=(10, 8))
        valid_data = df_success[[total_req_ram_col, actual_ram_col, actor_col]].dropna()
        
        scatter = plt.scatter(valid_data[total_req_ram_col], valid_data[actual_ram_col], 
                             c=valid_data[actor_col], cmap='viridis', alpha=0.7, s=50)
        
        # Add diagonal line (perfect prediction)
        max_val = max(valid_data[total_req_ram_col].max(), valid_data[actual_ram_col].max())
        plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction (1:1)')
        
        plt.colorbar(scatter, label='N ModelActors')
        plt.xlabel('Total Requested RAM (MB)')
        plt.ylabel('Process Tree RAM (MB)')
        plt.title(f'Requested vs Actual RAM\n{model_name} ({trial_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_name = "requested_vs_actual_ram.png"
        plot_path = os.path.join(output_dir, plot_name) if output_dir else plot_name
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log(f"Requested vs Actual RAM plot saved to {plot_path}")

    # 4. Throughput vs Actor Count (boxplot)
    if actor_col in df_success.columns:
        plt.figure(figsize=(10, 6))
        actor_values = sorted(df_success[actor_col].dropna().unique())
        data_for_boxplot = [df_success[df_success[actor_col] == a]['Throughput (Stations/s)'].dropna() 
                           for a in actor_values]
        plt.boxplot(data_for_boxplot, labels=[int(a) for a in actor_values])
        plt.xlabel('N ModelActors')
        plt.ylabel('Throughput (Stations/s)')
        plt.title(f'Throughput Distribution by Actor Count\n{model_name} ({trial_type})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_name = "throughput_by_actors_boxplot.png"
        plot_path = os.path.join(output_dir, plot_name) if output_dir else plot_name
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log(f"Throughput by actors boxplot saved to {plot_path}")

    # =========================================================================
    # AGGREGATED STATISTICS
    # =========================================================================
    print("\n" + "-"*60)
    print("AGGREGATED METRICS (Averages by Hardware Configuration)")
    print("-"*60)
    
    agg_cols = {
        runtime_col: 'mean',
        'Throughput (Stations/s)': 'mean',
        actor_col: 'mean',
        actual_ram_col: 'mean',
    }
    if is_gpu_trial and actual_vram_col in df_success.columns:
        agg_cols[actual_vram_col] = 'mean'
    
    agg_stats = df_success.groupby([cpu_col, 'GPU Count']).agg(agg_cols).round(2)
    agg_stats.columns = ['Avg Runtime (s)', 'Avg Throughput', 'Avg Actors', 'Avg RAM (MB)']
    if is_gpu_trial:
        agg_stats.columns = list(agg_stats.columns) + ['Avg VRAM (MB)']
    agg_stats = agg_stats.reset_index()
    print(agg_stats.to_string(index=False))

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    csv_name = "efficiency_analysis_results.csv"
    txt_name = "efficiency_summary.txt"
    
    if output_dir:
        csv_path_out = os.path.join(output_dir, csv_name)
        txt_path_out = os.path.join(output_dir, txt_name)
    else:
        csv_path_out = csv_name
        txt_path_out = txt_name

    df_success.to_csv(csv_path_out, index=False)
    with open(txt_path_out, "w") as f:
        f.write(summary_buffer.getvalue())
    
    print(f"\nDetailed CSV saved to {csv_path_out}")
    print(f"Summary TXT saved to {txt_path_out}")
    print(f"\nAnalysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze EQCCTPro GPU/CPU Trial Efficiency and Memory Usage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_trial_results_efficiency.py csv/eval_gpu_eqcct/gpu_test_results.csv
  python analyze_trial_results_efficiency.py csv/eval_cpu_eqcct/cpu_test_results.csv --output_dir analysis/
        """
    )
    parser.add_argument('csv_path', type=str, help='Path to the results CSV file')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Directory to save results (default: current directory)')
    args = parser.parse_args()
    
    analyze_efficiency(args.csv_path, args.output_dir)
