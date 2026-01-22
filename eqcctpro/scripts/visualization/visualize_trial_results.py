"""
Interactive Trial Results Visualization with Plotly
====================================================

This script creates interactive 3D scatter plots for EQCCTPro evaluation results,
supporting both CPU and GPU trial data.

Features:
- Automatic detection of trial type (CPU vs GPU)
- Interactive 3D visualizations with Plotly
- Multiple plot types: Runtime, RAM, VRAM (GPU only), Memory Efficiency
- Threshold filtering options
- HTML export for sharing

Usage:
------
python visualize_trial_results.py <csv_path> [options]

Examples:
---------
# Visualize GPU trial results
python visualize_trial_results.py results/csv/eval_gpu_eqcct/gpu_test_results.csv --output_dir vis/

# Visualize CPU trial results
python visualize_trial_results.py results/csv/eval_cpu_eqcct/cpu_test_results.csv --output_dir vis/

# With custom model name and runtime threshold
python visualize_trial_results.py gpu_test_results.csv --model "PhaseNet-GPU" --threshold 30
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import ast
import os
import argparse
import numpy as np


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


def detect_trial_type(df):
    """Detect if trials are CPU-based or GPU-based."""
    if 'GPUs Used' not in df.columns:
        return 'cpu'
    
    df_temp = df.copy()
    df_temp['GPU Count'] = df_temp['GPUs Used'].apply(parse_gpu_list)
    
    if df_temp['GPU Count'].max() > 0:
        return 'gpu'
    return 'cpu'


def visualize_trials(csv_path, model_name=None, output_dir="visualizations", 
                     filter_threshold=None, success_only=True):
    """
    Reads trial results from a CSV and creates interactive Plotly 3D scatter plots.
    Automatically detects CPU vs GPU trials and adjusts visualizations accordingly.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing trial results.
    model_name : str, optional
        Name to display in plot titles. Auto-detected if not provided.
    output_dir : str
        Directory to save HTML visualization files.
    filter_threshold : float, optional
        If provided, creates additional filtered plots for runtime <= threshold.
    success_only : bool
        If True, only visualize successful trials.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # =========================================================================
    # COLUMN DEFINITIONS (Updated for CANONICAL_CSV_HEADER)
    # =========================================================================
    cpu_col = 'Number of CPUs Allocated for Ray to Use'
    task_col = 'Number of Concurrent Station Tasks'
    actor_col = 'N ModelActors'
    runtime_col = 'Total Run time for Picker (s)'
    station_col = 'Number of Stations Used'
    
    # Memory columns (new PID-isolated tracking)
    total_req_vram_col = 'Total Requested VRAM (MB)'
    total_req_ram_col = 'Total Requested RAM (MB)'
    actual_vram_col = 'Process Tree VRAM (MB)'
    actual_ram_col = 'Process Tree RAM (MB)'
    ram_util_col = 'RAM Utilization (%)'
    vram_util_col = 'VRAM Utilization (%)'
    ram_overhead_col = 'RAM Overhead (MB)'
    
    # =========================================================================
    # DATA PREPROCESSING
    # =========================================================================
    # Detect trial type
    trial_type = detect_trial_type(df)
    is_gpu_trial = trial_type == 'gpu'
    
    # Parse GPU count
    df['GPU Count'] = df['GPUs Used'].apply(parse_gpu_list) if 'GPUs Used' in df.columns else 0
    
    # Auto-detect model name
    if model_name is None:
        model_name = df['Model Used'].iloc[0] if 'Model Used' in df.columns else "Unknown"
        model_name = f"{model_name}-{'GPU' if is_gpu_trial else 'CPU'}"
    
    print(f"Detected trial type: {trial_type.upper()}")
    print(f"Model: {model_name}")
    
    # Filter successful trials
    if success_only and 'Trial Success' in df.columns:
        df = df[df['Trial Success'] == 1.0]
        print(f"Filtering to successful trials: {len(df)} rows")
    
    # Convert numeric columns
    numeric_cols = [cpu_col, task_col, actor_col, runtime_col, station_col,
                    actual_ram_col, actual_vram_col, ram_util_col, vram_util_col,
                    total_req_ram_col, total_req_vram_col]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Define required columns based on trial type
    required_cols = [cpu_col, task_col, runtime_col, station_col, actual_ram_col]
    if is_gpu_trial:
        required_cols.append(actual_vram_col)
    
    # Drop rows with missing required values
    existing_required = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_required)
    
    if df.empty:
        print("Error: No valid data after filtering. Check your CSV structure.")
        return
    
    print(f"Valid trials for visualization: {len(df)}")

    # =========================================================================
    # MARKER AND COLOR CONFIGURATION
    # =========================================================================
    # Symbol mapping for different GPU/Actor counts
    symbol_map = {
        0: 'x',
        1: 'circle',
        2: 'diamond',
        3: 'square',
        4: 'cross',
        5: 'pentagon',
        6: 'star',
        7: 'hexagram',
        8: 'triangle-up'
    }

    # =========================================================================
    # PLOT CONFIGURATIONS
    # =========================================================================
    plot_configs = [
        {
            'z_col': runtime_col, 
            'title': 'Runtime vs Resources', 
            'z_label': 'Runtime (s)', 
            'file_name': 'runtime_3d',
            'show_threshold': True
        },
        {
            'z_col': actual_ram_col, 
            'title': 'Process Tree RAM vs Resources', 
            'z_label': 'Process Tree RAM (MB)', 
            'file_name': 'ram_3d',
            'show_threshold': False
        },
    ]
    
    # Add GPU-specific plots
    if is_gpu_trial and actual_vram_col in df.columns:
        plot_configs.append({
            'z_col': actual_vram_col, 
            'title': 'Process Tree VRAM vs Resources', 
            'z_label': 'Process Tree VRAM (MB)', 
            'file_name': 'vram_3d',
            'show_threshold': False
        })
    
    # Add memory efficiency plots if utilization data exists
    if ram_util_col in df.columns:
        plot_configs.append({
            'z_col': ram_util_col, 
            'title': 'RAM Utilization vs Resources', 
            'z_label': 'RAM Utilization (%)', 
            'file_name': 'ram_utilization_3d',
            'show_threshold': False
        })
    
    if is_gpu_trial and vram_util_col in df.columns:
        plot_configs.append({
            'z_col': vram_util_col, 
            'title': 'VRAM Utilization vs Resources', 
            'z_label': 'VRAM Utilization (%)', 
            'file_name': 'vram_utilization_3d',
            'show_threshold': False
        })

    # =========================================================================
    # DATASET VARIANTS (with/without filtering)
    # =========================================================================
    datasets = [{'df': df, 'suffix': '', 'apply_threshold': False}]
    
    if filter_threshold is not None:
        filtered_df = df[df[runtime_col] <= filter_threshold]
        if len(filtered_df) > 0:
            datasets.append({
                'df': filtered_df, 
                'suffix': f'_filtered_{int(filter_threshold)}s', 
                'apply_threshold': True
            })
            print(f"Creating filtered dataset with runtime <= {filter_threshold}s: {len(filtered_df)} trials")

    # =========================================================================
    # GENERATE 3D SCATTER PLOTS
    # =========================================================================
    # Determine grouping column (GPU Count for GPU trials, N ModelActors for CPU)
    if is_gpu_trial:
        group_col = 'GPU Count'
        group_label = 'GPUs'
    else:
        group_col = actor_col if actor_col in df.columns else cpu_col
        group_label = 'Actors' if actor_col in df.columns else 'CPUs'

    for dataset in datasets:
        curr_df = dataset['df']
        if curr_df.empty:
            continue
        
        title_suffix = f" (Filtered <= {filter_threshold}s)" if dataset['suffix'] else ""

        for config in plot_configs:
            # Skip if column doesn't exist
            if config['z_col'] not in curr_df.columns:
                continue
            
            # Skip if all values are NaN or zero
            if curr_df[config['z_col']].isna().all() or (curr_df[config['z_col']] == 0).all():
                continue
                
            fig = go.Figure()

            # Add traces for each group (GPU count or Actor count)
            unique_groups = sorted(curr_df[group_col].dropna().unique())
            for g_val in unique_groups:
                subset = curr_df[curr_df[group_col] == g_val]
                
                fig.add_trace(go.Scatter3d(
                    x=subset[cpu_col],
                    y=subset[task_col],
                    z=subset[config['z_col']],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=subset[station_col],
                        colorscale='Turbo',
                        colorbar=dict(title="Stations"),
                        cmin=curr_df[station_col].min(),
                        cmax=curr_df[station_col].max(),
                        opacity=0.7,
                        symbol=symbol_map.get(int(g_val) if not pd.isna(g_val) else 0, 'circle')
                    ),
                    name=f'{int(g_val)} {group_label}',
                    hovertemplate=(
                        f"<b>{int(g_val)} {group_label}</b><br>"
                        f"CPUs: %{{x}}<br>"
                        f"Concurrent Tasks: %{{y}}<br>"
                        f"{config['z_label']}: %{{z:.2f}}<br>"
                        f"Stations: %{{marker.color}}<br>"
                        "<extra></extra>"
                    )
                ))

            # Add threshold lines for runtime plots
            if config['z_col'] == runtime_col and dataset['apply_threshold'] and filter_threshold:
                threshold_val = filter_threshold * 0.5  # Show line at half the filter threshold
                x_max = curr_df[cpu_col].max()
                y_max = curr_df[task_col].max()
                
                # Back wall line
                fig.add_trace(go.Scatter3d(
                    x=[0, x_max], y=[y_max, y_max], z=[threshold_val, threshold_val],
                    mode='lines',
                    line=dict(color='red', width=4, dash='dash'),
                    name=f'{threshold_val:.0f}s Target',
                    showlegend=True
                ))
                # Left wall line
                fig.add_trace(go.Scatter3d(
                    x=[0, 0], y=[0, y_max], z=[threshold_val, threshold_val],
                    mode='lines',
                    line=dict(color='red', width=4, dash='dash'),
                    showlegend=False
                ))

            # Calculate axis ranges
            x_range = [0, curr_df[cpu_col].max() * 1.1]
            y_range = [0, curr_df[task_col].max() * 1.1]
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"[{model_name}]{title_suffix}<br>{config['title']}",
                    x=0.5,
                    xanchor='center'
                ),
                scene=dict(
                    xaxis=dict(title='CPUs Allocated', range=x_range),
                    yaxis=dict(title='Concurrent Tasks', range=y_range),
                    zaxis=dict(title=config['z_label']),
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=0.8)
                ),
                margin=dict(l=0, r=0, b=0, t=60),
                legend=dict(
                    title=f"{group_label} Count",
                    x=0.85,
                    y=0.95,
                    bgcolor='rgba(255,255,255,0.8)'
                )
            )

            # Save as HTML
            output_file = os.path.join(output_dir, f"{config['file_name']}{dataset['suffix']}.html")
            fig.write_html(output_file)
            print(f"Saved: {output_file}")

    # =========================================================================
    # ADDITIONAL 2D VISUALIZATIONS
    # =========================================================================
    
    # 1. Runtime vs Stations (2D scatter with actor coloring)
    if actor_col in df.columns:
        fig = px.scatter(
            df, 
            x=station_col, 
            y=runtime_col,
            color=actor_col,
            size=cpu_col,
            hover_data=[cpu_col, task_col, actual_ram_col],
            title=f"[{model_name}] Runtime vs Workload Size",
            labels={
                station_col: 'Number of Stations',
                runtime_col: 'Runtime (s)',
                actor_col: 'N ModelActors',
                cpu_col: 'CPUs'
            },
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            xaxis_title='Number of Stations',
            yaxis_title='Runtime (s)'
        )
        output_file = os.path.join(output_dir, "runtime_vs_stations_2d.html")
        fig.write_html(output_file)
        print(f"Saved: {output_file}")

    # 2. Memory Efficiency: Requested vs Actual RAM
    if total_req_ram_col in df.columns and actual_ram_col in df.columns:
        valid_mem = df[[total_req_ram_col, actual_ram_col, actor_col, station_col]].dropna()
        if len(valid_mem) > 0:
            fig = px.scatter(
                valid_mem,
                x=total_req_ram_col,
                y=actual_ram_col,
                color=actor_col if actor_col in valid_mem.columns else None,
                size=station_col,
                hover_data=[station_col],
                title=f"[{model_name}] Requested vs Actual RAM",
                labels={
                    total_req_ram_col: 'Total Requested RAM (MB)',
                    actual_ram_col: 'Process Tree RAM (MB)',
                    actor_col: 'N ModelActors'
                },
                color_continuous_scale='Plasma'
            )
            # Add 1:1 reference line
            max_val = max(valid_mem[total_req_ram_col].max(), valid_mem[actual_ram_col].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='1:1 (Perfect Prediction)'
            ))
            output_file = os.path.join(output_dir, "requested_vs_actual_ram_2d.html")
            fig.write_html(output_file)
            print(f"Saved: {output_file}")

    # 3. Throughput analysis
    df['Throughput (Stations/s)'] = df[station_col] / df[runtime_col]
    
    fig = px.scatter(
        df,
        x=task_col,
        y='Throughput (Stations/s)',
        color=cpu_col,
        size=station_col,
        hover_data=[runtime_col, actor_col if actor_col in df.columns else cpu_col],
        title=f"[{model_name}] Throughput vs Concurrency",
        labels={
            task_col: 'Concurrent Tasks',
            'Throughput (Stations/s)': 'Throughput (Stations/s)',
            cpu_col: 'CPUs Allocated'
        },
        color_continuous_scale='Turbo'
    )
    output_file = os.path.join(output_dir, "throughput_vs_concurrency_2d.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")

    # 4. GPU-specific: VRAM requested vs actual
    if is_gpu_trial and total_req_vram_col in df.columns and actual_vram_col in df.columns:
        valid_vram = df[[total_req_vram_col, actual_vram_col, 'GPU Count', station_col]].dropna()
        valid_vram = valid_vram[valid_vram[actual_vram_col] > 0]
        
        if len(valid_vram) > 0:
            fig = px.scatter(
                valid_vram,
                x=total_req_vram_col,
                y=actual_vram_col,
                color='GPU Count',
                size=station_col,
                title=f"[{model_name}] Requested vs Actual VRAM",
                labels={
                    total_req_vram_col: 'Total Requested VRAM (MB)',
                    actual_vram_col: 'Process Tree VRAM (MB)',
                    'GPU Count': 'GPU Count'
                },
                color_continuous_scale='Plasma'
            )
            # Add 1:1 reference line
            max_val = max(valid_vram[total_req_vram_col].max(), valid_vram[actual_vram_col].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='1:1 (Perfect Prediction)'
            ))
            output_file = os.path.join(output_dir, "requested_vs_actual_vram_2d.html")
            fig.write_html(output_file)
            print(f"Saved: {output_file}")

    print(f"\nVisualization complete! All files saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize EQCCTPro Trial Results with Interactive Plotly Charts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GPU trial visualization
  python visualize_trial_results.py results/csv/eval_gpu_eqcct/gpu_test_results.csv

  # CPU trial visualization with custom output directory
  python visualize_trial_results.py results/csv/eval_cpu_eqcct/cpu_test_results.csv --output_dir cpu_vis/

  # With runtime threshold filter and custom model name
  python visualize_trial_results.py gpu_test_results.csv --model "PhaseNet-GPU" --threshold 60
        """
    )
    parser.add_argument('csv_path', type=str, help='Path to the results CSV file')
    parser.add_argument('--model', type=str, default=None, 
                       help='Model name for plot titles (auto-detected if not provided)')
    parser.add_argument('--output_dir', type=str, default='visualizations', 
                       help='Directory to save HTML files (default: visualizations)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Runtime threshold in seconds for filtered plots (e.g., 60)')
    parser.add_argument('--include-failed', action='store_true',
                       help='Include failed trials in visualization (default: success only)')
    
    args = parser.parse_args()
    
    visualize_trials(
        csv_path=args.csv_path, 
        model_name=args.model, 
        output_dir=args.output_dir,
        filter_threshold=args.threshold,
        success_only=not args.include_failed
    )
