"""
Interactive Trial Results Visualization with Plotly
====================================================

This script creates interactive 3D scatter plots for EQCCTPro evaluation results,
supporting both CPU and GPU trials, and both ModelActor and Ripper execution modes.

Features:
- Automatic detection of trial type (CPU vs GPU) and execution mode (ModelActor vs Ripper)
- Interactive 3D visualizations with Plotly
- Multiple plot types: Runtime, RAM, VRAM (GPU only), Memory Efficiency
- Threshold filtering options
- HTML export for sharing
- Batch visualization of multiple result directories
- ModelActor vs Ripper comparison visualizations

Usage:
------
# Single file visualization
python visualize_trial_results.py <csv_path> [options]

# Batch visualization of all results
python visualize_trial_results.py --batch --results_root results/csv/ --output_dir visualizations/

# Compare ModelActor vs Ripper
python visualize_trial_results.py --compare --model eqcct --trial_type cpu --results_root results/csv/

Examples:
---------
# GPU trial visualization (ModelActor)
python visualize_trial_results.py results/csv/eval_gpu_eqcct_modelactor/gpu_test_results.csv --output_dir vis/

# CPU trial visualization (Ripper)
python visualize_trial_results.py results/csv/eval_cpu_eqcct_ripper/cpu_test_results.csv --output_dir vis/

# With custom model name and runtime threshold
python visualize_trial_results.py gpu_test_results.csv --model "PhaseNet-GPU" --threshold 30

# Batch visualization
python visualize_trial_results.py --batch --results_root results/csv/ --output_dir batch_vis/

# ModelActor vs Ripper comparison
python visualize_trial_results.py --compare --model phasenet_original --trial_type cpu --results_root results/csv/
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ast
import os
import argparse
import numpy as np
import glob


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


def detect_execution_mode(df):
    """
    Detect if trials use ModelActor or Ripper execution mode.
    
    Returns: 'modelactor', 'ripper', or 'mixed'
    """
    if 'N ModelActors' not in df.columns:
        return 'unknown'
    
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
    """
    if execution_mode == 'ripper':
        if 'Actual Ripper Concurrent Tasks' in df.columns:
            ripper_col = df['Actual Ripper Concurrent Tasks'].fillna(0)
            if (ripper_col > 0).any():
                return 'Actual Ripper Concurrent Tasks'
        return 'Number of Concurrent Station Tasks'
    else:
        return 'N ModelActors'


def visualize_trials(csv_path, model_name=None, output_dir="visualizations", 
                     filter_threshold=None, success_only=True):
    """
    Reads trial results from a CSV and creates interactive Plotly 3D scatter plots.
    Automatically detects CPU vs GPU trials and ModelActor vs Ripper execution mode.
    
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
    # COLUMN DEFINITIONS (CANONICAL_CSV_HEADER)
    # =========================================================================
    cpu_col = 'Number of CPUs Allocated for Ray to Use'
    task_col = 'Number of Concurrent Station Tasks'
    actor_col = 'N ModelActors'
    ripper_task_col = 'Actual Ripper Concurrent Tasks'
    runtime_col = 'Total Run time for Picker (s)'
    station_col = 'Number of Stations Used'
    
    # Memory columns (PID-isolated tracking)
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
    # Detect trial type and execution mode
    trial_type = detect_trial_type(df)
    is_gpu_trial = trial_type == 'gpu'
    execution_mode = detect_execution_mode(df)
    concurrency_col = get_concurrency_column(df, execution_mode)
    
    # Parse GPU count
    df['GPU Count'] = df['GPUs Used'].apply(parse_gpu_list) if 'GPUs Used' in df.columns else 0
    
    # Auto-detect model name
    if model_name is None:
        model_name = df['Model Used'].iloc[0] if 'Model Used' in df.columns else "Unknown"
        model_name = f"{model_name}-{'GPU' if is_gpu_trial else 'CPU'}-{execution_mode.upper()}"
    
    print(f"Detected trial type: {trial_type.upper()}")
    print(f"Detected execution mode: {execution_mode.upper()}")
    print(f"Model: {model_name}")
    
    # Filter successful trials
    if success_only and 'Trial Success' in df.columns:
        df = df[df['Trial Success'] == 1.0]
        print(f"Filtering to successful trials: {len(df)} rows")
    
    # Convert numeric columns
    numeric_cols = [cpu_col, task_col, actor_col, ripper_task_col, runtime_col, station_col,
                    actual_ram_col, actual_vram_col, ram_util_col, vram_util_col,
                    total_req_ram_col, total_req_vram_col]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create unified concurrency column
    if execution_mode == 'ripper':
        df['Effective Concurrency'] = df[task_col].fillna(1)
    else:
        df['Effective Concurrency'] = df[actor_col].fillna(1)
    
    # Define required columns
    required_cols = [cpu_col, task_col, runtime_col, station_col, actual_ram_col]
    if is_gpu_trial:
        required_cols.append(actual_vram_col)
    
    existing_required = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_required)
    
    if df.empty:
        print("Error: No valid data after filtering. Check your CSV structure.")
        return
    
    print(f"Valid trials for visualization: {len(df)}")

    # =========================================================================
    # MARKER AND COLOR CONFIGURATION
    # =========================================================================
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
    
    # Add memory efficiency plots
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
    # Determine grouping column based on execution mode
    if execution_mode == 'ripper':
        group_col = 'Effective Concurrency'
        group_label = 'Tasks'
    elif is_gpu_trial:
        group_col = 'GPU Count'
        group_label = 'GPUs'
    else:
        group_col = 'Effective Concurrency'
        group_label = 'Actors'

    for dataset in datasets:
        curr_df = dataset['df']
        if curr_df.empty:
            continue
        
        title_suffix = f" (Filtered <= {filter_threshold}s)" if dataset['suffix'] else ""
        mode_suffix = f"_{execution_mode}"

        for config in plot_configs:
            if config['z_col'] not in curr_df.columns:
                continue
            
            if curr_df[config['z_col']].isna().all() or (curr_df[config['z_col']] == 0).all():
                continue
                
            fig = go.Figure()

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
                threshold_val = filter_threshold * 0.5
                x_max = curr_df[cpu_col].max()
                y_max = curr_df[task_col].max()
                
                fig.add_trace(go.Scatter3d(
                    x=[0, x_max], y=[y_max, y_max], z=[threshold_val, threshold_val],
                    mode='lines',
                    line=dict(color='red', width=4, dash='dash'),
                    name=f'{threshold_val:.0f}s Target',
                    showlegend=True
                ))
                fig.add_trace(go.Scatter3d(
                    x=[0, 0], y=[0, y_max], z=[threshold_val, threshold_val],
                    mode='lines',
                    line=dict(color='red', width=4, dash='dash'),
                    showlegend=False
                ))

            x_range = [0, curr_df[cpu_col].max() * 1.1]
            y_range = [0, curr_df[task_col].max() * 1.1]
            
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

            output_file = os.path.join(output_dir, f"{config['file_name']}{mode_suffix}{dataset['suffix']}.html")
            fig.write_html(output_file)
            print(f"Saved: {output_file}")

    # =========================================================================
    # ADDITIONAL 2D VISUALIZATIONS
    # =========================================================================
    
    # 1. Runtime vs Stations (2D scatter with concurrency coloring)
    fig = px.scatter(
        df, 
        x=station_col, 
        y=runtime_col,
        color='Effective Concurrency',
        size=cpu_col,
        hover_data=[cpu_col, task_col, actual_ram_col],
        title=f"[{model_name}] Runtime vs Workload Size",
        labels={
            station_col: 'Number of Stations',
            runtime_col: 'Runtime (s)',
            'Effective Concurrency': f'{"Actors" if execution_mode == "modelactor" else "Tasks"}',
            cpu_col: 'CPUs'
        },
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        xaxis_title='Number of Stations',
        yaxis_title='Runtime (s)'
    )
    output_file = os.path.join(output_dir, f"runtime_vs_stations_2d_{execution_mode}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")

    # 2. Memory Efficiency: Requested vs Actual RAM
    if total_req_ram_col in df.columns and actual_ram_col in df.columns:
        valid_mem = df[[total_req_ram_col, actual_ram_col, 'Effective Concurrency', station_col]].dropna()
        if len(valid_mem) > 0:
            fig = px.scatter(
                valid_mem,
                x=total_req_ram_col,
                y=actual_ram_col,
                color='Effective Concurrency',
                size=station_col,
                hover_data=[station_col],
                title=f"[{model_name}] Requested vs Actual RAM",
                labels={
                    total_req_ram_col: 'Total Requested RAM (MB)',
                    actual_ram_col: 'Process Tree RAM (MB)',
                    'Effective Concurrency': f'{"Actors" if execution_mode == "modelactor" else "Tasks"}'
                },
                color_continuous_scale='Plasma'
            )
            max_val = max(valid_mem[total_req_ram_col].max(), valid_mem[actual_ram_col].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='1:1 (Perfect Prediction)'
            ))
            output_file = os.path.join(output_dir, f"requested_vs_actual_ram_2d_{execution_mode}.html")
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
        hover_data=[runtime_col, 'Effective Concurrency'],
        title=f"[{model_name}] Throughput vs Concurrency",
        labels={
            task_col: 'Concurrent Tasks',
            'Throughput (Stations/s)': 'Throughput (Stations/s)',
            cpu_col: 'CPUs Allocated'
        },
        color_continuous_scale='Turbo'
    )
    output_file = os.path.join(output_dir, f"throughput_vs_concurrency_2d_{execution_mode}.html")
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
            max_val = max(valid_vram[total_req_vram_col].max(), valid_vram[actual_vram_col].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='1:1 (Perfect Prediction)'
            ))
            output_file = os.path.join(output_dir, f"requested_vs_actual_vram_2d_{execution_mode}.html")
            fig.write_html(output_file)
            print(f"Saved: {output_file}")

    print(f"\nVisualization complete! All files saved to: {output_dir}/")


def compare_modelactor_vs_ripper(modelactor_csv, ripper_csv, output_dir="visualizations"):
    """
    Create comparative visualizations between ModelActor and Ripper execution modes.
    """
    if not os.path.exists(modelactor_csv):
        print(f"Error: ModelActor CSV not found: {modelactor_csv}")
        return
    if not os.path.exists(ripper_csv):
        print(f"Error: Ripper CSV not found: {ripper_csv}")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n{'='*70}")
    print("MODELACTOR VS RIPPER COMPARISON VISUALIZATION")
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
    runtime_col = 'Total Run time for Picker (s)'
    station_col = 'Number of Stations Used'
    actual_ram_col = 'Process Tree RAM (MB)'
    actual_vram_col = 'Process Tree VRAM (MB)'
    task_col = 'Number of Concurrent Station Tasks'
    cpu_col = 'Number of CPUs Allocated for Ray to Use'
    
    # Add execution mode labels
    df_ma['Execution Mode'] = 'ModelActor'
    df_rp['Execution Mode'] = 'Ripper'
    
    # Calculate throughput
    df_ma['Throughput (Stations/s)'] = df_ma[station_col] / df_ma[runtime_col]
    df_rp['Throughput (Stations/s)'] = df_rp[station_col] / df_rp[runtime_col]
    
    # Combine datasets
    df_combined = pd.concat([df_ma, df_rp], ignore_index=True)
    
    # =========================================================================
    # COMPARISON VISUALIZATIONS
    # =========================================================================
    
    # 1. Throughput Distribution Comparison
    fig = px.histogram(
        df_combined,
        x='Throughput (Stations/s)',
        color='Execution Mode',
        barmode='overlay',
        opacity=0.7,
        title=f"[{model_name} - {trial_type}] Throughput Distribution: ModelActor vs Ripper",
        labels={'Throughput (Stations/s)': 'Throughput (Stations/s)'}
    )
    output_file = os.path.join(output_dir, f"comparison_throughput_dist_{model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # 2. Runtime vs Stations Comparison
    fig = px.scatter(
        df_combined,
        x=station_col,
        y=runtime_col,
        color='Execution Mode',
        size=cpu_col,
        hover_data=[task_col, 'Throughput (Stations/s)'],
        title=f"[{model_name} - {trial_type}] Runtime vs Workload Size: ModelActor vs Ripper",
        labels={
            station_col: 'Number of Stations',
            runtime_col: 'Runtime (s)',
            cpu_col: 'CPUs'
        }
    )
    output_file = os.path.join(output_dir, f"comparison_runtime_vs_stations_{model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # 3. Throughput Box Plot by Concurrent Tasks
    fig = px.box(
        df_combined,
        x=task_col,
        y='Throughput (Stations/s)',
        color='Execution Mode',
        title=f"[{model_name} - {trial_type}] Throughput by Concurrency: ModelActor vs Ripper",
        labels={
            task_col: 'Concurrent Tasks',
            'Throughput (Stations/s)': 'Throughput (Stations/s)'
        }
    )
    output_file = os.path.join(output_dir, f"comparison_throughput_by_tasks_{model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # 4. RAM Usage Comparison
    fig = px.box(
        df_combined,
        x='Execution Mode',
        y=actual_ram_col,
        color='Execution Mode',
        title=f"[{model_name} - {trial_type}] RAM Usage: ModelActor vs Ripper",
        labels={actual_ram_col: 'Process Tree RAM (MB)'}
    )
    output_file = os.path.join(output_dir, f"comparison_ram_usage_{model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # 5. Throughput Scaling by Concurrent Tasks (Line Chart)
    ma_by_task = df_ma.groupby(task_col)['Throughput (Stations/s)'].agg(['mean', 'std']).reset_index()
    ma_by_task['Execution Mode'] = 'ModelActor'
    rp_by_task = df_rp.groupby(task_col)['Throughput (Stations/s)'].agg(['mean', 'std']).reset_index()
    rp_by_task['Execution Mode'] = 'Ripper'
    
    scaling_df = pd.concat([ma_by_task, rp_by_task], ignore_index=True)
    
    fig = px.line(
        scaling_df,
        x=task_col,
        y='mean',
        color='Execution Mode',
        error_y='std',
        markers=True,
        title=f"[{model_name} - {trial_type}] Throughput Scaling: ModelActor vs Ripper",
        labels={
            task_col: 'Concurrent Tasks',
            'mean': 'Mean Throughput (Stations/s)'
        }
    )
    output_file = os.path.join(output_dir, f"comparison_throughput_scaling_{model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # 6. 3D Comparison: Runtime vs CPUs vs Tasks
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=['ModelActor', 'Ripper']
    )
    
    fig.add_trace(
        go.Scatter3d(
            x=df_ma[cpu_col],
            y=df_ma[task_col],
            z=df_ma[runtime_col],
            mode='markers',
            marker=dict(
                size=5,
                color=df_ma[station_col],
                colorscale='Turbo',
                opacity=0.7
            ),
            name='ModelActor'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter3d(
            x=df_rp[cpu_col],
            y=df_rp[task_col],
            z=df_rp[runtime_col],
            mode='markers',
            marker=dict(
                size=5,
                color=df_rp[station_col],
                colorscale='Turbo',
                opacity=0.7
            ),
            name='Ripper'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"[{model_name} - {trial_type}] 3D Runtime Comparison",
        scene=dict(
            xaxis_title='CPUs',
            yaxis_title='Concurrent Tasks',
            zaxis_title='Runtime (s)'
        ),
        scene2=dict(
            xaxis_title='CPUs',
            yaxis_title='Concurrent Tasks',
            zaxis_title='Runtime (s)'
        )
    )
    output_file = os.path.join(output_dir, f"comparison_3d_runtime_{model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # 7. Summary Statistics Table
    summary_data = {
        'Metric': [
            'Mean Throughput (st/s)',
            'Median Throughput (st/s)',
            'Max Throughput (st/s)',
            'Mean Runtime (s)',
            'Min Runtime (s)',
            'Mean RAM (MB)',
            'Trial Count'
        ],
        'ModelActor': [
            f"{df_ma['Throughput (Stations/s)'].mean():.2f}",
            f"{df_ma['Throughput (Stations/s)'].median():.2f}",
            f"{df_ma['Throughput (Stations/s)'].max():.2f}",
            f"{df_ma[runtime_col].mean():.2f}",
            f"{df_ma[runtime_col].min():.2f}",
            f"{df_ma[actual_ram_col].mean():.1f}",
            str(len(df_ma))
        ],
        'Ripper': [
            f"{df_rp['Throughput (Stations/s)'].mean():.2f}",
            f"{df_rp['Throughput (Stations/s)'].median():.2f}",
            f"{df_rp['Throughput (Stations/s)'].max():.2f}",
            f"{df_rp[runtime_col].mean():.2f}",
            f"{df_rp[runtime_col].min():.2f}",
            f"{df_rp[actual_ram_col].mean():.1f}",
            str(len(df_rp))
        ]
    }
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>ModelActor</b>', '<b>Ripper</b>'],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=14)
        ),
        cells=dict(
            values=[summary_data['Metric'], summary_data['ModelActor'], summary_data['Ripper']],
            fill_color='lavender',
            align='left',
            font=dict(size=12)
        )
    )])
    fig.update_layout(
        title=f"[{model_name} - {trial_type}] Summary Statistics: ModelActor vs Ripper"
    )
    output_file = os.path.join(output_dir, f"comparison_summary_table_{model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    print(f"\nComparison visualization complete! Files saved to: {output_dir}/")


def batch_visualize(results_root, output_dir="visualizations"):
    """
    Batch visualize all result directories.
    """
    if not os.path.exists(results_root):
        print(f"Error: Results root not found: {results_root}")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n{'='*70}")
    print("BATCH VISUALIZATION OF ALL TRIAL RESULTS")
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
    for result_dir in sorted(result_dirs):
        dir_path = os.path.join(results_root, result_dir)
        
        # Find CSV file
        csv_files = glob.glob(os.path.join(dir_path, '*_test_results.csv'))
        if not csv_files:
            print(f"  Skipping {result_dir}: No test results CSV found")
            continue
        
        csv_path = csv_files[0]
        
        # Create output directory for this visualization
        vis_output = os.path.join(output_dir, result_dir)
        
        print(f"\nVisualizing: {result_dir}")
        visualize_trials(csv_path, output_dir=vis_output)
    
    print(f"\n{'='*70}")
    print(f"Batch visualization complete! All files saved to: {output_dir}/")


def find_comparison_files(results_root, model, trial_type):
    """
    Find ModelActor and Ripper CSV files for a given model and trial type.
    """
    trial_type = trial_type.lower()
    model = model.lower()
    
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
        description='Visualize EQCCTPro Trial Results with Interactive Plotly Charts (ModelActor & Ripper)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file visualization
  python visualize_trial_results.py results/csv/eval_gpu_eqcct_modelactor/gpu_test_results.csv
  python visualize_trial_results.py results/csv/eval_cpu_eqcct_ripper/cpu_test_results.csv --output_dir vis/

  # With custom model name and runtime threshold
  python visualize_trial_results.py gpu_test_results.csv --model "PhaseNet-GPU" --threshold 60

  # Batch visualization of all results
  python visualize_trial_results.py --batch --results_root results/csv/ --output_dir batch_vis/

  # Compare ModelActor vs Ripper
  python visualize_trial_results.py --compare --model eqcct --trial_type cpu --results_root results/csv/
  python visualize_trial_results.py --compare --model phasenet_original --trial_type gpu --results_root results/csv/
        """
    )
    
    # Single file mode
    parser.add_argument('csv_path', type=str, nargs='?', default=None,
                       help='Path to the results CSV file')
    parser.add_argument('--model', type=str, default=None, 
                       help='Model name for plot titles (auto-detected if not provided)')
    parser.add_argument('--output_dir', type=str, default='visualizations', 
                       help='Directory to save HTML files (default: visualizations)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Runtime threshold in seconds for filtered plots (e.g., 60)')
    parser.add_argument('--include-failed', action='store_true',
                       help='Include failed trials in visualization (default: success only)')
    
    # Batch mode
    parser.add_argument('--batch', action='store_true',
                       help='Run batch visualization on all result directories')
    parser.add_argument('--results_root', type=str, default='results/csv/',
                       help='Root directory containing result subdirectories (for batch/compare modes)')
    
    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                       help='Compare ModelActor vs Ripper for a specific model')
    parser.add_argument('--trial_type', type=str, default=None, choices=['cpu', 'gpu'],
                       help='Trial type for comparison (cpu or gpu)')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch visualization mode
        batch_visualize(args.results_root, args.output_dir)
    elif args.compare:
        # Comparison mode
        if not args.model or not args.trial_type:
            print("Error: --compare requires --model and --trial_type arguments")
            parser.print_help()
        else:
            modelactor_csv, ripper_csv = find_comparison_files(args.results_root, args.model, args.trial_type)
            if modelactor_csv and ripper_csv:
                compare_modelactor_vs_ripper(modelactor_csv, ripper_csv, args.output_dir)
            else:
                print(f"Error: Could not find both ModelActor and Ripper results for {args.model} ({args.trial_type})")
                if not modelactor_csv:
                    print(f"  Missing: ModelActor CSV")
                if not ripper_csv:
                    print(f"  Missing: Ripper CSV")
    elif args.csv_path:
        # Single file visualization mode
        visualize_trials(
            csv_path=args.csv_path, 
            model_name=args.model, 
            output_dir=args.output_dir,
            filter_threshold=args.threshold,
            success_only=not args.include_failed
        )
    else:
        parser.print_help()
