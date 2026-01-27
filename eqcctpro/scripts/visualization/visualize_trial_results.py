"""
Interactive Trial Results Visualization with Plotly
====================================================

This script creates interactive 2D and 3D scatter plots for EQCCTPro evaluation results,
supporting both CPU and GPU trials, and both ModelActor and Ripper execution modes.

Features:
- Automatic detection of trial type (CPU vs GPU) and execution mode (ModelActor vs Ripper)
- Unified "Effective Concurrency" metric for hardware-aware resource scaling
- Interactive 3D visualizations: Runtime, RAM, VRAM, and Utilization vs CPUs & Workload
- Interactive 2D visualizations:
    - Runtime vs Workload (with optional --desired_runtime target line)
    - Memory Efficiency: Requested vs Actual RAM/VRAM (10K step size, 0-min axes)
    - Throughput vs Concurrency scaling
- Standardized Turbo (rainbow) color scale with 10-unit concurrency steps
- Comprehensive hover details including created actors, CPUs, and precise memory values
- Batch visualization: Automatically processes all trial directories into separate folders
- Comparison mode: Side-by-side ModelActor vs Ripper performance dashboards

Usage:
------
# Single file/directory visualization
python visualize_trial_results.py <csv_path_or_dir> [options]

# Batch visualization of all results in a root folder
python visualize_trial_results.py --batch --results_root results/csv/ --output_dir visualizations/

# Compare ModelActor vs Ripper for a specific model
python visualize_trial_results.py --compare --model eqcct --trial_type cpu --results_root results/csv/

Examples:
---------
# Single file visualization with desired runtime threshold
python visualize_trial_results.py results/csv/eval_cpu_eqcct_modelactor/ --desired_runtime 30

# GPU trial visualization (ModelActor)
python visualize_trial_results.py results/csv/eval_gpu_eqcct_modelactor/gpu_test_results.csv --output_dir vis/

# ModelActor vs Ripper comparison
python visualize_trial_results.py --compare --model phasenet_original --trial_type cpu
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
                     filter_threshold=None, success_only=True, desired_runtime=None):
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
    desired_runtime : float, optional
        If provided, adds a dashed red horizontal line to the 2D runtime plot.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Path not found at {csv_path}")
        return

    # If csv_path is a directory, try to find the test results CSV inside it
    if os.path.isdir(csv_path):
        csv_files = glob.glob(os.path.join(csv_path, '*_test_results.csv'))
        if not csv_files:
            print(f"Error: No '*_test_results.csv' file found in directory {csv_path}")
            return
        csv_path = csv_files[0]
        print(f"Detected CSV file in directory: {csv_path}")

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
    station_col = 'Number of Stations Used'
    
    # Timing columns
    total_trial_time_col = 'Total Trial Time (s)'           # Entire trial: setup + actor creation + processing
    actor_creation_time_col = 'Actor Creation Time (s)'     # Time to spin up ModelActors (empty for Ripper)
    avg_model_load_time_col = 'Avg Model Load Time (s)'     # Average model load time per task (Ripper only)
    waveform_proc_time_col = 'Waveform Processing Time (s)' # Average time to load waveforms per task
    picker_runtime_col = 'Total Run time for Picker (s)'    # Total time for all task processing
    runtime_col = total_trial_time_col  # Use total trial time for main plots
    
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
    numeric_cols = [cpu_col, task_col, actor_col, ripper_task_col, station_col,
                    total_trial_time_col, actor_creation_time_col, avg_model_load_time_col,
                    waveform_proc_time_col, picker_runtime_col,
                    actual_ram_col, actual_vram_col, ram_util_col, vram_util_col,
                    total_req_ram_col, total_req_vram_col]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate additional metrics for hover details
    if actor_creation_time_col in df.columns and actor_col in df.columns:
        df['Avg. ModelActor Creation Time (s)'] = df[actor_creation_time_col] / df[actor_col].replace(0, np.nan)
    else:
        df['Avg. ModelActor Creation Time (s)'] = np.nan
    
    # Create unified concurrency column
    df['Effective Concurrency'] = df[concurrency_col].fillna(1)
    
    # Calculate dynamic dtick for colorbar
    max_conc = df['Effective Concurrency'].max()
    if max_conc <= 15:
        cbar_dtick = 1
    elif max_conc <= 20:
        cbar_dtick = 5
    else:
        cbar_dtick = 10
    
    # Define symbol map for plotting
    # 3D scatter only supports: ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']
    symbol_map_dict = {
        0: 'circle',
        1: 'circle',
        2: 'cross',
        3: 'x',
        4: 'square',
        5: 'diamond',
        6: 'circle-open',
        7: 'square-open',
        8: 'diamond-open'
    }
    df['Marker Symbol'] = df['GPU Count'].apply(lambda x: symbol_map_dict.get(int(x), 'circle'))
    
    # Calculate CPU step size for plotting
    cpu_vals = sorted(df[cpu_col].unique())
    cpu_step = 1
    if len(cpu_vals) > 1:
        cpu_step = min(np.diff(cpu_vals))
    
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

    # Determine colorbar title based on execution mode
    cbar_title = "N Model Actors" if execution_mode == "modelactor" else "Concurrent Tasks"

    # =========================================================================
    # GENERATE 3D SCATTER PLOTS
    # =========================================================================
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

            # Set Z-axis dtick
            z_dtick = None
            if config['z_col'] == actual_ram_col:
                # Use 5K step if max RAM is < 40K, else 10K
                max_ram = curr_df[actual_ram_col].max()
                if max_ram < 40000:
                    z_dtick = 5000
                else:
                    z_dtick = 10000

            # Prepare hover template based on execution mode
            if execution_mode == 'modelactor':
                actor_hover = (
                    "Number of ModelActor's Created: %{customdata[3]}<br>"
                    "Avg. ModelActor Creation Time (s): %{customdata[6]:.4f}<br>"
                    "Total Actor Creation Time (s): %{customdata[7]:.2f}<br>"
                )
            else:
                actor_hover = ""
            
            # Add dummy traces for symbol legend if it's a GPU trial
            if is_gpu_trial:
                unique_gpus = sorted(curr_df['GPU Count'].unique())
                for gpu_count in unique_gpus:
                    symbol = symbol_map_dict.get(int(gpu_count), 'circle')
                    fig.add_trace(go.Scatter3d(
                        x=[None], y=[None], z=[None],
                        mode='markers',
                        marker=dict(
                            symbol=symbol,
                            color='rgba(0,0,0,0.5)', # Semi-transparent black
                            size=6,
                            line=dict(width=1, color='black')
                        ),
                        name=f"{int(gpu_count)} {'GPU' if gpu_count == 1 else 'GPUs'}",
                        legendgroup="GPU Count",
                        legendgrouptitle=dict(
                            text="GPUs Used",
                            font=dict(size=14) # Match colorbar title size
                        ),
                        showlegend=True
                    ))

            # Add a single trace for all data points to have a clean colorbar
            fig.add_trace(go.Scatter3d(
                x=curr_df[cpu_col],
                y=curr_df[station_col],
                z=curr_df[config['z_col']],
                mode='markers',
                marker=dict(
                    size=6,
                    color=curr_df['Effective Concurrency'],
                    colorscale='Turbo',
                    colorbar=dict(
                        title=dict(
                            text=cbar_title,
                            font=dict(size=14)
                        ),
                        dtick=cbar_dtick,
                        x=1.1, # Position colorbar
                        y=0.5,
                        len=0.9, # Increased length
                        yanchor='middle'
                    ),
                    cmin=curr_df['Effective Concurrency'].min(),
                    cmax=curr_df['Effective Concurrency'].max(),
                    opacity=0.8,
                    symbol=curr_df['Marker Symbol'],
                    line=dict(width=0)
                ),
                name='', # Removed 'Trials'
                showlegend=False, # Hide the main trace from legend
                hovertemplate=(
                    "<b>Trial Details</b><br>"
                    "Total Number of Stations to Process: %{y}<br>"
                    "CPUs: %{x}<br>"
                    "GPUs: %{customdata[0]}<br>"
                    "GPU IDs: %{customdata[1]}<br>"
                    "Concurrent Tasks Requested: %{customdata[2]}<br>"
                    + actor_hover +
                    "Avg. Waveform Processing Time (s): %{customdata[8]:.4f}<br>"
                    "Total Picking Time (s): %{customdata[9]:.2f}<br>"
                    "Total Trial Runtime (s): %{customdata[4]:.2f}<br>"
                    "Process Tree RAM (MB): %{customdata[5]:.2f}<br>"
                    "<extra></extra>"
                ),
                customdata=curr_df[['GPU Count', 'GPUs Used', task_col, actor_col, runtime_col, actual_ram_col, 
                                    'Avg. ModelActor Creation Time (s)', actor_creation_time_col, 
                                    waveform_proc_time_col, picker_runtime_col]].values
            ))

            # Add threshold lines for runtime plots
            if config['z_col'] == runtime_col and dataset['apply_threshold'] and filter_threshold:
                threshold_val = filter_threshold * 0.5
                x_max = curr_df[cpu_col].max()
                y_max = curr_df[station_col].max()
                
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
            y_range = [0, curr_df[station_col].max() * 1.1]
            
            fig.update_layout(
                title=dict(
                    text=f"[{model_name}]{title_suffix}<br>{config['title']}",
                    x=0.5,
                    xanchor='center'
                ),
                scene=dict(
                    xaxis=dict(title='CPUs Allocated', range=x_range, dtick=cpu_step),
                    yaxis=dict(title='Total Number of Stations to Process', range=y_range, dtick=10),
                    zaxis=dict(title=config['z_label'], dtick=z_dtick),
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=0.8)
                ),
                margin=dict(l=0, r=0, b=0, t=60),
                legend=dict(
                    x=1.05, # Pushed 0.02 back to the right
                    y=0.95, # Align with the top of the colorbar
                    xanchor='right',
                    yanchor='top',
                    bgcolor='rgba(255,255,255,0.5)',
                    font=dict(size=12) # Item font size
                ),
                showlegend=True
            )

            output_file = os.path.join(output_dir, f"{config['file_name']}{mode_suffix}{dataset['suffix']}.html")
            fig.write_html(output_file)
            print(f"Saved: {output_file}")

    # =========================================================================
    # ADDITIONAL 2D VISUALIZATIONS
    # =========================================================================
    
    # Common coloraxis settings
    coloraxes_dict = dict(
        colorbar=dict(
            title=cbar_title,
            dtick=cbar_dtick,
            x=1.1, # Closer to the plot
            y=0.5,
            len=0.75
        ),
        colorscale='Turbo'
    )

    # 1. Runtime vs Stations (2D scatter with concurrency coloring)
    if execution_mode == 'modelactor':
        actor_hover = (
            "Number of ModelActor's Created: %{customdata[3]}<br>"
            "Avg. ModelActor Creation Time (s): %{customdata[6]:.4f}<br>"
            "Total Actor Creation Time (s): %{customdata[7]:.2f}<br>"
        )
    else:
        actor_hover = ""
    fig = px.scatter(
        df, 
        x=station_col, 
        y=runtime_col,
        color='Effective Concurrency',
        size=runtime_col,
        symbol='GPU Count',
        symbol_map=symbol_map_dict,
        title=f"[{model_name}] Total Trial Runtime vs Workload Size",
        labels={'GPU Count': 'GPUs Used'}
    )
    fig.update_coloraxes(**coloraxes_dict)
    fig.update_traces(
        marker=dict(line=dict(width=0)),
        hovertemplate=(
            "<b>Trial Details</b><br>"
            "Total Number of Stations to Process: %{x}<br>"
            "CPUs: %{customdata[0]}<br>"
            "GPUs: %{customdata[1]}<br>"
            "GPU IDs: %{customdata[2]}<br>"
            "Concurrent Tasks Requested: %{customdata[4]}<br>"
            + actor_hover +
            "Avg. Waveform Processing Time (s): %{customdata[8]:.4f}<br>"
            "Total Picking Time (s): %{customdata[9]:.2f}<br>"
            "Total Trial Runtime (s): %{y:.2f}<br>"
            "Process Tree RAM (MB): %{customdata[5]:.2f}<br>"
            "<extra></extra>"
        ),
        customdata=df[[cpu_col, 'GPU Count', 'GPUs Used', actor_col, task_col, actual_ram_col, 
                       'Avg. ModelActor Creation Time (s)', actor_creation_time_col, 
                       waveform_proc_time_col, picker_runtime_col]].values
    )

    if desired_runtime is not None:
        x_min = df[station_col].min()
        x_max = df[station_col].max()
        fig.add_trace(go.Scatter(
            x=[x_min, x_max],
            y=[desired_runtime, desired_runtime],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name=f'Desired Runtime ({desired_runtime}s)'
        ))

    fig.update_layout(
        xaxis=dict(title='Total Number of Stations to Process', dtick=10),
        yaxis=dict(title='Total Trial Runtime (s)'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.5)'
        )
    )
    output_file = os.path.join(output_dir, f"runtime_vs_stations_2d_{execution_mode}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")

    # 2. Memory Efficiency: Requested vs Actual RAM
    if total_req_ram_col in df.columns and actual_ram_col in df.columns:
        valid_mem = df[[total_req_ram_col, actual_ram_col, 'Effective Concurrency', station_col, cpu_col, 
                       task_col, actor_col, runtime_col, 'GPU Count', 'GPUs Used', 'Marker Symbol',
                       'Avg. ModelActor Creation Time (s)', actor_creation_time_col, 
                       waveform_proc_time_col, picker_runtime_col]].dropna()
        if len(valid_mem) > 0:
            if execution_mode == 'modelactor':
                actor_hover = (
                    "Number of ModelActor's Created: %{customdata[4]}<br>"
                    "Avg. ModelActor Creation Time (s): %{customdata[7]:.4f}<br>"
                    "Total Actor Creation Time (s): %{customdata[8]:.2f}<br>"
                )
            else:
                actor_hover = ""
            fig = px.scatter(
                valid_mem,
                x=total_req_ram_col,
                y=actual_ram_col,
                color='Effective Concurrency',
                symbol='GPU Count',
                symbol_map=symbol_map_dict,
                title=f"[{model_name}] Requested vs Actual RAM",
                labels={'GPU Count': 'GPUs Used'}
            )
            fig.update_coloraxes(**coloraxes_dict)
            fig.update_traces(
                marker=dict(size=6, line=dict(width=0)),
                hovertemplate=(
                    "<b>Trial Details</b><br>"
                    "Total Number of Stations to Process: %{customdata[0]}<br>"
                    "CPUs: %{customdata[1]}<br>"
                    "GPUs: %{customdata[2]}<br>"
                    "GPU IDs: %{customdata[3]}<br>"
                    "Concurrent Tasks Requested: %{customdata[5]}<br>"
                    + actor_hover +
                    "Avg. Waveform Processing Time (s): %{customdata[9]:.4f}<br>"
                    "Total Picking Time (s): %{customdata[10]:.2f}<br>"
                    "Total Trial Runtime (s): %{customdata[6]:.2f}<br>"
                    "Process Tree RAM (MB): %{y:.2f}<br>"
                    "<extra></extra>"
                ),
                customdata=valid_mem[[station_col, cpu_col, 'GPU Count', 'GPUs Used', actor_col, task_col, runtime_col,
                                     'Avg. ModelActor Creation Time (s)', actor_creation_time_col, 
                                     waveform_proc_time_col, picker_runtime_col]].values
            )
            max_val = max(valid_mem[total_req_ram_col].max(), valid_mem[actual_ram_col].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Estimated Prediction RAM Cost'
            ))
            
            # Add desired runtime line if provided
            if desired_runtime is not None:
                fig.add_trace(go.Scatter(
                    x=[valid_mem[total_req_ram_col].min(), valid_mem[total_req_ram_col].max()],
                    y=[desired_runtime, desired_runtime],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    name=f'Desired Runtime ({desired_runtime}s)'
                ))
                
            fig.update_layout(
                xaxis=dict(title='Total Requested RAM (MB)', dtick=10000, range=[0, max_val * 1.05]),
                yaxis=dict(title='Process Tree RAM (MB)', dtick=10000, range=[0, max_val * 1.05]),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255,255,255,0.5)'
                )
            )
            output_file = os.path.join(output_dir, f"requested_vs_actual_ram_2d_{execution_mode}.html")
            fig.write_html(output_file)
            print(f"Saved: {output_file}")

    # 3. Throughput analysis
    df['Throughput (Stations/s)'] = df[station_col] / df[runtime_col]
    
    if execution_mode == 'modelactor':
        actor_hover = (
            "Number of ModelActor's Created: %{customdata[4]}<br>"
            "Avg. ModelActor Creation Time (s): %{customdata[7]:.4f}<br>"
            "Total Actor Creation Time (s): %{customdata[8]:.2f}<br>"
        )
    else:
        actor_hover = ""
    fig = px.scatter(
        df,
        x=task_col,
        y='Throughput (Stations/s)',
        color='Effective Concurrency',
        size=station_col,
        symbol='GPU Count',
        symbol_map=symbol_map_dict,
        title=f"[{model_name}] Throughput vs Concurrency",
        labels={'GPU Count': 'GPUs Used'}
    )
    fig.update_coloraxes(**coloraxes_dict)
    fig.update_traces(
        marker=dict(line=dict(width=0)),
        hovertemplate=(
            "<b>Trial Details</b><br>"
            "Total Number of Stations to Process: %{marker.size}<br>"
            "CPUs: %{customdata[0]}<br>"
            "GPUs: %{customdata[1]}<br>"
            "GPU IDs: %{customdata[2]}<br>"
            "Concurrent Tasks Requested: %{x}<br>"
            + actor_hover +
            "Avg. Waveform Processing Time (s): %{customdata[9]:.4f}<br>"
            "Total Picking Time (s): %{customdata[10]:.2f}<br>"
            "Total Trial Runtime (s): %{customdata[5]:.2f}<br>"
            "Process Tree RAM (MB): %{customdata[6]:.2f}<br>"
            "Throughput (Stations/s): %{y:.3f}<br>"
            "<extra></extra>"
        ),
        customdata=df[[cpu_col, 'GPU Count', 'GPUs Used', task_col, actor_col, runtime_col, actual_ram_col,
                       'Avg. ModelActor Creation Time (s)', actor_creation_time_col, 
                       waveform_proc_time_col, picker_runtime_col]].values
    )
    
    # Add desired runtime line if provided
    if desired_runtime is not None:
        fig.add_trace(go.Scatter(
            x=[df[task_col].min(), df[task_col].max()],
            y=[desired_runtime, desired_runtime],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name=f'Desired Runtime ({desired_runtime}s)'
        ))
        
    fig.update_layout(
        xaxis=dict(title='Concurrent Tasks Requested'),
        yaxis=dict(title='Throughput (Stations/s)'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.5)'
        )
    )
    output_file = os.path.join(output_dir, f"throughput_vs_concurrency_2d_{execution_mode}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")

    # 4. GPU-specific: VRAM requested vs actual
    if is_gpu_trial and total_req_vram_col in df.columns and actual_vram_col in df.columns:
        valid_vram = df[[total_req_vram_col, actual_vram_col, 'Effective Concurrency', station_col, cpu_col, 
                        task_col, actor_col, runtime_col, 'GPU Count', 'GPUs Used', 'Marker Symbol',
                        'Avg. ModelActor Creation Time (s)', actor_creation_time_col, 
                        waveform_proc_time_col, picker_runtime_col]].dropna()
        valid_vram = valid_vram[valid_vram[actual_vram_col] > 0]
        
        if len(valid_vram) > 0:
            if execution_mode == 'modelactor':
                actor_hover = (
                    "Number of ModelActor's Created: %{customdata[4]}<br>"
                    "Avg. ModelActor Creation Time (s): %{customdata[7]:.4f}<br>"
                    "Total Actor Creation Time (s): %{customdata[8]:.2f}<br>"
                )
            else:
                actor_hover = ""
            fig = px.scatter(
                valid_vram,
                x=total_req_vram_col,
                y=actual_vram_col,
                color='Effective Concurrency',
                symbol='GPU Count',
                symbol_map=symbol_map_dict,
                title=f"[{model_name}] Requested vs Actual VRAM",
                labels={'GPU Count': 'GPUs Used'}
            )
            fig.update_coloraxes(**coloraxes_dict)
            fig.update_traces(
                marker=dict(size=6, line=dict(width=0)),
                hovertemplate=(
                    "<b>Trial Details</b><br>"
                    "Total Number of Stations to Process: %{customdata[0]}<br>"
                    "CPUs: %{customdata[1]}<br>"
                    "GPUs: %{customdata[2]}<br>"
                    "GPU IDs: %{customdata[3]}<br>"
                    "Concurrent Tasks Requested: %{customdata[5]}<br>"
                    + actor_hover +
                    "Avg. Waveform Processing Time (s): %{customdata[9]:.4f}<br>"
                    "Total Picking Time (s): %{customdata[10]:.2f}<br>"
                    "Total Trial Runtime (s): %{customdata[6]:.2f}<br>"
                    "Process Tree VRAM (MB): %{y:.2f}<br>"
                    "<extra></extra>"
                ),
                customdata=valid_vram[[station_col, cpu_col, 'GPU Count', 'GPUs Used', actor_col, task_col, runtime_col,
                                      'Avg. ModelActor Creation Time (s)', actor_creation_time_col, 
                                      waveform_proc_time_col, picker_runtime_col]].values
            )
            max_val = max(valid_vram[total_req_vram_col].max(), valid_vram[actual_vram_col].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Estimated Prediction VRAM Cost'
            ))
            
            # Add desired runtime line if provided
            if desired_runtime is not None:
                fig.add_trace(go.Scatter(
                    x=[valid_vram[total_req_vram_col].min(), valid_vram[total_req_vram_col].max()],
                    y=[desired_runtime, desired_runtime],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    name=f'Desired Runtime ({desired_runtime}s)'
                ))
                
            fig.update_layout(
                xaxis=dict(title='Total Requested VRAM (MB)'),
                yaxis=dict(title='Process Tree VRAM (MB)'),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255,255,255,0.5)'
                )
            )
            output_file = os.path.join(output_dir, f"requested_vs_actual_vram_2d_{execution_mode}.html")
            fig.write_html(output_file)
            print(f"Saved: {output_file}")

    print(f"\nVisualization complete! All files saved to: {output_dir}")


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
    total_trial_time_col = 'Total Trial Time (s)'           # Entire trial
    picker_runtime_col = 'Total Run time for Picker (s)'    # Pure processing time
    actor_creation_time_col = 'Actor Creation Time (s)'     # Actor creation (empty for Ripper)
    avg_model_load_time_col = 'Avg Model Load Time (s)'     # Model load time (Ripper only)
    waveform_proc_time_col = 'Waveform Processing Time (s)' # Waveform load time
    runtime_col = picker_runtime_col  # Use picker runtime for 3D comparison plots
    station_col = 'Number of Stations Used'
    actual_ram_col = 'Process Tree RAM (MB)'
    actual_vram_col = 'Process Tree VRAM (MB)'
    task_col = 'Number of Concurrent Station Tasks'
    cpu_col = 'Number of CPUs Allocated for Ray to Use'
    actor_col = 'N ModelActors'
    ripper_task_col = 'Actual Ripper Concurrent Tasks'
    
    # Add execution mode labels
    df_ma['Execution Mode'] = 'ModelActor'
    df_rp['Execution Mode'] = 'Ripper'
    
    # Calculate throughput (using picker runtime - pure processing time)
    df_ma['Throughput (Stations/s)'] = df_ma[station_col] / df_ma[runtime_col]
    df_rp['Throughput (Stations/s)'] = df_rp[station_col] / df_rp[runtime_col]
    
    # Convert timing columns to numeric
    for col in [total_trial_time_col, picker_runtime_col, actor_creation_time_col, 
                avg_model_load_time_col, waveform_proc_time_col]:
        if col in df_ma.columns:
            df_ma[col] = pd.to_numeric(df_ma[col], errors='coerce')
        if col in df_rp.columns:
            df_rp[col] = pd.to_numeric(df_rp[col], errors='coerce')
    
    # Calculate additional metrics for hover details
    if actor_creation_time_col in df_ma.columns and actor_col in df_ma.columns:
        df_ma['Avg. ModelActor Creation Time (s)'] = df_ma[actor_creation_time_col] / df_ma[actor_col].replace(0, np.nan)
    else:
        df_ma['Avg. ModelActor Creation Time (s)'] = np.nan
    
    # Prepare hover strings based on trial type
    if not is_gpu:
        ma_hover = (
            "<b>ModelActor Method</b><br>"
            "Total Number of Stations to Process: %{y}<br>"
            "CPUs: %{x}<br>"
            "Concurrent Tasks Requested: %{customdata[2]}<br>"
            "Number of ModelActor's Created: %{customdata[3]}<br>"
            "Avg. ModelActor Creation Time (s): %{customdata[5]:.4f}<br>"
            "Total Actor Creation Time (s): %{customdata[6]:.2f}<br>"
            "Avg. Waveform Processing Time (s): %{customdata[7]:.4f}<br>"
            "Total Picking Time (s): %{customdata[8]:.2f}<br>"
            "Total Trial Runtime (s): %{customdata[9]:.2f}<br>"
            "Process Tree RAM (MB): %{customdata[4]:.2f}<br>"
            "<extra></extra>"
        )
        rp_hover = (
            "<b>Ripper Method</b><br>"
            "Total Number of Stations to Process: %{y}<br>"
            "CPUs: %{x}<br>"
            "Concurrent Tasks Requested: %{customdata[2]}<br>"
            "Concurrent Tasks Generated: %{customdata[5]}<br>"
            "Avg. Model Load Time (s): %{customdata[6]:.4f}<br>"
            "Avg. Waveform Processing Time (s): %{customdata[7]:.4f}<br>"
            "Total Picking Time (s): %{customdata[4]:.2f}<br>"
            "Total Trial Runtime (s): %{customdata[8]:.2f}<br>"
            "Process Tree RAM (MB): %{customdata[3]:.2f}<br>"
            "<extra></extra>"
        )
    else:
        ma_hover = (
            "<b>ModelActor Method</b><br>"
            "Total Number of Stations to Process: %{y}<br>"
            "CPUs: %{x}<br>"
            "GPUs: %{customdata[0]}<br>"
            "GPU IDs: %{customdata[1]}<br>"
            "Concurrent Tasks Requested: %{customdata[2]}<br>"
            "Number of ModelActor's Created: %{customdata[3]}<br>"
            "Avg. ModelActor Creation Time (s): %{customdata[5]:.4f}<br>"
            "Total Actor Creation Time (s): %{customdata[6]:.2f}<br>"
            "Avg. Waveform Processing Time (s): %{customdata[7]:.4f}<br>"
            "Total Picking Time (s): %{customdata[8]:.2f}<br>"
            "Total Trial Runtime (s): %{customdata[9]:.2f}<br>"
            "Process Tree RAM (MB): %{customdata[4]:.2f}<br>"
            "<extra></extra>"
        )
        rp_hover = (
            "<b>Ripper Method</b><br>"
            "Total Number of Stations to Process: %{y}<br>"
            "CPUs: %{x}<br>"
            "GPUs: %{customdata[0]}<br>"
            "GPU IDs: %{customdata[1]}<br>"
            "Concurrent Tasks Requested: %{customdata[2]}<br>"
            "Concurrent Tasks Generated: %{customdata[5]}<br>"
            "Avg. Model Load Time (s): %{customdata[6]:.4f}<br>"
            "Avg. Waveform Processing Time (s): %{customdata[7]:.4f}<br>"
            "Total Picking Time (s): %{customdata[4]:.2f}<br>"
            "Total Trial Runtime (s): %{customdata[8]:.2f}<br>"
            "Process Tree RAM (MB): %{customdata[3]:.2f}<br>"
            "<extra></extra>"
        )

    # Prepare unified generated tasks and labels for comparison
    df_ma['Generated Tasks'] = df_ma[actor_col]
    df_ma['Generated Label'] = "Number of ModelActor's Created:"
    
    # For ripper, use actual tasks if available, else requested
    if ripper_task_col in df_rp.columns and df_rp[ripper_task_col].notna().any():
        df_rp['Generated Tasks'] = df_rp[ripper_task_col]
    else:
        df_rp['Generated Tasks'] = df_rp[task_col]
    df_rp['Generated Label'] = "Concurrent Tasks Generated:"

    # Ensure all timing columns exist in both dataframes for combined analysis
    all_timing_cols = [total_trial_time_col, picker_runtime_col, actor_creation_time_col, 
                        avg_model_load_time_col, waveform_proc_time_col, 'Avg. ModelActor Creation Time (s)']
    for col in all_timing_cols:
        if col not in df_ma.columns: df_ma[col] = np.nan
        if col not in df_rp.columns: df_rp[col] = np.nan

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
        title=f"[{model_name} - {trial_type}] Throughput Distribution: ModelActor Method vs Ripper Method",
        labels={
            'Throughput (Stations/s)': 'Throughput (Total Stations/s)',
            'count': 'Number of Trials Achieved Benchmark'
        }
    )
    fig.update_traces(
        hovertemplate=(
            "<b>Performance Benchmark</b><br>"
            "Execution Mode: %{fullData.name}<br>"
            "Throughput (Total Stations/s): %{x}<br>"
            "Number of Trials Achieved Benchmark: %{y}<extra></extra>"
        )
    )
    fig.update_layout(
        yaxis_title="Number of Trials Achieved Benchmark",
        xaxis_title="Throughput (Total Stations/s)"
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
        size=runtime_col,
        title=f"[{model_name} - {trial_type}] Runtime vs Workload Size: ModelActor Method vs Ripper Method",
        labels={
            station_col: 'Total Number of Stations to Process',
            runtime_col: 'Total Trial Runtime (s)',
            cpu_col: 'CPUs'
        }
    )
    fig.update_traces(
        hovertemplate=(
            "<b>Trial Details</b><br>"
            "Execution Mode: %{customdata[0]}<br>"
            "Total Number of Stations to Process: %{x}<br>"
            "Number of Concurrent Station Tasks: %{customdata[1]}<br>"
            "%{customdata[2]} %{customdata[3]}<br>"
            "Avg. ModelActor Creation Time (s): %{customdata[5]:.4f}<br>"
            "Total Actor Creation Time (s): %{customdata[6]:.2f}<br>"
            "Avg. Model Load Time (s): %{customdata[10]:.4f}<br>"
            "Avg. Waveform Processing Time (s): %{customdata[7]:.4f}<br>"
            "Total Picking Time (s): %{customdata[8]:.2f}<br>"
            "Total Trial Runtime (s): %{y:.2f}<br>"
            "Throughput (Total Stations/s): %{customdata[4]:.2f}<br>"
            "<extra></extra>"
        ),
        customdata=df_combined[['Execution Mode', task_col, 'Generated Label', 'Generated Tasks', 'Throughput (Stations/s)',
                               'Avg. ModelActor Creation Time (s)', actor_creation_time_col, 
                               waveform_proc_time_col, picker_runtime_col, total_trial_time_col,
                               avg_model_load_time_col]].values
    )
    
    # Add desired runtime line if provided
    if desired_runtime is not None:
        fig.add_trace(go.Scatter(
            x=[df_combined[station_col].min(), df_combined[station_col].max()],
            y=[desired_runtime, desired_runtime],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name=f'Desired Runtime ({desired_runtime}s)'
        ))
        
    fig.update_layout(
        xaxis=dict(dtick=10)
    )
    output_file = os.path.join(output_dir, f"comparison_runtime_vs_stations_{model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # 3. RAM Usage Comparison
    ram_max = df_combined[actual_ram_col].max()
    ram_dtick = 5000 if ram_max < 40000 else 10000
    
    fig = px.box(
        df_combined,
        x='Execution Mode',
        y=actual_ram_col,
        color='Execution Mode',
        title=f"[{model_name} - {trial_type}] RAM Usage: ModelActor Method vs Ripper Method",
        labels={
            'Execution Mode': 'Execution Mode:',
            actual_ram_col: 'Process Tree RAM (MB):'
        }
    )
    fig.update_traces(
        hovertemplate="Execution Mode: %{x}<br>Process Tree RAM (MB): %{y:.2f}<extra></extra>"
    )
    fig.update_layout(
        yaxis=dict(dtick=ram_dtick)
    )
    output_file = os.path.join(output_dir, f"comparison_ram_usage_{model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # 5. Throughput Scaling by Concurrent Tasks (Line Chart)
    # Use the previously calculated 'Generated Tasks' and labels
    ma_by_task = df_ma.groupby(task_col).agg({
        'Throughput (Stations/s)': ['mean', 'std'],
        'Generated Tasks': 'mean'
    }).reset_index()
    ma_by_task.columns = [task_col, 'mean', 'std', 'Generated Tasks']
    ma_by_task['Execution Mode'] = 'ModelActor Method'
    ma_by_task['Generated Label'] = "Number of ModelActors Created:"
    
    rp_by_task = df_rp.groupby(task_col).agg({
        'Throughput (Stations/s)': ['mean', 'std'],
        'Generated Tasks': 'mean'
    }).reset_index()
    rp_by_task.columns = [task_col, 'mean', 'std', 'Generated Tasks']
    rp_by_task['Execution Mode'] = 'Ripper Method'
    rp_by_task['Generated Label'] = "Concurrent Tasks Generated:"
    
    scaling_df = pd.concat([ma_by_task, rp_by_task], ignore_index=True)
    
    fig = px.line(
        scaling_df,
        x=task_col,
        y='mean',
        color='Execution Mode',
        error_y='std',
        markers=True,
        title=f"[{model_name} - {trial_type}] Throughput Scaling: ModelActor Method vs Ripper Method",
        labels={
            task_col: 'Concurrent Tasks Requested',
            'mean': 'Mean Throughput (Total Stations/s)'
        }
    )
    fig.update_traces(
        hovertemplate=(
            "<b>Scaling Analysis</b><br>"
            "Execution Mode: %{fullData.name}<br>"
            "Concurrent Tasks Requested: %{x}<br>"
            "%{customdata[0]} %{customdata[1]:.0f}<br>"
            "Mean Throughput (Total Stations/s): %{y:.2f} ± %{customdata[2]:.2f}<br>"
            "<extra></extra>"
        ),
        customdata=scaling_df[['Generated Label', 'Generated Tasks', 'std']].values
    )
    fig.update_layout(
        xaxis=dict(dtick=10)
    )
    output_file = os.path.join(output_dir, f"comparison_throughput_scaling_{model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # Create symbol columns
    # 3D scatter only supports: ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']
    symbol_map_dict = {0: 'circle', 1: 'circle', 2: 'cross', 3: 'x', 4: 'square', 5: 'diamond', 6: 'circle-open', 7: 'square-open', 8: 'diamond-open'}
    df_ma['Marker Symbol'] = df_ma['GPU Count'].apply(lambda x: symbol_map_dict.get(int(x), 'circle'))
    df_rp['Marker Symbol'] = df_rp['GPU Count'].apply(lambda x: symbol_map_dict.get(int(x), 'circle'))
    
    # 6. 3D Comparison: Runtime vs CPUs vs Tasks
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=['ModelActor Method', 'Ripper Method']
    )
    
    # Calculate shared color scale limits
    conc_min = min(df_ma[actor_col].min() if not df_ma.empty else 0, 
                  df_rp[task_col].min() if not df_rp.empty else 0)
    conc_max = max(df_ma[actor_col].max() if not df_ma.empty else 1, 
                  df_rp[task_col].max() if not df_rp.empty else 1)
    
    # Calculate dtick for comparison colorbar
    if conc_max <= 15:
        comp_dtick = 1
    elif conc_max <= 20:
        comp_dtick = 5
    else:
        comp_dtick = 10

    fig.add_trace(
        go.Scatter3d(
            x=df_ma[cpu_col],
            y=df_ma[station_col],
            z=df_ma[runtime_col],
            mode='markers',
            marker=dict(
                size=5,
                color=df_ma[actor_col],
                colorscale='Turbo',
                colorbar=dict(
                    title=dict(
                        text="Effective Concurrency",
                        font=dict(size=14)
                    ), 
                    dtick=comp_dtick, 
                    x=1.1, 
                    y=0.5, 
                    len=0.9,
                    yanchor='middle'
                ),
                cmin=conc_min,
                cmax=conc_max,
                opacity=0.8,
                symbol=df_ma['Marker Symbol'],
                line=dict(width=0) # Removed outline
            ),
            name='ModelActor Method',
            showlegend=False, # Removed dot in legend
            hovertemplate=ma_hover,
            customdata=df_ma[['GPU Count', 'GPUs Used', task_col, actor_col, actual_ram_col,
                             'Avg. ModelActor Creation Time (s)', actor_creation_time_col, 
                             waveform_proc_time_col, picker_runtime_col, total_trial_time_col]].values
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter3d(
            x=df_rp[cpu_col],
            y=df_rp[station_col],
            z=df_rp[runtime_col],
            mode='markers',
            marker=dict(
                size=5,
                color=df_rp[task_col],
                colorscale='Turbo',
                cmin=conc_min,
                cmax=conc_max,
                opacity=0.8,
                symbol=df_rp['Marker Symbol'],
                line=dict(width=0) # Removed outline
            ),
            name='Ripper Method',
            showlegend=False, # Removed dot in legend
            hovertemplate=rp_hover,
            customdata=df_rp[['GPU Count', 'GPUs Used', task_col, actual_ram_col, picker_runtime_col,
                             ripper_task_col, avg_model_load_time_col, waveform_proc_time_col,
                             total_trial_time_col]].values
        ),
        row=1, col=2
    )
    
    # Add dummy traces for symbol legend if it's a GPU trial
    is_gpu = df_ma['GPU Count'].max() > 0 or df_rp['GPU Count'].max() > 0
    if is_gpu:
        all_unique_gpus = sorted(pd.concat([df_ma['GPU Count'], df_rp['GPU Count']]).unique())
        for gpu_count in all_unique_gpus:
            if gpu_count == 0: continue
            symbol = symbol_map_dict.get(int(gpu_count), 'circle')
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(
                    symbol=symbol,
                    color='rgba(0,0,0,0.5)',
                    size=6,
                    line=dict(width=1, color='black')
                ),
                name=f"{int(gpu_count)} {'GPU' if gpu_count == 1 else 'GPUs'}",
                legendgroup="GPU Count",
                legendgrouptitle=dict(
                    text="GPUs Used",
                    font=dict(size=14)
                ),
                showlegend=True
            ), row=1, col=1)

    fig.update_layout(
        title=f"[{model_name} - {trial_type}] 3D Picker Runtime Comparison",
        showlegend=True,
        legend=dict(
            x=1.05,
            y=0.95,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.5)',
            font=dict(size=12)
        ),
        scene=dict(
            xaxis=dict(title='CPUs Allocated', dtick=1),
            yaxis=dict(title='Total Number of Stations to Process', dtick=10),
            zaxis_title='Picker Runtime (s)'
        ),
        scene2=dict(
            xaxis=dict(title='CPUs Allocated', dtick=1),
            yaxis=dict(title='Total Number of Stations to Process', dtick=10),
            zaxis_title='Picker Runtime (s)'
        )
    )
    output_file = os.path.join(output_dir, f"comparison_3d_picker_runtime_{model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # 7. Summary Statistics Table - including new timing metrics
    # Helper function to safely format values
    def safe_format(series, fmt=".2f", default="N/A"):
        if series.notna().any():
            return f"{series.mean():{fmt}}"
        return default
    
    def safe_format_min(series, fmt=".2f", default="N/A"):
        if series.notna().any():
            return f"{series.min():{fmt}}"
        return default
    
    # Build summary table with comprehensive timing metrics
    summary_data = {
        'Metric': [
            '--- Throughput Metrics ---',
            'Mean Throughput (st/s)',
            'Median Throughput (st/s)',
            'Max Throughput (st/s)',
            '--- Runtime Metrics ---',
            'Mean Total Trial Time (s)',
            'Min Total Trial Time (s)',
            'Mean Picker Runtime (s)',
            'Min Picker Runtime (s)',
            '--- Setup Time Metrics ---',
            'Mean Actor Creation Time (s)',
            'Mean Avg Model Load Time (s)',
            'Mean Waveform Processing Time (s)',
            '--- Memory Metrics ---',
            'Mean RAM (MB)',
            '--- Trial Info ---',
            'Trial Count'
        ],
        'ModelActor': [
            '',
            f"{df_ma['Throughput (Stations/s)'].mean():.2f}",
            f"{df_ma['Throughput (Stations/s)'].median():.2f}",
            f"{df_ma['Throughput (Stations/s)'].max():.2f}",
            '',
            safe_format(df_ma[total_trial_time_col]) if total_trial_time_col in df_ma.columns else "N/A",
            safe_format_min(df_ma[total_trial_time_col]) if total_trial_time_col in df_ma.columns else "N/A",
            safe_format(df_ma[picker_runtime_col]) if picker_runtime_col in df_ma.columns else "N/A",
            safe_format_min(df_ma[picker_runtime_col]) if picker_runtime_col in df_ma.columns else "N/A",
            '',
            safe_format(df_ma[actor_creation_time_col]) if actor_creation_time_col in df_ma.columns else "N/A",
            safe_format(df_ma[avg_model_load_time_col]) if avg_model_load_time_col in df_ma.columns else "N/A",
            safe_format(df_ma[waveform_proc_time_col], ".4f") if waveform_proc_time_col in df_ma.columns else "N/A",
            '',
            f"{df_ma[actual_ram_col].mean():.1f}",
            '',
            str(len(df_ma))
        ],
        'Ripper': [
            '',
            f"{df_rp['Throughput (Stations/s)'].mean():.2f}",
            f"{df_rp['Throughput (Stations/s)'].median():.2f}",
            f"{df_rp['Throughput (Stations/s)'].max():.2f}",
            '',
            safe_format(df_rp[total_trial_time_col]) if total_trial_time_col in df_rp.columns else "N/A",
            safe_format_min(df_rp[total_trial_time_col]) if total_trial_time_col in df_rp.columns else "N/A",
            safe_format(df_rp[picker_runtime_col]) if picker_runtime_col in df_rp.columns else "N/A",
            safe_format_min(df_rp[picker_runtime_col]) if picker_runtime_col in df_rp.columns else "N/A",
            '',
            "N/A (no actors)",  # Ripper mode doesn't create actors
            safe_format(df_rp[avg_model_load_time_col]) if avg_model_load_time_col in df_rp.columns else "N/A",
            safe_format(df_rp[waveform_proc_time_col], ".4f") if waveform_proc_time_col in df_rp.columns else "N/A",
            '',
            f"{df_rp[actual_ram_col].mean():.1f}",
            '',
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


def batch_visualize(results_root, output_dir="visualizations", desired_runtime=None):
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
        visualize_trials(csv_path, output_dir=vis_output, desired_runtime=desired_runtime)
    
    print(f"\n{'='*70}")
    print(f"Batch visualization complete! All files saved to: {output_dir}")


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
    parser.add_argument('--desired_runtime', type=float, default=None,
                       help='Add a desired runtime horizontal line (s) to the 2D runtime plot')
    
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
        batch_visualize(args.results_root, args.output_dir, desired_runtime=args.desired_runtime)
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
            success_only=not args.include_failed,
            desired_runtime=args.desired_runtime
        )
    else:
        parser.print_help()
