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
- Optimal Configurations: 3D visualizations and comparison tables for optimal_configurations_*.csv files
    - 3D scatter plots: CPUs (x) vs Stations (y) vs Runtime/Picking Time (z)
    - Rainbow (Turbo) color scale for number of concurrent tasks
    - Different marker shapes for GPU count
    - Summary comparison tables

Usage:
------
# Single file/directory visualization
python visualize_trial_results.py <csv_path_or_dir> [options]

# Batch visualization of all results in a root folder
python visualize_trial_results.py --batch --results_root results/csv/ --output_dir visualizations/

# Compare ModelActor vs Ripper for a specific model
python visualize_trial_results.py --compare --model eqcct --trial_type cpu --results_root results/csv/

# Optimal Configurations Visualization
python visualize_trial_results.py --optimal <optimal_config_csv_path> --output_dir vis/optimal/

# Batch optimal configurations visualization (individual per-trial)
python visualize_trial_results.py --optimal --batch --results_root results/trials/ --output_dir vis/optimal/

# Compare optimal configs for a single model (CPU vs GPU, ModelActor vs Ripper)
python visualize_trial_results.py --optimal --compare --model eqcct --results_root results/trials/

# Batch optimal comparison (all models, auto-discovers from results_root)
python visualize_trial_results.py --optimal --compare --batch --results_root results/trials/ --output_dir vis/optimal_comparisons/

Examples:
---------
# Single file visualization with desired runtime threshold
python visualize_trial_results.py results/csv/eval_cpu_eqcct_modelactor/ --desired_runtime 30

# GPU trial visualization (ModelActor)
python visualize_trial_results.py results/csv/eval_gpu_eqcct_modelactor/gpu_test_results.csv --output_dir vis/

# ModelActor vs Ripper comparison
python visualize_trial_results.py --compare --model phasenet_original --trial_type cpu

# Single optimal config visualization
python visualize_trial_results.py --optimal results/trials/eval_cpu_eqcct_modelactor/optimal_configurations_cpu.csv

# Compare optimal configs for eqcct model
python visualize_trial_results.py --optimal --compare --model eqcct --results_root results/trials/
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
    if pd.isna(gpu_val) or gpu_val == '' or str(gpu_val).lower() == 'nan' or str(gpu_val).lower() == 'n/a':
        return 0
    
    s = str(gpu_val).strip()
    # Handle list format [0] or [0, 1]
    if s.startswith('[') and s.endswith(']'):
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return len(parsed)
        except:
            # Fallback for manually counting commas in [0, 1]
            s = s[1:-1]
            if not s.strip(): return 0
            return len([p.strip() for p in s.split(',') if p.strip()])
            
    # Handle comma separated 0,1 or single number
    try:
        if ',' in s:
            return len([p.strip() for p in s.split(',') if p.strip()])
        return 1 if int(float(s)) >= 0 else 0
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
                     filter_threshold=None, success_only=True, desired_runtime=None, 
                     dot_growth=False):
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
    dot_growth : bool
        If True, makes dot sizes grow with workload/runtime in 2D plots.
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
    if 'GPUs Used' in df.columns:
        df['GPU Count'] = df['GPUs Used'].apply(parse_gpu_list)
    else:
        df['GPU Count'] = 0
    
    # Auto-detect model name
    if model_name is None:
        model_name = df['Model Used'].iloc[0] if 'Model Used' in df.columns else "Unknown"
        model_name = f"{model_name}-{'GPU' if is_gpu_trial else 'CPU'}-{execution_mode.upper()}"
    
    # Sanitize model name for filenames (replace / with _)
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    
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
            'title': 'Total Trial Runtime vs Resources', 
            'z_label': 'Total Trial Runtime (s)', 
            'file_name': 'runtime_3d',
            'show_threshold': True
        },
        {
            'z_col': picker_runtime_col, 
            'title': 'Total Waveform Picking Time vs Resources', 
            'z_label': 'Total Waveform Picking Time (s)', 
            'file_name': 'picking_time_3d',
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
                    "Avg. ModelActor Creation Time (s): %{customdata[6]:.2f}<br>"
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
                    "Avg. Waveform Processing Time (s): %{customdata[8]:.2f}<br>"
                    "Total Waveform Picking Time (s): %{customdata[9]:.2f}<br>"
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
                    xaxis=dict(title='CPUs Allocated', range=x_range, dtick=1),
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
    point_size_runtime = runtime_col if dot_growth else None
    
    if execution_mode == 'modelactor':
        actor_hover = (
            "Number of ModelActor's Created: %{customdata[3]}<br>"
            "Avg. ModelActor Creation Time (s): %{customdata[6]:.2f}<br>"
            "Total Actor Creation Time (s): %{customdata[7]:.2f}<br>"
        )
    else:
        actor_hover = ""
    fig = px.scatter(
        df, 
        x=station_col, 
        y=runtime_col,
        color='Effective Concurrency',
        size=point_size_runtime,
        symbol='GPU Count',
        symbol_map=symbol_map_dict,
        custom_data=[cpu_col, 'GPU Count', 'GPUs Used', actor_col, task_col, actual_ram_col, 
                     'Avg. ModelActor Creation Time (s)', actor_creation_time_col, 
                     waveform_proc_time_col, picker_runtime_col],
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
            "Avg. Waveform Processing Time (s): %{customdata[8]:.2f}<br>"
            "Total Waveform Picking Time (s): %{customdata[9]:.2f}<br>"
            "Total Trial Runtime (s): %{y:.2f}<br>"
            "Process Tree RAM (MB): %{customdata[5]:.2f}<br>"
            "<extra></extra>"
        )
    )

    if desired_runtime is not None:
        x_min = df[station_col].min()
        x_max = df[station_col].max()
        fig.add_trace(go.Scatter(
            x=[x_min, x_max],
            y=[desired_runtime, desired_runtime],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            showlegend=False
        ))
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.02, y=-0.1,
            text=f"Target: {desired_runtime}s",
            showarrow=False,
            font=dict(color="red", size=13, family="Arial Black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
            xanchor="left",
            yanchor="top"
        )
        fig.update_layout(margin=dict(b=80))

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
                    "Avg. ModelActor Creation Time (s): %{customdata[7]:.2f}<br>"
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
                custom_data=[station_col, cpu_col, 'GPU Count', 'GPUs Used', actor_col, task_col, runtime_col,
                             'Avg. ModelActor Creation Time (s)', actor_creation_time_col, 
                             waveform_proc_time_col, picker_runtime_col],
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
                    "Avg. Waveform Processing Time (s): %{customdata[9]:.2f}<br>"
                    "Total Waveform Picking Time (s): %{customdata[10]:.2f}<br>"
                    "Total Trial Runtime (s): %{customdata[6]:.2f}<br>"
                    "Process Tree RAM (MB): %{y:.2f}<br>"
                    "<extra></extra>"
                )
            )
            max_val = max(valid_mem[total_req_ram_col].max(), valid_mem[actual_ram_col].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Estimated Prediction RAM Cost'
            ))
            
        # Add target info block (consistent positioning below color bar)
        if desired_runtime is not None:
            fig.add_annotation(
                xref="paper", yref="paper",
                x=1.02, y=-0.1,
                text=f"Target: {desired_runtime}s",
                showarrow=False,
                font=dict(color="red", size=13, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1,
                xanchor="left",
                yanchor="top"
            )
            fig.update_layout(margin=dict(b=80))

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
    
    point_size_throughput = station_col if dot_growth else None
    
    if execution_mode == 'modelactor':
        actor_hover = (
            "Number of ModelActor's Created: %{customdata[4]}<br>"
            "Avg. ModelActor Creation Time (s): %{customdata[7]:.2f}<br>"
            "Total Actor Creation Time (s): %{customdata[8]:.2f}<br>"
        )
    else:
        actor_hover = ""
    fig = px.scatter(
        df,
        x=task_col,
        y='Throughput (Stations/s)',
        color='Effective Concurrency',
        size=point_size_throughput,
        symbol='GPU Count',
        symbol_map=symbol_map_dict,
        custom_data=[cpu_col, 'GPU Count', 'GPUs Used', task_col, actor_col, runtime_col, actual_ram_col,
                     'Avg. ModelActor Creation Time (s)', actor_creation_time_col, 
                     waveform_proc_time_col, picker_runtime_col],
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
            "Avg. Waveform Processing Time (s): %{customdata[9]:.2f}<br>"
            "Total Waveform Picking Time (s): %{customdata[10]:.2f}<br>"
            "Total Trial Runtime (s): %{customdata[5]:.2f}<br>"
            "Process Tree RAM (MB): %{customdata[6]:.2f}<br>"
            "Throughput (Stations/s): %{y:.3f}<br>"
            "<extra></extra>"
        )
    )
    
    # Add target info block (consistent positioning below color bar)
    if desired_runtime is not None:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.02, y=-0.1,
            text=f"Target: {desired_runtime}s",
            showarrow=False,
            font=dict(color="red", size=13, family="Arial Black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
            xanchor="left",
            yanchor="top"
        )
        fig.update_layout(margin=dict(b=80))

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
                    "Avg. ModelActor Creation Time (s): %{customdata[7]:.2f}<br>"
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
                custom_data=[station_col, cpu_col, 'GPU Count', 'GPUs Used', actor_col, task_col, runtime_col,
                             'Avg. ModelActor Creation Time (s)', actor_creation_time_col, 
                             waveform_proc_time_col, picker_runtime_col],
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
                    "Avg. Waveform Processing Time (s): %{customdata[9]:.2f}<br>"
                    "Total Waveform Picking Time (s): %{customdata[10]:.2f}<br>"
                    "Total Trial Runtime (s): %{customdata[6]:.2f}<br>"
                    "Process Tree VRAM (MB): %{y:.2f}<br>"
                    "<extra></extra>"
                )
            )
            max_val = max(valid_vram[total_req_vram_col].max(), valid_vram[actual_vram_col].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Estimated Prediction VRAM Cost'
            ))
            
        # Add target info block (consistent positioning below color bar)
        if desired_runtime is not None:
            fig.add_annotation(
                xref="paper", yref="paper",
                x=1.02, y=-0.1,
                text=f"Target: {desired_runtime}s",
                showarrow=False,
                font=dict(color="red", size=13, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1,
                xanchor="left",
                yanchor="top"
            )
            fig.update_layout(margin=dict(b=80))

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


def compare_modelactor_vs_ripper(modelactor_csv, ripper_csv, output_dir="visualizations", desired_runtime=None):
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
    
    # Sanitize model name for filenames (replace / with _)
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    
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
    runtime_col = total_trial_time_col  # Use total trial time for runtime comparison plots
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
    
    # Calculate BOTH throughput metrics:
    # 1. Picker Throughput = Stations / Picker Runtime (pure processing speed, excludes setup)
    # 2. Total Throughput = Stations / Total Trial Time (end-to-end including setup)
    df_ma['Picker Throughput (st/s)'] = df_ma[station_col] / df_ma[picker_runtime_col]
    df_rp['Picker Throughput (st/s)'] = df_rp[station_col] / df_rp[picker_runtime_col]
    df_ma['Total Throughput (st/s)'] = df_ma[station_col] / df_ma[total_trial_time_col]
    df_rp['Total Throughput (st/s)'] = df_rp[station_col] / df_rp[total_trial_time_col]
    # Keep backward compatibility
    df_ma['Throughput (Stations/s)'] = df_ma['Total Throughput (st/s)']
    df_rp['Throughput (Stations/s)'] = df_rp['Total Throughput (st/s)']
    
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
            "Avg. ModelActor Creation Time (s): %{customdata[5]:.2f}<br>"
            "Total Actor Creation Time (s): %{customdata[6]:.2f}<br>"
            "Avg. Waveform Processing Time (s): %{customdata[7]:.2f}<br>"
            "Total Waveform Picking Time (s): %{customdata[8]:.2f}<br>"
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
            "Avg. Model Load Time (s): %{customdata[6]:.2f}<br>"
            "Avg. Waveform Processing Time (s): %{customdata[7]:.2f}<br>"
            "Total Waveform Picking Time (s): %{customdata[4]:.2f}<br>"
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
            "Avg. ModelActor Creation Time (s): %{customdata[5]:.2f}<br>"
            "Total Actor Creation Time (s): %{customdata[6]:.2f}<br>"
            "Avg. Waveform Processing Time (s): %{customdata[7]:.2f}<br>"
            "Total Waveform Picking Time (s): %{customdata[8]:.2f}<br>"
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
            "Avg. Model Load Time (s): %{customdata[6]:.2f}<br>"
            "Avg. Waveform Processing Time (s): %{customdata[7]:.2f}<br>"
            "Total Waveform Picking Time (s): %{customdata[4]:.2f}<br>"
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
        x='Picker Throughput (st/s)',
        color='Execution Mode',
        barmode='overlay',
        opacity=0.7,
        title=f"[{model_name} - {trial_type}] Picker Throughput Distribution: ModelActor Method vs Ripper Method",
        labels={
            'Picker Throughput (st/s)': 'Picker Throughput (Stations/Picker Runtime)',
            'count': 'Number of Trials Achieved Benchmark'
        }
    )
    fig.update_traces(
        hovertemplate=(
            "<b>Performance Benchmark</b><br>"
            "Execution Mode: %{fullData.name}<br>"
            "Picker Throughput (st/s): %{x}<br>"
            "Number of Trials Achieved Benchmark: %{y}<extra></extra>"
        )
    )
    fig.update_layout(
        yaxis_title="Number of Trials Achieved Benchmark",
        xaxis_title="Picker Throughput (Stations/Picker Runtime)"
    )
    output_file = os.path.join(output_dir, f"comparison_picker_throughput_dist_{safe_model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # 2. Runtime vs Stations Comparison
    fig = px.scatter(
        df_combined,
        x=station_col,
        y=runtime_col,
        color='Execution Mode',
        size=runtime_col,
        custom_data=['Execution Mode', task_col, 'Generated Label', 'Generated Tasks', 'Total Throughput (st/s)',
                     'Avg. ModelActor Creation Time (s)', actor_creation_time_col, 
                     waveform_proc_time_col, picker_runtime_col, total_trial_time_col,
                     avg_model_load_time_col, 'Picker Throughput (st/s)'],
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
            "Avg. ModelActor Creation Time (s): %{customdata[5]:.2f}<br>"
            "Total Actor Creation Time (s): %{customdata[6]:.2f}<br>"
            "Avg. Model Load Time (s): %{customdata[10]:.2f}<br>"
            "Avg. Waveform Processing Time (s): %{customdata[7]:.2f}<br>"
            "Total Waveform Picking Time (s): %{customdata[8]:.2f}<br>"
            "Total Trial Runtime (s): %{y:.2f}<br>"
            "--- Throughput ---<br>"
            "Picker Throughput (st/s): %{customdata[11]:.2f}<br>"
            "Total Throughput (st/s): %{customdata[4]:.2f}<br>"
            "<extra></extra>"
        ),
    )
    
    # Add desired runtime line if provided
    if desired_runtime is not None:
        fig.add_trace(go.Scatter(
            x=[df_combined[station_col].min(), df_combined[station_col].max()],
            y=[desired_runtime, desired_runtime],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            showlegend=False
        ))
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.02, y=-0.1,
            text=f"Target: {desired_runtime}s",
            showarrow=False,
            font=dict(color="red", size=13, family="Arial Black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
            xanchor="left",
            yanchor="top"
        )
        fig.update_layout(margin=dict(b=80))
        
    fig.update_layout(
        xaxis=dict(dtick=10)
    )
    output_file = os.path.join(output_dir, f"comparison_runtime_vs_stations_{safe_model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")

    # 2b. Picking Time vs Stations Comparison
    fig = px.scatter(
        df_combined,
        x=station_col,
        y=picker_runtime_col,
        color='Execution Mode',
        size=picker_runtime_col,
        custom_data=['Execution Mode', task_col, 'Generated Label', 'Generated Tasks', 'Total Throughput (st/s)',
                     'Avg. ModelActor Creation Time (s)', actor_creation_time_col, 
                     waveform_proc_time_col, picker_runtime_col, total_trial_time_col,
                     avg_model_load_time_col, 'Picker Throughput (st/s)'],
        title=f"[{model_name} - {trial_type}] Picking Time vs Workload Size: ModelActor Method vs Ripper Method",
        labels={
            station_col: 'Total Number of Stations to Process',
            picker_runtime_col: 'Total Waveform Picking Time (s)',
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
            "Avg. ModelActor Creation Time (s): %{customdata[5]:.2f}<br>"
            "Total Actor Creation Time (s): %{customdata[6]:.2f}<br>"
            "Avg. Model Load Time (s): %{customdata[10]:.2f}<br>"
            "Avg. Waveform Processing Time (s): %{customdata[7]:.2f}<br>"
            "Total Waveform Picking Time (s): %{y:.2f}<br>"
            "Total Trial Runtime (s): %{customdata[9]:.2f}<br>"
            "--- Throughput ---<br>"
            "Picker Throughput (st/s): %{customdata[11]:.2f}<br>"
            "Total Throughput (st/s): %{customdata[4]:.2f}<br>"
            "<extra></extra>"
        ),
    )

    # Add desired runtime line if provided
    if desired_runtime is not None:
        fig.add_trace(go.Scatter(
            x=[df_combined[station_col].min(), df_combined[station_col].max()],
            y=[desired_runtime, desired_runtime],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            showlegend=False
        ))
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.02, y=-0.1,
            text=f"Target: {desired_runtime}s",
            showarrow=False,
            font=dict(color="red", size=13, family="Arial Black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
            xanchor="left",
            yanchor="top"
        )
        fig.update_layout(margin=dict(b=80))

    fig.update_layout(
        xaxis=dict(dtick=10)
    )
    output_file = os.path.join(output_dir, f"comparison_picking_time_vs_stations_{safe_model_name}_{trial_type.lower()}.html")
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
            'Execution Mode': 'Execution Mode',
            actual_ram_col: 'Process Tree RAM (MB)'
        }
    )
    fig.update_traces(
        hovertemplate="Execution Mode: %{x}<br>Process Tree RAM (MB): %{y:.2f}<extra></extra>"
    )
    fig.update_layout(
        yaxis=dict(dtick=ram_dtick)
    )
    output_file = os.path.join(output_dir, f"comparison_ram_usage_{safe_model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # 3b. VRAM Usage Comparison (GPU only)
    if is_gpu and actual_vram_col in df_combined.columns:
        vram_max = df_combined[actual_vram_col].max()
        if pd.notna(vram_max) and vram_max > 0:
            fig = px.box(
                df_combined[df_combined[actual_vram_col] > 0],
                x='Execution Mode',
                y=actual_vram_col,
                color='Execution Mode',
                title=f"[{model_name} - {trial_type}] VRAM Usage: ModelActor Method vs Ripper Method",
                labels={
                    'Execution Mode': 'Execution Mode',
                    actual_vram_col: 'Process Tree VRAM (MB)'
                }
            )
            fig.update_traces(
                hovertemplate="Execution Mode: %{x}<br>Process Tree VRAM (MB): %{y:.2f}<extra></extra>"
            )
            output_file = os.path.join(output_dir, f"comparison_vram_usage_{safe_model_name}_{trial_type.lower()}.html")
            fig.write_html(output_file)
            print(f"Saved: {output_file}")
    
    # 5. Throughput Scaling by Concurrent Tasks (Line Chart) - Picker Throughput
    # Use the previously calculated 'Generated Tasks' and labels
    ma_by_task = df_ma.groupby(task_col).agg({
        'Picker Throughput (st/s)': ['mean', 'std'],
        'Generated Tasks': 'mean'
    }).reset_index()
    ma_by_task.columns = [task_col, 'mean', 'std', 'Generated Tasks']
    ma_by_task['Execution Mode'] = 'ModelActor Method'
    ma_by_task['Generated Label'] = "Number of ModelActors Created:"
    
    rp_by_task = df_rp.groupby(task_col).agg({
        'Picker Throughput (st/s)': ['mean', 'std'],
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
        custom_data=['Generated Label', 'Generated Tasks', 'std'],
        title=f"[{model_name} - {trial_type}] Picker Throughput Scaling: ModelActor Method vs Ripper Method",
        labels={
            task_col: 'Concurrent Tasks Requested',
            'mean': 'Mean Picker Throughput (st/s)'
        }
    )
    fig.update_traces(
        hovertemplate=(
            "<b>Scaling Analysis</b><br>"
            "Execution Mode: %{fullData.name}<br>"
            "Concurrent Tasks Requested: %{x}<br>"
            "%{customdata[0]} %{customdata[1]:.0f}<br>"
            "Mean Picker Throughput (st/s): %{y:.2f} ± %{customdata[2]:.2f}<br>"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        xaxis=dict(dtick=10)
    )
    output_file = os.path.join(output_dir, f"comparison_picker_throughput_scaling_{safe_model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # Create symbol columns
    # 3D scatter only supports: ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']
    symbol_map_dict = {0: 'circle', 1: 'circle', 2: 'cross', 3: 'x', 4: 'square', 5: 'diamond', 6: 'circle-open', 7: 'square-open', 8: 'diamond-open'}
    df_ma['Marker Symbol'] = df_ma['GPU Count'].apply(lambda x: symbol_map_dict.get(int(x), 'circle'))
    df_rp['Marker Symbol'] = df_rp['GPU Count'].apply(lambda x: symbol_map_dict.get(int(x), 'circle'))
    
    # 6. 3D Comparison: Total Trial Runtime vs CPUs vs Stations
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=['ModelActor Method', 'Ripper Method']
    )
    
    # Calculate shared color scale limits
    conc_min = min(df_ma['Generated Tasks'].min() if not df_ma.empty else 0, 
                  df_rp['Generated Tasks'].min() if not df_rp.empty else 0)
    conc_max = max(df_ma['Generated Tasks'].max() if not df_ma.empty else 1, 
                  df_rp['Generated Tasks'].max() if not df_rp.empty else 1)
    
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
            z=df_ma[total_trial_time_col],
            mode='markers',
            marker=dict(
                size=5,
                color=df_ma['Generated Tasks'],
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
                line=dict(width=0)
            ),
            name='ModelActor Method',
            showlegend=False,
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
            z=df_rp[total_trial_time_col],
            mode='markers',
            marker=dict(
                size=5,
                color=df_rp['Generated Tasks'],
                colorscale='Turbo',
                cmin=conc_min,
                cmax=conc_max,
                opacity=0.8,
                symbol=df_rp['Marker Symbol'],
                line=dict(width=0)
            ),
            name='Ripper Method',
            showlegend=False,
            hovertemplate=rp_hover,
            customdata=df_rp[['GPU Count', 'GPUs Used', task_col, actual_ram_col, picker_runtime_col,
                             'Generated Tasks', avg_model_load_time_col, waveform_proc_time_col,
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
        title=f"[{model_name} - {trial_type}] 3D Total Trial Runtime Comparison",
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
            zaxis_title='Total Trial Runtime (s)'
        ),
        scene2=dict(
            xaxis=dict(title='CPUs Allocated', dtick=1),
            yaxis=dict(title='Total Number of Stations to Process', dtick=10),
            zaxis_title='Total Trial Runtime (s)'
        )
    )
    output_file = os.path.join(output_dir, f"comparison_3d_total_trial_runtime_{safe_model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")

    # 6b. 3D Comparison: Total Picking Time vs CPUs vs Stations
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=['ModelActor Method', 'Ripper Method']
    )

    fig.add_trace(
        go.Scatter3d(
            x=df_ma[cpu_col],
            y=df_ma[station_col],
            z=df_ma[picker_runtime_col],
            mode='markers',
            marker=dict(
                size=5,
                color=df_ma['Generated Tasks'],
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
                line=dict(width=0)
            ),
            name='ModelActor Method',
            showlegend=False,
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
            z=df_rp[picker_runtime_col],
            mode='markers',
            marker=dict(
                size=5,
                color=df_rp['Generated Tasks'],
                colorscale='Turbo',
                cmin=conc_min,
                cmax=conc_max,
                opacity=0.8,
                symbol=df_rp['Marker Symbol'],
                line=dict(width=0)
            ),
            name='Ripper Method',
            showlegend=False,
            hovertemplate=rp_hover,
            customdata=df_rp[['GPU Count', 'GPUs Used', task_col, actual_ram_col, picker_runtime_col,
                             'Generated Tasks', avg_model_load_time_col, waveform_proc_time_col,
                             total_trial_time_col]].values
        ),
        row=1, col=2
    )
    
    # Add dummy traces for symbol legend if it's a GPU trial
    if is_gpu:
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
        title=f"[{model_name} - {trial_type}] 3D Total Waveform Picking Time Comparison",
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
            zaxis_title='Total Waveform Picking Time (s)'
        ),
        scene2=dict(
            xaxis=dict(title='CPUs Allocated', dtick=1),
            yaxis=dict(title='Total Number of Stations to Process', dtick=10),
            zaxis_title='Total Waveform Picking Time (s)'
        )
    )
    output_file = os.path.join(output_dir, f"comparison_3d_total_waveform_picking_time_{safe_model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # 7. Summary Statistics Table - including new timing metrics
    # Helper function to safely format values
    def safe_format(series, fmt=".3f", default="N/A"):
        if series.notna().any():
            return f"{series.mean():{fmt}}"
        return default
    
    def safe_format_min(series, fmt=".3f", default="N/A"):
        if series.notna().any():
            return f"{series.min():{fmt}}"
        return default
    
    # Calculate max concurrency metrics
    # ModelActor: max number of actors created
    ma_max_actors = int(df_ma[actor_col].max()) if actor_col in df_ma.columns else 0
    ma_max_actor_row = df_ma[df_ma[actor_col] == ma_max_actors].iloc[0] if ma_max_actors > 0 else None
    ma_ram_at_max = f"{ma_max_actor_row[actual_ram_col]:.3f}" if ma_max_actor_row is not None and actual_ram_col in df_ma.columns else "N/A"
    
    # Ripper: max number of concurrent tasks
    rp_max_tasks = int(df_rp[ripper_task_col].max()) if ripper_task_col in df_rp.columns else 0
    rp_max_task_row = df_rp[df_rp[ripper_task_col] == rp_max_tasks].iloc[0] if rp_max_tasks > 0 else None
    rp_ram_at_max = f"{rp_max_task_row[actual_ram_col]:.3f}" if rp_max_task_row is not None and actual_ram_col in df_rp.columns else "N/A"
    
    # Check if GPU trial for VRAM metrics
    is_gpu = trial_type.lower() == 'gpu'
    if is_gpu:
        ma_vram_at_max = f"{ma_max_actor_row[actual_vram_col]:.3f}" if ma_max_actor_row is not None and actual_vram_col in df_ma.columns else "N/A"
        rp_vram_at_max = f"{rp_max_task_row[actual_vram_col]:.3f}" if rp_max_task_row is not None and actual_vram_col in df_rp.columns else "N/A"
        ma_mean_vram = f"{df_ma[actual_vram_col].mean():.3f}" if actual_vram_col in df_ma.columns else "N/A"
        rp_mean_vram = f"{df_rp[actual_vram_col].mean():.3f}" if actual_vram_col in df_rp.columns else "N/A"
    
    # Build summary table with comprehensive timing metrics
    summary_metrics = [
        '--- Picker Throughput (st/Picker Runtime) ---',
        'Mean Picker Throughput (st/s)',
        'Median Picker Throughput (st/s)',
        'Max Picker Throughput (st/s)',
        '--- Total Throughput (st/Total Trial Time) ---',
        'Mean Total Throughput (st/s)',
        'Median Total Throughput (st/s)',
        'Max Total Throughput (st/s)',
        '--- Runtime Metrics ---',
        'Mean Total Trial Time (s)',
        'Min Total Trial Time (s)',
        'Mean Picker Runtime (s)',
        'Min Picker Runtime (s)',
        '--- Setup Time Metrics ---',
        'Mean Actor Creation Time (s)',
        'Mean Avg Model Load Time (s)',
        'Mean Waveform Processing Time (s)',
        '--- Concurrency Metrics ---',
        'Max ModelActors / Concurrent Tasks',
        'RAM at Max Concurrency (MB)',
    ]
    
    ma_values = [
        '',
        f"{df_ma['Picker Throughput (st/s)'].mean():.3f}",
        f"{df_ma['Picker Throughput (st/s)'].median():.3f}",
        f"{df_ma['Picker Throughput (st/s)'].max():.3f}",
        '',
        f"{df_ma['Total Throughput (st/s)'].mean():.3f}",
        f"{df_ma['Total Throughput (st/s)'].median():.3f}",
        f"{df_ma['Total Throughput (st/s)'].max():.3f}",
        '',
        safe_format(df_ma[total_trial_time_col]) if total_trial_time_col in df_ma.columns else "N/A",
        safe_format_min(df_ma[total_trial_time_col]) if total_trial_time_col in df_ma.columns else "N/A",
        safe_format(df_ma[picker_runtime_col]) if picker_runtime_col in df_ma.columns else "N/A",
        safe_format_min(df_ma[picker_runtime_col]) if picker_runtime_col in df_ma.columns else "N/A",
        '',
        safe_format(df_ma[actor_creation_time_col]) if actor_creation_time_col in df_ma.columns else "N/A",
        safe_format(df_ma[avg_model_load_time_col]) if avg_model_load_time_col in df_ma.columns else "N/A",
        safe_format(df_ma[waveform_proc_time_col], ".3f") if waveform_proc_time_col in df_ma.columns else "N/A",
        '',
        f"{ma_max_actors} actors",
        ma_ram_at_max,
    ]
    
    rp_values = [
        '',
        f"{df_rp['Picker Throughput (st/s)'].mean():.3f}",
        f"{df_rp['Picker Throughput (st/s)'].median():.3f}",
        f"{df_rp['Picker Throughput (st/s)'].max():.3f}",
        '',
        f"{df_rp['Total Throughput (st/s)'].mean():.3f}",
        f"{df_rp['Total Throughput (st/s)'].median():.3f}",
        f"{df_rp['Total Throughput (st/s)'].max():.3f}",
        '',
        safe_format(df_rp[total_trial_time_col]) if total_trial_time_col in df_rp.columns else "N/A",
        safe_format_min(df_rp[total_trial_time_col]) if total_trial_time_col in df_rp.columns else "N/A",
        safe_format(df_rp[picker_runtime_col]) if picker_runtime_col in df_rp.columns else "N/A",
        safe_format_min(df_rp[picker_runtime_col]) if picker_runtime_col in df_rp.columns else "N/A",
        '',
        "N/A (no actors)",  # Ripper mode doesn't create actors
        safe_format(df_rp[avg_model_load_time_col]) if avg_model_load_time_col in df_rp.columns else "N/A",
        safe_format(df_rp[waveform_proc_time_col], ".3f") if waveform_proc_time_col in df_rp.columns else "N/A",
        '',
        f"{rp_max_tasks} tasks",
        rp_ram_at_max,
    ]
    
    # Add VRAM metrics for GPU trials
    if is_gpu:
        summary_metrics.append('VRAM at Max Concurrency (MB)')
        ma_values.append(ma_vram_at_max)
        rp_values.append(rp_vram_at_max)
    
    # Add memory metrics section
    summary_metrics.extend([
        '--- Memory Metrics ---',
        'Mean RAM (MB)',
    ])
    ma_values.extend([
        '',
        f"{df_ma[actual_ram_col].mean():.3f}",
    ])
    rp_values.extend([
        '',
        f"{df_rp[actual_ram_col].mean():.3f}",
    ])
    
    # Add mean VRAM for GPU trials
    if is_gpu:
        summary_metrics.append('Mean VRAM (MB)')
        ma_values.append(ma_mean_vram)
        rp_values.append(rp_mean_vram)
    
    # Add trial info section
    summary_metrics.extend([
        '--- Trial Info ---',
        'Trial Count'
    ])
    ma_values.extend([
        '',
        str(len(df_ma))
    ])
    rp_values.extend([
        '',
        str(len(df_rp))
    ])
    
    summary_data = {
        'Metric': summary_metrics,
        'ModelActor': ma_values,
        'Ripper': rp_values
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
    output_file = os.path.join(output_dir, f"comparison_summary_table_{safe_model_name}_{trial_type.lower()}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    print(f"\nComparison visualization complete! Files saved to: {output_dir}/")


def generate_aggregate_plots(all_data, output_dir):
    """
    Generate aggregate comparison plots across all models and execution modes.
    """
    if not all_data:
        return

    agg_df = pd.concat(all_data, ignore_index=True)
    
    # Define metrics to plot
    metrics = [
        ('Total Trial Time (s)', 'Mean Total Trial Time (s)', 'total_trial_time'),
        ('Total Run time for Picker (s)', 'Mean Picking Time (s)', 'picking_time'),
        ('Waveform Processing Time (s)', 'Mean Waveform Processing Time (s)', 'waveform_processing_time'),
        ('Process Tree RAM (MB)', 'Mean RAM (MB)', 'ram_usage'),
        ('Process Tree VRAM (MB)', 'Mean VRAM (MB)', 'vram_usage')
    ]

    aggregate_dir = os.path.join(output_dir, "aggregate_comparisons")
    if not os.path.exists(aggregate_dir):
        os.makedirs(aggregate_dir)

    for trial_type in agg_df['Trial Type'].unique():
        type_df = agg_df[agg_df['Trial Type'] == trial_type]
        
        for metric_col, metric_label, file_suffix in metrics:
            if metric_col not in type_df.columns:
                continue
            
            # Check if we have any non-null data for this metric
            if type_df[metric_col].isna().all():
                continue

            # Group by Model, Execution Mode, and Effective Concurrency
            plot_df = type_df.groupby(['Model', 'Execution Mode', 'Effective Concurrency'])[metric_col].mean().reset_index()
            plot_df.columns = ['Model', 'Execution Mode', 'Effective Concurrency', metric_label]
            
            # Create a combined label for the legend
            plot_df['Method Name'] = plot_df['Model'] + " (" + plot_df['Execution Mode'] + ")"
            
            # Prepare hover data
            plot_df['Concurrency Label'] = plot_df['Execution Mode'].apply(
                lambda x: 'Number of Actors' if x.lower() == 'modelactor' else 'Concurrent Tasks'
            )
            
            fig = px.line(
                plot_df,
                x='Effective Concurrency',
                y=metric_label,
                color='Method Name',
                markers=True,
                custom_data=['Model', 'Method Name', 'Concurrency Label'],
                title=f"Aggregate Comparison: {metric_label} vs Concurrency ({trial_type.upper()})",
                labels={
                    'Effective Concurrency': 'Number of Actors / Concurrent Tasks',
                    'Method Name': 'Method'
                }
            )
            
            # Customize hover template for conditional labels
            fig.update_traces(
                hovertemplate="<b>%{customdata[1]}</b><br>" + # Method Name
                              "Method: %{customdata[1]}<br>" +
                              "%{customdata[2]}: %{x}<br>" + # Dynamic Label: Value
                              "Mean " + metric_label + ": %{y:.2f}<extra></extra>"
            )
            
            fig.update_layout(
                xaxis=dict(dtick=10 if plot_df['Effective Concurrency'].max() > 20 else 1),
                legend=dict(
                    title_text="Method",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02,
                    bgcolor='rgba(255,255,255,0.5)'
                ),
                margin=dict(r=250) # Increased margin for the legend and labels
            )
            
            output_file = os.path.join(aggregate_dir, f"aggregate_{file_suffix}_{trial_type.lower()}.html")
            fig.write_html(output_file)
            print(f"Saved: {output_file}")


def batch_visualize(results_root, output_dir="visualizations", desired_runtime=None, 
                    dot_growth=False):
    """
    Batch visualize all result directories and generate aggregate comparison plots.
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
    
    all_data = []
    all_configs = []

    # Process each directory
    for result_dir in sorted(result_dirs):
        dir_path = os.path.join(results_root, result_dir)
        
        # Find CSV file
        csv_files = glob.glob(os.path.join(dir_path, '*_test_results.csv'))
        if not csv_files:
            print(f"  Skipping {result_dir}: No test results CSV found")
            continue
        
        csv_path = csv_files[0]
        
        # Load and collect data for aggregate plots
        try:
            df = pd.read_csv(csv_path)
            if 'Trial Success' in df.columns:
                df = df[df['Trial Success'] == 1.0].copy()
            
            trial_type = detect_trial_type(df)
            execution_mode = detect_execution_mode(df)
            concurrency_col = get_concurrency_column(df, execution_mode)
            df['Effective Concurrency'] = df[concurrency_col].fillna(1)
            
            # Ensure model name is present
            model_name = df['Model Used'].iloc[0] if 'Model Used' in df.columns else result_dir
            df['Model'] = model_name
            df['Trial Type'] = trial_type
            df['Execution Mode'] = execution_mode.capitalize()
            
            all_data.append(df)
        except Exception as e:
            print(f"  Error processing {result_dir} for aggregate data: {e}")

        # Create output directory for this visualization
        vis_output = os.path.join(output_dir, result_dir)
        
        print(f"\nVisualizing: {result_dir}")
        visualize_trials(csv_path, output_dir=vis_output, desired_runtime=desired_runtime,
                         dot_growth=dot_growth)
    
    # Generate aggregate plots
    if all_data:
        print(f"\nGenerating aggregate comparisons...")
        generate_aggregate_plots(all_data, output_dir)

    print(f"\n{'='*70}")
    print(f"Batch visualization complete! All files saved to: {output_dir}")


def find_all_model_files(results_root, model):
    """
    Find all 4 possible combinations of hardware and method for a given model.
    Ensures consistency by picking the best variant (Exact > Original > Others).
    """
    # Normalize model name: lower case and replace hyphens with underscores
    model = model.lower().replace('-', '_')
    user_parts = model.split('_')
    files = {
        'cpu_modelactor': None,
        'gpu_modelactor': None,
        'cpu_ripper': None,
        'gpu_ripper': None
    }
    
    if not os.path.exists(results_root):
        return files
        
    matches = []
    for item in os.listdir(results_root):
        item_lower = item.lower()
        if not item_lower.startswith('eval_'):
            continue
            
        # Also normalize directory name parts
        parts = item_lower.replace('-', '_').split('_')
        # Format: eval_{hw}_{model_parts...}_{method}
        method_idx = -1
        for i, p in enumerate(parts):
            if p in ['modelactor', 'ripper']:
                method_idx = i
                break
        if method_idx == -1:
            continue
        
        model_parts = parts[2:method_idx]
        
        # Check if user_parts is a prefix of model_parts
        if model_parts[:len(user_parts)] == user_parts:
            matches.append({
                'path': os.path.join(results_root, item),
                'model_parts': model_parts,
                'hardware': parts[1] if parts[1] in ['cpu', 'gpu'] else None,
                'method': parts[method_idx]
            })
            
    if not matches:
        # Fallback: exact substring match if the structured match fails
        for item in os.listdir(results_root):
            item_path = os.path.join(results_root, item)
            item_lower = item.lower()
            if not item_lower.startswith('eval_'): continue
            
            if model in item_lower.replace('-', '_'):
                hardware = 'cpu' if 'cpu' in item_lower else 'gpu' if 'gpu' in item_lower else None
                method = 'modelactor' if '_modelactor' in item_lower else 'ripper' if '_ripper' in item_lower else None
                if hardware and method:
                    key = f"{hardware}_{method}"
                    csv_files = glob.glob(os.path.join(item_path, '*_test_results.csv'))
                    if csv_files:
                        files[key] = csv_files[0]
        return files
        
    # Find the best model variant name among matches
    variants = {}
    for m in matches:
        v_key = tuple(m['model_parts'])
        if v_key not in variants:
            score = 1
            if list(m['model_parts']) == user_parts:
                score = 3  # Top priority: Exact match
            elif list(m['model_parts']) == user_parts + ['original']:
                score = 2  # 2nd priority: 'original' suffix
            variants[v_key] = score
                
    best_variant_key = max(variants, key=variants.get)
    
    # Now fill the files dict using only the best variant
    for m in matches:
        if tuple(m['model_parts']) == best_variant_key and m['hardware']:
            key = f"{m['hardware']}_{m['method']}"
            if key in files:
                csv_files = glob.glob(os.path.join(m['path'], '*_test_results.csv'))
                if csv_files:
                    files[key] = csv_files[0]
                    
    return files


def compare_hardware_and_methods(model_name, files, output_dir, desired_runtime=None, dot_growth=False):
    """
    Compare CPU vs GPU for both ModelActor and Ripper methods.
    Generates combined plots and two summary tables.
    """
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n{'='*70}")
    print(f"HARDWARE & METHOD COMPARISON: {model_name}")
    print(f"{'='*70}")

    # Load dataframes
    dfs = {}
    for key, path in files.items():
        if path:
            print(f"Loading {key}: {path}")
            df = pd.read_csv(path)
            # Detect success if column exists
            if 'Trial Success' in df.columns:
                df = df[df['Trial Success'] == 1.0]
            
            hardware, method = key.split('_')
            hw_str = hardware.upper()
            method_str = method.capitalize() if method == 'ripper' else 'ModelActor'
            
            df['Hardware'] = hw_str
            df['Execution Mode'] = method_str
            df['Label'] = f"{hw_str} - {method_str}"
            dfs[key] = df

    if not dfs:
        print("Error: No data found for comparison.")
        return

    # Column definitions
    total_trial_time_col = 'Total Trial Time (s)'
    picker_runtime_col = 'Total Run time for Picker (s)'
    station_col = 'Number of Stations Used'
    cpu_col = 'Number of CPUs Allocated for Ray to Use'
    actor_col = 'N ModelActors'
    ripper_task_col = 'Actual Ripper Concurrent Tasks'
    task_col = 'Number of Concurrent Station Tasks'
    actual_ram_col = 'Process Tree RAM (MB)'
    actual_vram_col = 'Process Tree VRAM (MB)'
    actor_creation_time_col = 'Actor Creation Time (s)'
    avg_model_load_time_col = 'Avg Model Load Time (s)'
    waveform_proc_time_col = 'Waveform Processing Time (s)'

    # Preprocess all dataframes
    for key, df in dfs.items():
        # Ensure GPU Count exists and is correctly parsed from string lists like '[0, 1]'
        if 'GPU Count' not in df.columns:
            if 'GPUs Used' in df.columns:
                df['GPU Count'] = df['GPUs Used'].apply(parse_gpu_list)
            else:
                df['GPU Count'] = 0
        else:
            # Re-parse if it exists to handle potential string inconsistencies
            if df['GPU Count'].dtype == object or df['GPU Count'].dtype == str:
                df['GPU Count'] = df['GPUs Used'].apply(parse_gpu_list) if 'GPUs Used' in df.columns else 0
        
        # Calculate throughputs
        df['Picker Throughput (st/s)'] = df[station_col] / df[picker_runtime_col]
        df['Total Throughput (st/s)'] = df[station_col] / df[total_trial_time_col]
        
        # Concurrency normalization
        if 'modelactor' in key:
            df['Generated Tasks'] = df[actor_col]
            df['Generated Label'] = "ModelActors Created"
        else:
            if ripper_task_col in df.columns and df[ripper_task_col].notna().any():
                df['Generated Tasks'] = df[ripper_task_col]
            else:
                df['Generated Tasks'] = df[task_col]
            df['Generated Label'] = "Ripper Tasks"

        # Ensure numeric
        for col in [total_trial_time_col, picker_runtime_col, station_col, cpu_col, 'GPU Count', 
                    actual_ram_col, actual_vram_col, 'Picker Throughput (st/s)']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Combined for plotting
    df_all = pd.concat(dfs.values(), ignore_index=True)

    # Helper for point size
    point_size = station_col if dot_growth else None

    # Standard hover configuration for all comparative scatter plots
    standard_hover = (
        "<b>%{customdata[0]}</b><br>"
        "Total Stations: %{x}<br>"
        "%{customdata[2]}: %{customdata[1]}<br>"
        "CPUs: %{customdata[5]} | GPUs: %{customdata[6]}<br>"
        "Total Trial Runtime: %{customdata[4]:.3f}s<br>"
        "Waveform Picking Time: %{customdata[3]:.3f}s<br>"
        "Picker Throughput: %{customdata[9]:.3f} st/s<br>"
        "RAM: %{customdata[7]:.3f} MB | VRAM: %{customdata[8]:.3f} MB<br>"
        "<extra></extra>"
    )
    standard_customdata_cols = ['Label', 'Generated Tasks', 'Generated Label', picker_runtime_col, 
                                total_trial_time_col, cpu_col, 'GPU Count', actual_ram_col, 
                                actual_vram_col, 'Picker Throughput (st/s)']

    # =========================================================================
    # 1. ALL METHOD COMPARISONS (Root output_dir/All_Method_Comparisons)
    # =========================================================================
    all_methods_dir = os.path.join(output_dir, "All_Method_Comparisons")
    os.makedirs(all_methods_dir, exist_ok=True)
    
    # 1. Total Runtime vs Stations Comparison (Scatter)
    fig = px.scatter(
        df_all,
        x=station_col,
        y=total_trial_time_col,
        color='Generated Tasks',
        symbol='Execution Mode',
        size=point_size,
        custom_data=standard_customdata_cols,
        title=f"[{model_name}] Total Trial Runtime vs Workload Size: Universal Comparison",
        labels={
            station_col: 'Total Number of Stations to Process',
            total_trial_time_col: 'Total Trial Runtime (s)',
            'Generated Tasks': 'Effective Concurrency',
            'Execution Mode': 'Method'
        }
    )
    fig.update_coloraxes(colorscale='Turbo', colorbar=dict(dtick=10, title="Effective Concurrency"))
    fig.update_traces(
        marker=dict(line=dict(width=0)),
        hovertemplate=standard_hover
    )
    fig.update_layout(
        xaxis=dict(title='Total Number of Stations to Process', dtick=10),
        yaxis=dict(title='Total Trial Runtime (s)'),
        legend=dict(
            title_text='Method',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.5)'
        )
    )
    if desired_runtime:
        fig.add_hline(y=desired_runtime, line_dash="dash", line_color="red")
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.02, y=-0.1,
            text=f"Target: {desired_runtime}s",
            showarrow=False,
            font=dict(color="red", size=13, family="Arial Black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
            xanchor="left",
            yanchor="top"
        )
        fig.update_layout(margin=dict(b=80))
    
    output_file = os.path.join(all_methods_dir, f"universal_comparison_total_runtime_{safe_model_name}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")

    # 2. Picker Runtime vs Stations Comparison (Scatter)
    fig = px.scatter(
        df_all,
        x=station_col,
        y=picker_runtime_col,
        color='Generated Tasks',
        symbol='Execution Mode',
        size=point_size,
        custom_data=standard_customdata_cols,
        title=f"[{model_name}] Total Waveform Picking Time vs Workload Size: Universal Comparison",
        labels={
            station_col: 'Total Number of Stations to Process',
            picker_runtime_col: 'Total Waveform Picking Time (s)',
            'Generated Tasks': 'Effective Concurrency',
            'Execution Mode': 'Method'
        }
    )
    fig.update_coloraxes(colorscale='Turbo', colorbar=dict(dtick=10, title="Effective Concurrency"))
    fig.update_traces(
        marker=dict(line=dict(width=0)),
        hovertemplate=standard_hover
    )
    fig.update_layout(
        xaxis=dict(title='Total Number of Stations to Process', dtick=10),
        yaxis=dict(title='Total Waveform Picking Time (s)'),
        legend=dict(
            title_text='Method',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.5)'
        )
    )
    if desired_runtime:
        fig.add_hline(y=desired_runtime, line_dash="dash", line_color="red")
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.02, y=-0.1,
            text=f"Target: {desired_runtime}s",
            showarrow=False,
            font=dict(color="red", size=13, family="Arial Black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
            xanchor="left",
            yanchor="top"
        )
        fig.update_layout(margin=dict(b=80))
        
    output_file = os.path.join(all_methods_dir, f"universal_comparison_picker_runtime_{safe_model_name}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")

    # 3. Throughput Scaling Comparison (Scatter)
    scaling_df = df_all.groupby(['Generated Tasks', 'Label', 'Hardware', 'Execution Mode'])['Picker Throughput (st/s)'].agg(['mean', 'std']).reset_index()
    fig = px.scatter(
        scaling_df,
        x='Generated Tasks',
        y='mean',
        color='Generated Tasks',
        symbol='Execution Mode',
        error_y='std',
        custom_data=['Label', 'std'],
        title=f"[{model_name}] Picker Throughput Scaling Comparison",
        labels={'Generated Tasks': 'Effective Concurrency', 'mean': 'Mean Picker Throughput (st/s)', 'Execution Mode': 'Method'}
    )
    fig.update_coloraxes(colorscale='Turbo', colorbar=dict(dtick=10, title="Effective Concurrency"))
    fig.update_traces(
        marker=dict(size=12, line=dict(width=0)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Effective Concurrency: %{x}<br>"
            "Mean Throughput: %{y:.3f} st/s<br>"
            "Std Dev: ±%{customdata[1]:.3f}<br>"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        xaxis=dict(title='Effective Concurrency', dtick=10),
        yaxis=dict(title='Mean Picker Throughput (st/s)'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.5)'
        )
    )
    if desired_runtime:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.02, y=-0.1,
            text=f"Target: {desired_runtime}s",
            showarrow=False,
            font=dict(color="red", size=13, family="Arial Black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
            xanchor="left",
            yanchor="top"
        )
        fig.update_layout(margin=dict(b=80))
    output_file = os.path.join(all_methods_dir, f"universal_comparison_throughput_scaling_{safe_model_name}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")

    # =========================================================================
    # 4. Two Comparison Tables: One for ModelActor, one for Ripper (Root)
    # =========================================================================
    def get_summary_values(df, hw_type, method):
        if df is None or df.empty:
            return ["N/A"] * 26 # Filler
        
        # Re-implement safe_format helpers locally
        def s_fmt(series, fmt=".3f"):
            return f"{series.mean():{fmt}}" if series.notna().any() else "N/A"
        def s_median(series, fmt=".3f"):
            return f"{series.median():{fmt}}" if series.notna().any() else "N/A"
        def s_max(series, fmt=".3f"):
            return f"{series.max():{fmt}}" if series.notna().any() else "N/A"
        def s_min(series, fmt=".3f"):
            return f"{series.min():{fmt}}" if series.notna().any() else "N/A"
        
        # Max concurrency metrics - handle NaNs safely
        m_actors = 0
        if actor_col in df.columns and df[actor_col].notna().any():
            m_actors = int(df[actor_col].max())
        
        m_tasks = 0
        if method == 'Ripper':
            if ripper_task_col in df.columns and df[ripper_task_col].notna().any():
                m_tasks = int(df[ripper_task_col].max())
            elif task_col in df.columns and df[task_col].notna().any():
                m_tasks = int(df[task_col].max())
        
        conc_val = m_actors if method == 'ModelActor' else m_tasks
        conc_label = "actors" if method == 'ModelActor' else "tasks"
        
        # Get row at max concurrency
        max_row = None
        if method == 'ModelActor':
            if m_actors > 0:
                matching_rows = df[df[actor_col] == m_actors]
                if not matching_rows.empty:
                    max_row = matching_rows.iloc[0]
        else:
            if m_tasks > 0:
                if ripper_task_col in df.columns:
                    matching_rows = df[df[ripper_task_col] == m_tasks]
                else:
                    matching_rows = df[df[task_col] == m_tasks]
                if not matching_rows.empty:
                    max_row = matching_rows.iloc[0]
            
        ram_at_max = f"{max_row[actual_ram_col]:.3f}" if max_row is not None and actual_ram_col in df.columns and pd.notna(max_row[actual_ram_col]) else "N/A"
        vram_at_max = f"{max_row[actual_vram_col]:.3f}" if max_row is not None and actual_vram_col in df.columns and pd.notna(max_row[actual_vram_col]) else "N/A"

        return [
            '',
            str(len(df)),
            '',
            s_fmt(df['Picker Throughput (st/s)']),
            s_median(df['Picker Throughput (st/s)']),
            s_max(df['Picker Throughput (st/s)']),
            '',
            s_fmt(df['Total Throughput (st/s)']),
            s_median(df['Total Throughput (st/s)']),
            s_max(df['Total Throughput (st/s)']),
            '',
            s_fmt(df[total_trial_time_col]),
            s_min(df[total_trial_time_col]),
            s_fmt(df[picker_runtime_col]),
            s_min(df[picker_runtime_col]),
            '',
            s_fmt(df[actor_creation_time_col]) if actor_creation_time_col in df.columns else "N/A",
            s_fmt(df[avg_model_load_time_col]) if avg_model_load_time_col in df.columns else "N/A",
            s_fmt(df[waveform_proc_time_col]) if waveform_proc_time_col in df.columns else "N/A",
            '',
            f"{conc_val} {conc_label}",
            ram_at_max,
            vram_at_max if hw_type == 'GPU' else "N/A",
            '',
            s_fmt(df[actual_ram_col], ".3f") if actual_ram_col in df.columns else "N/A",
            s_fmt(df[actual_vram_col], ".3f") if hw_type == 'GPU' and actual_vram_col in df.columns else "N/A"
        ]

    metrics = [
        '--- General Metrics ---',
        'Total Number of Trials',
        '--- Picker Throughput (st/Picker Runtime) ---',
        'Mean Picker Throughput (st/s)', 'Median Picker Throughput (st/s)', 'Max Picker Throughput (st/s)',
        '--- Total Throughput (st/Total Trial Time) ---',
        'Mean Total Throughput (st/s)', 'Median Total Throughput (st/s)', 'Max Total Throughput (st/s)',
        '--- Runtime Metrics ---',
        'Mean Total Trial Time (s)', 'Min Total Trial Time (s)', 'Mean Picker Runtime (s)', 'Min Picker Runtime (s)',
        '--- Setup Time Metrics ---',
        'Mean Actor Creation Time (s)', 'Mean Avg Model Load Time (s)', 'Mean Waveform Processing Time (s)',
        '--- Concurrency Metrics ---',
        'Max Concurrency Achieved', 'RAM at Max Concurrency (MB)', 'VRAM at Max Concurrency (MB)',
        '--- Memory Metrics ---',
        'Mean RAM consumption (MB)', 'Mean VRAM consumption (MB)'
    ]

    for method in ['ModelActor', 'Ripper']:
        method_key_cpu = f"cpu_{method.lower()}"
        method_key_gpu = f"gpu_{method.lower()}"
        
        df_cpu = dfs.get(method_key_cpu)
        df_gpu = dfs.get(method_key_gpu)
        
        if df_cpu is None and df_gpu is None:
            continue

        cpu_vals = get_summary_values(df_cpu, 'CPU', method)
        gpu_vals = get_summary_values(df_gpu, 'GPU', method)

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>CPU (Hardware)</b>', '<b>GPU (Hardware)</b>'],
                fill_color='paleturquoise', align='left', font=dict(size=14)
            ),
            cells=dict(
                values=[metrics, cpu_vals, gpu_vals],
                fill_color='lavender', align='left', font=dict(size=12)
            )
        )])
        fig.update_layout(title=f"[{model_name} - {method}] Hardware Comparison: CPU vs GPU")
        
        output_file = os.path.join(output_dir, f"universal_comparison_table_{method.lower()}_{safe_model_name}.html")
        fig.write_html(output_file)
        print(f"Saved: {output_file}")

    print(f"\nUniversal comparison visualization complete! Files saved to: {output_dir}/")

    # =========================================================================
    # HARDWARE-SPECIFIC COMPARISON PLOTS (Subfolders CPU/ and GPU/)
    # =========================================================================
    for hw_type in ['CPU', 'GPU']:
        hw_dir = os.path.join(output_dir, hw_type)
        os.makedirs(hw_dir, exist_ok=True)
        
        df_hw = df_all[df_all['Hardware'] == hw_type]
        if df_hw.empty:
            continue
            
        print(f"\nGenerating {hw_type}-specific comparisons in: {hw_dir}")

        # --- SPECIAL CASE: GPU Method-specific comparisons (per GPU Count) ---
        if hw_type == 'GPU':
            for method in ['ModelActor', 'Ripper']:
                method_gpu_dir = os.path.join(hw_dir, method)
                os.makedirs(method_gpu_dir, exist_ok=True)
                df_method_gpu = df_hw[df_hw['Execution Mode'] == method].copy()
                if df_method_gpu.empty: continue

                # Ensure GPU Count is string for categorical legend
                df_method_gpu = df_method_gpu.sort_values('GPU Count')
                df_method_gpu['GPU Config'] = df_method_gpu['GPU Count'].fillna(0).astype(int).astype(str) + " GPU(s)"
                
                # Define consistent symbol map for GPU counts
                gpu_symbol_map = {
                    "0 GPU(s)": "circle",
                    "1 GPU(s)": "circle",
                    "2 GPU(s)": "diamond",
                    "3 GPU(s)": "square",
                    "4 GPU(s)": "cross",
                    "5 GPU(s)": "x",
                    "6 GPU(s)": "triangle-up",
                    "7 GPU(s)": "triangle-down",
                    "8 GPU(s)": "star"
                }

                # Set method-specific concurrency label
                if method == 'ModelActor':
                    conc_label_display = "ModelActor's Created"
                else:
                    conc_label_display = "Concurrent Tasks Employeed"

                # 1. Total Runtime vs Stations Comparison (Scatter)
                fig = px.scatter(
                    df_method_gpu,
                    x=station_col,
                    y=total_trial_time_col,
                    color='Generated Tasks',
                    symbol='GPU Config',
                    symbol_map=gpu_symbol_map,
                    size=point_size,
                    custom_data=standard_customdata_cols,
                    title=f"[{model_name} - {method} - GPU] Total Trial Runtime vs GPU Count",
                    labels={
                        station_col: 'Total Number of Stations to Process', 
                        total_trial_time_col: 'Total Trial Runtime (s)',
                        'Generated Tasks': conc_label_display
                    }
                )
                fig.update_coloraxes(colorscale='Turbo', colorbar=dict(dtick=10, title=conc_label_display))
                fig.update_traces(
                    marker=dict(line=dict(width=0)),
                    hovertemplate=standard_hover
                )
                fig.update_layout(xaxis=dict(dtick=10), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.5)'))
                if desired_runtime:
                    fig.add_hline(y=desired_runtime, line_dash="dash", line_color="red")
                    fig.add_annotation(
                        xref="paper", yref="paper",
                        x=1.02, y=-0.1,
                        text=f"Target: {desired_runtime}s",
                        showarrow=False,
                        font=dict(color="red", size=13, family="Arial Black"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="red",
                        borderwidth=1,
                        xanchor="left",
                        yanchor="top"
                    )
                    fig.update_layout(margin=dict(b=80))
                fig.write_html(os.path.join(method_gpu_dir, "gpu_comparison_total_runtime.html"))

                # 2. Picker Runtime vs Stations Comparison (Scatter)
                fig = px.scatter(
                    df_method_gpu,
                    x=station_col,
                    y=picker_runtime_col,
                    color='Generated Tasks',
                    symbol='GPU Config',
                    symbol_map=gpu_symbol_map,
                    size=point_size,
                    custom_data=standard_customdata_cols,
                    title=f"[{model_name} - {method} - GPU] Total Waveform Picking Time vs GPU Count",
                    labels={
                        station_col: 'Total Number of Stations to Process', 
                        picker_runtime_col: 'Total Waveform Picking Time (s)',
                        'Generated Tasks': conc_label_display
                    }
                )
                fig.update_coloraxes(colorscale='Turbo', colorbar=dict(dtick=10, title=conc_label_display))
                fig.update_traces(
                    marker=dict(line=dict(width=0)),
                    hovertemplate=standard_hover
                )
                fig.update_layout(xaxis=dict(dtick=10), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.5)'))
                if desired_runtime:
                    fig.add_hline(y=desired_runtime, line_dash="dash", line_color="red")
                    fig.add_annotation(
                        xref="paper", yref="paper",
                        x=1.02, y=-0.1,
                        text=f"Target: {desired_runtime}s",
                        showarrow=False,
                        font=dict(color="red", size=13, family="Arial Black"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="red",
                        borderwidth=1,
                        xanchor="left",
                        yanchor="top"
                    )
                    fig.update_layout(margin=dict(b=80))
                fig.write_html(os.path.join(method_gpu_dir, "gpu_comparison_picker_runtime.html"))

                # 3. Throughput Scaling Comparison
                scaling_df_gpu = df_method_gpu.groupby(['GPU Config', 'GPU Count'])['Picker Throughput (st/s)'].agg(['mean', 'std']).reset_index().sort_values('GPU Count')
                fig = px.scatter(
                    scaling_df_gpu,
                    x='GPU Config',
                    y='mean',
                    color='GPU Config',
                    symbol='GPU Config',
                    symbol_map=gpu_symbol_map,
                    error_y='std',
                    custom_data=['std'],
                    title=f"[{model_name} - {method} - GPU] Picker Throughput vs GPU Count",
                    labels={'mean': 'Mean Picker Throughput (st/s)'}
                )
                fig.update_traces(
                    marker=dict(size=12, line=dict(width=0)),
                    hovertemplate="<b>%{x}</b><br>Mean Throughput: %{y:.3f} st/s<br>Std Dev: ±%{customdata[0]:.3f}<extra></extra>"
                )
                if desired_runtime:
                    fig.add_annotation(
                        xref="paper", yref="paper",
                        x=1.02, y=-0.1,
                        text=f"Target: {desired_runtime}s",
                        showarrow=False,
                        font=dict(color="red", size=13, family="Arial Black"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="red",
                        borderwidth=1,
                        xanchor="left",
                        yanchor="top"
                    )
                    fig.update_layout(margin=dict(b=80))

                # 4. GPU Count Comparison Table
                unique_gpu_counts = sorted(df_method_gpu['GPU Count'].unique())
                gpu_cols_data = []
                gpu_headers = [f"<b>{int(c)} GPU(s)</b>" for c in unique_gpu_counts]
                
                for count in unique_gpu_counts:
                    df_count = df_method_gpu[df_method_gpu['GPU Count'] == count]
                    gpu_cols_data.append(get_summary_values(df_count, 'GPU', method))

                fig = go.Figure(data=[go.Table(
                    header=dict(values=['<b>Metric</b>'] + gpu_headers, fill_color='paleturquoise', align='left', font=dict(size=14)),
                    cells=dict(values=[metrics] + gpu_cols_data, fill_color='lavender', align='left', font=dict(size=12))
                )])
                fig.update_layout(title=f"[{model_name} - {method}] GPU Scaling Table: performance vs GPU Count")
                fig.write_html(os.path.join(method_gpu_dir, "gpu_comparison_table.html"))

        # --- Standard Hardware Comparison Plots (CPU/GPU Root) ---
        # 1. Total Runtime vs Stations Comparison (Scatter)
        fig = px.scatter(
            df_hw,
            x=station_col,
            y=total_trial_time_col,
            color='Generated Tasks',
            symbol='Execution Mode',
            size=point_size,
            custom_data=standard_customdata_cols,
            title=f"[{model_name} - {hw_type}] Total Trial Runtime vs Workload Size",
            labels={
                station_col: 'Total Number of Stations to Process',
                total_trial_time_col: 'Total Trial Runtime (s)',
                'Generated Tasks': 'Effective Concurrency'
            }
        )
        fig.update_coloraxes(colorscale='Turbo', colorbar=dict(dtick=10, title="Effective Concurrency"))
        fig.update_traces(
            marker=dict(line=dict(width=0)),
            hovertemplate=standard_hover
        )
        fig.update_layout(
            xaxis=dict(title='Total Number of Stations to Process', dtick=10),
            yaxis=dict(title='Total Trial Runtime (s)'),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.5)')
        )
        if desired_runtime:
            fig.add_hline(y=desired_runtime, line_dash="dash", line_color="red")
            fig.add_annotation(
                xref="paper", yref="paper",
                x=1.02, y=-0.1,
                text=f"Target: {desired_runtime}s",
                showarrow=False,
                font=dict(color="red", size=13, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1,
                xanchor="left",
                yanchor="top"
            )
            fig.update_layout(margin=dict(b=80))

        output_file = os.path.join(hw_dir, f"{hw_type.lower()}_comparison_total_runtime.html")
        fig.write_html(output_file)
        
        # 2. Picker Runtime vs Stations Comparison (Scatter)
        fig = px.scatter(
            df_hw,
            x=station_col,
            y=picker_runtime_col,
            color='Generated Tasks',
            symbol='Execution Mode',
            size=point_size,
            custom_data=standard_customdata_cols,
            title=f"[{model_name} - {hw_type}] Total Waveform Picking Time vs Workload Size",
            labels={
                station_col: 'Total Number of Stations to Process',
                picker_runtime_col: 'Total Waveform Picking Time (s)',
                'Generated Tasks': 'Effective Concurrency'
            }
        )
        fig.update_coloraxes(colorscale='Turbo', colorbar=dict(dtick=10, title="Effective Concurrency"))
        fig.update_traces(
            marker=dict(line=dict(width=0)),
            hovertemplate=standard_hover
        )
        fig.update_layout(
            xaxis=dict(title='Total Number of Stations to Process', dtick=10),
            yaxis=dict(title='Total Waveform Picking Time (s)'),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.5)')
        )
        if desired_runtime:
            fig.add_hline(y=desired_runtime, line_dash="dash", line_color="red")
            fig.add_annotation(
                xref="paper", yref="paper",
                x=1.02, y=-0.1,
                text=f"Target: {desired_runtime}s",
                showarrow=False,
                font=dict(color="red", size=13, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1,
                xanchor="left",
                yanchor="top"
            )
            fig.update_layout(margin=dict(b=80))

        output_file = os.path.join(hw_dir, f"{hw_type.lower()}_comparison_picker_runtime.html")
        fig.write_html(output_file)

        # 3. Throughput Scaling Comparison
        scaling_df_hw = df_hw.groupby(['Generated Tasks', 'Execution Mode'])['Picker Throughput (st/s)'].agg(['mean', 'std']).reset_index()
        fig = px.scatter(
            scaling_df_hw,
            x='Generated Tasks',
            y='mean',
            color='Generated Tasks',
            symbol='Execution Mode',
            error_y='std',
            custom_data=['std'],
            title=f"[{model_name} - {hw_type}] Picker Throughput Scaling Comparison",
            labels={'Generated Tasks': 'Effective Concurrency', 'mean': 'Mean Picker Throughput (st/s)'}
        )
        fig.update_coloraxes(colorscale='Turbo', colorbar=dict(dtick=10, title="Effective Concurrency"))
        fig.update_traces(
            marker=dict(size=12, line=dict(width=0)),
            hovertemplate="<b>%{symbol}</b><br>Concurrency: %{x}<br>Throughput: %{y:.3f} st/s<br>Std Dev: ±%{customdata[0]:.3f}<extra></extra>"
        )
        fig.update_layout(
            xaxis=dict(title='Effective Concurrency', dtick=10),
            yaxis=dict(title='Mean Picker Throughput (st/s)'),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.5)')
        )
        if desired_runtime:
            fig.add_annotation(
                xref="paper", yref="paper",
                x=1.02, y=-0.1,
                text=f"Target: {desired_runtime}s",
                showarrow=False,
                font=dict(color="red", size=13, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1,
                xanchor="left",
                yanchor="top"
            )
            fig.update_layout(margin=dict(b=80))
        output_file = os.path.join(hw_dir, f"{hw_type.lower()}_comparison_throughput_scaling.html")
        fig.write_html(output_file)

        # 4. RAM Usage Box Plot
        ram_max = df_hw[actual_ram_col].max()
        ram_dtick = 5000 if ram_max < 40000 else 10000
        fig = px.box(
            df_hw,
            x='Execution Mode',
            y=actual_ram_col,
            color='Execution Mode',
            points="all",
            title=f"[{model_name} - {hw_type}] RAM Usage Comparison",
            labels={'Execution Mode': 'Method', actual_ram_col: 'Process Tree RAM (MB)'}
        )
        fig.update_layout(yaxis=dict(dtick=ram_dtick))
        output_file = os.path.join(hw_dir, f"{hw_type.lower()}_comparison_ram_usage.html")
        fig.write_html(output_file)

        # 5. VRAM Usage Box Plot (GPU only)
        if hw_type == 'GPU' and actual_vram_col in df_hw.columns:
            df_vram = df_hw[df_hw[actual_vram_col] > 0]
            if not df_vram.empty:
                fig = px.box(
                    df_vram,
                    x='Execution Mode',
                    y=actual_vram_col,
                    color='Execution Mode',
                    points="all",
                    title=f"[{model_name} - {hw_type}] VRAM Usage Comparison",
                    labels={'Execution Mode': 'Method', actual_vram_col: 'Process Tree VRAM (MB)'}
                )
                output_file = os.path.join(hw_dir, f"{hw_type.lower()}_comparison_vram_usage.html")
                fig.write_html(output_file)

    # =========================================================================
    # METHOD-SPECIFIC HARDWARE COMPARISONS (Subfolders CPUvsGPU/ModelActor/ and CPUvsGPU/Ripper/)
    # =========================================================================
    hardware_comp_root = os.path.join(output_dir, "CPUvsGPU")
    os.makedirs(hardware_comp_root, exist_ok=True)
    
    for method in ['ModelActor', 'Ripper']:
        method_dir = os.path.join(hardware_comp_root, method)
        os.makedirs(method_dir, exist_ok=True)
        
        df_method = df_all[df_all['Execution Mode'] == method]
        if df_method.empty:
            continue
            
        print(f"Generating CPU vs GPU comparisons for {method} in: {method_dir}")
        
        # Set method-specific concurrency label
        if method == 'ModelActor':
            conc_label_display = "ModelActor's Created"
        else:
            conc_label_display = "Concurrent Tasks Employeed"

        # 1. Total Runtime vs Stations (Scatter)
        fig = px.scatter(
            df_method,
            x=station_col,
            y=total_trial_time_col,
            color='Generated Tasks',
            symbol='Hardware',
            size=point_size,
            custom_data=standard_customdata_cols,
            title=f"[{model_name} - {method}] CPU vs GPU: Total Trial Runtime",
            labels={
                station_col: 'Total Number of Stations to Process',
                total_trial_time_col: 'Total Trial Runtime (s)',
                'Generated Tasks': conc_label_display
            }
        )
        fig.update_coloraxes(colorscale='Turbo', colorbar=dict(dtick=10, title=conc_label_display))
        fig.update_traces(
            marker=dict(line=dict(width=0)),
            hovertemplate=standard_hover
        )
        fig.update_layout(
            xaxis=dict(title='Total Number of Stations to Process', dtick=10),
            yaxis=dict(title='Total Trial Runtime (s)'),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.5)')
        )
        if desired_runtime:
            fig.add_hline(y=desired_runtime, line_dash="dash", line_color="red")
            fig.add_annotation(
                xref="paper", yref="paper",
                x=1.02, y=-0.1,
                text=f"Target: {desired_runtime}s",
                showarrow=False,
                font=dict(color="red", size=13, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1,
                xanchor="left",
                yanchor="top"
            )
            fig.update_layout(margin=dict(b=80))

        output_file = os.path.join(method_dir, f"hardware_comparison_total_runtime.html")
        fig.write_html(output_file)
        
        # 2. Picker Runtime vs Stations (Scatter)
        fig = px.scatter(
            df_method,
            x=station_col,
            y=picker_runtime_col,
            color='Generated Tasks',
            symbol='Hardware',
            size=point_size,
            custom_data=standard_customdata_cols,
            title=f"[{model_name} - {method}] CPU vs GPU: Total Waveform Picking Time",
            labels={
                station_col: 'Total Number of Stations to Process',
                picker_runtime_col: 'Total Waveform Picking Time (s)',
                'Generated Tasks': conc_label_display
            }
        )
        fig.update_coloraxes(colorscale='Turbo', colorbar=dict(dtick=10, title=conc_label_display))
        fig.update_traces(
            marker=dict(line=dict(width=0)),
            hovertemplate=standard_hover
        )
        fig.update_layout(
            xaxis=dict(title='Total Number of Stations to Process', dtick=10),
            yaxis=dict(title='Total Waveform Picking Time (s)'),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.5)')
        )
        if desired_runtime:
            fig.add_hline(y=desired_runtime, line_dash="dash", line_color="red")
            fig.add_annotation(
                xref="paper", yref="paper",
                x=1.02, y=-0.1,
                text=f"Target: {desired_runtime}s",
                showarrow=False,
                font=dict(color="red", size=13, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1,
                xanchor="left",
                yanchor="top"
            )
            fig.update_layout(margin=dict(b=80))

        output_file = os.path.join(method_dir, f"hardware_comparison_picker_runtime.html")
        fig.write_html(output_file)

        # 3. Throughput Scaling Comparison
        scaling_df_method = df_method.groupby(['Generated Tasks', 'Hardware'])['Picker Throughput (st/s)'].agg(['mean', 'std']).reset_index()
        fig = px.scatter(
            scaling_df_method,
            x='Generated Tasks',
            y='mean',
            color='Generated Tasks',
            symbol='Hardware',
            error_y='std',
            custom_data=['std'],
            title=f"[{model_name} - {method}] CPU vs GPU: Picker Throughput Scaling",
            labels={'Generated Tasks': conc_label_display, 'mean': 'Mean Picker Throughput (st/s)'}
        )
        fig.update_coloraxes(colorscale='Turbo', colorbar=dict(dtick=10, title=conc_label_display))
        fig.update_traces(
            marker=dict(size=12, line=dict(width=0)),
            hovertemplate="<b>%{symbol}</b><br>" + f"{conc_label_display}: %{{x}}<br>" + "Throughput: %{y:.3f} st/s<br>Std Dev: ±%{customdata[0]:.3f}<extra></extra>"
        )
        fig.update_layout(
            xaxis=dict(title=conc_label_display, dtick=10),
            yaxis=dict(title='Mean Picker Throughput (st/s)'),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.5)')
        )
        if desired_runtime:
            fig.add_annotation(
                xref="paper", yref="paper",
                x=1.02, y=-0.1,
                text=f"Target: {desired_runtime}s",
                showarrow=False,
                font=dict(color="red", size=13, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1,
                xanchor="left",
                yanchor="top"
            )
            fig.update_layout(margin=dict(b=80))
        output_file = os.path.join(method_dir, f"hardware_comparison_throughput_scaling.html")
        fig.write_html(output_file)

        # 4. Hardware Comparison Table (CPU vs GPU for this Method)
        method_key_cpu = f"cpu_{method.lower()}"
        method_key_gpu = f"gpu_{method.lower()}"
        df_cpu = dfs.get(method_key_cpu)
        df_gpu = dfs.get(method_key_gpu)
        
        cpu_vals = get_summary_values(df_cpu, 'CPU', method)
        gpu_vals = get_summary_values(df_gpu, 'GPU', method)

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>CPU (Hardware)</b>', '<b>GPU (Hardware)</b>'],
                fill_color='paleturquoise', align='left', font=dict(size=14)
            ),
            cells=dict(
                values=[metrics, cpu_vals, gpu_vals],
                fill_color='lavender', align='left', font=dict(size=12)
            )
        )])
        fig.update_layout(title=f"[{model_name} - {method}] Hardware Comparison Table: CPU vs GPU")
        output_file = os.path.join(method_dir, f"hardware_comparison_table.html")
        fig.write_html(output_file)
        print(f"Saved: {output_file}")

    print(f"\nHardware and Method comparisons complete!")


# =============================================================================
# OPTIMAL CONFIGURATIONS VISUALIZATION
# =============================================================================

def visualize_optimal_configurations(csv_path, model_name=None, output_dir="visualizations"):
    """
    Visualize optimal configurations CSV files with 3D scatter plots and summary tables.
    
    Creates:
    - 3D scatter plots: CPUs (x) vs Stations (y) vs Runtime/Picking Time (z)
    - Rainbow (Turbo) color scale for number of concurrent tasks
    - Different marker shapes for GPU count
    - Summary comparison table
    
    Parameters:
    -----------
    csv_path : str
        Path to the optimal_configurations_cpu.csv or optimal_configurations_gpu.csv file
    model_name : str, optional
        Name to display in plot titles. Auto-detected if not provided.
    output_dir : str
        Directory to save HTML visualization files.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Path not found at {csv_path}")
        return
    
    # If csv_path is a directory, try to find the optimal config CSV inside it
    if os.path.isdir(csv_path):
        csv_files = glob.glob(os.path.join(csv_path, 'optimal_configurations_*.csv'))
        if not csv_files:
            print(f"Error: No 'optimal_configurations_*.csv' file found in directory {csv_path}")
            return
        csv_path = csv_files[0]
        print(f"Detected optimal config CSV file in directory: {csv_path}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Loading optimal configurations from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Column definitions (same as test_results)
    cpu_col = 'Number of CPUs Allocated for Ray to Use'
    task_col = 'Number of Concurrent Station Tasks'
    actor_col = 'N ModelActors'
    ripper_task_col = 'Actual Ripper Concurrent Tasks'
    station_col = 'Number of Stations Used'
    total_trial_time_col = 'Total Trial Time (s)'
    picker_runtime_col = 'Total Run time for Picker (s)'
    actual_ram_col = 'Process Tree RAM (MB)'
    actual_vram_col = 'Process Tree VRAM (MB)'
    actor_creation_time_col = 'Actor Creation Time (s)'
    waveform_proc_time_col = 'Waveform Processing Time (s)'
    
    # Detect trial type and execution mode
    trial_type = detect_trial_type(df)
    is_gpu_trial = trial_type == 'gpu'
    execution_mode = detect_execution_mode(df)
    concurrency_col = get_concurrency_column(df, execution_mode)
    
    # Parse GPU count
    if 'GPUs Used' in df.columns:
        df['GPU Count'] = df['GPUs Used'].apply(parse_gpu_list)
    else:
        df['GPU Count'] = 0
    
    # Auto-detect model name
    if model_name is None:
        model_name = df['Model Used'].iloc[0] if 'Model Used' in df.columns else "Unknown"
        model_name = f"{model_name}-{'GPU' if is_gpu_trial else 'CPU'}-{execution_mode.upper()}-Optimal"
    
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    
    print(f"Detected trial type: {trial_type.upper()}")
    print(f"Detected execution mode: {execution_mode.upper()}")
    print(f"Model: {model_name}")
    print(f"Optimal configurations loaded: {len(df)} rows")
    
    # Convert numeric columns
    numeric_cols = [cpu_col, task_col, actor_col, ripper_task_col, station_col,
                    total_trial_time_col, actor_creation_time_col,
                    waveform_proc_time_col, picker_runtime_col,
                    actual_ram_col, actual_vram_col]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
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
    
    # Symbol map for GPU count (3D scatter compatible symbols)
    # 0 GPUs (CPU) gets circle, GPU trials get different shapes
    symbol_map_dict = {
        0: 'circle',         # CPU (0 GPUs)
        1: 'diamond',        # 1 GPU
        2: 'square',         # 2 GPUs
        3: 'cross',          # 3 GPUs
        4: 'x',              # 4 GPUs
        5: 'circle-open',    # 5 GPUs
        6: 'diamond-open',   # 6 GPUs
        7: 'square-open',    # 7 GPUs
        8: 'cross-open'      # 8 GPUs
    }
    df['Marker Symbol'] = df['GPU Count'].apply(lambda x: symbol_map_dict.get(int(min(x, 8)), 'circle'))
    
    # Calculate additional metrics
    if actor_creation_time_col in df.columns and actor_col in df.columns:
        df['Avg. ModelActor Creation Time (s)'] = df[actor_creation_time_col] / df[actor_col].replace(0, np.nan)
    else:
        df['Avg. ModelActor Creation Time (s)'] = np.nan
    
    # Calculate throughputs
    df['Picker Throughput (st/s)'] = df[station_col] / df[picker_runtime_col].replace(0, np.nan)
    df['Total Throughput (st/s)'] = df[station_col] / df[total_trial_time_col].replace(0, np.nan)
    
    # Colorbar title
    cbar_title = "N Model Actors" if execution_mode == "modelactor" else "Concurrent Tasks"
    
    # =========================================================================
    # 3D SCATTER PLOTS
    # =========================================================================
    plot_configs = [
        {
            'z_col': total_trial_time_col,
            'title': 'Optimal Config: Total Trial Runtime vs Resources',
            'z_label': 'Total Trial Runtime (s)',
            'file_name': 'optimal_runtime_3d'
        },
        {
            'z_col': picker_runtime_col,
            'title': 'Optimal Config: Total Waveform Picking Time vs Resources',
            'z_label': 'Total Waveform Picking Time (s)',
            'file_name': 'optimal_picking_time_3d'
        },
    ]
    
    for config in plot_configs:
        if config['z_col'] not in df.columns:
            continue
        
        if df[config['z_col']].isna().all() or (df[config['z_col']] == 0).all():
            continue
        
        fig = go.Figure()
        
        # Prepare hover template based on execution mode
        if execution_mode == 'modelactor':
            actor_hover = (
                "Number of ModelActor's Created: %{customdata[3]}<br>"
                "Avg. ModelActor Creation Time (s): %{customdata[6]:.2f}<br>"
                "Total Actor Creation Time (s): %{customdata[7]:.2f}<br>"
            )
        else:
            actor_hover = ""
        
        # Add dummy traces for symbol legend showing GPU count
        unique_gpus = sorted(df['GPU Count'].unique())
        for gpu_count in unique_gpus:
            symbol = symbol_map_dict.get(int(min(gpu_count, 8)), 'circle')
            if gpu_count == 0:
                label = "CPU (0 GPUs)"
            else:
                label = f"{int(gpu_count)} GPU{'s' if gpu_count > 1 else ''}"
            
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(
                    symbol=symbol,
                    color='rgba(100,100,100,0.7)',
                    size=8,
                    line=dict(width=1, color='black')
                ),
                name=label,
                legendgroup="Hardware",
                legendgrouptitle=dict(
                    text="Hardware",
                    font=dict(size=14)
                ),
                showlegend=True
            ))
        
        # Prepare custom data
        customdata_cols = ['GPU Count', 'GPUs Used', task_col, actor_col, total_trial_time_col, actual_ram_col,
                          'Avg. ModelActor Creation Time (s)', actor_creation_time_col,
                          waveform_proc_time_col, picker_runtime_col]
        customdata = []
        for col in customdata_cols:
            if col in df.columns:
                customdata.append(df[col].values)
            else:
                customdata.append(np.full(len(df), np.nan))
        customdata = np.column_stack(customdata)
        
        # Add main scatter trace
        fig.add_trace(go.Scatter3d(
            x=df[cpu_col],
            y=df[station_col],
            z=df[config['z_col']],
            mode='markers',
            marker=dict(
                size=6,
                color=df['Effective Concurrency'],
                colorscale='Turbo',
                colorbar=dict(
                    title=dict(
                        text=cbar_title,
                        font=dict(size=14)
                    ),
                    dtick=cbar_dtick,
                    x=1.1,
                    y=0.5,
                    len=0.9,
                    yanchor='middle'
                ),
                cmin=df['Effective Concurrency'].min(),
                cmax=df['Effective Concurrency'].max(),
                opacity=0.8,
                symbol=df['Marker Symbol'],
                line=dict(width=0)
            ),
            name='',
            showlegend=False,
            hovertemplate=(
                "<b>Optimal Configuration</b><br>"
                "Total Number of Stations: %{y}<br>"
                "CPUs: %{x}<br>"
                "GPUs: %{customdata[0]}<br>"
                "GPU IDs: %{customdata[1]}<br>"
                "Concurrent Tasks Requested: %{customdata[2]}<br>"
                + actor_hover +
                "Avg. Waveform Processing Time (s): %{customdata[8]:.2f}<br>"
                "Total Waveform Picking Time (s): %{customdata[9]:.2f}<br>"
                "Total Trial Runtime (s): %{customdata[4]:.2f}<br>"
                "Process Tree RAM (MB): %{customdata[5]:.2f}<br>"
                "<extra></extra>"
            ),
            customdata=customdata
        ))
        
        x_range = [0, df[cpu_col].max() * 1.1]
        y_range = [0, df[station_col].max() * 1.1]
        
        fig.update_layout(
            title=dict(
                text=f"[{model_name}]<br>{config['title']}",
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(title='CPUs Allocated', range=x_range, dtick=1),
                yaxis=dict(title='Total Number of Stations', range=y_range, dtick=10),
                zaxis=dict(title=config['z_label']),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.8)
            ),
            margin=dict(l=0, r=0, b=0, t=60),
            legend=dict(
                x=1.05,
                y=0.95,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.5)',
                font=dict(size=12)
            ),
            showlegend=True
        )
        
        output_file = os.path.join(output_dir, f"{config['file_name']}_{execution_mode}.html")
        fig.write_html(output_file)
        print(f"Saved: {output_file}")
    
    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    generate_optimal_config_table(df, model_name, execution_mode, is_gpu_trial, output_dir)
    
    print(f"\nOptimal configuration visualization complete! Files saved to: {output_dir}")
    return df


def generate_optimal_config_table(df, model_name, execution_mode, is_gpu_trial, output_dir):
    """
    Generate a summary table for optimal configurations.
    Groups data by station count and shows the optimal configuration for each.
    """
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    
    cpu_col = 'Number of CPUs Allocated for Ray to Use'
    station_col = 'Number of Stations Used'
    total_trial_time_col = 'Total Trial Time (s)'
    picker_runtime_col = 'Total Run time for Picker (s)'
    actual_ram_col = 'Process Tree RAM (MB)'
    actual_vram_col = 'Process Tree VRAM (MB)'
    actor_col = 'N ModelActors'
    task_col = 'Number of Concurrent Station Tasks'
    ripper_task_col = 'Actual Ripper Concurrent Tasks'
    
    actor_creation_time_col = 'Actor Creation Time (s)'
    avg_model_load_time_col = 'Avg Model Load Time (s)'
    
    # Group by station count and get the best config for each
    grouped = df.groupby(station_col).agg({
        cpu_col: 'first',
        'GPU Count': 'first',
        actor_col: 'first' if actor_col in df.columns else lambda x: 0,
        task_col: 'first' if task_col in df.columns else lambda x: 0,
        total_trial_time_col: 'first',
        picker_runtime_col: 'first',
        actual_ram_col: 'first',
        'Picker Throughput (st/s)': 'first',
        'Total Throughput (st/s)': 'first',
        actor_creation_time_col: 'first' if actor_creation_time_col in df.columns else lambda x: np.nan,
        avg_model_load_time_col: 'first' if avg_model_load_time_col in df.columns else lambda x: np.nan
    }).reset_index()
    
    # Add VRAM for GPU trials
    if actual_vram_col in df.columns:
        grouped_vram = df.groupby(station_col)[actual_vram_col].first().reset_index()
        grouped = grouped.merge(grouped_vram, on=station_col, how='left')
    
    # Create table data
    if execution_mode == 'modelactor':
        conc_col_name = 'N ModelActors'
        conc_values = grouped[actor_col].fillna(0).astype(int).tolist()
    else:
        if ripper_task_col in df.columns:
            ripper_grouped = df.groupby(station_col)[ripper_task_col].first().reset_index()
            grouped = grouped.merge(ripper_grouped, on=station_col, how='left', suffixes=('', '_ripper'))
            conc_col_name = 'Concurrent Tasks'
            conc_values = grouped[ripper_task_col].fillna(grouped[task_col]).fillna(0).astype(int).tolist()
        else:
            conc_col_name = 'Concurrent Tasks'
            conc_values = grouped[task_col].fillna(0).astype(int).tolist()
    
    table_headers = [
        '<b>Stations</b>',
        '<b>CPUs</b>',
        '<b>GPUs</b>',
        f'<b>{conc_col_name}</b>',
        '<b>Total Runtime (s)</b>',
        '<b>Picking Time (s)</b>',
        '<b>RAM (MB)</b>'
    ]
    
    table_values = [
        grouped[station_col].astype(int).tolist(),
        grouped[cpu_col].astype(int).tolist(),
        grouped['GPU Count'].astype(int).tolist(),
        conc_values,
        [f"{v:.2f}" for v in grouped[total_trial_time_col]],
        [f"{v:.2f}" for v in grouped[picker_runtime_col]],
        [f"{v:.1f}" for v in grouped[actual_ram_col]]
    ]
    
    # Add VRAM column
    if actual_vram_col in grouped.columns:
        table_headers.append('<b>VRAM (MB)</b>')
        table_values.append([f"{v:.1f}" if pd.notna(v) else "N/A" for v in grouped[actual_vram_col]])
        
    # Add Actor Creation Time
    if actor_creation_time_col in grouped.columns and execution_mode == 'modelactor':
        table_headers.append('<b>Actor Creation (s)</b>')
        table_values.append([f"{v:.2f}" if pd.notna(v) else "N/A" for v in grouped[actor_creation_time_col]])
        
    # Add Model Load Time
    if avg_model_load_time_col in grouped.columns:
        table_headers.append('<b>Model Load (s)</b>')
        table_values.append([f"{v:.2f}" if pd.notna(v) else "N/A" for v in grouped[avg_model_load_time_col]])
        
    # Add throughput at the end
    table_headers.append('<b>Throughput (st/s)</b>')
    table_values.append([f"{v:.3f}" for v in grouped['Picker Throughput (st/s)']])
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=table_headers,
            fill_color='paleturquoise',
            align='left',
            font=dict(size=13)
        ),
        cells=dict(
            values=table_values,
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title=f"[{model_name}] Optimal Configurations Summary Table",
        height=min(800, 100 + len(grouped) * 30)
    )
    
    output_file = os.path.join(output_dir, f"optimal_config_table_{execution_mode}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")


def batch_visualize_optimal(results_root, output_dir="visualizations"):
    """
    Batch visualize all optimal configuration files in result directories.
    Creates a subfolder structure similar to batch_visualize.
    """
    if not os.path.exists(results_root):
        print(f"Error: Results root not found: {results_root}")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n{'='*70}")
    print("BATCH VISUALIZATION OF OPTIMAL CONFIGURATIONS")
    print(f"{'='*70}")
    print(f"Results root: {results_root}")
    
    # Find all result directories with optimal_configurations files
    result_dirs = []
    for item in os.listdir(results_root):
        item_path = os.path.join(results_root, item)
        if os.path.isdir(item_path) and item.startswith('eval_'):
            optimal_files = glob.glob(os.path.join(item_path, 'optimal_configurations_*.csv'))
            if optimal_files:
                result_dirs.append((item, optimal_files[0]))
    
    print(f"Found {len(result_dirs)} directories with optimal configurations")
    
    all_optimal_dfs = []
    
    for result_dir, csv_path in sorted(result_dirs):
        vis_output = os.path.join(output_dir, result_dir, "optimal_configs")
        
        print(f"\nVisualizing optimal configs: {result_dir}")
        df = visualize_optimal_configurations(csv_path, output_dir=vis_output)
        if df is not None:
            df['Source Directory'] = result_dir
            all_optimal_dfs.append(df)
    
    # Generate aggregate optimal configuration comparison
    if all_optimal_dfs:
        print(f"\nGenerating aggregate optimal configuration comparisons...")
        generate_aggregate_optimal_plots(all_optimal_dfs, output_dir)
    
    print(f"\n{'='*70}")
    print(f"Batch optimal configuration visualization complete! Files saved to: {output_dir}")


def generate_aggregate_optimal_plots(all_dfs, output_dir):
    """
    Generate aggregate comparison plots across all optimal configurations.
    """
    if not all_dfs:
        return
    
    aggregate_dir = os.path.join(output_dir, "aggregate_optimal_comparisons")
    if not os.path.exists(aggregate_dir):
        os.makedirs(aggregate_dir)
    
    # Combine all dataframes
    for df in all_dfs:
        if 'Model Used' in df.columns:
            df['Model'] = df['Model Used']
        df['Trial Type'] = 'GPU' if df['GPU Count'].max() > 0 else 'CPU'
    
    agg_df = pd.concat(all_dfs, ignore_index=True)
    
    station_col = 'Number of Stations Used'
    cpu_col = 'Number of CPUs Allocated for Ray to Use'
    total_trial_time_col = 'Total Trial Time (s)'
    picker_runtime_col = 'Total Run time for Picker (s)'
    
    # Create comparison scatter plots
    for trial_type in agg_df['Trial Type'].unique():
        type_df = agg_df[agg_df['Trial Type'] == trial_type]
        
        # Throughput comparison by model
        if 'Picker Throughput (st/s)' in type_df.columns:
            plot_df = type_df.groupby(['Model', 'Source Directory'])['Picker Throughput (st/s)'].mean().reset_index()
            
            fig = px.bar(
                plot_df,
                x='Source Directory',
                y='Picker Throughput (st/s)',
                color='Model',
                title=f"Optimal Configuration Throughput Comparison ({trial_type})",
                labels={'Picker Throughput (st/s)': 'Mean Picker Throughput (st/s)'}
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            
            output_file = os.path.join(aggregate_dir, f"aggregate_optimal_throughput_{trial_type.lower()}.html")
            fig.write_html(output_file)
            print(f"Saved: {output_file}")
        
        # Runtime comparison by station count
        runtime_df = type_df.groupby(['Model', station_col])[total_trial_time_col].mean().reset_index()
        
        fig = px.line(
            runtime_df,
            x=station_col,
            y=total_trial_time_col,
            color='Model',
            markers=True,
            title=f"Optimal Config: Mean Runtime vs Station Count ({trial_type})",
            labels={
                station_col: 'Number of Stations',
                total_trial_time_col: 'Mean Total Runtime (s)'
            }
        )
        fig.update_layout(
            xaxis=dict(dtick=10),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02)
        )
        
        output_file = os.path.join(aggregate_dir, f"aggregate_optimal_runtime_{trial_type.lower()}.html")
        fig.write_html(output_file)
        print(f"Saved: {output_file}")


def discover_models_with_optimal_configs(results_root):
    """
    Discover all unique model names that have optimal configuration files.
    Returns a sorted list of model strings (e.g., ['eqcct', 'phasenet_original']).
    """
    if not os.path.exists(results_root):
        return []
    
    seen_models = set()
    
    for item in os.listdir(results_root):
        item_lower = item.lower()
        if not item_lower.startswith('eval_'):
            continue
        
        item_path = os.path.join(results_root, item)
        if not os.path.isdir(item_path):
            continue
        
        optimal_files = glob.glob(os.path.join(item_path, 'optimal_configurations_*.csv'))
        if not optimal_files:
            continue
        
        parts = item_lower.replace('-', '_').split('_')
        method_idx = -1
        for i, p in enumerate(parts):
            if p in ['modelactor', 'ripper']:
                method_idx = i
                break
        if method_idx == -1:
            continue
        
        model_parts = parts[2:method_idx]
        model_name = '_'.join(model_parts)
        seen_models.add(model_name)
    
    return sorted(seen_models)


def batch_compare_optimal_configs(results_root, output_dir="visualizations"):
    """
    Run optimal configuration comparison for all models found in results_root.
    Each model's comparison is saved to output_dir/<model_name>/.
    """
    if not os.path.exists(results_root):
        print(f"Error: Results root not found: {results_root}")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    models = discover_models_with_optimal_configs(results_root)
    
    if not models:
        print(f"No models with optimal configuration files found in {results_root}")
        return
    
    print(f"\n{'='*70}")
    print("BATCH OPTIMAL CONFIGURATION COMPARISON")
    print(f"{'='*70}")
    print(f"Results root: {results_root}")
    print(f"Output dir: {output_dir}")
    print(f"Found {len(models)} model(s): {', '.join(models)}")
    
    for model in models:
        files = find_optimal_config_files(results_root, model)
        if not any(files.values()):
            print(f"  Skipping {model}: No optimal config files found")
            continue
        
        model_output = os.path.join(output_dir, model)
        print(f"\nComparing optimal configs for: {model}")
        compare_optimal_configs(model, files, model_output)
    
    print(f"\n{'='*70}")
    print(f"Batch optimal comparison complete! Files saved to: {output_dir}")


def find_optimal_config_files(results_root, model):
    """
    Find optimal configuration files for a given model.
    Returns dict with keys: 'cpu_modelactor', 'gpu_modelactor', 'cpu_ripper', 'gpu_ripper'
    """
    model = model.lower().replace('-', '_')
    user_parts = model.split('_')
    files = {
        'cpu_modelactor': None,
        'gpu_modelactor': None,
        'cpu_ripper': None,
        'gpu_ripper': None
    }
    
    if not os.path.exists(results_root):
        return files
    
    for item in os.listdir(results_root):
        item_lower = item.lower()
        if not item_lower.startswith('eval_'):
            continue
        
        item_path = os.path.join(results_root, item)
        if not os.path.isdir(item_path):
            continue
        
        parts = item_lower.replace('-', '_').split('_')
        
        method_idx = -1
        for i, p in enumerate(parts):
            if p in ['modelactor', 'ripper']:
                method_idx = i
                break
        if method_idx == -1:
            continue
        
        model_parts = parts[2:method_idx]
        
        if model_parts[:len(user_parts)] == user_parts:
            hardware = parts[1] if parts[1] in ['cpu', 'gpu'] else None
            method = parts[method_idx]
            
            if hardware and method:
                key = f"{hardware}_{method}"
                optimal_files = glob.glob(os.path.join(item_path, 'optimal_configurations_*.csv'))
                if optimal_files:
                    files[key] = optimal_files[0]
    
    return files


def compare_optimal_configs(model_name, files, output_dir):
    """
    Compare optimal configurations across hardware (CPU/GPU) and methods (ModelActor/Ripper).
    """
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n{'='*70}")
    print(f"OPTIMAL CONFIGURATION COMPARISON: {model_name}")
    print(f"{'='*70}")
    
    # Load dataframes
    dfs = {}
    for key, path in files.items():
        if path:
            print(f"Loading {key}: {path}")
            df = pd.read_csv(path)
            
            hardware, method = key.split('_')
            hw_str = hardware.upper()
            method_str = method.capitalize() if method == 'ripper' else 'ModelActor'
            
            df['Hardware'] = hw_str
            df['Execution Mode'] = method_str
            df['Label'] = f"{hw_str} - {method_str}"
            
            # Parse GPU count
            if 'GPUs Used' in df.columns:
                df['GPU Count'] = df['GPUs Used'].apply(parse_gpu_list)
            else:
                df['GPU Count'] = 0
            
            dfs[key] = df
    
    if not dfs:
        print("Error: No optimal configuration data found for comparison.")
        return
    
    # Column definitions
    cpu_col = 'Number of CPUs Allocated for Ray to Use'
    station_col = 'Number of Stations Used'
    total_trial_time_col = 'Total Trial Time (s)'
    picker_runtime_col = 'Total Run time for Picker (s)'
    actual_ram_col = 'Process Tree RAM (MB)'
    actual_vram_col = 'Process Tree VRAM (MB)'
    actor_creation_time_col = 'Actor Creation Time (s)'
    avg_model_load_time_col = 'Avg Model Load Time (s)'
    actor_col = 'N ModelActors'
    task_col = 'Number of Concurrent Station Tasks'
    
    # Preprocess dataframes
    for key, df in dfs.items():
        df['Picker Throughput (st/s)'] = df[station_col] / df[picker_runtime_col].replace(0, np.nan)
        df['Total Throughput (st/s)'] = df[station_col] / df[total_trial_time_col].replace(0, np.nan)
        
        if 'modelactor' in key:
            df['Effective Concurrency'] = df[actor_col]
        else:
            ripper_col = 'Actual Ripper Concurrent Tasks'
            if ripper_col in df.columns and df[ripper_col].notna().any():
                df['Effective Concurrency'] = df[ripper_col]
            else:
                df['Effective Concurrency'] = df[task_col]
        
        for col in [total_trial_time_col, picker_runtime_col, station_col, cpu_col, actual_ram_col, 
                    actual_vram_col, actor_creation_time_col, avg_model_load_time_col]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Combined dataframe
    df_all = pd.concat(dfs.values(), ignore_index=True)
    
    # Combined symbol map: encodes both Execution Mode AND GPU Count
    # Format: (Execution Mode, GPU Count) -> symbol
    # ModelActor uses filled shapes, Ripper uses open shapes
    # Different GPU counts use different base shapes
    combined_symbol_map = {
        # ModelActor (filled shapes)
        ('ModelActor', 0): 'circle',        # CPU ModelActor (0 GPUs)
        ('ModelActor', 1): 'diamond',       # 1 GPU ModelActor
        ('ModelActor', 2): 'square',        # 2 GPUs ModelActor
        ('ModelActor', 3): 'cross',         # 3 GPUs ModelActor
        ('ModelActor', 4): 'x',             # 4 GPUs ModelActor
        # Ripper (open shapes)
        ('Ripper', 0): 'circle-open',       # CPU Ripper (0 GPUs)
        ('Ripper', 1): 'diamond-open',      # 1 GPU Ripper
        ('Ripper', 2): 'square-open',       # 2 GPUs Ripper
        ('Ripper', 3): 'cross-open',        # 3 GPUs Ripper (note: cross-open may not exist, fallback)
        ('Ripper', 4): 'x-open',            # 4 GPUs Ripper (note: x-open may not exist, fallback)
    }
    
    # Apply combined symbol mapping
    def get_combined_symbol(row):
        mode = row['Execution Mode']
        gpu_count = int(row['GPU Count'])
        # Clamp GPU count to max 4 for symbol mapping
        gpu_count = min(gpu_count, 4)
        key = (mode, gpu_count)
        return combined_symbol_map.get(key, 'circle')
    
    df_all['Marker Symbol'] = df_all.apply(get_combined_symbol, axis=1)
    
    # Calculate colorbar dtick
    max_conc = df_all['Effective Concurrency'].max()
    if max_conc <= 15:
        cbar_dtick = 1
    elif max_conc <= 20:
        cbar_dtick = 5
    else:
        cbar_dtick = 10
    
    # Get unique combinations for legend
    unique_combos = df_all[['Execution Mode', 'GPU Count', 'Marker Symbol']].drop_duplicates()
    unique_combos = unique_combos.sort_values(['Execution Mode', 'GPU Count'], ascending=[False, True])
    
    # =========================================================================
    # 3D COMPARISON PLOT: All Methods Together - Total Runtime
    # =========================================================================
    fig = go.Figure()
    
    # Add legend traces for each unique combination of Execution Mode + GPU Count
    for _, row in unique_combos.iterrows():
        mode = row['Execution Mode']
        gpu_count = int(row['GPU Count'])
        symbol = row['Marker Symbol']
        
        if gpu_count == 0:
            label = f"{mode} (CPU)"
        else:
            label = f"{mode} ({gpu_count} GPU{'s' if gpu_count > 1 else ''})"
        
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(
                symbol=symbol,
                color='rgba(100,100,100,0.7)',
                size=8,
                line=dict(width=1, color='black')
            ),
            name=label,
            legendgroup="Config",
            legendgrouptitle=dict(
                text="Method & Hardware",
                font=dict(size=14)
            ),
            showlegend=True
        ))
    
    # Add main scatter trace with color scale for concurrency
    fig.add_trace(go.Scatter3d(
        x=df_all[cpu_col],
        y=df_all[station_col],
        z=df_all[total_trial_time_col],
        mode='markers',
        marker=dict(
            size=6,
            color=df_all['Effective Concurrency'],
            colorscale='Turbo',
            colorbar=dict(
                title=dict(
                    text="Concurrent Tasks",
                    font=dict(size=14)
                ),
                dtick=cbar_dtick,
                x=1.15,
                y=0.5,
                len=0.9,
                yanchor='middle'
            ),
            cmin=df_all['Effective Concurrency'].min(),
            cmax=df_all['Effective Concurrency'].max(),
            opacity=0.8,
            symbol=df_all['Marker Symbol'],
            line=dict(width=0)
        ),
        name='',
        showlegend=False,
        hovertemplate=(
            "<b>Optimal Configuration</b><br>"
            "Method: %{customdata[0]}<br>"
            "Stations: %{y}<br>"
            "CPUs: %{x}<br>"
            "GPUs: %{customdata[1]}<br>"
            "Concurrent Tasks: %{customdata[2]:.0f}<br>"
            "Runtime: %{z:.2f}s<br>"
            "<extra></extra>"
        ),
        customdata=np.column_stack([df_all['Label'], df_all['GPU Count'], df_all['Effective Concurrency']])
    ))
    
    fig.update_layout(
        title=dict(
            text=f"[{model_name}] Optimal Configurations Comparison<br>Total Runtime vs Resources",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title='CPUs Allocated', dtick=1),
            yaxis=dict(title='Number of Stations', dtick=10),
            zaxis=dict(title='Total Runtime (s)'),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.8)
        ),
        legend=dict(
            x=1.05,
            y=0.95,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.5)',
            font=dict(size=12)
        ),
        showlegend=True
    )
    
    output_file = os.path.join(output_dir, f"optimal_comparison_runtime_3d_{safe_model_name}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # =========================================================================
    # 3D COMPARISON PLOT: Picking Time
    # =========================================================================
    fig = go.Figure()
    
    # Add legend traces for each unique combination of Execution Mode + GPU Count
    for _, row in unique_combos.iterrows():
        mode = row['Execution Mode']
        gpu_count = int(row['GPU Count'])
        symbol = row['Marker Symbol']
        
        if gpu_count == 0:
            label = f"{mode} (CPU)"
        else:
            label = f"{mode} ({gpu_count} GPU{'s' if gpu_count > 1 else ''})"
        
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(
                symbol=symbol,
                color='rgba(100,100,100,0.7)',
                size=8,
                line=dict(width=1, color='black')
            ),
            name=label,
            legendgroup="Config",
            legendgrouptitle=dict(
                text="Method & Hardware",
                font=dict(size=14)
            ),
            showlegend=True
        ))
    
    # Add main scatter trace with color scale for concurrency
    fig.add_trace(go.Scatter3d(
        x=df_all[cpu_col],
        y=df_all[station_col],
        z=df_all[picker_runtime_col],
        mode='markers',
        marker=dict(
            size=6,
            color=df_all['Effective Concurrency'],
            colorscale='Turbo',
            colorbar=dict(
                title=dict(
                    text="Concurrent Tasks",
                    font=dict(size=14)
                ),
                dtick=cbar_dtick,
                x=1.15,
                y=0.5,
                len=0.9,
                yanchor='middle'
            ),
            cmin=df_all['Effective Concurrency'].min(),
            cmax=df_all['Effective Concurrency'].max(),
            opacity=0.8,
            symbol=df_all['Marker Symbol'],
            line=dict(width=0)
        ),
        name='',
        showlegend=False,
        hovertemplate=(
            "<b>Optimal Configuration</b><br>"
            "Method: %{customdata[0]}<br>"
            "Stations: %{y}<br>"
            "CPUs: %{x}<br>"
            "GPUs: %{customdata[1]}<br>"
            "Concurrent Tasks: %{customdata[2]:.0f}<br>"
            "Picking Time: %{z:.2f}s<br>"
            "<extra></extra>"
        ),
        customdata=np.column_stack([df_all['Label'], df_all['GPU Count'], df_all['Effective Concurrency']])
    ))
    
    fig.update_layout(
        title=dict(
            text=f"[{model_name}] Optimal Configurations Comparison<br>Picking Time vs Resources",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title='CPUs Allocated', dtick=1),
            yaxis=dict(title='Number of Stations', dtick=10),
            zaxis=dict(title='Picking Time (s)'),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.8)
        ),
        legend=dict(
            x=1.05,
            y=0.95,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.5)',
            font=dict(size=12)
        ),
        showlegend=True
    )
    
    output_file = os.path.join(output_dir, f"optimal_comparison_picking_3d_{safe_model_name}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    def get_optimal_summary(df_subset, label):
        if df_subset is None or df_subset.empty:
            return ['N/A'] * 13
        
        # Determine execution mode from label
        is_modelactor = 'ModelActor' in label
        
        vram_mean = f"{df_subset[actual_vram_col].mean():.1f}" if actual_vram_col in df_subset.columns else "N/A"
        creation_time_mean = f"{df_subset[actor_creation_time_col].mean():.2f}" if (actor_creation_time_col in df_subset.columns and is_modelactor) else "N/A"
        load_time_mean = f"{df_subset[avg_model_load_time_col].mean():.2f}" if avg_model_load_time_col in df_subset.columns else "N/A"
        
        return [
            f"{df_subset[total_trial_time_col].min():.2f}",
            f"{df_subset[total_trial_time_col].mean():.2f}",
            f"{df_subset[picker_runtime_col].min():.2f}",
            f"{df_subset[picker_runtime_col].mean():.2f}",
            f"{df_subset['Picker Throughput (st/s)'].max():.3f}",
            f"{df_subset['Picker Throughput (st/s)'].mean():.3f}",
            f"{df_subset[actual_ram_col].min():.1f}",
            f"{df_subset[actual_ram_col].mean():.1f}",
            vram_mean,
            creation_time_mean,
            load_time_mean,
            f"{df_subset['Effective Concurrency'].mean():.1f}",
            str(len(df_subset))
        ]
    
    metrics = [
        'Min Total Runtime (s)',
        'Mean Total Runtime (s)',
        'Min Picking Time (s)',
        'Mean Picking Time (s)',
        'Max Throughput (st/s)',
        'Mean Throughput (st/s)',
        'Min RAM (MB)',
        'Mean RAM (MB)',
        'Mean VRAM (MB)',
        'Mean Actor Creation (s)',
        'Mean Model Load (s)',
        'Mean Concurrency',
        'Configuration Count'
    ]
    
    table_data = {'Metric': metrics}
    # Desired order: 
    # 1. CPU - ModelActor
    # 2. GPU (1) - ModelActor, GPU (2) - ModelActor...
    # 3. CPU - Ripper
    # 4. GPU (1) - Ripper, GPU (2) - Ripper...
    
    for method_key in ['modelactor', 'ripper']:
        method_name = 'ModelActor' if method_key == 'modelactor' else 'Ripper'
        
        # CPU
        cpu_key = f"cpu_{method_key}"
        if cpu_key in dfs and not dfs[cpu_key].empty:
            label = f"CPU - {method_name}"
            table_data[label] = get_optimal_summary(dfs[cpu_key], label)
            
        # GPUs
        gpu_key = f"gpu_{method_key}"
        if gpu_key in dfs and not dfs[gpu_key].empty:
            df_gpu = dfs[gpu_key]
            unique_gpu_counts = sorted(df_gpu['GPU Count'].unique())
            for gpu_count in unique_gpu_counts:
                if gpu_count == 0: continue # Handled by CPU
                gpu_df = df_gpu[df_gpu['GPU Count'] == gpu_count]
                if gpu_df.empty: continue
                label = f"GPU ({int(gpu_count)}) - {method_name}"
                table_data[label] = get_optimal_summary(gpu_df, label)
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Metric</b>'] + [f'<b>{k}</b>' for k in table_data.keys() if k != 'Metric'],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=13)
        ),
        cells=dict(
            values=[table_data[k] for k in table_data.keys()],
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title=f"[{model_name}] Optimal Configurations Comparison Table"
    )
    
    output_file = os.path.join(output_dir, f"optimal_comparison_table_{safe_model_name}.html")
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    print(f"\nOptimal configuration comparison complete! Files saved to: {output_dir}")


def find_comparison_files(results_root, model, trial_type):
    """
    Find ModelActor and Ripper CSV files for a given model and trial type.
    Ensures consistency by picking the best variant (Exact > Original > Others).
    """
    trial_type = trial_type.lower()
    # Normalize model name: lower case and replace hyphens with underscores
    model = model.lower().replace('-', '_')
    user_parts = model.split('_')
    
    if not os.path.exists(results_root):
        return None, None
        
    matches = []
    for item in os.listdir(results_root):
        item_lower = item.lower()
        if not item_lower.startswith('eval_') or trial_type not in item_lower:
            continue
            
        # Also normalize directory name parts
        parts = item_lower.replace('-', '_').split('_')
        method_idx = -1
        for i, p in enumerate(parts):
            if p in ['modelactor', 'ripper']:
                method_idx = i
                break
        if method_idx == -1:
            continue
        
        model_parts = parts[2:method_idx]
        
        # Check if user_parts is a prefix of model_parts
        if model_parts[:len(user_parts)] == user_parts:
            matches.append({
                'path': os.path.join(results_root, item),
                'model_parts': model_parts,
                'method': parts[method_idx]
            })
            
    if not matches:
        # Fallback: exact substring match
        modelactor_csv = None
        ripper_csv = None
        for item in os.listdir(results_root):
            item_path = os.path.join(results_root, item)
            item_lower = item.lower()
            if not item_lower.startswith('eval_') or trial_type not in item_lower: continue
            
            if model in item_lower.replace('-', '_'):
                csv_files = glob.glob(os.path.join(item_path, '*_test_results.csv'))
                if not csv_files: continue
                if "_modelactor" in item_lower:
                    modelactor_csv = csv_files[0]
                elif "_ripper" in item_lower:
                    ripper_csv = csv_files[0]
        return modelactor_csv, ripper_csv
        
    # Find the best model variant among matches
    variants = {}
    for m in matches:
        v_key = tuple(m['model_parts'])
        if v_key not in variants:
            score = 1
            if list(m['model_parts']) == user_parts:
                score = 3
            elif list(m['model_parts']) == user_parts + ['original']:
                score = 2
            variants[v_key] = score
                
    best_variant_key = max(variants, key=variants.get)
    
    modelactor_csv = None
    ripper_csv = None
    for m in matches:
        if tuple(m['model_parts']) == best_variant_key:
            csv_files = glob.glob(os.path.join(m['path'], '*_test_results.csv'))
            if not csv_files:
                continue
            if m['method'] == 'modelactor':
                modelactor_csv = csv_files[0]
            elif m['method'] == 'ripper':
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

  # Compare ModelActor vs Ripper (specific hardware)
  python visualize_trial_results.py --compare --model eqcct --trial_type cpu --results_root results/csv/

  # Universal Comparison (CPU vs GPU across both methods)
  # triggered by omitting --trial_type
  python visualize_trial_results.py --compare --model eqcct --results_root results/csv/ --output_dir vis/universal/

  # Optimal Configurations Visualization
  # Single file
  python visualize_trial_results.py --optimal results/trials/eval_cpu_eqcct_modelactor/optimal_configurations_cpu.csv

  # Batch optimal visualization (individual per-trial)
  python visualize_trial_results.py --optimal --batch --results_root results/trials/ --output_dir vis/optimal/

  # Compare optimal configs for a single model
  python visualize_trial_results.py --optimal --compare --model eqcct --results_root results/trials/

  # Batch optimal comparison (all models, CPU vs GPU, ModelActor vs Ripper)
  python visualize_trial_results.py --optimal --compare --batch --results_root results/trials/ --output_dir vis/optimal_comparisons/
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
    parser.add_argument('--dot_growth', action='store_true',
                       help='Make dot size grow with workload in comparison plots (default: fixed size)')
    
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
    
    # Optimal configurations mode
    parser.add_argument('--optimal', action='store_true',
                       help='Visualize optimal configuration files (optimal_configurations_*.csv)')
    
    args = parser.parse_args()
    
    if args.optimal:
        # Optimal configurations visualization mode
        if args.batch:
            if args.compare:
                # Batch optimal comparison: run comparison for all models
                batch_compare_optimal_configs(args.results_root, args.output_dir)
            else:
                # Batch optimal visualization (individual per-trial)
                batch_visualize_optimal(args.results_root, args.output_dir)
        elif args.compare:
            # Compare optimal configs for a single model
            if not args.model:
                print("Error: --optimal --compare requires the --model argument (or use --batch for all models)")
                parser.print_help()
            else:
                files = find_optimal_config_files(args.results_root, args.model)
                if any(files.values()):
                    compare_optimal_configs(args.model, files, args.output_dir)
                else:
                    print(f"Error: Could not find any optimal configuration files for model '{args.model}' in {args.results_root}")
        elif args.csv_path:
            # Single optimal config file visualization
            visualize_optimal_configurations(
                csv_path=args.csv_path,
                model_name=args.model,
                output_dir=args.output_dir
            )
        else:
            print("Error: --optimal requires either a csv_path, --batch, or --compare with --model")
            parser.print_help()
    elif args.batch:
        # Batch visualization mode
        batch_visualize(args.results_root, args.output_dir, 
                        desired_runtime=args.desired_runtime,
                        dot_growth=args.dot_growth)
    elif args.compare:
        # Comparison mode
        if not args.model:
            print("Error: --compare requires at least the --model argument")
            parser.print_help()
        elif args.trial_type:
            # Case 1: Compare ModelActor vs Ripper for a SPECIFIC trial type (CPU or GPU)
            modelactor_csv, ripper_csv = find_comparison_files(args.results_root, args.model, args.trial_type)
            if modelactor_csv and ripper_csv:
                compare_modelactor_vs_ripper(modelactor_csv, ripper_csv, args.output_dir, desired_runtime=args.desired_runtime)
            else:
                print(f"Error: Could not find both ModelActor and Ripper results for {args.model} ({args.trial_type})")
                if not modelactor_csv:
                    print(f"  Missing: ModelActor CSV")
                if not ripper_csv:
                    print(f"  Missing: Ripper CSV")
        else:
            # Case 2: Universal Comparison - Compare CPU vs GPU across both methods
            # This is triggered when --compare and --model are passed without --trial_type
            files = find_all_model_files(args.results_root, args.model)
            if any(files.values()):
                compare_hardware_and_methods(args.model, files, args.output_dir, 
                                           desired_runtime=args.desired_runtime,
                                           dot_growth=args.dot_growth)
            else:
                print(f"Error: Could not find any result directories for model '{args.model}' in {args.results_root}")
    elif args.csv_path:
        # Single file visualization mode
        visualize_trials(
            csv_path=args.csv_path, 
            model_name=args.model, 
            output_dir=args.output_dir,
            filter_threshold=args.threshold,
            success_only=not args.include_failed,
            desired_runtime=args.desired_runtime,
            dot_growth=args.dot_growth
        )
    else:
        parser.print_help()
