# **EQCCTPro: A Powerful Seismic Event Detection & Performance Optimization Toolkit**

EQCCTPro is a high-performance seismic event detection and processing framework designed to bridge the gap between deep learning models and large-scale seismic data processing. It natively supports **EQCCT** (TensorFlow) and the **SeisBench** ecosystem (PyTorch), including models like **PhaseNet**, **EQTransformer**, **GPD**, and **CRED**. 

EQCCTPro is engineered for **real-time performance**, identifying the optimal parallelization configurations for your specific hardware (CPU and Multi-GPU) to minimize runtime and maximize station throughput. EQCCTPro has enabled seismic networks, like the Texas Seismological Research Group (TexNet), to enable their DL picking model EQCCT to run operationally, in real-time, over its network of over 250+ seismic stations. More information on the architecture and application of EQCCTPro can be found in our upcoming publication [here](https://github.com/ut-beg-texnet/eqcct/blob/main/eqcctpro/OptimizedEQCCT_Paper.pdf).

## **Project Structure**

The repository is organized as follows:

```text
eqcctpro/
├── data/                       # Dataset files and creation scripts
│   ├── 230_stations_1_min_dt/  # Sample mseed waveforms
│   └── scripts/                # Tools for data acquisition (create_dataset.py)
├── docs/                       # Documentation and publications
│   └── OptimizedEQCCT_Paper.pdf
├── eqcctpro/                   # Core Python package source code
├── experiments/                # Pipeline execution and benchmarking
│   ├── main/                   # Production scripts (run.py)
│   └── workbench/              # Performance evaluation (e.g. test_cpus_and_gpus.py, scaling/run_requested_benchmarks.py)
├── models/                     # Pre-trained model weights (.h5 files)
│   └── EQCCT/                  # EQCCT specific model checkpoints
├── results/                    # Output artifacts
│   └── csv/                    # Benchmarking results and logs
│       ├── eval_cpu_eqcct/     # CPU-only trial results
│       ├── eval_gpu_eqcct/     # GPU-accelerated trial results
│       └── logs/               # Detailed execution logs and predictions
├── scripts/                    # Utility scripts
│   ├── analysis/               # Efficiency analysis tools
│   └── visualization/          # Plotting and GPU monitoring tools
├── environment.yml             # Conda environment configuration
├── pyproject.toml              # Build system requirements
└── README.md                   # Main project documentation
```

## **Features**
- **Multi-Model Support**: Integrated with **EQCCT** and **SeisBench** (PhaseNet, EQTransformer, etc.).
- **Hybrid Parallelism**: Optimized for both CPU-only and Multi-GPU environments using Ray.
- **Intelligent Benchmarking**: Automated system evaluation with 20% step-size concurrency testing and redundancy filtering.
- **Advanced VRAM Management**: Per-worker memory slicing and aggregate pool safety caps to prevent OOM errors.
- **Automated Dataset Creation**: Workflow-ready data retrieval and denoising via FDSNWS connection.
- **Resource Selection**: Fine-grained control over CPU affinity binding and specific GPU selection.

# **Installation Guide**

EQCCTPro requires a specific dependency stack to ensure compatibility between TensorFlow, PyTorch, and CUDA libraries.

### **Requirements**
- **Python**: 3.10.14+
- **TensorFlow**: 2.20.0
- **PyTorch**: 2.5.1 + cu121
- **SeisBench**: 0.10.2
- **NVIDIA Driver**: Compatible with CUDA 12.1+

### **Standard Installation (Recommended)**
The easiest way to install EQCCTPro with its sample data and all dependencies provided via the `environment.yml` file can be found below:

```sh
# Clone the repository
git clone https://github.com/ut-beg-texnet/eqcct.git
cd eqcct/eqcctpro

# Create and activate the environment
conda env create -f environment.yml
conda activate eqcctpro
```

### **Pip installation**
EQCCTPro is also maintained on the PyPI website, which can be found [here](https://pypi.org/project/eqcctpro/). 
You can install the EQCCTPro package via: 

```sh
pip install eqcctpro
```

---

# Understanding the Waveform Data Input Style to EQCCTPro

1-minute long sample seismic waveforms from 229 TexNet stations have been provided in the repository under the `230_stations_1_min_dt.zip` file to help users understand the EQCCTPro waveform input style. 

After downloading the `.zip` file from the repository, run:
```sh
unzip 230_stations_1_min_dt.zip
```

Inside the zip foilder, we have can see a single timechunk subdirectory, which is comprised of 229 station subdirectories that contain three-component waveforms:

```sh
[skevofilaxc 230_stations_1_min_dt]$ ls
20241215T120000Z_20241215T120100Z

[skevofilaxc 230_stations_1_min_dt]$ cd 20241215T120000Z_20241215T120100Z
237B  BP01  CT02  DG02  DG10  EE04  EF07  EF54  EF63  EF69  EF77   FOAK3  FW06  FW14  
HBVL  LWM2  MB05  MB12  MB19   MBBB3  MID03  NM01  OG02  PB05  PB11  PB19  PB26  PB34  
PB41  PB51  PB57  PH03  SA06  SGCY  SN02  SN10  WB03  WB09  YK01
435B  BRDY  CV01  DG04  DKNS  EF02  EF08  EF56  EF64  EF71  ELG6   FOAK4  FW07  FW15  
HNDO  LWM3  MB06  MB13  MB21   MBBB5  MLDN   NM02  OG04  PB06  PB12  PB21  PB28  PB35  
PB42  PB52  PB58  PL01  SA07  SM01  SN03  SNAG  WB04  WB10
ALPN  BW01  CW01  DG05  DRIO  EF03  EF09  EF58  EF65  EF72  ET02   FW01   FW09  GV01  
HP01  MB01  MB07  MB15  MB22   MBBB6  MNHN   NM03  OZNA  PB07  PB14  PB22  PB29  PB37  
PB43  PB53  PB59  PLPT  SA09  SM02  SN04  TREL  WB05  WB11
APMT  CF01  DB02  DG06  DRZT  EF04  EF51  EF59  EF66  EF74  FLRS   FW02   FW11  GV02  
HP02  MB02  MB08  MB16  MB25   MG01   MO01   ODSA  PB01  PB08  PB16  PB23  PB30  PB38  
PB44  PB54  PCOS  POST  SAND  SM03  SN07  VHRN  WB06  WB12
AT01  CRHG  DB03  DG07  EE02  EF05  EF52  EF61  EF67  EF75  FOAK1  FW04   FW12  GV03  
INDO  MB03  MB09  MB17  MBBB1  MID01  NGL01  OE01  PB03  PB09  PB17  PB24  PB32  PB39  
PB46  PB55  PECS  SA02  SD01  SM04  SN08  VW01  WB07  WTFS
BB01  CT01  DB04  DG09  EE03  EF06  EF53  EF62  EF68  EF76  FOAK2  FW05   FW13  GV04  
LWM1  MB04  MB11  MB18  MBBB2  MID02  NGL02  OG01  PB04  PB10  PB18  PB25  PB33  PB40  
PB47  PB56  PH02  SA04  SE01  SMWD  SN09  WB02  WB08  WW01

[skevofilaxc PB35]$ ls
TX.PB35.00.HH1__20241215T115800Z__20241215T120100Z.mseed  TX.PB35.00.
HHZ__20241215T115800Z__20241215T120100Z.mseed
TX.PB35.00.HH2__20241215T115800Z__20241215T120100Z.mseed
```
EQCCT requires at least one pose per station for detection, but 
using multiple poses enhances P and S wave directionality.

Where each subdirectory is named after station code. If you wish to use create your own input directory with custom waveform mSEED files, **please follow the above naming conventions.** Otherwise, EQCCTPro will **not** work. 
Create subdirectories for each timechunk (sub-parent directories) and for each station (child directories). The station directories should be named as shown above. Each timechunk directory spans from the **start of the analysis period minus the waveform overlap** to the **end of the analysis period**, based on the defined timechunk duration.

For example: 
```sh
[skevofilaxc 230_stations_2hr_1_hr_dt]$ ls
20241215T115800Z_20241215T130000Z  20241215T125800Z_20241215T140000Z
```
The timechunk time length is 1 hour long. At the same time, we use a waveform overlap of 2 minutes. Hence: `20241215T115800Z_20241215T130000Z` spans from `11:58:00 to 13:00:00 UTC on Dec 15, 2024` and `20241215T125800Z_20241215T140000Z` spans from `12:58:00 to 14:00:00 UTC on Dec 15, 2024`. 

## **Dataset creation using a FDSNWS connection**

Through the help of [Donavin97](https://github.com/Donavin97), it is now possible to create the necessary dataset structure with your own data using the 
provided `create_dataset.py` script.

`create_dataset.py` can:
1. Retrieves waveform data from a user defined FDSNWS webservice.
2. Selects data according to network, station, channel and location codes.
3. Has the option for defining time chunks according to the users requirements.
4. Automatically downloads and creates the required folder structure for eqcctpro.
5. Optionally denoises the data using seisbench as backend.

An example is provided below:
```sh
python create_dataset.py -h

usage: create_dataset.py [-h] [--start START] [--end END] [--networks NETWORKS] 
[--stations STATIONS] [--locations LOCATIONS]
                         [--channels CHANNELS] [--host HOST] [--output OUTPUT] [--chunk 
                         CHUNK] [--denoise]

Download FDSN waveforms in equal-time chunks.

options:
  -h, --help            show this help message and exit
  --start START         Start time, e.g. 2024-12-03T00:00:00Z
  --end END             End time, e.g. 2024-12-03T02:00:00Z
  --networks NETWORKS   Comma-separated network codes or *
  --stations STATIONS   Comma-separated station codes or *
  --locations LOCATIONS
                        Comma-separated location codes or *
  --channels CHANNELS   Comma-separated channel codes or *
  --host HOST           FDSNWS base URL
  --output OUTPUT       Base output directory
  --chunk CHUNK         Chunk size in minutes. Splits start■end into N windows.
  --denoise             If set, apply seisbench.DeepDenoiser to each chunk.
```

An example to download waveforms from a local FDSNWS server is given below:
```sh
python create_dataset.py --start 2025-10-31T00:00 --end 2025-10-31T04:00 --networks TX 
--stations "*" --locations "*" --channels HH?,HN? --host http://localhost:8080 --output 
waveforms_directory --chunk 60
```

The resulting output folder contains the data to be processed by EQCCTPro.

**Note:** Please make sure that you set a consistant chunk size in the download script, as well as in EQCCTPro itself to avoid issues.
E.g.: If you set a time chunk of 20 minutes in the download script, then also use 20 minutes as chunk size when calling EQCCTPro. This is so that data won't be processed erroneously.

---

# **1. Processing mSEED Data (RunEQCCTPro)**

The `RunEQCCTPro` class is the primary interface for running seismic detection on your data. It handles model loading (TensorFlow or PyTorch), waveform segmenting, and parallel pick generation.

### **Example: Running SeisBench PhaseNet on GPU**
```python
from eqcctpro import RunEQCCTPro

runner = RunEQCCTPro(
    model_type='seisbench',           # 'eqcct' or 'seisbench'
    seisbench_parent_model='PhaseNet',# SeisBench class
    seisbench_child_model='original', # Pretrained version
    Detection_threshold=0.3,          # SeisBench detection threshold
    use_gpu=True,
    selected_gpus=[0, 1],             # Use multiple GPUs
    vram_mb=2500,                     # VRAM budget per station task
    number_of_concurrent_station_predictions=10,
    number_of_concurrent_timechunk_predictions=2,
    start_time='2024-12-15 12:00:00',
    end_time='2024-12-15 13:00:00',
    timechunk_dt=30, 
    waveform_overlap=2
)

runner.run_eqcctpro()
```

### **Parameter Definitions**

#### **Model Configuration**
- **`model_type (str)`**: Choice of `'eqcct'` (for the original EQCCT model) or `'seisbench'` (for SeisBench-based models).
- **`seisbench_parent_model (str)`**: (SeisBench only) The model architecture (e.g., `PhaseNet`, `EQTransformer`).
- **`seisbench_child_model (str)`**: (SeisBench only) The pretrained weights (e.g., `original`, `stead`, `ethz`).
- **`Detection_threshold (float)`**: (SeisBench only) The probability threshold for detection traces. Default: `0.3`.
- **`P_threshold (float)`**: (EQCCT only) Arrival probability threshold for P-waves. Default: `0.001`.
- **`S_threshold (float)`**: (EQCCT only) Arrival probability threshold for S-waves. Default: `0.02`.
- **`p_model_filepath / s_model_filepath (str)`**: (EQCCT only) Paths to the `.h5` model files.

#### **Hardware & Parallelism**
- **`use_gpu (bool)`**: Enables GPU acceleration. 
- **`selected_gpus (list)`**: List of GPU indices (e.g., `[0, 1]`) to utilize.
- **`vram_mb (float)`**: The hard VRAM limit allocated to **each** station prediction task.
- **`gpu_vram_safety_cap (float)`**: The fraction of VRAM (0.0 to 1.0) EQCCTPro is allowed to use for the total pool. Default: `0.95`.
- **`cudnn_headroom (float)`**: The fraction of GPU VRAM (0.0 to 0.80) to reserve specifically for cuDNN workspace overhead during concurrent predictions. Default: `0.25` (25%).
- **`cpu_id_list (list)`**: Specific CPU core IDs to bind the process to (e.g., `range(0, 16)`).
- **`intra_threads (int)`**: Default = 1; Controls how many intra-parallelism threads Tensorflow can use
- **`inter_threads (int)`**: Default = 1; Controls how many inter-parallelism threads Tensorflow can use
- **`number_of_concurrent_station_predictions (int)`**: How many stations to process in parallel per timechunk.
- **`number_of_concurrent_timechunk_predictions (int)`**: How many timechunks to process in parallel.

#### **Workflow & Data**
- **`input_dir / output_dir (str)`**: Paths for input mSEED files and output pick results.
- **`start_time / end_time (str)`**: Analysis window (Format: `YYYY-MM-DD HH:MM:SS`).
- **`timechunk_dt (int)`**: Duration of each processing chunk in minutes.
- **`waveform_overlap (int)`**: Overlap between chunks in minutes to ensure no events are missed at boundaries.
- **`best_usecase_config (bool)`**: If `True`, overrides parallelism settings with the optimal values found by `EvaluateSystem`.

#### **Waveform band-limiting (EQCCT + SeisBench paths)**

Before resampling to 100 Hz, traces are filtered using [ObsPy’s `Stream.filter`](https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.filter.html). You can set this from `RunEQCCTPro` / `EvaluateSystem` (it is forwarded into each Ray worker’s `args` dict):

- **`waveform_filter_type (str)`**: One of **`bandpass`**, **`bandstop`**, **`lowpass`**, **`highpass`** (same set as ObsPy exposes for these modes). Default: **`bandpass`** (legacy behavior).
- **`waveform_filter_freqmin (float)`**: For **`bandpass`** / **`bandstop`**, the lower corner (Hz). For **`highpass`**, the corner frequency (Hz). Default: **`1.0`**.
- **`waveform_filter_freqmax (float)`**: For **`bandpass`** / **`bandstop`**, the upper corner (Hz). For **`lowpass`**, the corner frequency (Hz). Default: **`45.0`**.
- **`waveform_filter_corners (int)`**: Number of corners (filter order per ObsPy). Default: **`2`**.
- **`waveform_filter_zerophase (bool)`**: If `True`, apply the forward-backward filter for zero phase shift (ObsPy `zerophase=True`). Default: **`True`**.

If the worker **`args`** dictionary includes **`stations_filters`** (pandas `DataFrame` with columns **`sta`**, **`hp`**, **`lp`**), per-station **`hp`** / **`lp`** still override **`waveform_filter_freqmin`** / **`waveform_filter_freqmax`** for that station, matching the previous hard-coded override behavior (same mechanism as **`mseed_predictor(..., stations_filters=...)`**).

#### **Pick output format**

Each station directory gets `<station>_outputs/` with a single result file:

- **`pick_output_format (str)`**: **`xml`** (default), **`ascii`**, or **`csv`**.
  - **XML**: UTF-8 document `X_prediction_results.xml` with one `<pick>` element per row; child tags match the historical CSV columns (`file_name`, `network`, `station`, …).
  - **ASCII**: file `X_prediction_results.ascii`; currently the same comma-separated / quoted row layout as CSV (a dedicated fixed-width layout may be added later).
  - **CSV**: legacy `X_prediction_results.csv`.

The same option applies in **Ripper** mode and for **EvaluateSystem** trials that call `mseed_predictor`.

---

# **2. System Evaluation (EvaluateSystem)**

Before running large-scale production jobs, use `EvaluateSystem` to benchmark your hardware. It autonomously runs trials across different concurrency levels to find the "sweet spot" for your system.

### **Trial timing and result CSV metrics**

Trial durations are measured with **`eqcctpro.timing_util.monotonic_s()`**, which wraps Python’s **`time.perf_counter()`**: the clock is monotonic and high-resolution, so elapsed times are not skewed by NTP or manual clock changes the way wall-clock **`time.time()`** can be.

For **SeisBench (PyTorch) on GPU**, **`eqcctpro.timing_util.cuda_synchronize_best_effort()`** runs after weights are moved to the device and again after **`classify`** inside Ripper tasks and inside **`SeisBenchModelActor`**. PyTorch schedules GPU work asynchronously; without an explicit device synchronize, stopwatches can stop before kernels finish. That policy keeps **model load** and **inference** intervals aligned with completed CUDA work where applicable. **EQCCT (TensorFlow)** Ripper paths rely on the standard **`predict`** path, which already blocks the host until batch work completes for the generator-driven workflow.

Trial outputs use the canonical CSV header defined in **`eqcctpro/tools.py`** (see **`CANONICAL_CSV_HEADER`**). The primary timing columns have the following meaning:

| Column | Meaning |
|--------|---------|
| **Total Trial Time (s)** | Wall time from the trial’s **initial** timestamp (driver/worker entry for the native process) through the end of the picking phase. |
| **Total Run time for Picker (s)** | Wall time for the **picker phase** only: from just before the driver merges station mSEED (and registers shared waveform refs) through completion of all station tasks. Compare this across ModelActor, Ripper, and serial baselines. |
| **Actor Creation Time (s)** | Time to create and warm **ModelActor** (or SeisBench actor) instances; empty in Ripper-only rows. |
| **Avg Model Load Time (s)** | Mean **per-task** model load interval in **Ripper** mode (each task loads its own model). |
| **Waveform Processing Time (s)** | Mean **per-station-task** time spent preparing waveforms in the worker (e.g. resolving shared streams and running the same preprocessing wedge as production picking), not including the one-time driver-side merge of all stations into the object store. |

The **`experiments/workbench/scaling/run_requested_benchmarks.py`** script uses the same **`timing_util`** helpers for **raw serial** and **reload-per-waveform** baselines so those CSV rows remain comparable to **`EvaluateSystem`** trial exports for the same column names.

### **Key Benchmark Optimizations**
- **20% Step Size**: Automatically tests station concurrency at 20%, 40%, 60%, 80%, and 100% levels.
- **Redundancy Filtering**: Skips configurations that are already in the results CSV, allowing for interrupted evaluations to resume instantly.
- **Memory-Aware Dynamic Actor Pool**: EQCCTPro automatically calculates how many model instances can safely fit on each GPU based on the specific model's VRAM requirements and the available hardware memory. It utilizes Ray's fractional GPU allocation to stack multiple actors on a single GPU when possible, maximizing throughput while preventing memory conflicts.
- **cuDNN Stability Logic**: To prevent "DNN library is not found" and "BLAS operation" errors, EQCCTPro enforces a dynamic safety buffer (`cudnn_headroom`). This reserves VRAM specifically for the dynamic workspace memory that TensorFlow/Keras requires during actual inference, which is not captured by static model weights.
- **Automatic Ray Restart for OOM Prevention**: When increasing concurrency between trials, actors from previous trials may remain in GPU/RAM. If spawning new actors alongside existing ones would exceed the user-defined VRAM/RAM capacity (`max_vram_mb` / `ram_safety_cap`), Ray is automatically restarted to clear memory before the trial runs. This prevents unexpected OOM crashes during benchmark progression.

### **Deep Dive: The cuDNN Safety Calculation**

The calculation logic for GPU actor capping and concurrent prediction limits transforms a static hardware limit into a dynamic, stability-aware execution plan. Here is the step-by-step breakdown:

#### **1. Defining the Total Pool**
The `max_vram_mb` (or the physical total) is the budget for the **entire system**. 
*   **Example**: If you have 2 GPUs and set `max_vram_mb = 93100`, the budget per card is calculated as:
    `Per-GPU Budget = 93100 / 2 = 46550 MB`

#### **2. Theoretical Capacity**
The system determines how many models could *physically* fit if memory usage were perfectly static (model weights only). It uses the `per_actor_vram_mb`, which already includes a fixed Ray/CUDA buffer (e.g., 1024 MB).
*   **Example**: `46550 MB / 2756 MB per actor ≈ 16.9 actors`.

#### **3. Applying the cuDNN Headroom (ModelActor Mode)**
This is the most critical step for stability in **ModelActor mode**. TensorFlow/Keras models allocate **dynamic workspace memory** during actual prediction (for convolution algorithms and BLAS operations). This memory is not reserved at startup. 

The `cudnn_headroom` (default **25%**) is applied to the theoretical capacity to reserve a "cushion" for these dynamic spikes:
*   `Safe Max Per GPU = floor(16.9 × 0.75) = 12 actors`
*   `Max Actors Total = 12 × 2 GPUs = 24 actors`

**Note**: For **Ripper Mode**, a different, more optimized calculation is used (see below).

#### **4. Why this prevents OOM**
The `max_vram_mb` acts as the ceiling, but the **Headroom** acts as the safety net. 
*   **Without Headroom**: If you loaded 16 models on one 46GB GPU, you would use ~44GB for weights. When those 16 models all start a prediction simultaneously, they each try to grab ~500MB+ for cuDNN workspaces ($16 \times 500\text{MB} = 8\text{GB}$ extra). Total needed: $44\text{GB} + 8\text{GB} = 52\text{GB}$, leading to an **immediate crash (OOM)**.
*   **With 25% Headroom**: By capping at 12 models, you use ~33GB for weights. This leaves **~13GB of free VRAM** specifically for the dynamic cuDNN workspaces. Even if all 12 models hit a heavy convolution at the same time, they have plenty of room to "breathe."

#### **5. Interaction with Concurrency Limits**
The final piece ensures that `max_pending_tasks` (the number of stations processed at once) matches this safety calculation. EQCCTPro caps the task submission to the `Max Actors Total` (e.g., 26). Even if the user requested 100 concurrent tasks, the code only allows 26 to enter the GPU pipeline at a time. This prevents the "Idle Actor" problem, where actors exist and consume VRAM for weights but starve the active tasks of the dynamic memory they need to finish.

### **Optimal Actor Pool Architecture (Technical Details)**

EQCCTPro uses a hardware-aware actor pool design for maximum efficiency:

| Mode | Actor Creation | Concurrency Method |
|------|----------------|-------------------|
| **GPU** | Multiple ModelActors per physical GPU (VRAM dependent) | Task queue with round-robin |
| **CPU** | 1 ModelActor per physical CPU core (RAM dependent) | Task queue with round-robin |

**Why not create N actors for N concurrent tasks?**
- **GPU**: Deep learning frameworks (TensorFlow/PyTorch) allocate GPU memory based on their internal runtimes. EQCCTPro calculates a `MIN_FRACTIONAL_GPU` value dynamically for each model (e.g., PhaseNet needs ~1.6% of a 93GB GPU, while EQCCT needs ~3%). By accurately matching the actor count to available VRAM, we achieve maximum parallelism without OOM.
- **CPU**: Each actor loads a full model copy into RAM. EQCCTPro caps the actor count based on available system RAM and a safety buffer to prevent thrashing and system instability.

**How concurrency works:**
```
Example: 1 GPU (93GB), PhaseNet model (~1.5GB VRAM), 60 stations to process
- EQCCTPro calculates that ~57 actors can fit on the GPU (95% headroom / 1.6% per actor)
- 60 tasks submitted to queue
- Tasks are distributed across the 57 actors
- Result: High degree of parallelism (~57x) on a single GPU
```

The `Number of Concurrent Station Tasks` column in CSV reports the *requested* concurrency, while `N ModelActors` reports the *actual* actors created (capped to hardware and memory limits). When these differ, the `Comments` column explains the constraint (e.g., "Requested 100 actors, created 57 (VRAM limited to 93100 MB, 1524 MB/actor)").

---

### **Ripper Mode: Task-Based Alternative**

EQCCTPro offers an alternative parallelization strategy called **Ripper Mode**, which uses the older task-based approach instead of persistent ModelActors.

#### **Enabling Ripper Mode**
```python
# For EvaluateSystem
eval_gpu = EvaluateSystem(
    eval_mode='gpu',
    model_type='eqcct',
    # ... other parameters ...
    ripper=True  # Enable Ripper Mode
)

# For RunEQCCTPro
runner = RunEQCCTPro(
    model_type='eqcct',
    # ... other parameters ...
    ripper=True  # Enable Ripper Mode
)
```

#### **Comparison: ModelActor vs Ripper Mode**

| Aspect | ModelActor (Default) | Ripper Mode |
|--------|---------------------|-------------|
| **Model Loading** | Once per actor (persistent) | Per task (load/unload each time) |
| **Memory Management** | Actors hold memory until Ray session ends | Memory released after each task completes |
| **GPU Scheduling** | 1 actor per physical GPU, round-robin tasks | Multiple tasks can share GPU memory dynamically |
| **Overhead** | Lower (no repeated model loading) | Higher (model loaded per task) |
| **VRAM Flexibility** | Fixed actor pool, constrained by `MIN_FRACTIONAL_GPU` | More dynamic memory sharing |
| **Timing Pattern** | High upfront `Actor Creation Time`, then fast processing | Zero actor creation, but consistent `Avg Model Load Time` per task |
| **Best For** | Production workloads, consistent throughput | Experimentation, memory-constrained environments |

#### **When to Use Ripper Mode**

- **Testing/Experimentation**: When you want to compare performance between approaches
- **Memory-Constrained Systems**: When you need more flexible GPU memory sharing
- **Variable Workloads**: When task memory requirements vary significantly
- **Legacy Compatibility**: Matches the original EQCCTPro behavior before ModelActor optimization

#### **Memory Management in Ripper Mode (Empirically Calibrated)**

Ripper mode uses a simplified but highly optimized memory calculation derived from actual GPU test trials. Unlike ModelActor mode which stacks multiple fixed buffers, Ripper mode uses **model-specific initialization multipliers** and a fixed **concurrency headroom**.

##### **1. Per-Task Initialization Multiplier**
Each Ripper task loads its own model. The peak memory during initialization (graph construction, library loading, and weight allocation) is higher than the steady-state inference memory. We use empirically-calibrated multipliers to account for this:

| Model | Multiplier | Rationale |
|-------|------------|-----------|
| **EQCCT** | **2.0×** | Empirical 1.94× measured + safety margin |
| **EQTransformer** | **2.0×** | Large model architecture, similar to EQCCT |
| **PhaseNet / Light** | **1.7×** | Smaller models, higher relative CUDA context overhead |
| **GPD** | **1.8×** | Medium-sized model |

##### **2. Concurrency Headroom (10%)**
When many Ripper tasks run **simultaneously**, they compete for cuDNN workspace memory and CUDA context switching. To prevent `CUDNN_STATUS_EXECUTION_FAILED` errors during high-concurrency peaks, EQCCTPro reserves a fixed **10% concurrency headroom**.

##### **3. GPU Ripper Concurrency Calculation**
Before each trial, EQCCTPro queries **actual free VRAM** from the GPU(s) using `pynvml` and applies the following logic:

```text
# 1. Get the pool
effective_vram_pool = min(user_defined_pool, actual_free_vram)

# 2. Reserve 10% for concurrent task interference
usable_vram = effective_vram_pool × (1.0 - 0.10)

# 3. Calculate tasks per GPU using the init multiplier
vram_per_task = base_model_vram × model_multiplier
max_concurrent = floor(usable_vram / vram_per_task)
```

**Why this is better:** 
- **No Double-Counting**: It avoids stacking `VRAM_BUFFER`, `cudnn_headroom`, and `ripper_multiplier` additively.
- **Empirically Proven**: This logic enabled **24-26 concurrent EQCCT tasks** on a dual-49GB GPU system, whereas conservative stacking would have limited it to ~20.
- **Smart Slicing**: Each task is assigned a `gpu_memory_limit_mb` (TensorFlow soft cap) equal to the fair share of the usable VRAM pool.

##### **4. CPU Ripper Mode: RAM-Aware Concurrency Limiting**
Before each trial, ripper queries **actual available RAM** using `psutil`:
```text
usable_ram = actual_available_ram × ram_safety_cap
max_concurrent = floor(usable_ram / ram_per_task)
```
This prevents system instability and swap-thrashing when many tasks load models simultaneously.

##### **Tunable Headroom Variables**

| Variable | Parameter | Default | Mode | Description |
|----------|-----------|---------|------|-------------|
| `cudnn_headroom` | `cudnn_headroom` | `0.25` | GPU | **(ModelActor Only)** Reserved for cuDNN workspace |
| `Ripper Headroom` | *Internal* | `0.10` | GPU | **(Ripper Only)** Reserved for concurrency interference |
| `ram_safety_cap` | `ram_safety_cap` | `0.90` | CPU | Fraction of Total System RAM to use |

##### **Recommended Values and Risks**

| cuDNN Headroom | Use Case | Risk Level |
|----------------|----------|------------|
| **0.25 - 0.30** | Shared systems, other GPU-intensive processes running | Very Safe |
| **0.25** | Dedicated systems with moderate background processes | Safe (Default) |
| **0.15 - 0.20** | Dedicated systems with minimal background processes | Moderate Risk |
| **< 0.05** | Not recommended | High Risk of OOM |

**Risks of Higher Values:**
- **GPU (VRAM)**: Memory fragmentation can cause `CUDNN_STATUS_EXECUTION_FAILED` or `DNN library is not found` errors even when theoretical VRAM is available. TensorFlow's XLA compilation can cause temporary memory spikes.
- **CPU (RAM)**: System may become unresponsive if RAM is exhausted. Linux OOM killer may terminate processes unexpectedly. Swap usage causes severe performance degradation.

**Recommendation**: Start with the defaults (`0.25` for GPU, `0.90` for CPU). If you encounter OOM errors, increase the headroom (reduce `ram_safety_cap`). If tasks complete reliably, you can gradually decrease headroom to maximize throughput.

##### **Other OOM Prevention Mechanisms**

1. **Per-Trial Ray Restart (Ripper Mode Only)**: In ripper mode, Ray is **restarted at the start of every trial** to ensure a completely fresh GPU/RAM state. This eliminates memory fragmentation issues that can accumulate when Ray workers are reused between trials. This is the primary mechanism that prevents OOM in ripper mode.

2. **Automatic Ray Restart (Fallback)**: When increasing concurrency between trials, if the estimated memory footprint would exceed capacity, Ray is automatically restarted to clear GPU/RAM memory. This applies to both standard ModelActor mode and ripper mode as an additional safety mechanism.

3. **Task-Level Cleanup**: Each task explicitly releases model memory after completion:
   - TensorFlow: `del model` + `tf.keras.backend.clear_session()`
   - PyTorch: `del model` + `torch.cuda.empty_cache()`

4. **CSV Tracking**: The `Actual Ripper Concurrent Tasks` column shows the real concurrency used after memory limiting. Compare with `Number of Concurrent Station Tasks` (requested) to see if memory constraints reduced concurrency.

**Note**: The per-trial Ray restart in ripper mode adds ~1-2 seconds overhead per trial but ensures reliable execution by preventing memory fragmentation. This is the recommended approach for maximum stability when running many consecutive trials.

---

### **Example: Evaluating GPU Performance**
```python
from eqcctpro import EvaluateSystem

eval_gpu = EvaluateSystem(
    eval_mode='gpu',
    model_type='seisbench',
    seisbench_parent_model='PhaseNet',
    seisbench_child_model='original',
    selected_gpus=[0, 1],
    max_vram_mb=48000,                # Total VRAM pool to test across all GPUs
    gpu_vram_safety_cap=0.95,         # Reserve 5% VRAM for system stability
    ram_safety_cap=0.95,              # Reserve 5% System RAM for machine stability
    stations2use=100,                 # Max stations to test
    cpu_id_list=range(0, 8),          # CPUs available for Ray management
    input_dir='/path/to/mseed',
    csv_dir='/path/to/results'
)
eval_gpu.evaluate()
```

### **Evaluation Parameters**
- **`eval_mode (str)`**: `'cpu'` or `'gpu'`.
- **`max_vram_mb (float)`**: The total aggregate VRAM budget across all GPUs for the evaluation. If not provided, it is calculated from physical VRAM.
- **`gpu_vram_safety_cap (float)`**: The fraction of VRAM (0.0 to 1.0) EQCCTPro is allowed to use for the total pool. Default: `0.95`.
- **`cudnn_headroom (float)`**: The fraction of GPU VRAM (0.0 to 0.80) to reserve specifically for cuDNN workspace overhead during concurrent predictions. Default: `0.25` (25%).
- **`ram_safety_cap (float)`**: The fraction of Total System RAM (0.0 to 0.98) EQCCTPro is allowed to use. Default: `0.90`.
- **`stations2use (int)`**: The maximum number of stations to test in the benchmark.
- **`min_cpu_amount / cpu_test_step_size (int)`**: Controls the iterative testing of CPU core counts.
- **`starting_amount_of_stations / station_list_step_size (int)`**: Controls the iterative testing of total workload size.

---

# **3. Numerical Efficiency Analysis**

The `analyze_trial_results_efficiency.py` script provides deep insights into your benchmarking results, including throughput gains, memory utilization, and comparative performance between ModelActor and Ripper modes.

### **Basic Usage (Single CSV)**
```sh
python scripts/analysis/analyze_trial_results_efficiency.py /path/to/results/cpu_test_results.csv --output_dir analysis/
```
*   **`--desired_runtime`**: (Optional) Adds a red dashed horizontal line to the runtime plot at the specified seconds (e.g., `--desired_runtime 30`).

### **Batch Analysis**
Analyze all result subdirectories within a results folder simultaneously to generate a global summary.
```sh
python scripts/analysis/analyze_trial_results_efficiency.py --batch --results_root results/csv/ --output_dir batch_analysis/
```

### **Mode Comparison**
Compare the performance of **ModelActor** vs. **Ripper** mode side-by-side for a specific model and hardware type.
```sh
python scripts/analysis/analyze_trial_results_efficiency.py --compare --model eqcct --trial_type cpu --results_root results/csv/
```

### **Key Metrics Defined**
- **Effective Concurrency**: A unified metric for parallelism. In **ModelActor** mode, it represents the **number of persistent actors spawned**. In **Ripper** mode, it represents the **actual number of concurrent tasks executed**.
- **Throughput (Stations/s)**: The total number of stations processed divided by the total runtime.
- **RAM Utilization (%)**: A measure of memory efficiency, calculated as `(Process Tree RAM / Total Requested RAM) × 100`. Values < 100% indicate efficient memory sharing (e.g., via Copy-on-Write), while values > 100% indicate that overhead (Ray/Framework) has exceeded the allocated safety buffers.
- **Resource Cost Score**: A weighted score (CPUs + GPUs × 10) to evaluate the efficiency of hardware utilization.

### **Generated Artifacts**
The script generates the following in your output directory:
1.  **`efficiency_summary_*.txt`**: A comprehensive text report with throughput formulas, memory usage summaries, and performance lever correlations.
2.  **`correlation_matrix_*.png`**: Visualizes how resources correlate with runtime. **Note**: In ModelActor mode, requested concurrency is excluded in favor of `Effective Concurrency`.
3.  **`runtime_vs_stations_by_concurrency_*.png`**: Static scatter plot showing runtime scaling across workload sizes.
4.  **`requested_vs_actual_ram_*.png`**: Analyzes theoretical vs. actual memory footprints.
5.  **`efficiency_analysis_results_*.csv`**: A processed version of trial data with derived metrics.

---

# **4. Interactive Visualization (Plotly)**

The `visualize_trial_results.py` script generates high-fidelity, interactive visualizations (HTML) for exploring multidimensional trial data.

### **Usage Examples**
```sh
# Single file visualization (auto-detects model name from CSV)
python scripts/visualization/visualize_trial_results.py /path/to/results/cpu_test_results.csv --output_dir vis/

# Batch visualization (creates separate folders per trial subdirectory)
python scripts/visualization/visualize_trial_results.py --batch --results_root results/csv/ --output_dir batch_vis/

# Compare ModelActor vs Ripper performance for a specific model (fixed hardware)
python scripts/visualization/visualize_trial_results.py --compare --model phasenetlight --trial_type cpu

# Universal Comparison (CPU vs GPU across both methods)
# Omitting --trial_type triggers a comprehensive 4-way comparison
python scripts/visualization/visualize_trial_results.py --compare --model eqcct --output_dir vis/universal/

# Optimal Configuration Visualization (from optimal_configurations_*.csv files)
# Single model comparison (CPU vs GPU, ModelActor vs Ripper)
python scripts/visualization/visualize_trial_results.py --optimal --compare --model eqcct --results_root results/trials/ --output_dir vis/optimal_comparisons/

# Batch optimal comparison (all models, auto-discovers from results/trials/)
python scripts/visualization/visualize_trial_results.py --optimal --compare --batch --results_root results/trials/ --output_dir visualizations/optimal_comparisons/

# Batch optimal comparison via shell script (alternative; edit MODELS in script for custom list)
./scripts/visualization/batch_optimal_comparison.sh results/trials/ visualizations/optimal_comparisons/
```

### **Optimal Configuration Plots**
- **3D Scatter**: CPUs (x) vs Stations (y) vs Runtime/Picking Time (z), with rainbow color scale for concurrent tasks and marker shapes for GPU count and execution mode.
- **Comparison Table**: GPU columns separated by count (e.g., GPU (1), GPU (2)), plus Mean VRAM, Actor Creation Time, and Model Load Time.

### **Interactive Plot Features**
- **3D Resource Plots**: Explore the relationship between **CPUs**, **Total Stations**, and **Runtime/RAM/VRAM** in interactive 3D space.
- **Memory Efficiency (2D)**: Requested vs. Actual RAM/VRAM plots with fixed 10K step sizes and 0-min axes for clarity.
- **Throughput Scaling**: Visualize how `Effective Concurrency` impacts stations-per-second.
- **Customizable Overlays**: Pass `--desired_runtime X` to add a labeled target runtime line to 2D runtime plots.
- **Hover Details**: Comprehensive tooltips showing station counts, CPUs, requested vs. created actors, and precise memory/runtime values.
- **Hardware-Aware Scales**: Automatically uses "N Model Actors" for ModelActor trials and "Concurrent Tasks" for Ripper trials, with a standardized rainbow color scale.

---

# **5. Finding Optimal Configurations**

Once the evaluation is complete, use the configuration finders to extract the best settings. Results are now automatically grouped by the model used during testing.

```python
from eqcctpro import OptimalGPUConfigurationFinder

# results_dir should contain 'gpu_test_results.csv'
finder = OptimalGPUConfigurationFinder(results_dir='/path/to/results')

# 1. Get the fastest overall config for a balanced workload
best_config = finder.find_best_overall_usecase()

# 2. Get the optimal config for a specific resource limit
# Example: What is the fastest way to process 50 stations using 4 CPUs and GPU 0?
specific_config = finder.find_optimal_for(num_cpus=4, gpu_list=[0], station_count=50)
```

There are more examples on how to use EQCCTPro using different SeisBench and EQCCT models in `run.py` file.

---

# **6. Resource Requirement Calibration**

EQCCTPro relies on accurate memory footprint estimates to perform memory-aware parallelization and prevent Out-Of-Memory (OOM) errors. These estimates are stored as lookup tables in `eqcctpro/parallelization.py`.

### **Measuring Model Memory Usage**

If you are using a new hardware environment or a custom model, you should calibrate these values using the provided profiling script:

```sh
python measure_model_memory_usage.py
```

This script:
1.  Spawns each model (SeisBench and EQCCT) in an **isolated subprocess**.
2.  Measures the "first-load" footprint (Library initialization + Weights + Inference buffers).
3.  Outputs formatted Python dictionaries that can be copied directly into the codebase.

### **Updating Lookup Tables**

After running the script, open `eqcctpro/parallelization.py` and update the following constants based on the script's output:

#### **1. SEISBENCH_MODEL_VRAM_MB**
- **What it is**: The **VRAM (GPU memory)** required for one SeisBench model actor when running in **GPU mode**.
- **Why it matters**: Includes the model weights and the CUDA context overhead (~500 MB) that each fresh process incurs.

#### **2. SEISBENCH_MODEL_RAM_MB**
- **What it is**: The **System RAM** required for one SeisBench model actor when running in **GPU mode**.
- **Why it matters**: Even in GPU mode, models require system RAM for the Python interpreter, PyTorch libraries, ObsPy, and data pre-processing.

#### **3. SEISBENCH_MODEL_CPU_RAM_MB**
- **What it is**: The **System RAM** required for one SeisBench model actor when running in **CPU mode**.
- **Why it matters**: This is the primary memory constraint for CPU-only evaluation. It typically excludes the CUDA runtime overhead found in `SEISBENCH_MODEL_RAM_MB`.

#### **EQCCT Specific Constants**
Similarly, you should update the EQCCT-specific constants:
- **`EQCCT_GPU_VRAM_MB`**: VRAM requirement for EQCCT in GPU mode.
- **`EQCCT_GPU_RAM_MB`**: System RAM requirement for EQCCT in GPU mode (often higher due to XLA compilation).
- **`EQCCT_CPU_RAM_MB`**: System RAM requirement for EQCCT in CPU mode.

---

# **Understanding Evaluation Results (CSV Columns)**

The `EvaluateSystem` functionality generates a CSV file (e.g., `cpu_test_results.csv` or `gpu_test_results.csv`) with **PID-isolated absolute memory tracking**. This ensures accurate measurements even when running multiple EvaluateSystem instances in parallel on the same machine.

### **Core Configuration**
- **`Trial Number`**: The sequential ID of the benchmark test.
- **`Model Used`**: The specific model/weights pair tested (e.g., `PhaseNet/stead`).
- **`Number of Stations Used`**: Total number of stations processed.
- **`Number of CPUs Allocated for Ray to Use`**: The CPU affinity limit set for the Ray cluster. If `N ModelActors` exceeds this count, actors will automatically share cores via fractional CPU allocation.
- **`GPUs Used`**: List of physical GPU IDs utilized (e.g., `[0, 1]`).
- **`N ModelActors`**: The **actual** number of Ray ModelActors created for parallel inference. This may be less than the requested amount if VRAM/RAM constraints limit how many models can be loaded simultaneously. Will be set to 0 if not using ModelActor mode (IE. Ripper Mode).
- **`Number of Concurrent Station Tasks`**: The **requested** concurrency level for station predictions. This is what the user asked for, but the actual number of actors created (`N ModelActors`) may be lower due to memory constraints.
- **`Actual Ripper Concurrent Tasks`**: **(Ripper mode only)** The **actual** number of concurrent tasks used after memory-aware limiting. In GPU ripper mode, this may be lower than requested if actual free VRAM is insufficient. In CPU ripper mode, this may be lower if actual available RAM is insufficient. This column is empty for standard ModelActor mode (use `N ModelActors` instead).
- **`Concurrent Timechunks Used`**: Number of timechunks processed in parallel.
- **`Total Number of Timechunks`**: Number of temporal segments the data was split into.
- **`Length of Timechunk (min)`**: Duration of a single timechunk.
- **`Total Waveform Analysis Timespace (min)`**: Total duration of the analyzed waveforms.

### **Model-Requested Memory (Theoretical)**
*These values represent the expected memory footprint based on empirical model data, calibrated from GPU test trials (2x 49GB GPUs, 93100 MB pool).*
- **`Requested VRAM per Actor (MB)`**: Theoretical VRAM required for one model instance (GPU trials only). **Includes safety buffer** (1024 MB, empirically calibrated).
- **`Requested RAM per Actor (MB)`**: Theoretical system RAM required for one model instance. In CPU trials, this is the primary memory footprint; in GPU trials, this tracks the system RAM overhead for the GPU process. **Includes safety buffer** (1536 MB).
- **`Total Requested VRAM (MB)`**: The total theoretical VRAM footprint expected (`N ModelActors × Requested per Actor + N × 128 MB overhead`).
- **`Total Requested RAM (MB)`**: The total theoretical RAM footprint expected (`N ModelActors × Requested per Actor + N × 256 MB overhead`).

### **Actual Memory Used (PID-Isolated Absolute Values)**
*These are absolute values for THIS specific process tree only. Unlike deltas, these will never be 0 when workers are reused.*

- **`Process Tree VRAM (MB)`**: The **absolute** VRAM used by this process tree (main PID + all child workers) on the assigned GPUs. Calculated using `pynvml` to query which of our specific PIDs are running on the GPU and their exact VRAM consumption.
- **`Num GPU Processes`**: Number of processes from this trial found running on the GPUs.
- **`Process Tree RAM (MB)`**: The **absolute** combined RSS (Resident Set Size) of the main process and all its child processes (Ray workers). This is the total RAM this process tree is currently consuming.
- **`Peak RAM (MB)`**: The **maximum** RSS observed during the trial, captured via `getrusage()`. Useful for understanding peak memory pressure during inference.
- **`RAM Growth (MB)`**: The increase in RAM during the trial (`Process Tree RAM (after) - Process Tree RAM (before)`). Shows how much additional RAM was allocated during the trial.
- **`Num Worker Processes`**: Total number of Ray worker processes in this process tree.

### **Memory Overhead (Ray + Framework Cost)**
*Overhead represents the difference between the actual memory consumed and the theoretical model requirements. This includes memory used by Ray infrastructure (Raylets, worker processes), TensorFlow/PyTorch framework buffers, CUDA context, and other runtime overhead.*

- **`VRAM Overhead (MB)`**: The additional VRAM consumed beyond what was theoretically requested (`Process Tree VRAM - Total Requested VRAM`). A **positive value** indicates that Ray/CUDA/framework infrastructure uses more memory than just the model weights. A **negative value** would indicate that the actual usage is less than expected (rare).
- **`RAM Overhead (MB)`**: The additional system RAM consumed beyond what was theoretically requested (`Process Tree RAM - Total Requested RAM`). This is calculated for **both** CPU and GPU trials since Ray workers and data processing always use RAM regardless of whether models run on GPU.

**Overhead Breakdown:**
The overhead typically consists of:
1. **CUDA Context**: ~400-500 MB per GPU for CUDA library initialization.
2. **Ray Raylets**: ~100-200 MB per Raylet process for Ray's distributed runtime.
3. **Ray Workers**: ~50-150 MB per worker process for Python interpreter and framework imports.
4. **TensorFlow/PyTorch Buffers**: Variable memory for gradient caching, intermediate tensors, and framework-specific optimizations.
5. **Safety Buffer**: ~1024 MB (1 GB) for VRAM and ~1536 MB (1.5 GB) for RAM reserved for operational stability and framework overhead. These values are empirically calibrated from GPU test trials.
6. **Per-Task Overheads**: ~128 MB VRAM and ~256 MB RAM per concurrent task for waveform data handling and processing.

### **Efficiency & Performance**
      - **`Effective Concurrency`**: A unified metric representing the degree of parallelism used in the trial. 
        - In **ModelActor mode**, this equals `N ModelActors` (the number of persistent models loaded).
        - In **Ripper mode**, this equals either `Actual Ripper Concurrent Tasks` (if available) or `Number of Concurrent Station Tasks` (the number of simultaneous tasks requested).
- **`VRAM Utilization (%)`**: How much VRAM is actually being used relative to what was requested (`Process Tree VRAM / Total Requested VRAM × 100`). Values **>100%** indicate overhead exceeds buffer allocations; values **<100%** indicate underutilization or efficient memory sharing.
- **`RAM Utilization (%)`**: How much RAM is actually being used relative to what was requested (`Process Tree RAM / Total Requested RAM × 100`). Similar interpretation to VRAM utilization.
- **`Total Trial Time (s)`**: The absolute wall-clock duration of the entire trial, from the very start of the `run_eqcctpro` call to its completion. This includes setup, ModelActor creation (if applicable), and all waveform processing.
- **`Actor Creation Time (s)`**: 
    - In **ModelActor mode**: The time taken to spin up persistent Ray actors and load their models into memory.
    - In **Ripper mode**: This field is empty (N/A).
- **`Avg Model Load Time (s)`**: 
    - In **ModelActor mode**: This field is empty (N/A).
    - In **Ripper mode**: The average time taken to load the model into memory *per station task*. Since models are loaded/unloaded for every station in Ripper mode, this represents the recurring overhead.
- **`Waveform Processing Time (s)`**: The average time to load waveforms (mSEED files) into memory per station task. This measures the `_mseed2nparray` (EQCCT) or `mseed2stream_3c` (SeisBench) call duration.
- **`Total Run time for Picker (s)`**: The total wall-clock time for all station task processing (task submission, waveform loading, and inference combined).
- **`Intra-parallelism Threads`**: Number of threads TensorFlow uses for individual operations.
- **`Inter-parallelism Threads`**: Number of threads TensorFlow uses for independent operations in parallel.
- **`Inference Actor Memory Limit (MB)`**: Legacy VRAM ceiling per shared inference actor.

### **Trial Outcome**
- **`Trial Success`**: `1` for success, `0` for failure.
- **`Error Message`**: Detailed exception info if the trial failed (e.g., OOM prevention details). It also contains informational notes like **`[RAY RESTART]`** if Ray was restarted before the trial to clear memory and prevent OOM.
- **`Comments`**: Notes on actor capping due to memory constraints. When `N ModelActors` is less than `Number of Concurrent Station Tasks`, this column explains why (e.g., "Requested 100 actors, created 1 (VRAM limited to 2756 MB, 2756 MB/actor)"). Empty when the requested concurrency was achievable.
- **`Stations Used`**: (Legacy) List of specific station codes processed.

**Note on Optimal Configuration Files:**
The configuration finders (`OptimalGPUConfigurationFinder`, etc.) generate summary CSVs (`optimal_configurations_*.csv` and `best_overall_usecase_*.csv`). These files use the same headers as above, with the exception of the concurrency column, which is labeled as **`Number of Concurrent Station Tasks per Timechunk`** for clarity in those reports.

---

# **Understanding Memory Consumption Patterns**

When analyzing evaluation results, it's important to understand why memory values vary between trials. This section explains the patterns you'll observe.

## How Memory Values Are Calculated

| Column | Formula | Description |
|--------|---------|-------------|
| **Requested RAM per Actor** | `Model_RAM + RAM_BUFFER (1536 MB)` | Theoretical per-actor cost from isolated testing (empirically calibrated) |
| **Total Requested RAM** | `N × Requested_per_Actor + N × 256 MB` | Worst-case estimate for N fresh actors |
| **Process Tree RAM** | `sum(RSS of main + all children)` | Actual total footprint at snapshot time |
| **RAM Growth** | `Process_Tree_RAM(after) - Process_Tree_RAM(before)` | What THIS specific trial added |

## Why RAM Growth Varies Between Trials

### Pattern 1: New Actor Spawned → Large Growth (+1000-3000 MB)

When a trial needs **more actors than currently exist in the Ray session**, new ones are spawned:

| Trial | N Actors | RAM Growth | What Happened |
|-------|----------|------------|---------------|
| 1 | 1 | +1128 MB | First actor spawned |
| 3 | 2 | +1596 MB | Second actor spawned |
| 6 | 3 | +2931 MB | Third actor spawned |

Each new actor costs ~1500-3000 MB (model weights + TensorFlow runtime + Ray worker).

### Pattern 2: Existing Actors Reused → Minimal Growth (<100 MB)

When a trial needs **fewer or equal actors** than already exist, Ray reuses them:

| Trial | N Actors | RAM Growth | What Happened |
|-------|----------|------------|---------------|
| 4 | 1 | -61 MB | Reused existing actor |
| 7 | 1 | +3.74 MB | Reused existing actor |
| 8 | 2 | +1.94 MB | Reused existing actors |

**Idle actors stay in memory** - Ray doesn't deallocate them between trials within the same session.

### Pattern 3: Process Tree RAM Shows "High Water Mark"

Since `ray.shutdown()` is only called **once per CPU count configuration** (not between individual trials), the Process Tree RAM reflects the **maximum actors ever spawned** in that session:

```
Trial 1:  Spawn 1 actor  → Process Tree RAM ~2500 MB
Trial 3:  Spawn 2 actors → Process Tree RAM ~4000 MB  
Trial 6:  Spawn 3 actors → Process Tree RAM ~6100 MB  ← Peak for 3 actors
Trial 7:  Use only 1     → Process Tree RAM ~3867 MB  ← Still shows accumulated memory
```

## Why "Actual < Requested" (Negative Overhead)

You may see **RAM Utilization < 100%** or negative overhead. This is normal because:

1. **Ray Worker Pooling**: Multiple actors share Python libraries, TensorFlow runtime, and Ray infrastructure
2. **Copy-on-Write**: Linux shares memory pages between forked processes until modified
3. **Conservative Buffers**: The 1.5 GB buffer is a worst-case estimate; actual usage is often lower

## Quick Reference: Reading the CSV

| If you see... | It means... |
|---------------|-------------|
| Large RAM Growth (+1000+ MB) | New ModelActor was spawned |
| Small RAM Growth (<100 MB) | Existing actors were reused |
| Process Tree RAM much higher than Requested | Accumulated from previous trials in session |
| RAM Utilization < 100% | Memory sharing between actors (efficient) |
| RAM Utilization > 100% | Overhead exceeded buffers (rare, may need larger safety cap) |

---

# **License & Citation**
EQCCTPro is provided under an open-source license. If you use this software in your research, please cite our work:
[Optimized EQCCT Paper](https://github.com/ut-beg-texnet/eqcct/blob/main/eqcctpro/OptimizedEQCCT_Paper.pdf) (Currently in Review).

# **Contact**
**Constantinos Skevofilax**: constantinos.skevofilax@austin.utexas.edu  
**Victor Salles**: victor.salles@beg.utexas.edu
