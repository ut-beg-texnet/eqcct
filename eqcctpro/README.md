# **EQCCTPro: A Powerful Seismic Event Detection & Performance Optimization Toolkit**

EQCCTPro is a high-performance seismic event detection and processing framework designed to bridge the gap between deep learning models and large-scale seismic data processing. It natively supports **EQCCT** (TensorFlow) and the **SeisBench** ecosystem (PyTorch), including models like **PhaseNet**, **EQTransformer**, **GPD**, and **CRED**. 

EQCCTPro is engineered for **real-time performance**, identifying the optimal parallelization configurations for your specific hardware (CPU and Multi-GPU) to minimize runtime and maximize station throughput. EQCCTPro has enabled seismic networks, like the Texas Seismological Research Group (TexNet), to enable their DL picking model EQCCT to run operationally, in real-time, over its network of over 250+ seismic stations. More information on the architecture and application of EQCCTPro can be found in our upcoming publication [here](https://github.com/ut-beg-texnet/eqcct/blob/main/eqcctpro/OptimizedEQCCT_Paper.pdf).

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

After donwloading the `.zip` file from the repository, run:
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

Through the help of [Donavin97](https://github.com/Donavin97), it is now possible to create the necesary dataset structure with your own data using the 
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
E.g.: If you set a time chunk of 20 minutes in the download script, then also use 20 minutes as chunk size when calling EQCCTPro. This is so that data won't be processed eroniusly.

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

---

# **2. System Evaluation (EvaluateSystem)**

Before running large-scale production jobs, use `EvaluateSystem` to benchmark your hardware. It autonomously runs trials across different concurrency levels to find the "sweet spot" for your system.

### **Key Benchmark Optimizations**
- **20% Step Size**: Automatically tests station concurrency at 20%, 40%, 60%, 80%, and 100% levels.
- **Redundancy Filtering**: Skips configurations that are already in the results CSV, allowing for interrupted evaluations to resume instantly.
- **GPU Resource Slicing**: Dynamically calculates per-task VRAM limits based on an aggregate pool.

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
- **`gpu_vram_safety_cap (float)`**: The fraction of VRAM (0.0 to 1.0) EQCCTPro is allowed to use. Default: `0.95`.
- **`stations2use (int)`**: The maximum number of stations to test in the benchmark.
- **`min_cpu_amount / cpu_test_step_size (int)`**: Controls the iterative testing of CPU core counts.
- **`starting_amount_of_stations / station_list_step_size (int)`**: Controls the iterative testing of total workload size.

---

# **3. Finding Optimal Configurations**

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


# **License & Citation**
EQCCTPro is provided under an open-source license. If you use this software in your research, please cite our work:
[Optimized EQCCT Paper](https://github.com/ut-beg-texnet/eqcct/blob/main/eqcctpro/OptimizedEQCCT_Paper.pdf) (Currently in Review).

# **Contact**
**Constantinos Skevofilax**: constantinos.skevofilax@austin.utexas.edu  
**Victor Salles**: victor.salles@beg.utexas.edu
