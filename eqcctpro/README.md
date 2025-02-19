# EQCCTPro: powerful seismic event detection toolkit

EQCCTPro is a high-performace seismic event detection and processing framework that leverages EQCCT to process seismic data efficiently. It enables users to fully leverage the computational ability of their computing resources for maximum performance for simultaneous seismic waveform processing, achieving real-time performance by identifying and utilizing the optimal computational configurations for their hardware. More information about the development, capabilities, and real-world applications about EQCCTPro can be read about in our research publication here.  

## Features
- Supports both CPU and GPU execution
- Configurable parallelism execution for optimized performance
- Includes tools for evaluating system performance for optimal use-case configurations
- Automatic selection of best-use-case configurations
- Efficient handling of large-scale seismic data

## Installation
To install the necessary dependencies, create a conda environment using:

```sh
conda env create -f environment.yml
conda activate eqcctpro
```
After creating and activating the conda environment, install eqcctpro Python package using the following command: 
```sh
pip install eqcctpro
```
More information on the package can be found at our PyPi project link [eqcctpro](https://pypi.org/project/eqcctpro/).


## Usage
There are three main capabilities of EQCCTPro: 
1. Process mSEED data from singular or multiple seismic stations using either CPUs or GPUs 
2. Evaluate your system to identify the optimal paralleization configurations needed to get the minimum runtime performance out of your system
3. Identify and return back the optimal parallelization configurations for both specific and general-use use-cases for both CPU and GPU applications 

These capabilities are achieved by the following functions in order respect to the above descriptions: 
EQCCTMSeedRunner (1), EvaluateSystem (2), OptimalCPUConfigurationFinder (3a), OptimalGPUConfigurationFinder (3b).

### Processing mSEED data using EQCCTPro (EQCCTMSeedRunner) 
To use EQCCTPro to process mSEED from various seismic stations, use the EQCCTMSeedRunner class. 
EQCCTMSeedRunner enables users to process multiple mSEED from a given input directory. The input directory is made up of station directories such as: 

```sh
[skevofilaxc@BEGE-TEXA75553X sample_1_minute_data]$ ls
AT01  CF01  DG05  EF54  EF76   HBVL  MB09  MB21   MID02  ODSA  PB16  PB25  PB35  PB52  PH02  SM03  WB11
BB01  CT02  DG09  EF63  FOAK4  HNDO  MB13  MB25   MID03  PB04  PB17  PB26  PB39  PB54  PL01  SMWD  WB12
BP01  DB02  EF02  EF75  FW13   MB06  MB19  MID01  MO01   PB11  PB18  PB34  PB42  PECS  SM02  WB06
```
Where each subdirectory is named after station code. If you wish to use create your own input directory with custom information, please follow the above naming convention. Otherwise, EQCCTPro will not work. 

Within each subdirectory, such as PB35, it is made up of mseed files. EQCCTPro only needs one pose for the detection to occur, however the more the merrier. 

```sh
[skevofilaxc@BEGE-TEXA75553X PB35]$ ls
TX.PB35.00.HH1__20241215T115800Z__20241215T120100Z.mseed  TX.PB35.00.HHZ__20241215T115800Z__20241215T120100Z.mseed
TX.PB35.00.HH2__20241215T115800Z__20241215T120100Z.mseed
```

After setting up or utilizing the provided sample waveform directory, import EQCCTMseedRunner as show below: 

```python
from eqcctpro import EQCCTMSeedRunner

eqcct_runner = EQCCTMSeedRunner(
    use_gpu=True,
    intra_threads=1,
    inter_threads=1,
    cpu_id_list=[0,1,2,3,4],
    input_dir='/path/to/mseed',
    output_dir='/path/to/outputs',
    log_filepath='/path/to/outputs/eqcctpro.log',
    P_threshold=0.001,
    S_threshold=0.02,
    p_model_filepath='/path/to/model_p.h5',
    s_model_filepath='/path/to/model_s.h5',
    number_of_concurrent_predictions=5,
    best_usecase_config=True,
    csv_dir='/path/to/csv',
    selected_gpus=[0],
    set_vram_mb=24750,
    specific_stations='AT01, BP01, DG05'
)
eqcct_runner.run_eqcctpro()
```

### Evaluating System Performance
To evaluate the system’s GPU performance:

```python
from eqcctpro import EvaluateSystem

eval_gpu = EvaluateSystem(
    mode='gpu',
    intra_threads=1,
    inter_threads=1,
    input_dir='/path/to/mseed',
    output_dir='/path/to/outputs',
    log_filepath='/path/to/outputs/eqcctpro.log',
    csv_dir='/path/to/csv',
    P_threshold=0.001,
    S_threshold=0.02,
    p_model_filepath='/path/to/model_p.h5',
    s_model_filepath='/path/to/model_s.h5',
    stations2use=2,
    cpu_id_list=[0,1],
    set_vram_mb=24750,
    selected_gpus=[0]
)
eval_gpu.evaluate()
```

### Finding Optimal CPU/GPU Configurations
To determine the best CPU or GPU configuration:

```python
from eqcctpro import OptimalCPUConfigurationFinder, OptimalGPUConfigurationFinder

csv_filepath = '/path/to/csv'

cpu_finder = OptimalCPUConfigurationFinder(csv_filepath)
best_cpu_config = cpu_finder.find_best_overall_usecase()
print(best_cpu_config)

optimal_cpu_config = cpu_finder.find_optimal_for(cpu=3, station_count=2)
print(optimal_cpu_config)

gpu_finder = OptimalGPUConfigurationFinder(csv_filepath)
best_gpu_config = gpu_finder.find_best_overall_usecase()
print(best_gpu_config)

optimal_gpu_config = gpu_finder.find_optimal_for(num_cpus=1, gpu_list=[0], station_count=1)
print(optimal_gpu_config)
```

## Configuration
The `environment.yml` file specifies the dependencies required to run EQCCTPro. Ensure you have the correct versions installed by using the provided conda environment setup.

## License
EQCCTPro is provided under an open-source license. See LICENSE for details.

## Contact
For inquiries or issues, please contact constantinos.skevofilax@austin.utexas.edu.

