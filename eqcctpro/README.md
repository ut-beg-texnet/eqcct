# **EQCCTPro: powerful seismic event detection toolkit**

EQCCTPro is a high-performace seismic event detection and processing framework that leverages DL-pickers, like EQCCT, to process seismic data efficiently. It enables users to fully leverage the computational ability of their computing resources for maximum performance for simultaneous seismic waveform processing, achieving real-time performance by identifying and utilizing the optimal computational configurations for their hardware. More information about the development, capabilities, and real-world applications about EQCCTPro can be read about in our research publication here.  

## **Features**
- Supports both CPU and GPU execution
- Configurable parallelism execution for optimized performance
- Includes tools for evaluating system performance for optimal usecase configurations
- Automatic selection of best-usecase configurations
- Efficient handling of large-scale seismic data

# **Installation Guide**
There are **two installation methods** for EQCCTPro:

1. **Method 1: Install EQCCTPro out of the box** (for experienced users)
2. **Method 2: Install EQCCTPro with sample waveform data** (recommended for first-time users)

It is **highly recommended** that first-time users pull the `EQCCTPro` folder, which includes sample waveform data and code to help get acquainted with **EQCCTPro**.

---

## **Method 1: Install EQCCTPro (No Sample Data)**
This method installs only the EQCCTPro package **without** the sample waveform data.

### **Step 1: Create a Clean Conda Environment for the Install**
EQCCTPro **requires Python 3.10.14 or higher as well as minimum Tensorflow packages**. If you have a clean working environment, you can simply run `pip install eqcctpro`. However, if you have a nonclean environment, its highly recommended to create a new conda environment so that you can install the necessary packages safely with no issues. You can create a new conda environment with the correct Python version by using the following commands:

```sh
[skevofilaxc] conda create --name yourenvironemntname python=3.10.14 -y
[skevofilaxc] conda activate yourenvironemntname 
[skevofilaxc] python3 --version
```
Expected output:
```
Python 3.10.14
```

After activating your new conda environment, run the following command:  
```sh
[skevofilaxc] pip install eqcctpro
```
You will have access to EQCCTPro and its functionality. However you will not have immediate access to the provided sample waveform data to use for testing. Youcan pull the waveform data either by downloading the .zip file from the repository or by following step 3. 

### **Step 3 (Optional): Pull the EQCCTPro Folder**
Although not required, **it is highly recommended** to pull the `EQCCTPro` folder to gain access to sample waveform data for testing.

```sh
[skevofilaxc] mkdir my_work_directory
[skevofilaxc] cd my_work_directory
[skevofilaxc] git clone --depth 1 --filter=tree:0 https://github.com/ut-beg-texnet/eqcct.git --sparse
[skevofilaxc] cd eqcct
[skevofilaxc] git sparse-checkout set eqcctpro
```

---

## **Method 2: Install EQCCTPro with Sample Data (Recommended for First-Time Users)**
This method sets up EQCCTPro **with a pre-created conda environment and sample waveform data**.

### **Step 1: Clone the EQCCTPro Repository**
```sh
[skevofilaxc] mkdir my_work_directory
[skevofilaxc] cd my_work_directory
[skevofilaxc] git clone --depth 1 --filter=tree:0 https://github.com/ut-beg-texnet/eqcct.git --sparse
[skevofilaxc] cd eqcct
[skevofilaxc] git sparse-checkout set eqcctpro
```

### **Step 2: Create and Activate the Conda Environment**
A **pre-configured conda environment** is included in the repository to handle all dependencies.

```sh
[skevofilaxc] conda env create -f environment.yml
[skevofilaxc] conda activate eqcctpro
```

### **Step 3: Install EQCCTPro**
After activating the environment, install the EQCCTPro package:
```sh
[skevofilaxc] pip install eqcctpro
```

This will install any remaining dependencies needed for **EQCCTPro**.

---

## **More Information**
For additional details and package updates, visit the **EQCCTPro PyPI page**:  
🔗 [EQCCTPro on PyPI](https://pypi.org/project/eqcctpro/)

---

### **Using Sample Waveform Data**
To understand how **EQCCTPro** works, it is **highly recommended** to use provided sample seismic waveform data as the data source when testing the package. 

Sample seismic waveform data from 50 TexNet stations have provided in the repository under `sample_1_minute_data.zip`. 

### **Step 1: Unzip the Sample Wavefrom Data**
After downloading the `.zip` file through the GitHub methods above, run:
```sh
[skevofilaxc] unzip sample_1_minute_data.zip
```
### **Step 2: Check and Understand the Directory Structure**
The extracted data will contain multiple station directories:
```sh
[skevofilaxc sample_1_minute_data]$ ls
AT01  CF01  DG05  EF54  EF76   HBVL  MB09  MB21   MID02  ODSA  PB16  PB25  PB35  PB52  PH02  SM03  WB11
BB01  CT02  DG09  EF63  FOAK4  HNDO  MB13  MB25   MID03  PB04  PB17  PB26  PB39  PB54  PL01  SMWD  WB12
BP01  DB02  EF02  EF75  FW13   MB06  MB19  MID01  MO01   PB11  PB18  PB34  PB42  PECS  SM02  WB06
```
Each subdirectory contains **mSEED** files of different waveform components:
```sh
[skevofilaxc PB35]$ ls
TX.PB35.00.HH1__20241215T115800Z__20241215T120100Z.mseed  TX.PB35.00.HHZ__20241215T115800Z__20241215T120100Z.mseed
TX.PB35.00.HH2__20241215T115800Z__20241215T120100Z.mseed
```
EQCCT (i.e., the ML model) requires at least one pose per station for detection, but using multiple poses enhances P and S wave directionality.

You have successfully installed EQCCTPro and set up the required sample waveform dataset for testing.

---
### **Using EQCCTPro**
There are three main capabilities of EQCCTPro: 
1. **Process mSEED data from singular or multiple seismic stations using either CPUs or GPUs** 
2. **Evaluate your system to identify the optimal parallelization configurations needed to get the minimum runtime performance out of your system**
3. **Identify and return back the optimal parallelization configurations for both specific and general-use usecases for both CPU (a) and GPU applications (b)**


These capabilities are achieved using the following core functions:

- **EQCCTMSeedRunner** (for processing mSEED data)

- **EvaluateSystem** (for system evaluation)

- **OptimalCPUConfigurationFinder** (for CPU configuration optimization)

- **OptimalGPUConfigurationFinder** (for GPU configuration optimization)

---
### **Processing mSEED data using EQCCTPro (EQCCTMSeedRunner)** 
To process mSEED from various seismic stations, use the **EQCCTMSeedRunner** class. 
**EQCCTMSeedRunner** enables users to process multiple mSEED from a given input directory, which consists of station directories formatted as follows:

```sh
[skevofilaxc sample_1_minute_data]$ ls
AT01  CF01  DG05  EF54  EF76   HBVL  MB09  MB21   MID02  ODSA  PB16  PB25  PB35  PB52  PH02  SM03  WB11
BB01  CT02  DG09  EF63  FOAK4  HNDO  MB13  MB25   MID03  PB04  PB17  PB26  PB39  PB54  PL01  SMWD  WB12
BP01  DB02  EF02  EF75  FW13   MB06  MB19  MID01  MO01   PB11  PB18  PB34  PB42  PECS  SM02  WB06
```
Where each subdirectory is named after station code. If you wish to use create your own input directory with custom waveform mSEED files, **please follow the above naming convention.** Otherwise, EQCCTPro will **not** work. 

Within each subdirectory, such as PB35, it is made up of mSEED files of different poses (EX. N, E, Z): 
```sh
[skevofilaxc PB35]$ ls
TX.PB35.00.HH1__20241215T115800Z__20241215T120100Z.mseed  TX.PB35.00.HHZ__20241215T115800Z__20241215T120100Z.mseed
TX.PB35.00.HH2__20241215T115800Z__20241215T120100Z.mseed
```
EQCCT only needs one pose for the detection to occur, however more poses allow for better detection of the direction of the P and S waves.

After setting up or utilizing the provided sample waveform directory, and install eqcctpro, import **EQCCTMseedRunner** as show below: 

```python
from eqcctpro import EQCCTMSeedRunner

eqcct_runner = EQCCTMSeedRunner(
    use_gpu=False,
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

**EQCCTMseedRunner** has multiple input paramters that need to be configured and are defined below: 

- **`use_gpu (bool)`: True or False** 
  - Tells Ray to use either the GPU(s) (True) or CPUs (False) on your computer to process the waveforms in the entire workflow
  - Further specification of which GPU(s) and CPU(s) are provided in the parameters below 
- **`intra_threads (int)`: default = 1**
  - Controls how many intra-parallelism threads Tensorflow can use 
- **`inter_threads (int)`: default = 1**
  - Controls how many inter-parallelism threads Tensorflow can use
- **`cpu_id_list (list)`: default = [1]**
  - List that defines which specific CPU cores that sched_setaffinity will allocate for executing the current EQCCTPro process.
  - Allows for specific allocation and limitation of CPUs for a given EQCCTPro process 
    - "I want this program to run only on these specific cores." 
- **`input_dir (str)`**
  - Directory path to the the mSEED directory
  - EX. `/home/skevofilaxc/my_work_directory/eqcct/eqcctpro/sample_1_minute_data`
- **`output_dir (str)`**
  - Directory path to where the output picks and logs will be sent 
  - Doesn't need to exist, will be created if doesn't exist 
  - Recommended to be in the same working directory as the input directory for convience
- **`log_filepath (str)`**
  - Filepath to where the EQCCTPro log will be written to and stored
  - Doesn't need to exist, will be created if doesn't exist
  - Recommended to be **in** the **output directory** and called **eqcctpro.log**, however the name can be changed for your own purposes 
- **`P_threshold (float)`: default = 0.001**
  - Threshold in which the P probabilities above it will be considered as P arrival
- **`S_threshold (float)`: default = 0.02**
  - Threshold in which the S probabilities above it will be considered as S arrival
- **`p_model_filepath (str)`**
  - Filepath to where the P EQCCT detection model is stored
- **`s_model_filepath (str)`**
  - Filepath to where the S EQCCT detection model is stored
- **`number_of_concurrent_predictions (int)`**
  - The number of concurrent EQCCT detection tasks that can happen simultaneously on a given number of resources
  - EX. if number_of_concurrent_predictions = 5, there will be up to 5 EQCCT instances analyzing 5 different waveforms at the sametime
  - Best to use the optimal amount for your hardware, which can be identified using **EvaluateSystem** (below)
- **`best_usecase_config (bool)`: default = False**
  - If True, will override inputted cpu_id_list, number_of_concurrent_predictions, intra_threads, inter_threads values for the best overall usecase configurations 
  - Best overall usecase configurations are defined as the best overall input configurations that minimize runtime while doing the most amount of processing with your available hardware 
  - Can only be used if EvaluateSystem has been run 
- **`csv_dir (str)`**
  - Directory path containing the CSV's outputted by EvaluateSystem that contain the trial data that will be used to find the best_usecase_config
  - Script will look for specific files, will only exist if EvaluateSystem has been run 
- **`selected_gpus (list)`: default = None**
  - List of GPU IDs on your computer you want to use if `use_gpu = True`
  - None existing GPU IDs will cause the code to exit 
- **`set_vram_mb (float)`**
  - Value of the maximum amount of VRAM EQCCTPro can use 
  - Must be a real value that is based on your hardware's physical memory space, if it exceeds the space the code will break due to **OutOfMemoryError**
- **`specific_stations (str)`: default = None**
  - String that contains the "list" of stations you want to only analyze 
  - EX. Out of the 50 sample stations in `sample_1_minute_data`, if I only want to analyze AT01, BP01, DG05, then specific_stations='AT01, BP01, DG05'. 
  - Removes the need to move station directories around to be used as input, can contain all stations in one directory for access
- **`cpu_id_list (list)`: default = [1]**
  - List that defines which specific CPU cores that sched_setaffinity will allocate for executing the current EQCCTPro process.
  - Allows for specific allocation and limitation of CPUs for a given EQCCTPro process 
    - "I want this program to run only on these specific cores." 

---

### **Evaluating Your Systems Runtime Performance Capabilites**
To evaluate your system’s runtime performance capabilites for both your CPU(s) and GPU(s), the **EvaluateSystem** class allows you to autonomously evaluate your system:

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

```python
from eqcctpro import EvaluateSystem

eval_cpu = EvaluateSystem(
                mode='cpu',
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
                stations2use=12,
                cpu_id_list=range(87,102), 
                starting_amount_of_stations=2, 
                station_list_step_size=1,
                min_cpu_amount=15,
                min_conc_predictions=2,
                conc_predictions_step_size=1)
eval_cpu.evaluate()
```

**EvaluateSystem** will iterate through different combinations of CPU(s), Concurrent Predictions, and Workloads (stations), as well as GPU(s), and the amount of VRAM (MB) each Concurrent Prediction can use. 
**EvaluateSystem** will take time, depending on the number of CPU/GPUs, the amount of VRAM available, and the total workload that needs to be tested. However, after doing the testing once for most if not all usecases, 
the trial data will be available and can be used to identify the optimal input parallelization configurations for **EQCCTMSeedRunner** to use to get the maximum amount of processing out of your system in the shortest amonut of time. 

The following input parameters need to be configurated for **EvaluateSystem** to evaluate your system based on your desired utilization of EQCCTPro: 

- **`mode (str)`**
  - Can be either `cpu` or `gpu`
  - Tells `EvaluateSystem` which configuration trials should it iterate through
- **`intra_threads (int)`: default = 1**
  - Controls how many intra-parallelism threads Tensorflow can use 
- **`inter_threads (int)`: default = 1**
  - Controls how many inter-parallelism threads Tensorflow can use 
- **`input_dir (str)`**
  - Directory path to the the mSEED directory
  - EX. /home/skevofilaxc/my_work_directory/eqcct/eqcctpro/sample_1_minute_data
- **`output_dir (str)`**
  - Directory path to where the output picks and logs will be sent 
  - Doesn't need to exist, will be created if doesn't exist 
  - Recommended to be in the same working directory as the input directory for convience
- **`log_filepath (str)`**
  - Filepath to where the EQCCTPro log will be written to and stored
  - Doesn't need to exist, will be created if doesn't exist
  - Recommended to be **in** the **output directory** and called **eqcctpro.log**, however the name can be changed for your own purposes 
- **`csv_dir (str)`**
  - Directory path where the CSV's outputted by EvaluateSystem will be saved 
  - Doesn't need to exist, will be created if doesn't exist
- **`P_threshold (float)`: default = 0.001**
  - Threshold in which the P probabilities above it will be considered as P arrival
- **`S_threshold (float)`: default = 0.02**
  - Threshold in which the S probabilities above it will be considered as S arrival
- **`p_model_filepath (str)`**
  - Filepath to where the P EQCCT detection model is stored
- **`s_model_filepath (str)`**
  - Filepath to where the S EQCCT detection model is stored
- **`stations2use (int)`: default = None**
  - Controls the maximum amount of stations EvaluateSystem can use in its trial iterations 
  - Sample data has been provided so that the maximum is 50, however, if using custom data, configure for your specific usecase 
- **`cpu_id_list (list)`: default = [1]**
  - List that defines which specific CPU cores that sched_setaffinity will allocate for executing the current EQCCTPro process and **is the maximum amount of cores EvaluteSystem can use in its trial iterations**
  - Allows for specific allocation and limitation of CPUs for a given EQCCTPro process 
    - "I want this program to run only on these specific cores." 
  - Must be at least 1 CPU if using GPUs (Ray needs CPUs to manage the Raylets (concurrent tasks), however the processing of the waveform is done on the GPU)
- **`starting_amount_of_stations (int)`: default = 1** 
  - For evaluating your system, you have the option to set a starting amount of stations you want to use in the test
  - By default, the test will start using 1 station but now is configurable 
- **`station_list_step_size (int)`: default = 1** 
  - You can set a step size for the station list that is generated 
  - For example if the stepsize is set to 10 and you start with 50 stations with a max of 100, then your list would be: [50, 60, 70, 80, 80, 100]
  - Using 1 will use the default step size of 1-10, then step size of 5 up to station2use
- **`min_cpu_amount (int)`: default = 1**
  - Is the minimum amount of CPUs you want to start your trials with 
  - By default, trials will start iterating with 1 CPU up to the maximum allocated 
  - Can now set a value as the starting point, such as 15 CPUs up to the maximum of for instance 25
- **`min_conc_predictions (int)`: default = 1** 
  - Is the minimum amount of concurrent predictions you want each trial iteration to start with 
  - By default, if `min_conc_predictions` and `conc_predictions_step_size` are set to 1, a custom step size iteration will be applied to test the 50 sample waveforms. The sequence follows: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, n+5, 50].
- **`conc_predictions_step_size (int)`: default = 1** 
  - Is the concurrent predictions step size you want each trial iteration to iterate with with 
  - By default, if `min_conc_predictions` and `conc_predictions_step_size` are set to 1, a custom step size iteration will be applied to test the 50 sample waveforms. The sequence follows: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, n+5, 50]
- **`set_vram_mb (float)`**
  - Value of the maximum amount of VRAM EQCCTPro can use 
  - Must be a real value that is based on your hardware's physical memory space, if it exceeds the space the code will break due to OutOfMemoryError 
- **`selected_gpus (list)`: default = None**
  - List of GPU IDs on your computer you want to use if `mode = 'gpu'`
  - Non-existing GPU IDs will cause the code to exit 


---
### **Finding Optimal CPU/GPU Configurations**
After running **EvalutateSystem**, you can use either the **OptimalCPUConfigurationFinder** or the **OptimalGPUConfigurationFinder** determine the best CPU or GPU configurations (respectively) for your specific usecase:

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
Both **OptimalCPUConfigurationFinder** and **OptimalGPUConfigurationFinder** each have two usecases: 

1. **`find_best_overall_usecase`**
  - Returns the best overall usecase configuration 
    - Uses middle 50% of CPUs for moderate, balanced CPU usage, with the maximum amount of stations processed with the minimum runtime 
2. **`find_optimal_for`**
  - Return the paralleliztion configurations (EX. concurrent predictions, intra/inter thread counts, vram, etc.) for a given number of CPU(s)/GPU(s) and stations
    - Enables users to quickly identify which input parameters should be used for the given amount of resources and workload they have for the minimum runtime possible on their computer

A input CSV directory path must be passed for the classes to use as a reference point: 
- **`csv_filepath (str)`**
  - Directory path where the CSV's outputted by EvaluateSystem are

Using **OptimalCPUConfigurationFinder.find_best_overall_usecase()**, no input parameters are needed. It will return back the best usecase parameters. 

For **OptimalCPUConfigurationFinder.find_optimal_for()**, the function requires two input parameters: 
- **`cpu (int)`**
  - The number of CPU(s) you want to use in your application
- **`station_count (int)`**
  - The number of station(s) you want to use in your application

**OptimalCPUConfigurationFinder.find_optimal_for()** will return back a trial data point containing the mimimum runtime based on your input paramters 

Similar to **OptimalCPUConfigurationFinder.find_best_overall_usecase()**, **OptimalGPUConfigurationFinder.find_best_overall_usecase()** will return back the best usecase parameters and no input parameters are needed. 

For **OptimalGPUConfigurationFinder.find_optimal_for()**, the function requires three input parameters: 
- **`cpu (int)`**
  - The number of CPU(s) you want to use in your application
- **`gpu_list (list)`**
  - The specific GPU ID(s) you want to use in your application
  - Useful if you have multiple GPUs available and want to use/dedicate a specific one to using EQCCTPro
- **`station_count (int)`**
  - The number of station(s) you want to use in your application

## **Configuration**
The `environment.yml` file specifies the dependencies required to run EQCCTPro. Ensure you have the correct versions installed by using the provided conda environment setup.

## **License**
EQCCTPro is provided under an open-source license. See LICENSE for details.

## **Contact**
For inquiries or issues, please contact constantinos.skevofilax@austin.utexas.edu or victor.salles@beg.utexas.edu.

