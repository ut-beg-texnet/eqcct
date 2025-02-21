# EQCCT
**EQCCT**, or the **EarthQuake Compact Convolutional Transformer**, is a highly affective, generalizable, accurate, and production-ready, machine-learning algorithm designed to detect of seismic phases for earthquake detection. 

EQCCT was developed by Saad, O.M., Chen, Y.F., Siervo, D., Zhang, F., Savvaidis, A., Huang, G., Igonin, N., Fomel, S., and Chen, Y., (2023), as a highly accurate and generalizable machine-learning seismic event detection algorithm. More information regarding the purpose, design, functionality, results, and real-world implementation and application can be read about in their research paper [here](https://ieeexplore.ieee.org/document/10264212).

**If you desire to use and pull only EQCCT (EQCCTOne) as originally developed in the 2023 research paper, use the following commands:** 
```sh
[skevofilaxc] mkdir my_work_directory
[skevofilaxc] cd my_work_directory
[skevofilaxc] git clone --depth 1 --filter=tree:0 https://github.com/ut-beg-texnet/eqcct.git --sparse
[skevofilaxc] cd eqcct
[skevofilaxc] git sparse-checkout set eqcctone
```
This should pull only the eqcctone folder.

Further documentation and the originally developed source-code can be found in the `eqcctone` subfolder. 


# EQCCTPro
**EQCCTPro** built upon the accurate seismic event detection achievements of EQCCT, developing a highly efficient detection framework that utilizes EQCCT to process seismic waveforms from a large seismic network in real-time. By utilizing parallelization frameworks like [Ray](https://docs.ray.io/en/latest/index.html) to parallelize the processing of waveforms and identifying the optimal hardware configurations for any given computer that maximized the Ray's simultaneous processing capabilites, EQCCTPro enables users to utilize EQCCT to quickly, accurately, and efficiently process their seismic waveform data in real-time, with capabilities for real-time continous monitoring. 

EQCCTPro was created by Skevofilax, C., Salles, V., Munoz, C., Siervo, D., Saad, O.M., Chen, Y., and Savvaidis, A., (2025), and can be read about here. 

To install `EQCCTPro`, there are two installation approaches: 
1. Install **EQCCTPro** out the box with no sample waveform data to test the application with
2. Install **EQCCTPro** with the sample waveform data as provided from the Github folder

It is **highly** recommended you pull the `EQCCTPro` folder to gain access to the sample waveform data to help you get acquainted with **EQCCTPro** and its capabilites.

If you wish to install **only the EQCCTPro Python package and use it out of the box (method 1)**, run **`pip install eqcctpro`**. **You must have at least Python verison 3.10.14 for the application to run**. 
You can install Python 3.10.14 using either traditional methods or do the following commands: 

```sh
[skevofilaxc] conda create --name yourname python=3.10.14 -y
[skevofilaxc] conda activate yourname 
[skevofilaxc] python3 --version
Python 3.10.14 (it should return)
[skevofilaxc] pip install eqcctpro
```
You will have access to **EQCCTPro** and its capabilities, however, it is **highly** recommended you pull the `EQCCTPro` folder to gain access to the sample waveform data and to get acquainted with **EQCCTPro**. 
You can pull the `EQCCTPro` folder by running the following commands: 

```sh
[skevofilaxc] mkdir my_work_directory
[skevofilaxc] cd my_work_directory
[skevofilaxc] git clone --depth 1 --filter=tree:0 https://github.com/ut-beg-texnet/eqcct.git --sparse
[skevofilaxc] cd eqcct
[skevofilaxc] git sparse-checkout set eqcctpro
```


If you wish to install **EQCCTPro** with the sample waveform data as originally intended, and or are having trouble installing Python 3.10.14, there has been a precreated conda environment under the `EQCCTPro` folder that will install the necessary packages
and dependencies needed for **EQCCTPro** to run (method 2). 
You can pull the `EQCCTPro` folder, create the precreated conda environment, and activate it using the following commands: 

```sh
[skevofilaxc] mkdir my_work_directory
[skevofilaxc] cd my_work_directory
[skevofilaxc] git clone --depth 1 --filter=tree:0 https://github.com/ut-beg-texnet/eqcct.git --sparse
[skevofilaxc] cd eqcct
[skevofilaxc] git sparse-checkout set eqcctpro
[skevofilaxc] conda env create -f environment.yml
[skevofilaxc] conda activate eqcctpro
```

After creating and activating the conda environment, install the **EQCCTPro Python package** using the following command: 
```sh
[skevofilaxc] pip install eqcctpro
```
The pip package will install the remaining packages needed for **EQCCTPro** to work. More information on the package can be found at our PyPi project link [EQCCTPro](https://pypi.org/project/eqcctpro/).

Further documentation and source-code can be found in the `eqcctpro` subfolder.


# Downloading and using EQCCTOne/Pro
If you would like to use specific versions/implementations of EQCCT, read the above instructions on how to pull the specific folder containing the version you want to pull. 

If you want to pull the whole repository, run the following command: 

```sh
git clone https://github.com/ut-beg-texnet/eqcct.git
```

# Contact Information
If you wish to contact the developers of EQCCTOne, please email `chenyk2016@gmail.com`.

If you wish to contact the developers of EQCCTPro, please email `constantinos.skevofilax@austin.utexas.edu` or `victor.salles@beg.utexas.edu`. 