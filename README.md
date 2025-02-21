# EQCCT
**EQCCT**, or the **EarthQuake Compact Convolutional Transformer**, is a highly effective, accurate, generalizable, and production-ready machine-learning algorithm designed to autonomously detect seismic phases for earthquake detection. 
**EQCCT** was developed by Saad, O.M., Chen, Y.F., Siervo, D., Zhang, F., Savvaidis, A., Huang, G., Igonin, N., Fomel, S., and Chen, Y., (2023). More information regarding the purpose, design, functionality, results, and real-world implementation and application of EQCCT can be read about in their research paper [here](https://ieeexplore.ieee.org/document/10264212).

**If you desire to use and pull only EQCCT (EQCCTOne) as originally developed in the 2023 research paper, use the following commands:** 
```sh
[skevofilaxc] mkdir my_work_directory
[skevofilaxc] cd my_work_directory
[skevofilaxc] git clone --depth 1 --filter=tree:0 https://github.com/ut-beg-texnet/eqcct.git --sparse
[skevofilaxc] cd eqcct
[skevofilaxc] git sparse-checkout set eqcctone
```
This should pull only the `eqcctone` folder, which contains all the source and demo code as designed by Yangkang Chen.
Further documentation can be found in the `eqcctone` subfolder. 


# EQCCTPro
**EQCCTPro** builds on **EQCCT's** accurate seismic event detection, creating a highly efficient framework for real-time waveform processing across large seismic networks. By leveraging parallelization frameworks like [Ray](https://docs.ray.io/en/latest/index.html) and optimizing hardware configurations, **EQCCTPro** enables fast, accurate, and efficient seismic data processing with continuous real-time monitoring capabilities.

**EQCCTPro** was created by Skevofilax, C., Salles, V., Munoz, C., Siervo, D., Saad, O.M., Chen, Y., and Savvaidis, A., (2025), and can be read about here. 

To install `EQCCTPro`, there are two installation approaches: 
1. Install **EQCCTPro** out the box with no sample waveform data to test the application with (experts)
2. Install **EQCCTPro** with the sample waveform data as provided from the Github folder (first-time users)

It is **highly** recommended you pull the `EQCCTPro` folder to gain access to the sample waveform data and code to help you get acquainted with **EQCCTPro** and its capabilites.

However, if you wish to install **only the EQCCTPro Python package and use it out of the box** (method 1), run:
```sh
**`pip install eqcctpro`**
```
**You must have at least Python verison 3.10.14 for the application to run**. 

You can install Python 3.10.14 using either traditional methods or do the following commands: 
```sh
[skevofilaxc] conda create --name yourenvironemntname python=3.10.14 -y
[skevofilaxc] conda activate yourenvironemntname 
[skevofilaxc] python3 --version
Python 3.10.14 (it should return)
[skevofilaxc] pip install eqcctpro
```
You will have access to **EQCCTPro** and its capabilities, however, it is **highly** recommended you pull the `EQCCTPro` folder to gain access to the sample waveform data to help you get acquainted with **EQCCTPro's** functionality. 
You can pull the `EQCCTPro` folder by running the following commands: 

```sh
[skevofilaxc] mkdir my_work_directory
[skevofilaxc] cd my_work_directory
[skevofilaxc] git clone --depth 1 --filter=tree:0 https://github.com/ut-beg-texnet/eqcct.git --sparse
[skevofilaxc] cd eqcct
[skevofilaxc] git sparse-checkout set eqcctpro
```

If you wish to install **EQCCTPro** with the sample waveform data as **originally intended for first-time users**, and or are having trouble installing Python 3.10.14, there has been a precreated conda environment under the `EQCCTPro` folder that will install the necessary packages
and dependencies needed for **EQCCTPro** to run (method 2). 

You can pull the `EQCCTPro` folder, create the precreated conda environment, and activate it as originally intended for first-time users using the following commands: 
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
The pip package will install the remaining packages needed for **EQCCTPro** to work. More information on the eqcctpro pip package can be found at our PyPi project link here [(EQCCTPro)](https://pypi.org/project/eqcctpro/).

Further documentation and source-code can be found in the `eqcctpro` subfolder.


# Downloading and using EQCCTOne/Pro
If you would like to use specific versions/implementations of EQCCT, read the above instructions on how to pull the specific folder containing the version you want to pull. 

If you want to pull the whole repository, run the following command: 

```sh
git clone https://github.com/ut-beg-texnet/eqcct.git
```

# Contact Information
If you wish to contact the developers of EQCCTOne, please email `yangkang.chen@beg.utexas.edu`.

If you wish to contact the developers of EQCCTPro, please email `constantinos.skevofilax@austin.utexas.edu` or `victor.salles@beg.utexas.edu`. 