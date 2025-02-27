# EQCCT
**EQCCT**, or the **EarthQuake Compact Convolutional Transformer**, is a highly effective, accurate, generalizable, and production-ready machine-learning algorithm designed to autonomously detect seismic phases for earthquake detection. 
**EQCCT** was developed by Saad, O.M., Chen, Y.F., Siervo, D., Zhang, F., Savvaidis, A., Huang, G., Igonin, N., Fomel, S., and Chen, Y., (2023). More information regarding the purpose, design, functionality, results, and real-world implementation and application of EQCCT can be read about in their research paper [here](https://ieeexplore.ieee.org/document/10264212).

# **Installation Guide**
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

# **Installation Guide**
There are **two installation methods** for EQCCTPro:

1. **Method 1: Install EQCCTPro out of the box** (for experienced users)
2. **Method 2: Install EQCCTPro with sample waveform data** (recommended for first-time users)

It is **highly recommended** that first-time users pull the `EQCCTPro` folder, which includes sample waveform data and code to help get acquainted with **EQCCTPro**.

---

## **Method 1: Install EQCCTPro (No Sample Data)**
This method installs only the EQCCTPro package **without** the sample waveform data.

### **Step 1: Install EQCCTPro**
Run the following command:
```sh
pip install eqcctpro
```

### **Step 2: Ensure Python 3.10.14 is Installed**
EQCCTPro **requires Python 3.10.14 or higher**. If you don’t have it installed, you can create a conda environment with the correct Python version:

```sh
[skevofilaxc] conda create --name yourenvironemntname python=3.10.14 -y
[skevofilaxc] conda activate yourenvironemntname 
[skevofilaxc] python3 --version
```
Expected output:
```
Python 3.10.14
```

Now, reinstall EQCCTPro:
```sh
[skevofilaxc] pip install eqcctpro
```

### **Step 3 (Optional): Pull the EQCCTPro Folder**
Although not required, **it is highly recommended** to pull the `EQCCTPro` folder to gain access to sample waveform data.

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