# EQCCT
**EQCCT**, or the **EarthQuake Compact Convolutional Transformer**, is a highly affective, generalizable, accurate, and production-ready, machine-learning algorithm designed to detect of seismic phases for earthquake detection. 

EQCCT was developed by Saad, O.M., Chen, Y.F., Siervo, D., Zhang, F., Savvaidis, A., Huang, G., Igonin, N., Fomel, S., and Chen, Y., (2023), as a highly accurate and generalizable machine-learning seismic event detection algorithm. More information regarding the purpose, design, functionality, results, and real-world implementation and application can be read about in their research paper [here](https://ieeexplore.ieee.org/document/10264212).

EQCCT (EQCCTOne), can be used out-of-the-box as developed originally in the `eqcctone` subdirectory. 

If you desire to use and pull only EQCCTOne, use the following command: 
```sh
git clone --depth 1 --filter=tree:0 https://github.com/ut-beg-texnet/eqcct.git --sparse
git sparse-checkout set eqcctone
```

# EQCCTPro
**EQCCTPro** built upon the accurate seismic event detection achievements of EQCCT, developing a highly efficient detection framework that utilizes EQCCT to process seismic waveforms from a large seismic network in real-time. By utilizing parallelization frameworks like [Ray](https://docs.ray.io/en/latest/index.html) to parallelize the processing of waveforms and identifying the optimal hardware configurations for any given computer that maximized the Ray's simultaneous processing capabilites, EQCCTPro enables users to utilize EQCCT to quickly, accurately, and efficiently process their seismic waveform data in real-time, with capabilities for real-time continous monitoring. 

EQCCTPro was created by Skevofilax, C., Salles, V., Munoz, C., Siervo, D., Saad, O.M., Chen, Y., and Savvaidis, A., (2025), and can be read about here. 

EQCCTPro can be used-out-of-the box by using: 
`pip install eqcctpro`
Further documentation and source-code can be found in the `eqcctpro` subdirectory.

If you desire to use and pull only EQCCTPro, use the following command: 
```sh
git clone --depth 1 --filter=tree:0 https://github.com/ut-beg-texnet/eqcct.git --sparse
git sparse-checkout set eqcctpro
```

# Downloading and using EQCCTOne/Pro
If you would like to use specific versions/implementations of EQCCT, read the above instructions on how to pull the specific folder containing the version you want to pull. 

If you want to pull the whole repository, run the following command: 

```sh
git clone https://github.com/ut-beg-texnet/eqcct.git
```

# Contact Information
If you wish to contact the developers of EQCCTOne, please email `chenyk2016@gmail.com`.

If you wish to contact the developers of EQCCTPro, please email `constantinos.skevofilax@austin.utexas.edu`. 