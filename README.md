# EQCCT (EQCCTOne)
EQCCT, or the EarthQuake Compact Convolutional Transformer, is a highly affective, generalizable, accurate, and production-ready, machine-learning algorithm designed to detect of seismic phases for earthquake detection. 

EQCCT was developed by Saad, O.M., Chen, Y.F., Siervo, D., Zhang, F., Savvaidis, A., Huang, G., Igonin, N., Fomel, S., and Chen, Y., (2023), as a highly accurate and generalizable machine-learning seismic event detection algorithm. More information regarding the purpose, design, functionality, results, and real-world implementation and application can be read about in their research paper [here](https://ieeexplore.ieee.org/document/10264212).

EQCCT can be used out-of-the-box as developed originally in the `eqcctone` subdirectory. 

# EQCCTPro
EQCCTPro built upon the accurate seismic event detection achievements of EQCCT, developing a highly efficient detection framework that utilizes EQCCT to process seismic waveforms from a large seismic network in real-time. By utilizing parallelization frameworks like [Ray](https://docs.ray.io/en/latest/index.html) to parallelize the processing of waveforms and identifying the optimal hardware configurations for any given computer that maximized the Ray's simultaneous processing capabilites, EQCCTPro enables users to utilize EQCCT to quickly, accurately, and efficiently process their seismic waveform data in real-time, with capabilities for real-time continous monitoring. 

EQCCTPro was created by Skevofilax, C., Salles, V., Munoz, C., Siervo, D., Saad, O.M., Chen, Y., and Savvaidis, A., (2025), and can be read about here. 

EQCCTPro can be used-out-of-the box by using: `pip install eqcctpro`
Further documentation and source-code can be found in the `eqcctpro` subdirectory.