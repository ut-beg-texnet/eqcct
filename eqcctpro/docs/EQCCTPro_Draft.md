# RAPID: A Generalized High-Performance Parallelization Framework for Real-Time Deep Learning Seismic Phase Picking

## Abstract

Deep learning has transformed seismic phase picking, yet deploying these models at network scale in real time remains an unresolved engineering challenge. While unified libraries such as SeisBench excel at high-throughput offline batch inference, these methods are ill-suited for continuous monitoring, where waveforms arrive asynchronously from individual stations and cannot be held in memory until a full batch is assembled. The naive alternative, dispatching each station as an independent parallel task, forces repeated framework initialization for every waveform, inflating wall times and consuming much of the available compute time budget unless concurrency and hardware are tuned carefully. To resolve this, we introduce **RAPID (Resource-Aware Parallel Inference Dispatcher)**, a generalized orchestration framework that implements two task parallelization strategies: **Ripper**, an ephemeral task-based approach, and **Model-Actor**, which keeps persistent inference instances loaded in memory across an entire processing window. We benchmarked RAPID across five deep learning phase pickers (PhaseNet, PhaseNetLight, EQTransformer, EQTransformer-NonConservative, and EQCCT) using 228 three-component 60-second waveforms from the Texas Seismological Network under various hardware constraints that reflect real operational deployment scenarios. Across all configurations, the Model-Actor strategy achived real-time processing, reducing processing time by roughly **50–82%** relative to the best Ripper totals at 228 stations for the same model–hardware pairings. These results show that the dominant bottleneck in network-scale DL picking is not raw inference complexity but orchestration overhead, and that persistent-actor lifecycle management delivers substantially faster totals and larger headroom under the 30-second target than even the best tuned ephemeral-task baseline.

## 1. Introduction

Accurate and timely identification of seismic phase arrivals is fundamental to earthquake monitoring. For decades, seismic networks have relied on energy-ratio algorithms, primarily the Short-Term Average/Long-Term Average (STA/LTA), to automate event detection. While computationally inexpensive, these algorithms are sensitive to background noise, requiring station-specific tuning, and often leading to high rates of false-positives in complex noise environments.

Over the past decade, deep learning (DL) has emerged as a powerful alternative. Models such as PhaseNet (Zhu & Beroza, 2018), EQTransformer (Mousavi et al., 2020), and EQCCT (Saad et al., 2023) consistently outperform traditional picking methods in both precision and recall, particularly in low signal-to-noise conditions. SeisBench (Woollam et al., 2022) has consolidated many of these models into a unified interface along with several datasets (Münchmeyer et al., 2022), establishing itself as the standard library for DL-based phase-picking research.

This progress has naturally led researchers to incorporate these pickers into production seismic workflows with real-time operations in mind. We define *real-time processing* as returning picks for a full station selection within 30 seconds of a 60-second processing window. Chen, Savvaidis, et al. (2024) describe a near real-time workflow with EQCCT in SeisComP as their main picking algorithm, together with association, relocation, and catalog quality control; their evaluation foregrounds catalog quality and how the workflow positively affects analysts, rather than strictly bounding end-to-end picking time over the whole network. Waveforms arrive through FDSN archive access and can lag live streams by the order of ten minutes, which is reasonable for post-processing but is not the same as picking traces as they arrive in time. Yeck et al. (2020) sketch a related deployment at NEIC with compact CNNs that refine STA/LTA detections before association. The timing metrics they report are from replayed automatic feeds, so they are not a live wall-clock measure of full-window deep learning across the entire network. Sheen et al. (2023) replace classical single-channel Earthworm pickers with a module that walks 30 s windows every second on WAVE_RING using ordinary Earthworm utilities—fast for that ring-based path, though the paper still centers on lightweight classical picking rather than coordinating regional deep-learning models in bulk.

None of these quite address our core question: how to run today's most widely used models under strict CPU, GPU, memory, and timing constraints, rather than settling for near real time through slower ingestion, replayed feeds, or simplified stacks. We took that question up by starting from SeisBench, where those models already live alongside training weights and standard inference entry points, and ask what must be added once station count and the 30-second window are both binded by hardware constraints.

Currently, SeisBench supports two primary waveform processing modes. The first, *offline batch mode*, processes waveforms in a single `annotate()` call. This method uses internal batching to achieve sub-second runtimes for entire networks (0.22–1.22 s for 228 stations). While highly effective for post-event analysis, it is incompatible with real-time monitoring. Operationally, waveforms arrive asynchronously from individual stations due to factors such as network latency or power failures. As a result, incoming waveforms cannot be held in memory until a full batch is assembled because the processing window will continue to advance forward in time.

The second mode, *per-station streaming*, uses the `classify()` method to process one waveform at a time. This method windows the incoming trace, runs a forward pass, and extracts phase arrivals. Since the model remains in memory between calls, no per-station reinitialization cost is paid. For the four SeisBench-integrated architectures tested, sequential processing of 228 stations requires 1.43–33.70 s. While these times might seem sufficient, `classify()` has four structural limitations:

(1) **Hardware non-scalability**. Because `classify()` processes stations sequentially, adding more CPU cores or GPUs does not improve throughput. Runtime remains limited by the performance of a single model instance.

(2) **Operational scale failure**. Scaling to even larger network sizes, PhaseNet running on a single CPU reaches 35.85 s for 250 stations (TexNet) and 80.23 s for 580 stations (NCSN), already exceeding real-time deadlines.

(3) **Shared inference budget**. The 30-second real-time compute window must also accommodate data quality checks, phase association, and alert dispatch. A 2-second margin is insufficient for operational stability.

(4) **Registry limitations**. Models not yet integrated into SeisBench, such as EQCCT or custom institutional pickers, lack a `classify()` path, forcing networks to find other multi-waveform processing alternatives.

While parallelization is the intuitive remedy for these sequential processing constraints, treating each station as a separate parallel task forces the machine learning framework to re-initialize for every station. On our hardware that cost dominates runtimes at 228 stations, as will be discussed later, leaving little margin for association, quality control, and dispatch.

To address these issues, we introduce RAPID (Resource-Aware Parallel Inference Dispatcher), a generalized, resource-aware parallelization framework for seismic phase picking. RAPID is designed to complement SeisBench by using its native `from_pretrained()`, `annotate()`, and `classify()` interfaces while providing the necessary orchestration for real-time operations. We implement two parallelization strategies: Model-Actor (persistent instances) and Ripper (ephemeral tasks). We benchmarked these across five pickers and two hardware configurations. The Model-Actor method reduced total runtimes by roughly 50–82% compared with the best Ripper totals at 228 stations for the same model–device pairs, demonstrating that persistent-actor orchestration yields faster end-to-end times and more comfortable real-time margin than tuned ephemeral-task Ripper on real-world network workloads.

## 2. Methodology

The methodology is organized as follows: dataset and workload simulation (§2.1), model selection (§2.2), hardware environment and resource control (§2.3), orchestration strategies (§2.4), performance metrics (§2.5), and study protocol (§2.6).

### 2.1 Dataset and Workload Simulation

To evaluate our proposed parallelization strategies, we used data from the Texas Seismological Network (TexNet) to simulate network-scale processing. From TexNet's 250 stations, we retrieved 228 unique three-component (3-C) 60-second waveforms, sampled at 100 Hz, for an M4.29 event that occurred on 26 January 2026 in West Texas. The 60-second window was chosen as it is the minimum input requirement for EQCCT (Saad et al., 2023) as well as matches TexNet's standard operational interval. Although models such as PhaseNet can accept shorter window durations, we used a uniform one-minute input across all models to ensure a fair comparison of orchestration performance under identical workload conditions.

Waveforms were pre-downloaded as miniSEED files to exclude network latency from the benchmark. During execution, waveforms are converted to NumPy arrays and stored in memory-resident Python dictionaries. This design isolates inference and orchestration costs from disk I/O. A 1-45 Hz bandpass filter was applied to all data during inference as a standard preprocessing step for high-frequency phase identification.

### 2.2 Model Selection

To demonstrate that the primary performace bottleneck is overhead rather than model architecture, we benchmarked the strategies across five architectures:

1. **PhaseNet** (U-Net; Zhu & Beroza, 2018),
2. **PhaseNetLight** (Optimized PhaseNet; Woollam et al., 2022)
3. **EQTransformer** (Attention-based; Mousavi et al., 2020)
4. **EQTransformer-NonConservative** (High-sensitivity configuration; Woollam et al., 2022)
5. **EQCCT** (Transformer-based; Saad et al., 2023).

These models utilize two machine learning frameworks: PyTorch (PhaseNet and EQTransformer variants) and TensorFlow (EQCCT). This selection of models covers a range of architecture designers: from lightweight U-Nets to computationally intensive transformers, allowing us to evaluate how framework choice and model complexity affect orchestration overhead.

### 2.3 Hardware Environment and Resource Control

Experiments were conducted on a workstation equipped with dual AMD Ryzen Threadripper PRO 7985WX CPUs (128 total cores) and 512 GB of DDR5 RAM. GPU trials utilized two NVIDIA RTX 6000 Ada Generation GPUs, each with 49 GB of VRAM.

To simulate the resource-constrained environments of operational seismic networks, we implemented strict hardware isolation. Using the Linux `sched_setaffinity` utility, we bound each trial to a specific set of CPU cores and GPUs. CPU allocations were varied from 5 to 20 cores in steps of 3 (5, 8, 11, 14, 17, 20). For GPU trials, the same CPU allocation schedule was applied while inference was restricted to one or two GPUs. These constraints reflect the hardware typical of real-time deployment systems: for example, TexNet has only up to 20 CPUs that can be dedicated for DL operational deployment as well as potentially up to two GPUs.

### 2.4 Architectural Implementation of Orchestration Strategies

We evaluated three orchestration strategies that define how waveforms are processed across an station network to offer a fair comparison between processing methodologies.

#### 2.4.1 Sequential Processing (Serial)

The serial workflow (Figure 1) loads a single model instance once and processes each waveform in order. Within SeisBench, two serial modes exist. *Offline batch mode* uses the `annotate()` method to process the entire array in a single forward pass. While extremely efficient (0.22–1.22 s for 228 stations, Table 1), this approach is incompatible with real-time streaming where waveforms arrive asynchronously. Because it requires a fixed number of stations to complete a batch, input delays cause the system to wait, leading to unpredictable, growing run-times. The alternative, *Per-station streaming mode*, invokes the `classify()` method for each incoming waveform. While this eliminates batching delays, it incurs the full cost of SeisBench’s preprocessing and windowing pipeline for every station (1.43–33.70 s total; Table 1). Although this mode provides the best available single-process solution for real-time use, runtimes scale linearly with the station count, reaching 35.85 s for 250 stations (TexNet) and 80.23 s for 580 stations (NCSN).

![Serial baseline workflow. Waveforms are processed sequentially by a single DL model instance. Each waveform must wait for the previous one to complete, creating a linear bottleneck that scales poorly with network size.](figures/fig1.JPG)

#### 2.4.2 Ephemeral Task-Based Parallelism (Ripper)

One of our proposed parallelization methods, the Ripper method (Figure 2), uses a task-parallel strategy; each station is handled as an independent task that is managed by the parallelization Python library Ray (Moritz et al., 2018). Each task performs a full work cycle: the model is loaded into memory, performs inference on the given waveform set (3-components from a given station), and the model instance is released from memory when the task ends. Consequently, every task pays the full framework initialization cost (to be discussed in §3.2). Given that Pytorch and TensorFlow have extensive memory costs when loading models into memory (CUDA, XLA-compiled graph, etc.), we cap how many Ripper tasks may run at once using calibrated per-task memory budgets (Table 2) so that concurrent model loads do not exhaust available RAM or VRAM. For GPU trials, Ripper assigns a higher effective VRAM requirement per task than Model-Actor to account for fragmentation and repeated CUDA context creation; we applied scaling factors (1.7 for PhaseNet and PhaseNetLight, 2.0 for EQCCT and EQTransformer) derived from isolated testing, plus a small headroom when many tasks overlap.

**Parallel Task Scheduling.** To avoid resource exhaustion, Ripper eschews unbounded fan-out. Instead, it employs a controlled concurrency approach: Let *N* be the number of stations in the window and *R* the memory-limited cap on concurrent station tasks. The driver code first submits up to min(*R*, *N*) station tasks, then enters a loop: wait until any task finishes, collect its result, and if any stations remain, submit the next one in queue. In-flight task count therefore stays at most *R* until the tail of the run, when fewer than *R* jobs are left, so concurrent model loads stay within memory limits.

**Waveform Referencing.** On the driver, miniSEED files for all stations are read once and assembled into merged three-component ObsPy `Stream` objects, together with a small argument bundle for filters and paths. Those objects are registered in Ray’s object store using `put`. Each station task receives only the resulting object references, calls `get` to access the shared stream, and subsets traces to its station (matching network and station codes to the directory naming used in the dataset) before SeisBench or EQCCT preprocessing and inference. Workers therefore do not each re-read the full miniSEED tree from disk for the same window. Reference counting keeps a single merged copy in cluster memory while any task still needs it.

![Ripper workflow. Each task independently loads the model, performs inference, and unloads. Multiple tasks run in parallel but incur repeated framework initialization overhead per station.](figures/fig2.JPG)

#### 2.4.3 Persistent Inference Actors (Model-Actor)

The final parallelization method we propose is the Model-Actor method (Figure 3). It uses Ray *actors*—stateful, long-lived workers that persist across calls. By handling a stream of remote method invocations, each inference instance loads its PyTorch or TensorFlow weights only once, remaining resident in RAM or VRAM throughout the processing window. The system initializes the requested number of actors, capping the count if host memory is limited. After creation, the driver blocks until each actor answers a lightweight `ready()` call, which returns only after the model has finished loading on that worker, so station work does not start against cold actors. Setup and teardown overhead is concentrated in actor creation, as model weights must be fully loaded before processing can begin.

**Parallel Task Scheduling.** Station work is still expressed as Ray tasks, but these tasks are thin: they subset the shared waveform data for one station, build inputs, and call `remote` inference on an actor. The driver assigns station *i* to actor *i* mod *M* in round-robin order over *M* actors so load spreads across hot models. Each prediction task does not reload the full model; only the actor holds the long-lived weights and kernels until all tasks are finished, in which all actors are released from memory. As with Ripper, we limit how many station prediction tasks are in flight at once to a cap derived from hardware budgets and stability rules (including extra headroom where multiple GPU-bound actors share the same device). The driver keeps submitting prediction tasks while under that cap and uses `wait` to drain completions when the cap is reached, so the cluster never schedules an unbounded pile of concurrent forward passes.

**Waveform Referencing.** Model-Actor uses the same driver-side read-once, `put`, and reference handoff as Ripper: one merged `Stream` and one argument object in the store, subset per station inside each prediction task before the actor runs inference.

![Model-Actor workflow. Persistent inference actors maintain loaded models in memory (or GPU VRAM). Waveforms are dispatched to actors as a stream, eliminating load/unload overhead and maximizing throughput.](figures/fig3.JPG)

### 2.5 Performance Metrics

We recorded two primary timing metrics: **Total Trial Time and Total Run Time for Picker. Total Trial Time** is the wall-clock duration from initial waveform structuring through result saving. This includes setup costs, model loading, and all orchestration overhead. **Total Run Time for Picker** is the cumulative time spent exclusively on inference and preprocessing. This metric isolates the computational cost of the picking algorithm itself.

Memory consumption was monitored continuously throughout each trial using `psutil`, the Python systems library for measuring RAM usage, and `pynvml`, the NVIDIA Management Library (NVML) for VRAM. Per-worker memory budgets were derived from isolated-process measurements of framework initialization, weights, and inference buffers. We added safety buffers (1024 MB VRAM, 1536 MB RAM) to account for Ray overhead and long-lived memory spikes that may exceed available system memory. Final concurrency limits were computed as the minimum of available RAM or VRAM divided by these budgets, which was subject to an 95% safety cap to further prevent OOM errors (Table 2).

### 2.6 Study Protocol

Each configuration was run across a grid of station counts. From the 228 waveforms we acquired from TexNet, we tested counts starting at 10 and stepping by 5 toward 228, with the last step being 3 stations long to reach the total 228 amount. Concurrency was varied in 20% increments of the maximum supported workload for each approach. Trials resulting in OOM errors or system instability were excluded from our results and were restarted to ensure we tested the given workload configuration. The optimal parallelization configuration for each model was defined as the one achieving the minimum total trial time for the 228-station workload among successful runs. No other processes were active on the workstation during testing.

## 3. Results

### 3.1 Single-Process Inference Baselines

To establish a reference for the parallel strategies, we measured the two single-process inference methods for the four SeisBench-integrated models: offline batch method (a single `annotate()` call for all 228 stations) and per-station streaming method (228 sequential `classify()` calls on one loaded model). EQCCT was excluded from this baseline because it uses a custom TensorFlow interface without a SeisBench-compatible stream pipeline. All measurements in Table 1 were performed on a warm cache with five repeated runs; minimum times are reported.

\newpage

**Table 1.** Single-process inference baselines for 228 stations across SeisBench-compatible models. Load Time is warm-cache model initialization. Annotate-All is the total inference duration for a single `annotate()` call with all 228 station traces combined. Classify-Per-Station is the total duration of 228 sequential `classify()` calls on individual station streams. All inference values are minimum across five repeated runs with pre-copied streams. EQCCT is omitted because it does not provide a SeisBench-compatible streaming interface.




| Model     | Dev | CPUs | GPUs | Load (s) | Annotate-All (s) | Classify-Per-Stn (s) |
| --------- | --- | ---- | ---- | -------- | ---------------- | -------------------- |
| PhaseNet  | CPU | 1    | 0    | 1.264    | 0.343            | 33.70                |
| PhaseNet  | GPU | 1    | 1    | 1.309    | 0.224            | 27.11                |
| PNLight   | CPU | 1    | 0    | 1.184    | 0.315            | 1.43                 |
| PNLight   | GPU | 1    | 1    | 1.180    | 0.216            | 1.43                 |
| EQT       | CPU | 1    | 0    | 1.197    | 1.216            | 12.20                |
| EQT       | GPU | 1    | 1    | 1.215    | 0.513            | 12.22                |
| EQT-NC    | CPU | 1    | 0    | 1.190    | 1.182            | 8.19                 |
| EQT-NC    | GPU | 1    | 1    | 1.171    | 0.458            | 8.18                 |


Warm-cache model initalization times were consistent across all architectures, ranged from 1.17 s to 1.31 s. Because this cost is incurred only once per session, it is not a bottleneck in single-process execution. Offline batch inference via `annotate()` was the fastest method overall, completing 228-station processing in 0.22–1.22 s. Lighter models (PhaseNet and PhaseNetLight) finished in 216 to 343 ms, while the heavier EQTransformer variants required 458 ms to 1.22 s. However, as discussed earlier, this method is not viable for real-time operational workflows.

Per-station streaming via `classify()` was substantially slower, ranging from 1.43 to 33.70 s. While similar in architecture, PhaseNetLight remained efficient compared to PhaseNet on CPU, who required 33.70 s to compute the 228 workload, exceeding the 30-second deadline. Although both models use identical windowing parameters (3001 samples, 1500 sample overlap) and the same asyncio batching pipeline, PhaseNet applies an additional "blinding" step that discards 250 samples from each side of every window during its default classification pass. This interaction with SeisBench's asyncio infrastructure produces variable per-call overhead (40 to 350 ms per station), confirming that the 34 s runtime is an inherent characteristic of the `classify()` pipeline rather than a benchmarking artifact. The transformer models fell between these extremes, requiring 8.18 to 12.22 s to process the network.

The per-instance memory budgets referenced in §2.5—which set upper bounds on concurrent Ripper tasks and Model-Actor instances—are tabulated below (Table 2).

**Table 2.** Per-instance memory budgets (MB) used by RAPID to cap concurrency and avoid out-of-memory errors. RAPID divides available RAM and VRAM by these values to determine how many Model-Actors or Ripper tasks can run at once.

*Note:* **Host RAM** is system memory used by the Python process, including model framework (PyTorch/TensorFlow), weights (or host-side copies when using GPU), inference buffers, and CUDA driver context when GPU-accelerated. **GPU VRAM** is memory on the GPU itself, holding model weights, CUDA context, and activation buffers. "Base" is the measured footprint of a single loaded model; "Per Actor" adds safety buffers (1536 MB host RAM, 1024 MB VRAM) for Ray workers and runtime spikes. For host RAM, both Model-Actor and Ripper use the same per-instance budget (Base + buffer). For GPU VRAM, the per-instance budget differs by strategy: Actors use Base + 1024 MB buffer; Ripper uses Base × scaling factor (1.7 for PhaseNet/PhaseNetLight, 2.0 for EQCCT and EQTransformer) to account for fragmentation and repeated initialization in load/unload cycles.


| Model            | Framework  | Host RAM (CPU - MB)     | Host RAM (GPU - MB)     | GPU VRAM (MB)                 |
| ---------------- | ---------- | ----------------------- | ----------------------- | ----------------------------- |
|                  |            | Base / Per Ripper/Actor | Base / Per Ripper/Actor | Base / Per Ripper / Per Actor |
| PhaseNet         | PyTorch    | 502 / 2038              | 870 / 2406              | 500 / 850 / 1524              |
| PhaseNetLight    | PyTorch    | 502 / 2038              | 861 / 2397              | 500 / 850 / 1524              |
| EQTransformer    | PyTorch    | 521 / 2057              | 1001 / 2537             | 528 / 1056 / 1552             |
| EQT-NC | PyTorch    | 524 / 2060              | 1017 / 2553             | 530 / 1060 / 1554             |
| EQCCT            | TensorFlow | 728 / 2264              | 2311 / 3847             | 1732 / 3464 / 2756            |


### 3.2 Ripper: Ephemeral Task-Based Parallelism

The best 228-station Ripper runtime totals fall between 34.3–76.1 s on CPU and 52.5–70.3 s on GPU for the configurations in Table 3. EQCCT produced the highest Ripper totals, driven by TensorFlow’s XLA-compiled graph being reinitialized on every ephemeral task. The PyTorch-backed SeisBench models span about 34–38 s on CPU Ripper. On GPUs, Ripper totals for those same SeisBench models span about 52.5–56.3 s.

Notably, where both CPU and GPU Ripper are listed for the same model, GPU totals remain slower than CPU, mainly from repeated framework and CUDA setup per task. For example, PhaseNet GPU Ripper is about 64% slower than its best CPU Ripper counterpart at 228 stations.

Table 3. Best successful Ripper total time at 228 stations per model and device (picking time equals total time for Ripper). Each entry is the minimum successful end-to-end time over Ripper experiments.


| Model              | Device | CPUs | GPUs | Conc. Tasks | Ripper Picking/Total (s) |
| ------------------ | ------ | ---- | ---- | ----------- | ------------------------ |
| PhaseNet           | CPU    | 20   | 0    | 91          | 34.25                    |
| PhaseNetLight      | CPU    | 20   | 0    | 91          | 35.38                    |
| EQTransformer-NC | CPU    | 20   | 0    | 228         | 35.45                    |
| EQTransformer      | CPU    | 20   | 0    | 137         | 38.42                    |
| EQTransformer      | GPU    | 20   | 1    | 22          | 52.48                    |
| EQTransformer-NC | GPU    | 20   | 1    | 22          | 53.31                    |
| PhaseNetLight      | GPU    | 20   | 1    | 22          | 55.95                    |
| PhaseNet           | GPU    | 20   | 2    | 44          | 56.27                    |
| EQCCT              | GPU    | 40   | 2    | 24          | 70.33                    |
| EQCCT              | CPU    | 20   | 0    | 137         | 76.07                    |


### 3.3 Model-Actor: Persistent Inference Actors

The Model-Actor method produced the lowest total runtimes across all hardware configurations among the parallel orchestration strategies (Table 4). While the hardware in this study supports over 200 concurrent CPU actors, optima for the models tested use 46 persistent actors for the best CPU totals on PhaseNet, PhaseNetLight, EQTransformer, and EQTransformer-NC, 22 actors for the best GPU totals on those SeisBench models, 12 actors for EQCCT on GPU, and 46 actors with 14 Ray CPUs for EQCCT on CPU.

Optimal total runtimes in Table 4 range from 10.97 s (PhaseNet GPU) to 25.01 s (EQCCT CPU). EQCCT in GPU mode benefits strongly from paying the TensorFlow/XLA compilation cost only once at actor creation: Model-Actor totals 12.47 s versus 70.33 s for the best Ripper GPU total in Table 3—about an 82% reduction.

Model-Actor requires a one-time actor initialization period before waveform processing begins. This setup cost ranged from about 4.6 s to 12.9 s in the configurations listed in Table 4, depending on model and actor count, constituting roughly 37–73% of total trial time depending on model and device. While this pre-computing cost is significant, it is a fixed overhead paid once per processing window rather than once per station. For continuous monitoring, where the actors will continue to stay "on", the amortized setup cost per station will approach zero because actors will not be turned off after being created. Similarly, Ray has built in functionality to "revive" dead Raylets (workers). While the initialization cost will be needed to be paid again once revived, the amortized setup cost will again approach zero.

\newpage

Table 4. Optimal Model-Actor performance for the full 228-station workload, ranked by minimum total runtime. Values represent the best successful result at 228 stations for each model–hardware configuration.

*Note: CPUs and GPUs are the number of cores and GPUs allocated at the optimal configuration for each trial (CPU trials use 0 GPUs; GPU trials used one or two GPUs depending on the optimal result). Actors is the number of persistent model instances.*


| Model    | Dev | CPUs | GPUs | Actors | Setup (s) | Pick (s) | Total (s) | Setup OH (%) |
| -------- | --- | ---- | ---- | ------ | --------- | -------- | --------- | ------------ |
| PhaseNet | GPU | 20   | 1    | 22     | 6.70      | 4.27     | 10.97     | 61.0         |
| PNLight  | CPU | 20   | 0    | 46     | 7.04      | 4.28     | 11.47     | 61.4         |
| PhaseNet | CPU | 20   | 0    | 46     | 7.21      | 4.16     | 11.48     | 62.8         |
| PNLight  | GPU | 20   | 1    | 22     | 6.96      | 4.56     | 11.53     | 60.4         |
| EQT      | CPU | 20   | 0    | 46     | 6.92      | 4.70     | 11.63     | 59.5         |
| EQT-NC   | GPU | 20   | 1    | 22     | 6.91      | 5.19     | 12.11     | 57.1         |
| EQCCT    | GPU | 14   | 1    | 12     | 4.64      | 7.82     | 12.47     | 37.2         |
| EQT      | GPU | 20   | 1    | 22     | 7.12      | 6.92     | 14.05     | 50.7         |
| EQT-NC   | CPU | 20   | 0    | 46     | 12.88     | 4.70     | 17.74     | 72.6         |
| EQCCT    | CPU | 14   | 0    | 46     | 12.73     | 12.23    | 25.01     | 50.9         |


### 3.4 Comparative Analysis

Comparing Model-Actor picking times to the SeisBench per-station streaming baseline (Tables 1 and 4, and Figures 4 and 5), the EQTransformer architectures still show large reductions in picking time versus per-station `classify()` on CPU and GPU for some configurations; total runtimes (including actor setup), however, vary by architecture. EQTransformer-NC is illustrative: the best CPU Model-Actor total (17.74 s, Table 4) exceeds the 8.19 s streaming baseline because actor setup (12.88 s) dominates inference time; EQTransformer-NC GPU Model-Actor (12.11 s) can likewise exceed the 8.18 s streaming baseline when actor setup is large relative to the lightweight streaming pass (Table 1 vs. Table 4). EQTransformer (original) CPU Model-Actor remains faster than its 12.20 s streaming baseline when all costs are included (11.63 s). For PhaseNet CPU, the persistent-actor approach remains far faster than per-station streaming on picking time (4.16 s vs. 33.70 s), and the end-to-end Model-Actor total (11.48 s) is well below the best Ripper CPU total at 228 stations (34.25 s, Table 3). For lightweight models like PhaseNetLight, the single-process baseline is still faster in raw picking (1.43 s vs. 4.28 s) because actor setup and IPC dominate. For EQCCT, which lacks a streaming baseline, Model-Actor is the only viable parallelization strategy among the methods we highlight.

The SeisBench offline batch mode remains the fastest option for pre-collected data. However, RAPID's value lies in distributed resource management: it supports continuous streaming without pre-collection, provides automatic memory budgeting, and remains model-agnostic across both TensorFlow and PyTorch.

Figure 4 shows total trial runtime versus station count (markers every 10 stations from 10 to 220, plus 228) for Ripper and Model-Actor. Figure 5 compares the best 228-station totals for each strategy side by side.

<!-- **Table 5.** Optimal performance at 228 stations for the Ripper and Model-Actor methods. All values are from the configuration achieving the minimum picking time across the full concurrency and CPU-count exploration.

*Note: Ripper Picking (s) and Ripper Total (s) are equal (no pre-loading phase). R. Tasks is the number of parallel Ripper worker tasks at the optimal concurrency level. CPUs and GPUs are the number of cores and GPUs allocated at the optimal configuration for the Model-Actor trial (Ripper and M.A. may use different configs; these values apply to the M.A. trial). MA Pick (s) is the cumulative inference time for all 228 stations at the optimal actor count. MA Setup (s) is the actor initialization cost before inference begins. MA Total (s) is end-to-end wall-clock time. MA Actors is the number of persistent actors at the lowest total runtime; Max Tasks is the maximum the hardware allows given available RAM and VRAM (CPU: 90 pct of 512 GB; GPU: 95 pct of 49 GB per GPU). Reduction is computed as (1 - MA Total / Ripper Total) x 100 pct.*


| Model    | HW       | R. Tasks | R. Time (s) | MA Act. | MA Setup (s) | MA Pick (s) | MA Total (s) | Red. |
| -------- | -------- | -------- | ----------- | ------- | ------------ | ----------- | ------------- | ---- |
| PhaseNet | 20C/2G   | 20       | 46.58       | 22      | 7.02         | 3.48        | 10.51         | 77%  |
| EQCCT    | 20C/2G   | 22       | 87.53       | 12      | 4.56         | 6.17        | 10.75         | 88%  |
| EQT-NC   | 20C      | 90       | 33.13       | 45      | 7.16         | 3.81        | 11.07         | 67%  |
| EQT      | 20C      | 146      | 31.85       | 45      | 7.12         | 4.20        | 11.36         | 64%  |
| EQT-NC   | 20C/2G   | 20       | 49.08       | 22      | 7.69         | 5.10        | 12.80         | 74%  |
| EQT      | 20C/2G   | 22       | 52.48       | 22      | 6.92         | 7.12        | 14.05         | 73%  |
| PNLight  | 20C/2G   | 20       | 47.29       | 22      | 10.78        | 3.51        | 14.30         | 70%  |
| PNLight  | 20C      | 45       | 31.54       | 45      | 11.19        | 3.52        | 14.94         | 53%  |
| PhaseNet | 20C      | 120      | 26.72       | 45      | 11.25        | 4.66        | 16.05         | 40%  |
| EQCCT    | 20C      | 50       | 63.01       | 45      | 9.38         | 7.67        | 17.07         | 73%  | -->


\newpage

![Comparison of total trial runtime across all models and parallelization methods. Markers and line segments are shown every 10 stations (10–220) with a final point at 228; the vertical axis is 0–30 s (10 s ticks), with trajectories above 30 s clipped.](figures/fig4_runtime_3d.png){width=90%}

![Minimum total runtime at 228 stations for Ripper versus Model-Actor (CPU left, GPU right). X-axis labels show concurrent workers: R = Ripper tasks, MA = Model-Actor instances. Bars use the best successful end-to-end time per strategy at 228 stations. The dashed red line is the 30-second real-time target. Several Ripper bars exceed that line; Model-Actor totals in Table 4 remain below it for the pairings shown.](figures/fig5.png){width=90%}

### 3.5 Memory Utilization

Memory tracking confirmed that the pre-actor budgeting system prevented OOM errors. In CPU mode, the optimal rows for PhaseNet, PhaseNetLight, and both EQTransformer variants in Table 5 use 46 actors and report combined process-tree RAM near 49–52 GB against requested budgets of about 104–107 GB. EQTransformer-NC shows the largest tree footprint among those rows (about 56 GB tree RAM versus 106,536 MB requested), because setup-heavy actor waves leave more resident RSS in the process tree. The EQCCT CPU optimum (46 actors, 14 CPUs) shows a similar pattern (about 49 GB tree RAM, 115,920 MB requested). Preliminary runs without this budget triggered OOM failures during concurrent initialization. We implemented incremental testing based off of scaling factors to identify the minimal amount of buffer memory needed to maintain an actor in memory; our findings found that using less than 1.7 and 2.0 x of the given model's memory consumption caused OOM errors.

In GPU mode, the optimal 228-station trials use 22 actors for the SeisBench models and 12 for EQCCT (Table 4). Our two GPUs support up to 44 actors for the SeisBench models and 24 for EQCCT amongst themselves; the optimal configurations use fewer because additional actors did not improve total runtimes. Prior GPU benchmarks showed that increasing actors from the optimal level toward the hardware cap can increase both picking and total runtimes (for example, PhaseNet GPU total rising when moving from 22 toward 44 actors). For EQCCT, measured VRAM slightly exceeded the per-actor budget due to TensorFlow XLA workspace allocations, showing that our memory budgeting strategy is not always flawless. However, all reported values remained within the 49 GB per-GPU hardware limit.

\newpage

Table 5. Memory utilization at the optimal 228-station Model-Actor configuration. CPUs and GPUs are the number of cores and GPUs allocated at the optimal configuration. Peak RAM is the maximum single-process RSS. Process-Tree RAM/VRAM is the combined memory footprint of the main process and all Ray worker actors. Requested RAM/VRAM is the total memory pre-allocated based on per-actor budgets (Table 2) including safety buffers.

*Note: GPU process-tree VRAM is measured for the assigned GPUs via NVML PID lookup. CPU-based trials report zero VRAM. The large gap between Requested and Actual VRAM for SeisBench GPU models reflects the conservative 1024 MB per-actor safety buffer; EQCCT GPU actual VRAM slightly exceeds the base per-actor budget due to TensorFlow XLA workspace allocations, but remains within the 49 GB per-GPU hardware limit.*


| Model    | Dev | CPUs | GPUs | Act. | Req. RAM | Tree RAM | Req. VRAM | Tree VRAM | Peak RAM |
| -------- | --- | ---- | ---- | ---- | -------- | -------- | --------- | --------- | -------- |
| PhaseNet | GPU | 20   | 1    | 22   | 58,564   | 32,813   | 36,344    | 11,580    | 782      |
| PNLight  | CPU | 20   | 0    | 46   | 105,524  | 49,017   | 0         | 0         | 273      |
| PhaseNet | CPU | 20   | 0    | 46   | 105,524  | 51,694   | 0         | 0         | 274      |
| PNLight  | GPU | 20   | 1    | 22   | 58,366   | 39,068   | 36,344    | 11,580    | 280      |
| EQT      | CPU | 20   | 0    | 46   | 106,398  | 51,012   | 0         | 0         | 258      |
| EQT-NC   | GPU | 20   | 1    | 22   | 61,798   | 43,578   | 37,004    | 12,273    | 244      |
| EQCCT    | GPU | 14   | 1    | 12   | 49,236   | 15,363   | 34,608    | 40,316    | 456      |
| EQT      | GPU | 20   | 1    | 22   | 61,446   | 42,982   | 36,960    | 12,226    | 275      |
| EQT-NC   | CPU | 20   | 0    | 46   | 106,536  | 56,537   | 0         | 0         | 200      |
| EQCCT    | CPU | 14   | 0    | 46   | 115,920  | 48,701   | 0         | 0         | 202      |


![Peak memory at 228 stations for Ripper versus Model-Actor. Left panel shows CPU RAM (GB); right panel shows GPU VRAM (GB). X-axis labels show R = Ripper concurrent tasks, MA = Model-Actor actors. All instances were loaded simultaneously via Ray actors and memory was recorded with psutil (RAM) and pynvml (VRAM). When the instance count is the same (e.g., PhaseNetLight CPU, 45 for both strategies), memory is nearly identical, confirming that the per-instance footprint is strategy-independent; differences arise only when concurrency counts diverge.](figures/fig6.png){width=90%}

\newpage

![Serial baseline versus fastest Ripper configurations versus the Amdahl ideal limit, per CPU allocation (5, 8, 11, 14, 17, 20 CPUs). Serial curves use per-station streaming runtimes for Ripper variants that have SeisBench streaming baselines (PhaseNet, EQTransformer, and EQT-NC for the configurations plotted). Ripper curves are the models with the lowest mean successful total trial time over the station grid and Ray CPU allocations (5, 8, 11, 14, 17, 20) for CPU Ripper, one-GPU Ripper, and two-GPU Ripper respectively. They illustrate scaling with network size rather than reproducing every 228-station minimum in Table 3. The red dotted line is the batch-based Amdahl reference (T = load + (N × tbatch) / workers) from Table 1.](figures/fig7_serial_vs_ripper.png){width=78%}

![Serial baselines versus fastest Model Actor configurations versus the Amdahl ideal limit, shown per CPU allocation. Serial curves use per-station streaming baselines wherever the plotted Model Actor architecture has a SeisBench streaming entry. Parallel curves are chosen the same way as in Figure 7—lowest mean total trial time over the station grid and protocol CPU counts—for CPU Model Actor, one-GPU Model Actor, and two-GPU Model Actor. The red dotted line denotes the Amdahl ideal (T = load + (N × tbatch) / workers), representing the theoretical minimum runtime for all methods.](figures/fig8_serial_vs_modelactor.png){width=78%}

## 4. Discussion

### 4.1 The Case for Persistent-Actor Orchestration in Real-Time Seismic Networks

In our study, it is evident that the primary runtime bottleneck in stream-based seismic phase picking is per-task initialization cost rather than inference complexity. The Ripper method makes that cost explicit: with model load times between 0.92 and 1.31 s per task, aggregate initialization at 228 stations keeps GPU Ripper and EQCCT on CPU above the 30-second real-time target in the updated benchmarks, and the SeisBench CPU Ripper minima in Table 3 also land above that line—leaving Model-Actor with substantially more headroom for downstream processing. The Model-Actor strategy eliminates this bottleneck by loading model instances once and reusing them for all incoming waveforms. This reduces the marginal cost of each station to the forward-pass inference time plus inter-process communication (IPC) latency. The observed roughly 50–82% runtime reduction for Table 4 versus matching Table 3 pairings reflects a fundamental change in model lifecycle management rather than incremental optimization.

At 228 stations, Model-Actor delivers end-to-end runtimes under the 30-second real-time target for every Model-Actor pairing listed in Table 4. GPU Ripper and EQCCT on CPU remain above that line; the revised SeisBench CPU Ripper minima in Figure 5 / Table 3 also sit above it at the tested concurrencies, whereas persistent actors avoid paying full per-station initialization every time. Figures 7 and 8 plot Ripper and Model-Actor trajectories over station count (5, 10, …, 225, 228) at each CPU allocation tested, with the batch-based Amdahl reference from Table 1 overlaid in red. Persistent-actor curves track much closer to, or even outperform, streaming-like slopes, with a visible positive offset originating from one-time actor setup and Ray IPC. Negative offset can be attributed to the parallel distribution of inference tasks across multiple workers, which for compute-heavy models allows the marginal cost per station to drop significantly below the sequential baseline, eventually overcoming the initial setup overhead as the network size increases. The majority of the streaming and RAPID results don't come near the theoretically achievable runtime via idealized batch inference (Amdahl’s ideal curve; see Amdahl, 1967); however that gap is expected, as batch `annotate()` sidesteps the per-station classify pipeline entirely and our parallelization methods have large IPC latency. Ripper curves selected as fastest on one or two GPUs (Figure 7) can sit closer to that batch-based reference than heavier CPU-Ripper stacks, consistent with smaller per-task work on GPU, though total runtimes still remain well above the ideal curve. 

## 5. Conclusion

In conclusion, RAPID is a resource-aware parallelization framework that enables real-time, network-scale seismic phase picking. By combining persistent model actors with hardware-constrained memory budgeting, RAPID is able to process 228 stations between 10.97–25.01 s for the Model-Actor configurations summarized in Table 4, representing on the order of roughly 50–82% lower total runtime than the matching best Ripper totals at 228 stations for each Model-Actor row alongside its Table 3 counterpart (largest relative gaps where Ripper times are highest, for example EQCCT GPU at about 82%). It meets the 30-second real-time target for all Model-Actor models listed in Table 4, including heavy models like EQCCT, which have high individual initialization costs under ephemeral-task execution.

The results confirm that orchestration overhead—repeated loading and teardown of model instances—dominates wall time for ephemeral-task Ripper at network scale, leaving Table 3 CPU Ripper times well above the 30-second bound for the refreshed sweeps while Model-Actor retains sub-30 s totals. RAPID is model-agnostic and supports both PyTorch and TensorFlow through a unified interface. It is intended to complement existing toolkits like SeisBench by providing the orchestration layer required for streaming waveforms in real-time operational environments.

## 6. Limitations and Future Work

### 6.1 Current Limitations

This study evaluated performance on a single workstation. While internal testing at TexNet suggests these gains are consistent across similar hardware, formal multi-node evaluations for larger networks (e.g., NCSN or SCSN) have not yet been conducted. Additionally, waveforms were pre-loaded into RAM to isolate orchestration costs; live deployments would incur extra overhead for waveform decoding and network retrieval. The 20% step size in how we varied concurrency also means the reported optimal actor counts are approximations that could be refined with more granular testing. Finally, EQCCT is currently accessed through a custom TensorFlow loader; porting it to PyTorch for native SeisBench integration would likely reduce framework-specific initialization costs.

### 6.2 Operational Deployment via SeisComP Integration

The nearest-term deployment target is SeisComP, the open-source platform used by TexNet and other national networks. Chen, Savvaidis, et al. (2024) show how EQCCT already participates in SeisComP-driven association and catalog products. RAPID is meant to slip under that kind of messaging fabric: it governs Ray workers, batches data into SeisBench or TensorFlow, and reports feasible wall times when every station in a window must be picked before association. TexNet is extending this stack so live miniSEED can flow into Model-Actor-based picking within the same processing cycle as the rest of the catalog pipeline, without ad hoc resource management for each event.

### 6.3 EQCCT Integration into SeisBench

EQCCT is currently implemented via TensorFlow, while SeisBench requires PyTorch for integration with its training infrastructure and community registry. Porting EQCCT to PyTorch is underway at TexNet. Once integrated, EQCCT will benefit from SeisBench’s standardized preprocessing and weight management, while gaining immediate compatibility with RAPID's PyTorch-based memory budgeting. This will remove the TensorFlow-specific XLA compilation overhead and further reduce actor setup time.

### 6.4 RAPID as a SeisBench-Compatible Tool

A long-term goal is contributing RAPID to SeisBench as a standardized orchestration module. This would allow any SeisBench-compatible model to run in persistent-actor streaming mode with built-in hardware-aware memory budgeting and multi-GPU scheduling. This would lower the engineering barrier for networks looking to deploy DL picking in production settings.

### 6.5 Distributed and Dynamic Scaling

Ray natively supports multi-node clusters, which would allow RAPID to scale to networks exceeding single-machine capacity. Future work will investigate how memory budgeting behaves across heterogeneous nodes and how scheduling overhead impacts performance at very large scales. Additionally, a dynamic scaling mode that adjusts the actor pool based on real-time resource usage would remove the need for offline calibration and improve reliability on shared infrastructure. Finally, alternative parallelization toolkits must be explored to identify further techniques that will lower processing runtimes.

## Data and Resources

Seismic wave data and computational hardware were provided by the Texas Seismological Network and Seismology Research Team (TexNet). All seismic data were downloaded from TexNet’s FDSN network and are publicly available at http://rtserve.beg.utexas.edu/. EQCCT is an open-source machine learning model (Saad et al. (2023)) and is available on Github at (https://github.com/ut-beg-texnet/eqcct/tree/main). EQCCTOne and RAPID can be accessed at (https://github.com/ut-beg-texnet/eqcct/tree/main/eqcctone) and (https://github.com/ut-beg-texnet/eqcct/tree/main/eqcctpro), respectively.

## Acknowledgements

The authors would like to thank the Texas Seismological Network and Seismology Research (TexNet) group and the State of Texas that provided financial support for this publication under the University of Texas at Austin award #201503664. Special thanks to Elena Kalogirou and Peter Sarkis for revising this paper.

## References

Sheen, D.-H., Y.-J. Seong, S. Jung, and J.-h. Park (2023). Real-time application of deep learning seismic phase picker to earthquake monitoring. Poster S31G-0424, *AGU Fall Meeting*, San Francisco, CA, 11–15 December, session *Machine Learning–Driven Analysis of Geophysical Signals III*.

Ali, M. (2023, Jul). Distributed processing using ray framework in python.

Amdahl, G. M. (1967). Validity of the single processor approach to achieving large scale computing capabilities. In Proceedings of the April 18-20, 1967, Spring Joint Computer Conference, AFIPS ’67 (Spring), New York, NY, USA, pp. 483–485. Association for Computing Machinery.

Bates, D. and D. Watts (2008, 05). Nonlinear Regression Analysis and Its Applications, pp. 32 – 66.

Chen, Y., O. M. Saad, A. Savvaidis, F. Zhang, Y. Chen, D. Huang, H. Li, and F. Aziz Zanjani (2024). Deep learning for p-wave first-motion polarity determination and its application in focal mechanism inversion. IEEE Transactions on Geoscience and Remote Sensing 62, 1–11.

Chen, Y., Savvaidis, A., Siervo, D., Huang, D., & Saad, O. M. (2024). Near real-time earthquake monitoring in Texas using the highly precise deep learning phase picker. Earth and Space Science, 11, e2024EA003890. https://doi.org/10.1029/2024EA003890

Lim, C. S. Y., S. Lapins, M. Segou, and M. J. Werner (2025). Deep learning phase pickers: how well can existing models detect hydraulic-fracturing induced microseismicity from a borehole array? *Geophysical Journal International* 240(1), 535–549, https://doi.org/10.1093/gji/ggae386

Feng, T., S. Mohanna, and L. Meng (2022). Edgephase: A deep learning model for multi-station seismic phase picking. Geochemistry, Geophysics, Geosystems 23(11), e2022GC010453. e2022GC010453 2022GC010453.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics 29(5), 1189 – 1232.

Gustafson, J. L. (1988, May). Reevaluating amdahl’s law. Commun. ACM 31(5), 532–533. Herlihy, M. and N. Shavit (2012). The Art of Multiprocessor Programming, Revised Reprint. Morgan Kaufmann.

Johnston, J. and J. DiNardo (1997). Econometric Methods (4th ed.). McGraw-Hill.

McCool, M., J. Reinders, and A. Robison (2012). Structured Parallel Programming: Patterns for Efficient Computation. ITPro collection. Elsevier Science.

Moritz, P., R. Nishihara, S. Wang, A. Tumanov, R. Liaw, X. Liang, M. Elibol, Z. Yang, W. Paul, M. I. Jordan, and I. Stoica (2018). Ray: A Distributed Framework for Emerging AI Applications. In 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18), Carlsbad, CA, pp. 561–577. USENIX Association.

Mousavi, S., W. Ellsworth, Z. Weiqiang, L. Chuang, and G. Beroza (2020, 08). Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nature Communications 11, 3952.

Mousavi, S. M. and G. C. Beroza (2022). Deep-learning seismology. Science 377(6607), eabm4470.

Münchmeyer, J., J. Woollam, A. Rietbrock, F. Tilmann, D. Lange, T. Bornstein, T. Diehl, C. Giunchi, F. Haslinger, D. Jozinović, A. Michelini, J. Saul, and H. Soto (2022, 01). Which picker fits my data? a quantitative evaluation of deep learning based seismic pickers. Journal of Geophysical Research: Solid Earth 127.

Ray.io (n.d.). What is ray core? https://docs.ray.io/en/latest/ray-core/walkthrough.html. Accessed: 2023-11-13.

Saad, O. M. and Y. Chen (2022). Capsphase: Capsule neural network for seismic phase classification and picking. IEEE Transactions on Geoscience and Remote Sensing 60, 1–11.

Saad, O. M., Y. Chen, A. Savvaidis, W. Chen, F. Zhang, and Y. Chen (2022). Unsupervised deep learning for single-channel earthquake data denoising and its applications in event detection and fully automatic location. IEEE Transactions on Geoscience and Remote Sensing 60, 1–10.

Saad, O. M., Y. Chen, A. Savvaidis, S. Fomel, and Y. Chen (2022). Real-time earthquake detection and magnitude estimation using vision transformer. Journal of Geophysical Research: Solid Earth 127(5), e2021JB023657. e2021JB023657 2021JB023657.

Saad, O. M., Y. Chen, D. Siervo, F. Zhang, A. Savvaidis, G.-c. D. Huang, N. Igonin, S. Fomel, and Y. Chen (2023). Eqcct: A production-ready earthquake detection and phase-picking method using the compact convolutional transformer. IEEE Transactions on Geoscience and Remote Sensing 61, 1–15.

Saad, O. M., G. Huang, Y. Chen, A. Savvaidis, S. Fomel, N. Pham, and Y. Chen (2021). Scalodeep: A highly generalized deep learning framework for real-time earthquake detection. Journal of Geophysical Research: Solid Earth 126(4), e2020JB021473. e2020JB021473 2020JB021473.

Savvaidis, A., B. Young, G. D. Huang, and A. Lomax (2019, 06). Texnet: A statewide seismological network in texas. Seismological Research Letters 90(4), 1702–1715.

Tan, Y. J., F. Waldhauser, W. Ellsworth, M. Zhang, Z. Weiqiang, M. Michele, L. Chiaraluce, G. Beroza, and M. Segou (2021, 04). Machine-learning-based high-resolution earthquake catalog reveals how complex fault structures were activated during the 2016–2017 central italy sequence. The Seismic Record 1, 11–19.

Wang, T., D. Trugman, and Y. Lin (2021). Seismogen: Seismic waveform synthesis using gan with application to seismic data augmentation. Journal of Geophysical Research: Solid Earth 126(4), e2020JB020077. e2020JB020077 2020JB020077.

Woollam, J., J. Münchmeyer, F. Tilmann, A. Rietbrock, D. Lange, T. Bornstein, T. Diehl, C. Giunchi, F. Haslinger, and D. Jozinović (2022). SeisBench—A Toolbox for Machine Learning in Seismology. Seismological Research Letters 93(3), 1695–1709.

Xiao, Z., J. Wang, C. Liu, J. Li, L. Zhao, and Z. Yao (2021). Siamese earthquake transformer: A pair-input deep-learning model for earthquake detection and phase picking on a seismic arrival-time picking method. Journal of Geophysical Research: Solid Earth 126(5), e2020JB021444. e2020JB021444 2020JB021444.

Yeck, W. L., J. M. Patton, Z. E. Ross, G. P. Hayes, M. R. Guy, N. B. Ambruz, D. R. Shelly, H. M. Benz, and P. S. Earle (2020). Leveraging Deep Learning in Global 24/7 Real-Time Earthquake Monitoring at the National Earthquake Information Center, Seismol. Res. Lett. 92, 469–480, doi: 10.1785/0220200178.

Yu, Z., Y. Jiang, X. Jing, and H. Zheng (2024). Study on geomagnetic observations associated with three major earthquakes in southwest china through a novel deep learning framework. IEEE Transactions on Geoscience and Remote Sensing 62, 1–18.

Zhu, W. and G. C. Beroza (2018, 10). Phasenet: a deep-neural-network-based seismic arrival-time picking method. Geophysical Journal International 216(1), 261–273.

\newpage

## Supplemental Material

The following pages present enlarged, rotated versions of all results figures and expanded tables for improved readability.

\newpage
\begin{figure}[H]
\centering
\rotatebox{90}{%
  \begin{minipage}{0.85\textheight}
    \centering
    \includegraphics[width=\linewidth]{/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/docs/figures/fig4_runtime_3d.png}
    \caption*{Figure 4. Comparison of total trial runtime across all models and parallelization methods. Data points are plotted every 10 stations from 10 to 228. Fig.~5 and Table~3 summarize best 228-station totals.}
  \end{minipage}%
}
\end{figure}

\newpage
\begin{figure}[H]
\centering
\rotatebox{90}{%
  \begin{minipage}{0.85\textheight}
    \centering
    \includegraphics[width=\linewidth]{/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/docs/figures/fig5.png}
    \caption*{Figure 5. Minimum total runtime at 228 stations for Ripper versus Model-Actor (left: CPU, right: GPU). Bars are best successful end-to-end times at 228 stations per strategy. Dashed red line: 30~s target.}
  \end{minipage}%
}
\end{figure}

\newpage
\begin{figure}[H]
\centering
\rotatebox{90}{%
  \begin{minipage}{0.85\textheight}
    \centering
    \includegraphics[width=\linewidth]{/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/docs/figures/fig6.png}
    \caption*{Figure 6. Peak memory at 228 stations for Ripper (/// hatching) versus Model-Actor (dot hatching). Left panel shows CPU RAM (GB); right panel shows GPU VRAM (GB). X-axis labels show R = Ripper concurrent tasks, MA = Model-Actor actors.}
  \end{minipage}%
}
\end{figure}

\newpage
\begin{figure}[H]
\centering
\rotatebox{90}{%
  \begin{minipage}{0.85\textheight}
    \centering
    \includegraphics[width=\linewidth]{/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/docs/figures/fig7_serial_vs_ripper.png}
    \caption*{Figure 7. Serial baseline versus fastest Ripper configurations versus the batch-based Amdahl reference, per CPU allocation. Ripper curves use the lowest mean total trial time over the station grid and protocol CPU allocations for CPU, 1-GPU, and 2-GPU Ripper. Serial curves are per-station streaming for Ripper models with SeisBench baselines. The panel illustrates scaling with network size rather than every minimum reported in Table~3.}
  \end{minipage}%
}
\end{figure}

\newpage
\begin{figure}[H]
\centering
\rotatebox{90}{%
  \begin{minipage}{0.85\textheight}
    \centering
    \includegraphics[width=\linewidth]{/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/docs/figures/fig8_serial_vs_modelactor.png}
    \caption*{Figure 8. Serial baselines versus fastest Model Actor configurations versus the Amdahl ideal limit, shown per CPU allocation. Parallel curves are chosen like Figure 7 (minimum mean total trial time per hardware class). The red dotted line denotes the Amdahl ideal (T = load + (N $\times$ tbatch) / workers), representing the theoretical minimum runtime for all methods.}
  \end{minipage}%
}
\end{figure}

\newpage
\begin{landscape}

\begin{center}
\textbf{Table 1.} Single-process inference baselines for 228 stations across SeisBench-compatible models.

\vspace{1em}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Device} & \textbf{CPUs} & \textbf{GPUs} & \textbf{Load Time (s)} & \textbf{Annotate-All (s)} & \textbf{Classify-Per-Stn (s)} \\
\midrule
PhaseNet      & CPU & 1 & 0 & 1.264 & 0.343 & 33.70 \\
PhaseNet      & GPU & 1 & 1 & 1.309 & 0.224 & 27.11 \\
PhaseNetLight & CPU & 1 & 0 & 1.184 & 0.315 & 1.43  \\
PhaseNetLight & GPU & 1 & 1 & 1.180 & 0.216 & 1.43  \\
EQTransformer & CPU & 1 & 0 & 1.197 & 1.216 & 12.20 \\
EQTransformer & GPU & 1 & 1 & 1.215 & 0.513 & 12.22 \\
EQT-NC        & CPU & 1 & 0 & 1.190 & 1.182 & 8.19  \\
EQT-NC        & GPU & 1 & 1 & 1.171 & 0.458 & 8.18  \\
\bottomrule
\end{tabular}
\end{center}

\end{landscape}

\newpage
\begin{landscape}

\begin{center}
\textbf{Table 2.} Per-instance memory budgets (MB) used by RAPID to cap concurrency and avoid out-of-memory errors.

\vspace{1em}
\small
\begin{tabular}{llccccccc}
\toprule
\textbf{Model} & \textbf{Framework} & \multicolumn{2}{c}{\textbf{Host RAM -- CPU (MB)}} & \multicolumn{2}{c}{\textbf{Host RAM -- GPU (MB)}} & \multicolumn{3}{c}{\textbf{GPU VRAM (MB)}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-9}
 & & Base & Per Ripper/Actor & Base & Per Ripper/Actor & Base & Per Ripper & Per Actor \\
\midrule
PhaseNet      & PyTorch    & 502  & 2,038 & 870  & 2,406 & 500  & 850  & 1,524 \\
PhaseNetLight & PyTorch    & 502  & 2,038 & 861  & 2,397 & 500  & 850  & 1,524 \\
EQTransformer & PyTorch    & 521  & 2,057 & 1,001 & 2,537 & 528  & 1,056 & 1,552 \\
EQT-NC        & PyTorch    & 524  & 2,060 & 1,017 & 2,553 & 530  & 1,060 & 1,554 \\
EQCCT         & TensorFlow & 728  & 2,264 & 2,311 & 3,847 & 1,732 & 3,464 & 2,756 \\
\bottomrule
\end{tabular}
\end{center}

\end{landscape}

\newpage
\begin{landscape}

\begin{center}
\textbf{Table 3.} Best successful Ripper total time at 228 stations per model and device.

\vspace{0.8em}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Device} & \textbf{CPUs} & \textbf{GPUs} & \textbf{Conc. Tasks} & \textbf{Ripper Picking/Total (s)} \\
\midrule
PhaseNet         & CPU & 20 & 0 & 91  & 34.25 \\
PhaseNetLight    & CPU & 20 & 0 & 91  & 35.38 \\
EQTransformer-NC & CPU & 20 & 0 & 228 & 35.45 \\
EQTransformer    & CPU & 20 & 0 & 137 & 38.42 \\
EQTransformer    & GPU & 20 & 1 & 22  & 52.48 \\
EQTransformer-NC & GPU & 20 & 1 & 22  & 53.31 \\
PhaseNetLight    & GPU & 20 & 1 & 22  & 55.95 \\
PhaseNet         & GPU & 20 & 2 & 44  & 56.27 \\
EQCCT            & GPU & 40 & 2 & 24  & 70.33 \\
EQCCT            & CPU & 20 & 0 & 137 & 76.07 \\
\bottomrule
\end{tabular}
\end{center}

\end{landscape}

\newpage
\begin{landscape}

\begin{center}
\textbf{Table 4.} Optimal Model-Actor performance for the full 228-station workload, ranked by minimum total runtime.

\vspace{1em}
\small
\begin{tabular}{lcccccccc}
\toprule
\textbf{Model} & \textbf{Device} & \textbf{CPUs} & \textbf{GPUs} & \textbf{Actors} & \textbf{Setup (s)} & \textbf{Pick (s)} & \textbf{Total (s)} & \textbf{Setup OH} \\
\midrule
PhaseNet      & GPU & 20 & 1 & 22 & 6.70  & 4.27 & 10.97 & 61.0\% \\
PhaseNetLight & CPU & 20 & 0 & 46 & 7.04  & 4.28 & 11.47 & 61.4\% \\
PhaseNet      & CPU & 20 & 0 & 46 & 7.21  & 4.16 & 11.48 & 62.8\% \\
PhaseNetLight & GPU & 20 & 1 & 22 & 6.96  & 4.56 & 11.53 & 60.4\% \\
EQTransformer & CPU & 20 & 0 & 46 & 6.92  & 4.70 & 11.63 & 59.5\% \\
EQT-NC        & GPU & 20 & 1 & 22 & 6.91  & 5.19 & 12.11 & 57.1\% \\
EQCCT         & GPU & 14 & 1 & 12 & 4.64  & 7.82 & 12.47 & 37.2\% \\
EQTransformer & GPU & 20 & 1 & 22 & 7.12  & 6.92 & 14.05 & 50.7\% \\
EQT-NC        & CPU & 20 & 0 & 46 & 12.88 & 4.70 & 17.74 & 72.6\% \\
EQCCT         & CPU & 14 & 0 & 46 & 12.73 & 12.23 & 25.01 & 50.9\% \\
\bottomrule
\end{tabular}
\end{center}

\end{landscape}

\newpage
\begin{landscape}

\begin{center}
\textbf{Table 5.} Memory utilization at the optimal 228-station Model-Actor configuration. All memory values are in MB.

\vspace{1em}
\small
\begin{tabular}{lccccccccc}
\toprule
\textbf{Model} & \textbf{Device} & \textbf{CPUs} & \textbf{GPUs} & \textbf{Actors} & \textbf{Req. RAM} & \textbf{Tree RAM} & \textbf{Req. VRAM} & \textbf{Tree VRAM} & \textbf{Peak RAM} \\
\midrule
PhaseNet      & GPU & 20 & 1 & 22 & 58,564  & 32,813 & 36,344 & 11,580 & 782 \\
PhaseNetLight & CPU & 20 & 0 & 46 & 105,524 & 49,017 & 0      & 0      & 273 \\
PhaseNet      & CPU & 20 & 0 & 46 & 105,524 & 51,694 & 0      & 0      & 274 \\
PhaseNetLight & GPU & 20 & 1 & 22 & 58,366  & 39,068 & 36,344 & 11,580 & 280 \\
EQTransformer & CPU & 20 & 0 & 46 & 106,398 & 51,012 & 0      & 0      & 258 \\
EQT-NC        & GPU & 20 & 1 & 22 & 61,798  & 43,578 & 37,004 & 12,273 & 244 \\
EQCCT         & GPU & 14 & 1 & 12 & 49,236  & 15,363 & 34,608 & 40,316 & 456 \\
EQTransformer & GPU & 20 & 1 & 22 & 61,446  & 42,982 & 36,960 & 12,226 & 275 \\
EQT-NC        & CPU & 20 & 0 & 46 & 106,536 & 56,537 & 0      & 0      & 200 \\
EQCCT         & CPU & 14 & 0 & 46 & 115,920 & 48,701 & 0      & 0      & 202 \\
\bottomrule
\end{tabular}
\end{center}

\end{landscape}
