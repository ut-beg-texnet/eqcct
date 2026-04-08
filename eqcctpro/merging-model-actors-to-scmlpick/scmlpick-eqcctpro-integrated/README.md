# scmlpick + EQCCTPro ModelActor (reference integration)

This tree is a **working duplicate** of the scmlpick SeisComP module with **minimal changes** that wire **eqcctpro’s `ModelActor`** into the existing Ray `picker` path. It is intended as a **reference** for upstream scmlpick: copy patterns from here rather than maintaining this fork long-term.

## What changed

| Area | Change |
|------|--------|
| **`predictor/predictor.py`** | Drops duplicated Keras EQCCT graph; imports **`eqcctpro.eqcct_tf_models`** and **`eqcctpro.waveform_filter`**. **`prepare_station_chunk`** uses **`apply_waveform_filter`** / **`resolve_waveform_filter_params`** so filter type, corners, and zerophase match EQCCTPro. **`mseed_predictor`** / **`mseed_predictor_seisbench`** accept **`waveform_filter_*`** (forwarded from **`params`**). |
| **`seiscomp/bin/scmlpick`** | **`ray.init(num_gpus=len(ray_gpu_device_ids))`** with **`CUDA_VISIBLE_DEVICES`** from **`ray.gpuDeviceIds`** or **`eqcct.gpuID`**. **`ray.inferenceMode`**: **`modelActor`** spawns ModelActors; **`ripper`** skips actors. **`ray.numModelActors`** (0=auto) and **`ray.maxTasksQueue`** cap concurrency. CLI: **`--inference-mode`**, **`--gpu-devices`**, **`--num-model-actors`**. |
| **`etc/descriptions/scmlpick.xml`**, **`etc/defaults/scmlpick.cfg`** | Document **`ray.inferenceMode`**, **`ray.numModelActors`**, **`ray.gpuDeviceIds`**, **`ray.actorGpuMemoryLimitMB`**, **`ray.ripperGpuMemoryLimitMB`**, **`eqcct.waveformFilterType`**, **`eqcct.waveformFilterCorners`**, **`eqcct.waveformFilterZerophase`**. |

### Configuration (module, not station bindings)

| Key | Role |
|-----|------|
| **`ray.inferenceMode`** | **`modelActor`** (default) or **`ripper`**. |
| **`ray.maxTasksQueue`** | Max concurrent **`picker`** Ray tasks (unchanged; applies to both modes). |
| **`ray.numModelActors`** | ModelActor pool size; **`0`** = auto from **`maxTasksQueue`** and visible GPU count. |
| **`ray.gpuDeviceIds`** | e.g. **`0`** or **`0,1`** → sets **`CUDA_VISIBLE_DEVICES`** and **`ray.init(num_gpus=…)`**. If empty, uses **`eqcct.gpuID`** when **`≥ 0`**. |
| **`eqcct.gpuID`** | Primary device when **`gpuDeviceIds`** empty; **`-1`** = CPU-only. |
| **`ray.numCPUs`** | Ray CPU resource (unchanged). |
| **`ray.actorGpuMemoryLimitMB`** | Soft TF VRAM cap per **ModelActor** (**0** = unset). |
| **`ray.ripperGpuMemoryLimitMB`** | Soft TF VRAM cap per **Ripper** task when GPU visible (**0** = unset). |

## Deploy

1. Install **eqcctpro** (and its deps) in the **same** environment SeisComP uses for scmlpick.
2. From `seiscomp/share/scmlpick/tools/scmlpick-predicctor` run **`pip install -e .`**.
3. Install this tree’s `seiscomp/` layout into **`$SEISCOMP_ROOT`** (same as stock scmlpick), or run from a test SeisComP prefix.

## Fallback

If **`_spawn_eqcctpro_model_actors`** fails, **`model_actors`** is set to **`[]`** and **`mseed_predictor`** uses the **legacy** path (loads weights inside each picker worker).

## Limits

- This README’s “Limits” section is historical: the integrated **`scmlpick`** and **`predictor`** also support **SeisBench** pools when bindings set **`pickerModel`** (see **`INTEGRATION_GUIDE.md`**). Batch-style **`X_prediction_results.xml`** output remains **`RunEQCCTPro`** only; scmlpick uses **`scPhase`**.
- GPU **fractional** scheduling is simplified (`≤4` actors, `0.95/n` GPU fraction each); tune against your VRAM.
- **`eqcct.gpuLimit`** is passed through as **`gpu_memory_limit_mb`** to **`ModelActor`** when `> 0`; confirm units match your site config.

See also **[INTEGRATION_GUIDE.md](../INTEGRATION_GUIDE.md)** for the full narrative and file map.
