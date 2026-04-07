# scmlpick + EQCCTPro ModelActor (reference integration)

This tree is a **working duplicate** of the scmlpick SeisComP module with **minimal changes** that wire **eqcctpro’s `ModelActor`** into the existing Ray `picker` path. It is intended as a **reference** for upstream scmlpick: copy patterns from here rather than maintaining this fork long-term.

## What changed

| Area | Change |
|------|--------|
| **`predictor/predictor.py`** | Drops duplicated Keras EQCCT graph; imports **`eqcctpro.eqcct_tf_models`**. **`mseed_predictor(..., model_actor=, inference_mode=, ripper_gpu_memory_limit_mb=)`**. **`parallel_predict`** calls **`eqcctpro.tools.tf_environ`** when **Ripper + GPU** (per-task VRAM cap). |
| **`seiscomp/bin/scmlpick`** | **`ray.init(num_gpus=len(ray_gpu_device_ids))`** with **`CUDA_VISIBLE_DEVICES`** from **`ray.gpuDeviceIds`** or **`eqcct.gpuID`**. **`ray.inferenceMode`**: **`modelActor`** spawns ModelActors; **`ripper`** skips actors. **`ray.numModelActors`** (0=auto) and **`ray.maxTasksQueue`** cap concurrency. CLI: **`--inference-mode`**, **`--gpu-devices`**, **`--num-model-actors`**. |
| **`etc/descriptions/scmlpick.xml`**, **`etc/defaults/scmlpick.cfg`** | Document **`ray.inferenceMode`**, **`ray.numModelActors`**, **`ray.gpuDeviceIds`**, **`ray.actorGpuMemoryLimitMB`**, **`ray.ripperGpuMemoryLimitMB`**. |

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

- **SeisBench** is not wired in this reference (EQCCT + TensorFlow only).
- GPU **fractional** scheduling is simplified (`≤4` actors, `0.95/n` GPU fraction each); tune against your VRAM.
- **`eqcct.gpuLimit`** is passed through as **`gpu_memory_limit_mb`** to **`ModelActor`** when `> 0`; confirm units match your site config.

See also **[INTEGRATION_GUIDE.md](../INTEGRATION_GUIDE.md)** for the full narrative and file map.
