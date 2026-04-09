# Integrating EQCCTPro into scmlpick

This document explains how **scmlpick** can adopt the same patterns used in **EQCCTPro** (the **RAPID** stack—Resource-Aware Parallel Inference Dispatcher): persistent inference via **Ray actors**, a clean separation of **EQCCT TensorFlow** code, optional **SeisBench** PyTorch pickers, **per-station model selection** in SeisComP bindings, and **shared ObjectRefs** for waveforms—without giving up SeisComP messaging and real-time scheduling.

Nothing below changes what **stock scmlpick** is; it describes the **baseline** as shipped from its own repository, then what **EQCCTPro** brings, and how you can **merge** the two. A concrete **reference implementation**—a full copy of the repo tree with those merges applied—lives in `**[scmlpick-eqcctpro-integrated/](./scmlpick-eqcctpro-integrated/)`** for line-by-line porting or experiments.

---

## 1. What scmlpick does today

**scmlpick** is a SeisComP **StreamApplication**. It ingests waveforms, builds work in terms of network, station, and time window, and uses **Ray** to fan that work out to worker tasks. The heart of the picking pipeline is a remote function traditionally called `**picker`**: it receives a slice of stream data (often via `**ray.put**` so the full stream is not serialized for every task), unpacks parameters, and calls `**mseed_predictor**` from the `**predictor**` package that ships beside scmlpick (the `scmlpick-predicctor` layout).

In **upstream scmlpick**, `**mseed_predictor`** always drives `**parallel_predict**` in the worker process. `**parallel_predict**` loads EQCCT’s TensorFlow weights with `**load_eqcct_model**`, runs `**model.predict**` on a generator, and turns probabilities into pick dictionaries for `**scPhase**`. **There is no persistent ModelActor pool** in the stock design: each `**picker`** invocation that takes that path pays the full model load cost inside the worker, which is simple and stateless but expensive when many windows run back-to-back.

The **predictor** historically carries its **own copy** of large pieces of the EQCCT Keras graph and helpers. That duplicates logic that **EQCCTPro** now maintains in one place, so drift and bugfixes are harder to share until you align on a single import path.

---

## 2. What EQCCTPro (RAPID) adds

**EQCCTPro** is packaged as **RAPID** in `eqcctpro`’s metadata; the codebase is the same. It is meant for **resource-aware** parallel inference: Ray **actors** that hold models open, optional **Ripper** mode that loads per task, and shared helpers for **VRAM/RAM** thinking when you size pools.

At a high level, these pieces matter for scmlpick-style workflows:


| Layer                          | Role                                                                                                                                                                                |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `eqcctpro/eqcct_tf_models.py`  | Canonical EQCCT loading and inference: `**load_eqcct_model`**, `**PreLoadGeneratorTest**`, and related utilities.                                                                   |
| `eqcctpro/seisbench_models.py` | `**SeisBenchModels**` (`from_pretrained`, `**classify**` / `**annotate**`), plus `**mseed2stream_3c**` and `**process_raw_station_stream_3c**` for three-component ObsPy pipelines. |
| `eqcctpro/parallelization.py`  | `**ModelActor**`, `**SeisBenchModelActor**`, Ray helpers such as `**parallel_predict**` / `**parallel_predict_seisbench**`, and Ripper-oriented utilities.                          |
| `eqcctpro/waveform_filter.py` | Shared ObsPy filter helpers: `**resolve_waveform_filter_params**`, `**apply_waveform_filter**` (bandpass / bandstop / lowpass / highpass, corners, zerophase).                        |
| `eqcctpro/pick_output.py`      | Batch pick I/O: `**PickOutputSink**` (per-station XML/CSV rows), run-level **`summary_results.ascii`**, **`merge_ascii_summary_rows`**. `**pick_output_format**` = **`xml`** / **`csv`** / **`ascii`**; with **`ascii`**, **`ascii_station_pick_format`** chooses per-station **`xml`** vs **`csv`**. Not used on the scmlpick **`scPhase`** path. |
| `eqcctpro/functionality.py`    | **`RunEQCCTPro`** / **`EvaluateSystem`**: forwards **`waveform_filter_*`**, **`pick_output_format`**, **`ascii_station_pick_format`**, **`overwrite`** into **`mseed_predictor`**. |


**ModelActor** mode means one process (or a small pool) keeps weights resident; station tasks only ship arrays or preprocessed streams and call `**predict_from_arrays.remote`** (EQCCT) or `**classify.remote**` (SeisBench). **Ripper** means each task loads the model, infers, and tears down—closest to what **stock scmlpick already does** for EQCCT, but with EQCCTPro’s optional GPU memory tooling. The **Ray max-tasks** queue in scmlpick plays the same role as **backpressure** against how many of those tasks run at once, whether or not actors are in use.

**Enabling GPU inference (reference merge).** Stock scmlpick does not wire up CUDA for Ray-backed picking the way this merge does. In `**scmlpick-eqcctpro-integrated`**, we **now enable GPU processing** for both EQCCT (TensorFlow) and SeisBench (PyTorch): you choose which devices are visible, Ray schedules fractional GPU actors, and Ripper tasks can still use the GPU with optional VRAM caps.

---

## 3. SeisBench in scmlpick

**Current scmlpick implementation:** Upstream scmlpick only ever runs EQCCT (TensorFlow) - it is not integrated with SeisBench.

**After a RAPID-style merge:** You can still pick with EQCCT, but you may also pick with **SeisBench** models (PyTorch, via EQCCTPro). The important difference is **where** you choose the backend: not one switch for the whole module, but **per station** in the SeisComP **bindings** (each profile that has picking turned on can name its own model).

**In plain terms:**

- **EQCCT** — same family of models scmlpick already used; thresholds stay `**eqcctPthr`** and `**eqcctSthr**`.
- **SeisBench** — models such as PhaseNet or EQTransformer loaded through SeisBench; they use `**eqcct.detectionThreshold`** as the `**Detection_threshold**` argument to `**classify**`, in addition to the usual P/S probability settings passed through from config.

### Where you configure it (reference tree)


| Setting                                 | Role                                                                                                              |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `**profiles.*.pickerModel**` (binding)  | The model to use for that station profile when `**pickEnable**` is on. Example: `EQCCT` or `PhaseNet/original`.   |
| `**eqcct.defaultPickerModel**` (module) | Used only if a profile does not set `**pickerModel**`. Same style of values as the binding.                       |
| `**eqcct.detectionThreshold**` (module) | SeisBench-only tuning for `**classify**`. EQCCT picking ignores this and keeps `**eqcctPthr**` / `**eqcctSthr**`. |


**How the text you type is interpreted:** The app accepts either `**EQCCT`** or a **family/weights** pair written as `**PhaseNet/original`** (slash). Inside the driver, pairs become an internal label such as `**PhaseNet|original**` (pipe-separated) so Ray can attach the right **pool of SeisBench actors** (one pool per distinct pair). `**EQCCT`** stays a single label.

**Drop-down in scconfig:** In `**scmlpick.xml`**, the reference lists allowed `**pickerModel**` values in a `**values="..."**` attribute so the GUI can show a fixed list. If your SeisComP version does not support that attribute, keep the parameter as a free string and invalid values will fail at runtime with a clear error.

### What each Ray task receives

When `**picker**` runs, it gets a Python `**params**` dict (via `**ray.put**`) that includes, among other things:

- `**station_picker_model**` — a dictionary `**NET.STA` → backend label** built from the bindings (for example `**US.TX11` → `PhaseNet|original`**).
- `**default_picker_model_resolved**` — the fallback label if a station is missing from that map.
- `**Detection_threshold**` — copied from `**eqcct.detectionThreshold**` for SeisBench `**classify**`.
- `**model_actor_pools**` — for ModelActor mode, a dict whose keys are labels like `**EQCCT**` or `**PhaseNet|original**`, and whose values are lists of Ray actor handles for that backend. In Ripper mode this dict is empty and each task loads the model inside the worker instead.

The `**picker**` uses `**NET.STA**` for the current job to look up the label, then calls `**mseed_predictor**` (EQCCT) or `**mseed_predictor_seisbench**` (SeisBench) and passes the matching actor from the right pool when actors are enabled.

### What `**mseed_predictor_seisbench**` does (short path)

1. Resolve the **bandpass** for `**NET.STA`** from `**params["df_filters"]**`: the `**profiles.*.filter**` strings in bindings (`BW(2,hp,lp)`). The helper `**_bandpass_hz_from_scmlpick_bindings**` tries an **exact** match on `**key == NET.STA`**, then a station-code suffix match—so you get what was **entered in bindings**, not a silent generic default when a row exists. `**prepare_station_chunk`** is then called with `**stations_filters=None**` and `**default_band=(fmin, fmax)**` so that single resolved pair is what is applied. Only if there is **no** binding row (or an empty filter list) does the path fall back to broadband **1–45 Hz**, with a **warning** in the log.
2. After merge, taper, that bandpass, and **100 Hz** resampling inside `**prepare_station_chunk`**, assemble **three components** (E, N, Z) in `**_prepared_stream_to_seisbench_3c`**.
3. Run `**classify**` on a `**SeisBenchModelActor**` (ModelActor mode) or `**SeisBenchModels**` in-process (Ripper, with GPU if `**gpu_id` ≥ 0**).
4. Map `**ClassifyOutput`** to the same pick dict layout EQCCT uses (`**_classify_output_to_scmlpick_picks**`) so `**scPhase**` stays unchanged.

---

## 4. Integration strategies (what you should do)

These are three graduated ways to bring EQCCTPro ideas into scmlpick. Pick the one that matches how much risk and churn you can take in one step.

**Strategy A — Depend on `eqcctpro` and add Ray actor pools (recommended for production throughput).**  
You should (1) add `**eqcctpro`** to the `**scmlpick-predicctor**` `install_requires` so TensorFlow, Ray, PyTorch, and SeisBench stay version-aligned with `**eqcctpro/pyproject.toml**`; (2) replace duplicated Keras code in `**predictor.py**` with imports from `**eqcctpro.eqcct_tf_models**`; (3) after `**ray.init**` in the long-lived scmlpick process, spawn **driver-owned** actor pools (one pool for EQCCT `**ModelActor`**, and one pool per distinct SeisBench `**Parent|Child**` choice if you enable SeisBench); (4) pass `**job_idx**` (or similar) into `**picker**` so each task selects `**actors[job_idx % len(actors)]**`; (5) keep `**ray.put**` on the shared stream so you do not reserialize waveforms for every task. The **reference tree** implements this end-to-end.

**Strategy B — Minimal change: import EQCCT from `eqcctpro` only.**  
You should swap `**load_eqcct_model`** / generator usage to `**eqcctpro.eqcct_tf_models**` and delete the in-repo duplicate graph, but **leave** the per-task `**parallel_predict`** behavior unchanged. You get one source of truth and easier upgrades, without yet introducing ModelActors.

**Strategy C — Batch and playback regression via `RunEQCCTPro`.**  
For offline comparisons or benchmarks—not the hot real-time `**picker`** path—you should run `**RunEQCCTPro**` from `**functionality.py**` with aligned **`waveform_filter_*`** settings and the same Hz corners you expect from bindings (or document intentional differences), then diff **`scPhase`**-style results against batch outputs (**`pick_output_format`**, including **`ascii`** + **`ascii_station_pick_format`** when you want both a summary table and per-station XML/CSV) after changes to scmlpick’s predictor or Ray wiring.

---

## 5. Preprocessing consistency

For **EQCCT**, keep `**prepare_station_chunk`** windowing, overlap, and `**filterShift**` aligned with operational scmlpick unless you intentionally benchmark a change. Corner frequencies still come from `**_bandpass_hz_from_scmlpick_bindings**`: match the binding row whose `**key**` equals `**NET.STA**`, else the row whose key ends with the station code, else `**default_band**` (typically **1–45 Hz**).

**Waveform filter parity (scmlpick reference versus EQCCTPro batch).** Both code paths call **`eqcctpro.waveform_filter.resolve_waveform_filter_params`** / **`apply_waveform_filter`** so ObsPy **`Stream.filter`** behavior (type, corners count, zerophase) matches.

| Aspect | scmlpick (reference tree) | **`RunEQCCTPro`** / **`mseed_predictor`** |
|--------|---------------------------|-------------------------------------------|
| Filter **type** | **`eqcct.waveformFilterType`** in cfg/xml | **`waveform_filter_type`** (default **`bandpass`**) |
| **Frequency** corners (Hz) | From binding **`BW(hp,lp)`** / **`default_band`** | **`waveform_filter_freqmin`**, **`waveform_filter_freqmax`** (defaults **1.0** / **45.0**); optional pandas **`stations_filters`** (`sta`, `hp`, `lp`) overrides per station |
| **Corners** / **zerophase** | **`eqcct.waveformFilterCorners`**, **`eqcct.waveformFilterZerophase`** | **`waveform_filter_corners`**, **`waveform_filter_zerophase`** |

The reference **`predictor`** passes **`waveform_filter_*`** from **`params`** (filled in **`seiscomp/bin/scmlpick`** from module config) into **`prepare_station_chunk`** and into EQCCT / SeisBench **`mseed_predictor`** call paths.

For **SeisBench**, the same `**prepare_station_chunk`** pipeline applies **your binding `BW(...)` corners** via `**default_band`** from `**df_filters**`, then `**_prepared_stream_to_seisbench_3c**` only selects E/N/Z—no second filter pass.

**Batch / offline (`RunEQCCTPro` / `EvaluateSystem`):** `**pick_output_format**` is **`xml`**, **`ascii`**, or **`csv`**. For **`xml`** or **`csv`**, each `<station>_outputs/` directory gets **`X_prediction_results.xml`** or **`.csv`** only. For **`ascii`**, the driver always writes a run-level **`summary_results.ascii`** (or per-chunk **`summary_results_<timechunk_id>.ascii`**) **and** per-station detail files: use **`ascii_station_pick_format`** = **`xml`** (default) or **`csv`**. With **`overwrite=False`**, existing per-station files skip that task and the ASCII summary is **merged** by station name; **`overwrite=True`** deletes the summary at job start and workers may replace station outputs. Real-time scmlpick continues to emit picks through **`scPhase`** / SeisComP messaging, not those files.

---

## 6. Dependencies and packaging

You should install **one** Python environment for the SeisComP module, `**scmlpick`**, and `**eqcctpro**`, so you do not mix CUDA, TensorFlow, and PyTorch versions by accident. `**pip install eqcctpro**` (or an editable install from this repo) pulls the pins in `**eqcctpro/pyproject.toml**`, including `**numpy==1.26.4**`, TensorFlow, Ray, PyTorch, and **SeisBench**. The reference `**setup.py`** for `**scmlpick-predicctor**` declares `**eqcctpro**` and the usual ObsPy/Ray/TensorFlow pins; SeisBench enters transitively through `**eqcctpro**`.

---

## 7. Testing checklist

Before you rely on a merged tree in operations, you should: (1) verify `**eqcctpro.eqcct_tf_models.load_eqcct_model**` runs in the same venv as scmlpick; (2) smoke-test one `**ModelActor**` or `**SeisBenchModelActor**` with two overlapping remote calls; (3) run scmlpick **playback** on a short window and compare picks to a known baseline; (4) soak-test GPU memory so actor pools level off after warmup, while Ripper mode shows the expected load-per-task pattern; (5) run `**python3 -m py_compile`** on the modified `**predictor/predictor.py**` and `**seiscomp/bin/scmlpick**` so syntax errors never reach a SeisComP deployment.

---

## 8. Key file map (and where the logic lives)

Use this as a reading order: **file → main functions / classes to search for**.


| Topic                          | File                                                                 | What to read                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------ | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| EQCCT graph + weights          | `eqcctpro/eqcct_tf_models.py`                                        | `**load_eqcct_model`**, `**PreLoadGeneratorTest**`, model building helpers                                                                                                                                                                                                                                                                    |
| SeisBench + 3C IO              | `eqcctpro/seisbench_models.py`                                       | `**SeisBenchModels**`, `**process_raw_station_stream_3c**`, `**mseed2stream_3c**` (use `**waveform_filter_*`** in `args` like core EQCCTPro)                                                                                                                                                                                                    |
| Ray actors (EQCCT + SeisBench) | `eqcctpro/parallelization.py`                                        | `**ModelActor**`, `**SeisBenchModelActor**`, `**parallel_predict_seisbench**`, resource helpers                                                                                                                                                                                                                                               |
| Waveform filter + batch output | `eqcctpro/waveform_filter.py`, `eqcctpro/pick_output.py`              | scmlpick: `**prepare_station_chunk**` + module/binding Hz; batch: `**pick_output_format**`, `**ascii_station_pick_format**`, `**merge_ascii_summary_rows**`, `**overwrite`** in `**RunEQCCTPro**`                                                                                                                                                |
| Predictor (reference)          | `.../scmlpick-predicctor/predictor/predictor.py`                     | `**_bandpass_hz_from_scmlpick_bindings**`, `**prepare_station_chunk**`, `**mseed_predictor**`, `**parallel_predict**`, `**parallel_predict_with_actor**`, `**mseed_predictor_seisbench**`, `**_prepared_stream_to_seisbench_3c**`, `**_classify_output_to_scmlpick_picks**`, `**_readnparray**`, `**_picker**`, `**_output_dict_prediction**` |
| `**scmlpick**` app (reference) | `.../seiscomp/bin/scmlpick`                                          | `**init**` / `**initConfiguration**` (Ray + `**params**`), `**get_filters**`, `**get_station_picker_models**`, `**_normalize_picker_model_label**`, `**_spawn_eqcct_model_actors_only**`, `**_spawn_seisbench_model_actors_only**`, `**_spawn_inference_actor_pools**`, `**picker**`, `**run_picker**`, `**scPhase**`, `**scPick**`           |
| Defaults + XML                 | `.../etc/defaults/scmlpick.cfg`, `.../etc/descriptions/scmlpick.xml` | `**ray.***`, `**eqcct.***`, binding `**pickerModel**`                                                                                                                                                                                                                                                                                         |
| Stock scmlpick (baseline)      | `scmlpick/seiscomp/...`                                              | Compare to reference `**picker**` + `**mseed_predictor**` only (no RAPID pools)                                                                                                                                                                                                                                                               |


Paths prefixed with `**.../**` are under `**merging-model-actors-to-scmlpick/scmlpick-eqcctpro-integrated/seiscomp/**`.

---

## 9. Reference implementation: `scmlpick-eqcctpro-integrated/`

The folder `**[scmlpick-eqcctpro-integrated/](./scmlpick-eqcctpro-integrated/)**` is the **merged** tree: everything in section 1 still describes **upstream** scmlpick; this tree is what you **change toward** when you adopt section 4 Strategy A together with optional SeisBench bindings from section 3. Treat it as the merge candidate: EQCCT `**ModelActor`** pools, optional **Ripper**, `**ray.gpuDeviceIds`** / `**CUDA_VISIBLE_DEVICES**`, SeisBench `**SeisBenchModelActor**` pools keyed by model pair, and binding-driven `**pickerModel**`. After local edits, always `**py_compile**` `**predictor.py**` and `**scmlpick**` before installing under SeisComP.

**Predictor (`predictor/predictor.py`).** Imports `**eqcctpro.eqcct_tf_models`** instead of vendoring the full Keras graph. `**mseed_predictor**` accepts `**model_actor**`, `**inference_mode**`, and `**ripper_gpu_memory_limit_mb**`; with an actor it uses `**parallel_predict_with_actor**`, otherwise `**parallel_predict**` (Ripper uses `**eqcctpro.tools.tf_environ**` when GPU is enabled). `**mseed_predictor_seisbench**` implements the SeisBench path and `**_classify_output_to_scmlpick_picks**` so `**scPhase**` stays unchanged. Core EQCCT picking helpers (`**_readnparray**`, `**_picker**`, `**_output_dict_prediction**`) remain as in operational scmlpick.

**Application shell (`seiscomp/bin/scmlpick`).** Configures Ray from `**ray.gpuDeviceIds`** and `**ray.init(num_gpus=...)**`; builds `**params**` with `**station_picker_model**`, `**default_picker_model_resolved**`, `**Detection_threshold**`, and `**model_actor_pools**`. `**_spawn_eqcct_model_actors_only**` creates the EQCCT `**ModelActor**` list; `**_spawn_seisbench_model_actors_only**` does the same for each SeisBench pair; `**_spawn_inference_actor_pools**` collects every backend key from bindings plus the default picker model and spawns one pool per key. `**picker**` chooses `**mseed_predictor**` vs `**mseed_predictor_seisbench**` from the resolved spec for `**NET.STA**`. `**run_picker**` passes `**job_idx**` for round-robin actor choice.

**Packaging (`scmlpick-predicctor/setup.py`).** Declares `**eqcctpro`** and aligned pins; PyTorch and SeisBench come in via `**eqcctpro**`.

**Configuration (`etc/descriptions/scmlpick.xml`, `etc/defaults/scmlpick.cfg`).** Adds `**ray.inferenceMode**`, `**ray.maxTasksQueue**`, `**ray.numModelActors**`, `**ray.gpuDeviceIds**`, `**ray.actorGpuMemoryLimitMB**`, `**ray.ripperGpuMemoryLimitMB**`, `**eqcct.defaultPickerModel**`, `**eqcct.detectionThreshold**`, `**eqcct.waveformFilterType**`, `**eqcct.waveformFilterCorners**`, `**eqcct.waveformFilterZerophase**`, and binding `**pickerModel**` with optional `**values**` for the GUI.

**Pre-deploy syntax check (paths relative to this repo):**

```bash
python3 -m py_compile \
  merging-model-actors-to-scmlpick/scmlpick-eqcctpro-integrated/seiscomp/share/scmlpick/tools/scmlpick-predicctor/predictor/predictor.py \
  merging-model-actors-to-scmlpick/scmlpick-eqcctpro-integrated/seiscomp/bin/scmlpick
```

Adjust paths if your install tree differs.