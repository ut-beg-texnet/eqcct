# Merging EQCCTPro ModelActors into scmlpick

This folder holds **integration documentation** plus a **reference duplicate** of scmlpick with EQCCTPro **`ModelActor`** wiring.

## Contents

| Path | Purpose |
|------|---------|
| [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) | Architecture, strategies, dependencies, testing, preprocessing parity notes |
| [scmlpick-eqcctpro-integrated/](./scmlpick-eqcctpro-integrated/) | Runnable **scmlpick + eqcctpro** tree ([README](./scmlpick-eqcctpro-integrated/README.md)) |

## Quick orientation

- **EQCCT** lives in **`eqcctpro/eqcct_tf_models.py`**; **`eqcctpro.parallelization.ModelActor`** loads it once per Ray actor.
- **Waveform filtering** in the reference duplicate uses **`eqcctpro.waveform_filter`** (`**apply_waveform_filter`**, **`resolve_waveform_filter_params`**) inside **`prepare_station_chunk`**. Module keys **`eqcct.waveformFilterType`**, **`eqcct.waveformFilterCorners`**, **`eqcct.waveformFilterZerophase`** match **`RunEQCCTPro`** / **`mseed_predictor`** worker **`waveform_filter_*`**; corner **frequencies** still come from SeisComP binding **`BW(hp,lp)`** (or the SeisBench **`default_band`** fallback), while batch **`RunEQCCTPro`** uses **`waveform_filter_freqmin`** / **`waveform_filter_freqmax`** (and optional **`stations_filters`**) for Hz corners.
- **Batch pick files** (**`RunEQCCTPro`** / **`EvaluateSystem`** only, via **`eqcctpro.pick_output`**): **`pick_output_format`** is **`xml`**, **`csv`**, or **`ascii`**. **`ascii`** writes a run-level **`summary_results.ascii`** (and **`summary_results_<chunk>.ascii`** when multiple timechunks) **and** per-station **`X_prediction_results.xml`** or **`.csv`** controlled by **`ascii_station_pick_format`** (**`xml`** default, **`csv`** optional). With **`overwrite=False`**, skipped stations keep existing files and the ASCII summary is **merged** by station; **`overwrite=True`** removes the summary at job start and workers may replace per-station outputs. Real-time scmlpick still emits **`scPhase`** only.
- The **integrated duplicate** keeps scmlpick preprocessing (`_readnparray`) and swaps inference to **`predict_from_arrays`** when **`model_actors`** is non-empty in **`params`**.

Start with [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md), then diff against [scmlpick-eqcctpro-integrated/](./scmlpick-eqcctpro-integrated/) for line-level patches.
