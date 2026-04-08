# Merging EQCCTPro ModelActors into scmlpick

This folder holds **integration documentation** plus a **reference duplicate** of scmlpick with EQCCTPro **`ModelActor`** wiring.

## Contents

| Path | Purpose |
|------|---------|
| [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) | Architecture, strategies, dependencies, testing — includes **§10** listing every code change |
| [scmlpick-eqcctpro-integrated/](./scmlpick-eqcctpro-integrated/) | Runnable **scmlpick + eqcctpro** tree ([README](./scmlpick-eqcctpro-integrated/README.md)) |

## Quick orientation

- **EQCCT** lives in **`eqcctpro/eqcct_tf_models.py`**; **`eqcctpro.parallelization.ModelActor`** loads it once per Ray actor.
- **Waveform filtering** in the reference duplicate uses **`eqcctpro.waveform_filter`** in **`prepare_station_chunk`**, with **`eqcct.waveformFilter*`** module parameters (aligned with **`RunEQCCTPro`** / **`mseed_predictor`**). Batch **XML/ASCII/CSV** pick files are still **`RunEQCCTPro`** only (`**eqcctpro.pick_output**`).
- The **integrated duplicate** keeps scmlpick preprocessing (`_readnparray`) and swaps inference to **`predict_from_arrays`** when **`model_actors`** is non-empty in **`params`**.

Start with [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md), then diff against [scmlpick-eqcctpro-integrated/](./scmlpick-eqcctpro-integrated/) for line-level patches.
