# `summary_results.ascii` — miniSEED columns and error tags

When **`pick_output_format='ascii'`**, EQCCTPro writes **`summary_results.ascii`** (or **`summary_results_<timechunk_id>.ascii`** for multi-chunk runs). Besides station names, pick counts, and model metadata, each row includes three columns derived from the **driver-time miniSEED preload** for that timechunk:

| Column | Meaning |
|--------|---------|
| **Expected_Header_Samples** | Sum of **declared sample counts** from miniSEED headers (when available) for all `.mseed` files contributing to that station in the chunk. If only some files expose `npts` in the header scan, the cell may note **`partial; k/n files with header npts`**. If no header totals were obtained, **`?`**. |
| **Decoded_Samples** | Sum of **actually decoded** samples across those files after EQCCTPro’s strict and recovery read paths. |
| **MSEED_errors** | **`OK`** if every file decoded cleanly with no recovery tags. Otherwise a **` | `**-separated list of **`tag: human explanation`** entries (and possibly an extra **`decode_failed`** line if any file failed entirely). The machine-readable tag strings match driver logs and **`eqcctpro.pick_output.MSEED_ERROR_TAG_GLOSSARY`**. |

If a row was built without preload statistics (e.g. legacy path), these three cells may show **`not available (no driver preload stats)`**.

## Sidecar glossary file

Next to **`summary_results.ascii`**, the first successful write in that output directory also creates **`mseed_error_tags_reference.txt`** (once per directory), listing each tag and its short explanation.

## Tag reference (`MSEED_errors`)

These short codes show up in **`MSEED_errors`** or in logs. They are the same labels the software uses internally; below is what they mean in everyday terms.

- **`per_channel_longest_prefix`** — On at least one channel, the reader kept as much good data as it could from the start of the channel, but had to stop where the compressed data looked bad or truncated at the end. You usually still get a usable trace; it may not cover the full nominal length.

- **`whole_file_longest_prefix`** — The file could not be read cleanly from beginning to end. EQCCTPro used the longest stretch at the **start** of the file that could be decoded. What you get may be shorter than the length written in the file header.

- **`loose_obspy_read`** — The stricter read paths produced nothing useful, but a broader ObsPy read did return waveform data.

- **`loose_check_compression`** — Data was read only after turning off some compression checks that were blocking a normal read. Use the result knowing the file may be marginal.

- **`decoded_shorter_than_header`** — The number of samples actually read is **smaller** than the count the file header claims. Something stopped the decode early, or the file does not fully match its header.

- **`decode_failed`** — After all normal and backup read attempts, this file still produced **no** usable samples, or it was classified as a hard failure.

Multiple tags can apply to one file. In **`summary_results.ascii`**, one station row lists every tag that appeared on **any** of that station’s files in that chunk.
