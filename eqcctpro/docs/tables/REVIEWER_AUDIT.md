# Reviewer audit — `EQCCTPro_Draft.md` vs data and code

Companion to `FIGURE_VS_TABLES_CHECKLIST.md`. Numerical sources: `seisbench_table1_scaling_228_250_580.json`, `serial_classify_spotcheck.json`, `table3_ripper_228_stations.csv`, `table4_modelactor_228.csv`, `table5_modelactor_memory.csv`, `table1_memory_requirements.csv`, `paper_runtime_raw_dict` / figure scripts.

## Statement | Code / data | Verdict

| Statement in paper | Code output / source | Verdict |
|--------------------|-------------------------|---------|
| Offline `annotate()` ~**0.20–1.22 s** @228 merged stations (§1, §3.1) | `annotate_all_s` @228: **0.199–1.219 s** (Table 1 / JSON) | **Correct** (rounding) |
| Total sequential `classify()` @228 **~1.4–60 s** (CPU and GPU) | Totals in JSON: **1.38–59.57 s** | **Correct** |
| PhaseNet **~0.84 s/stn** @250, **~0.23 s/stn** @580 (§1) | 210.31/250 = **0.841**, 131.33/580 = **0.226** | **Correct** |
| Warm-cache **load ~1.16–1.24 s** (§3.1) | `load_s`: **1.163–1.238** | **Correct** |
| **50–87%** reduction vs best Ripper @228 per model–hardware pair (Abstract, §1, §5) | e.g. EQCCT GPU **87.0%**, EQT-NC CPU **~50%** (Tables 4–5) | **Correct** |
| CPU Ripper **34.3–76.1 s**; GPU Ripper **52.5–127.4 s** (§3.2) | Table 4: **34.25–76.07**, **52.48–127.35** | **Correct** |
| PyTorch CPU Ripper **~34–38.4 s**; GPU **52.5–67.6 s** (excl. EQCCT) | Table 4 rows | **Correct** |
| PhaseNet GPU Ripper **~64%** slower than CPU Ripper | (56.27−34.25)/34.25 = **64.3%** | **Correct** |
| Model-Actor **10.97–25.01 s**; setup **4.6–12.9 s**; OH **37–73%** (§3.3) | Table 5 | **Correct** |
| EQCCT MA **12.47 s** vs GPU Ripper **96.10 s** ≈ **87%** faster (§3.3) | (96.10−12.47)/96.10 = **87.0%** | **Correct** |
| Dataset window **15 Dec 2024**; chunk `20241215…` (§2.1) | Benchmarks / JSON `timechunk` | **Correct** |
| Station grid **5…228** for Figs 7–8; Fig. 4 uses **10…220, 228** (§2.6) | `generate_fig7…`, `generate_fig4…` | **Correct** |
| Embedded **Tables 1–6** vs CSV/JSON | `verify_figures_vs_tables.py` | **Correct** |
| **Fig. 5** bars vs trials | Checklist: all **yes** | **Correct** |
| **Fig. 6** bars vs `peak_memory_measured.json` | Checklist table; differs from Table 6 by design | **Correct** (legend: load-hold benchmark) |
| **Figs 7–8** serial from Table 2 / JSON; Ripper/MA = mean-min model per panel | Script + checklist | **Correct** |
| GPU vs CPU sequential “**order of magnitude**” for all models (§3.1, *before fix*) | Table 1: not true for EQTransformer / EQT-NC / PhaseNetLight ratios | **Was incorrect** — text updated |
| Table 5 **Setup OH** “difference vs Ripper runtime” (*before fix*) | Column is **Setup/Total × 100%** (MA trial only) | **Was incorrect** — caption fixed |
| Ripper VRAM scaling “§2.5” (*before fix*) | Factors stated in **§2.4.2** | **Was incorrect** — cross-ref fixed |

## Plots vs raw data (summary)

- **Fig. 4:** Subsamples trials to stations 10, 20, …, 220, 228; consistent with script.
- **Fig. 5:** Matches `paper_runtime_raw_dict` ↔ Tables 4–5 (checklist).
- **Fig. 6:** `peak_memory_measured.json`; not interchangeable with Table 6.
- **Figs 7–8:** Serial curves from spot-check JSON + interpolation; parallel curves from trial CSVs and mean-based model selection—see figure captions.

## Residual limitations (not errors)

- **Abstract “50–87%”** is a **range across pairings**, not a single paired experiment; wording already ties to “same model–hardware pairings.”
- **Operational narrative** (22 stations down, etc.) is plausible but not independently verified from files in-repo.
