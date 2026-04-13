#!/usr/bin/env python3
"""
Empirical serial spot-check: model load, annotate-all (merged stream), and sequential classify.

Per architecture and device:
  - Load time: min wall time over N_RUNS full load cycles (weights + optional .to(cuda)), same as Table~1.
  - **Two separate model sessions** (each load → sweep N → delete) so metrics do not interact:
    1) **Classify-only** — same as the legacy spotcheck: for each N, min time over N_RUNS full sequential
       ``classify()`` passes over ``streams[:N]`` (no annotate beforehand).
    2) **Annotate-all only** — fresh load, for each N merged ``streams[:N]``, min over N_RUNS ``annotate()``.

  Running annotate on the merged stream immediately before timing classify inflates classify times badly on
  CPU (thread oversubscription and cache pressure from back-to-back heavy inference). Background load
  alone rarely explains an order-of-magnitude gap; isolating passes fixes comparability with older JSON.

Datasets (same TIMECHUNK under data/<dataset>/):
  - 230_stations_1_min_dt — up to 228 stations (legacy classify curve for fig7)
  - 250_stations_1_min_dt — up to 250 stations
  - 580_stations_1_min_dt — extended station grid up to 580

Outputs:
  - docs/tables/serial_classify_spotcheck.json — format_version 3 (load_s + per-N annotate + classify)
  - docs/tables/serial_classify_spotcheck_cpu.json — legacy: classify-only flat models (230, CPU) for fig7
  - docs/tables/serial_classify_spotcheck_gpu.json — same for 230, GPU

Checkpointing: existing ``docs/tables/serial_classify_spotcheck.json`` is loaded when present. Each
(device, dataset, model) block is **skipped** if it already has ``load_s``, ``annotate_all_s``, and
``classify_total_s`` with entries for every requested station count. Use ``--rerun`` to ignore the
checkpoint and re-benchmark everything in the selected device/dataset loop.

Usage (from eqcctpro repo root):
  python experiments/workbench/scaling/benchmark_serial_classify_spotcheck.py
  python experiments/workbench/scaling/benchmark_serial_classify_spotcheck.py --devices cpu
  python experiments/workbench/scaling/benchmark_serial_classify_spotcheck.py --devices gpu
  python experiments/workbench/scaling/benchmark_serial_classify_spotcheck.py --rerun
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Selected high corner frequency")
warnings.filterwarnings("ignore", message="download precheck failed")
os.environ["SEISBENCH_LOG_LEVEL"] = "ERROR"

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO))

from eqcctpro.tools import build_station_list_from_dir  # noqa: E402
from eqcctpro.seisbench_models import SeisBenchModels, mseed2stream_3c  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

TIMECHUNK = "20241215T120000Z_20241215T120100Z"
N_RUNS = 5
FORMAT_VERSION = 3

# (parent, child, json_label) — same four SeisBench models as Fig. 7/8 serial table
MODEL_SPECS = [
    ("PhaseNet", "original", "PhaseNet"),
    ("PhaseNetLight", "stead", "PhaseNetLight"),
    ("EQTransformer", "original", "EQTransformer"),
    ("EQTransformer", "original_nonconservative", "EQT-NC"),
]

# Grid: original paper curve, then 250; 580 folder gets extra points to 580.
_COUNTS_CORE = [10, 20, 30] + list(range(40, 221, 10)) + [228, 250]
_COUNTS_580_TAIL = list(range(260, 571, 10)) + [580]

# (dataset_dir_name, max_stations_cap, station_counts)
DATASET_SPECS: list[tuple[str, int, list[int]]] = [
    ("230_stations_1_min_dt", 228, [n for n in _COUNTS_CORE if n <= 228]),
    ("250_stations_1_min_dt", 250, _COUNTS_CORE),
    ("580_stations_1_min_dt", 580, _COUNTS_CORE + _COUNTS_580_TAIL),
]


def _classify_kw(parent: str) -> dict:
    if parent == "PhaseNet":
        return dict(
            P_threshold=0.3,
            S_threshold=0.3,
            Detection_threshold=0.3,
            strict=False,
            flexible_horizontal_components=True,
        )
    return {}


def cuda_sync():
    if torch and torch.cuda.is_available():
        torch.cuda.synchronize()


def merge_streams_stream(streams: list) -> object:
    from obspy import Stream as ObsStream

    merged = ObsStream()
    for s in streams:
        merged += s
    return merged


def preload_streams(input_dir: Path, n_stations: int) -> list:
    stations = build_station_list_from_dir(str(input_dir))[:n_stations]
    streams = []
    for sta in stations:
        files = glob.glob(str(input_dir / sta / "*mseed"))
        if not files:
            continue
        try:
            stream, _, _ = mseed2stream_3c({}, files, sta)
            streams.append(stream)
        except Exception as e:
            print(f"  skip {sta}: {e}", flush=True)
    return streams


def bench_load(parent: str, child: str, use_gpu: bool) -> float:
    """Min over N_RUNS full load (+ cuda) cycles, model deleted each time (Table~1 style)."""
    times = []
    for _ in range(N_RUNS):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_gpu else "-1"
        gc.collect()
        cuda_sync()
        t0 = time.perf_counter()
        sb = SeisBenchModels(parent, child)
        model = sb.load_model()
        if use_gpu and torch and torch.cuda.is_available():
            model.to(torch.device("cuda"))
            cuda_sync()
        times.append(time.perf_counter() - t0)
        del model, sb
        gc.collect()
        cuda_sync()
    return min(times)


def _session_classify_only(
    all_streams: list,
    station_counts: list[int],
    parent: str,
    child: str,
    use_gpu: bool,
) -> dict[str, float]:
    """One load; for each N, min over N_RUNS sequential classify totals (legacy spotcheck semantics)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_gpu else "-1"
    kw = _classify_kw(parent)
    sb = SeisBenchModels(parent, child)
    model = sb.load_model()
    if use_gpu and torch and torch.cuda.is_available():
        model.to(torch.device("cuda"))
    cuda_sync()

    out: dict[str, float] = {}
    for n in station_counts:
        if n > len(all_streams):
            continue
        subset = all_streams[:n]
        print(f"  N={n} classify (sequential only)…", flush=True)
        times = []
        for _ in range(N_RUNS):
            cuda_sync()
            t0 = time.perf_counter()
            for stream in subset:
                model.classify(stream, **kw)
            cuda_sync()
            times.append(time.perf_counter() - t0)
        tmin = min(times)
        out[str(n)] = round(tmin, 3)
        print(f"    classify min: {tmin:.3f} s", flush=True)

    del model, sb
    gc.collect()
    cuda_sync()
    return out


def _session_annotate_only(
    all_streams: list,
    station_counts: list[int],
    parent: str,
    child: str,
    use_gpu: bool,
) -> dict[str, float]:
    """Fresh load; for each N, merged streams[:N] annotate min over N_RUNS (no classify)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_gpu else "-1"
    sb = SeisBenchModels(parent, child)
    model = sb.load_model()
    if use_gpu and torch and torch.cuda.is_available():
        model.to(torch.device("cuda"))
    cuda_sync()

    out: dict[str, float] = {}
    for n in station_counts:
        if n > len(all_streams):
            continue
        merged = merge_streams_stream(all_streams[:n])
        print(f"  N={n} annotate-all…", flush=True)
        times = []
        for _ in range(N_RUNS):
            cuda_sync()
            t0 = time.perf_counter()
            model.annotate(merged.copy())
            cuda_sync()
            times.append(time.perf_counter() - t0)
        tmin = min(times)
        out[str(n)] = round(tmin, 3)
        print(f"    annotate min: {tmin:.3f} s", flush=True)

    del model, sb
    gc.collect()
    cuda_sync()
    return out


def bench_model_session(
    all_streams: list,
    station_counts: list[int],
    parent: str,
    child: str,
    use_gpu: bool,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """bench_load; then classify-only session; then separate annotate-only session (no cross-contamination)."""
    print("  bench load (N_RUNS fresh loads)…", flush=True)
    load_s = bench_load(parent, child, use_gpu)
    print(f"    load min: {load_s:.3f} s", flush=True)

    print("  session A: classify-only (legacy spotcheck)…", flush=True)
    classify_series = _session_classify_only(
        all_streams, station_counts, parent, child, use_gpu
    )
    print("  session B: annotate-all only (fresh load)…", flush=True)
    annotate_series = _session_annotate_only(
        all_streams, station_counts, parent, child, use_gpu
    )
    return load_s, annotate_series, classify_series


def _spotcheck_entry_complete(entry: dict | None, need_counts: list[int]) -> bool:
    """True if v3 entry has load_s and full annotate/classify series for all need_counts."""
    if not isinstance(entry, dict):
        return False
    if "load_s" not in entry:
        return False
    ann = entry.get("annotate_all_s")
    cls = entry.get("classify_total_s")
    if not isinstance(ann, dict) or not isinstance(cls, dict):
        return False
    need = {str(n) for n in need_counts}
    if not need.issubset(ann.keys()) or not need.issubset(cls.keys()):
        return False
    return True


def _write_legacy_files(
    payload: dict,
    out_legacy_cpu: Path,
    out_legacy_gpu: Path,
) -> None:
    """Refresh legacy fig7 JSONs from payload when 230-station blocks exist."""
    res = payload.get("results") or {}
    if "CPU" in res and "230_stations_1_min_dt" in res["CPU"]:
        ds = res["CPU"]["230_stations_1_min_dt"]
        legacy = {
            "device": "CPU",
            "timechunk": payload.get("timechunk", TIMECHUNK),
            "dataset": "230_stations_1_min_dt",
            "n_runs": payload.get("n_runs", N_RUNS),
            "station_counts_requested": ds.get("station_counts_requested") or [],
            "models": _legacy_models_flat(ds.get("models") or {}),
        }
        out_legacy_cpu.write_text(json.dumps(legacy, indent=2))
        print(f"  (legacy) wrote {out_legacy_cpu}", flush=True)
    if "GPU" in res and "230_stations_1_min_dt" in res["GPU"]:
        ds = res["GPU"]["230_stations_1_min_dt"]
        legacy_g = {
            "device": "GPU",
            "timechunk": payload.get("timechunk", TIMECHUNK),
            "dataset": "230_stations_1_min_dt",
            "n_runs": payload.get("n_runs", N_RUNS),
            "station_counts_requested": ds.get("station_counts_requested") or [],
            "models": _legacy_models_flat(ds.get("models") or {}),
        }
        out_legacy_gpu.write_text(json.dumps(legacy_g, indent=2))
        print(f"  (legacy) wrote {out_legacy_gpu}", flush=True)


def _legacy_models_flat(models_block: dict) -> dict:
    """Fig. 7 legacy file: model_name -> station_str -> classify seconds only."""
    out = {}
    for name, entry in models_block.items():
        if isinstance(entry, dict) and "classify_total_s" in entry:
            out[name] = entry["classify_total_s"]
        elif isinstance(entry, dict) and "load_s" not in entry:
            # v2 flat classify-only
            out[name] = entry
        else:
            out[name] = entry.get("classify_total_s", entry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--devices",
        default="cpu,gpu",
        help="Comma-separated: cpu, gpu, or both (default: cpu,gpu)",
    )
    ap.add_argument(
        "--rerun",
        action="store_true",
        help="Ignore completed checkpoints and re-run all (device × dataset × model) trials",
    )
    args = ap.parse_args()
    want_cpu = "cpu" in {x.strip().lower() for x in args.devices.split(",")}
    want_gpu = "gpu" in {x.strip().lower() for x in args.devices.split(",")}
    gpu_ok = bool(torch and torch.cuda.is_available())
    if want_gpu and not gpu_ok:
        print("WARNING: CUDA not available — skipping GPU runs.", flush=True)
        want_gpu = False

    out_full = REPO / "docs" / "tables" / "serial_classify_spotcheck.json"
    out_legacy_cpu = REPO / "docs" / "tables" / "serial_classify_spotcheck_cpu.json"
    out_legacy_gpu = REPO / "docs" / "tables" / "serial_classify_spotcheck_gpu.json"
    out_full.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "format_version": FORMAT_VERSION,
        "timechunk": TIMECHUNK,
        "n_runs": N_RUNS,
        "results": {},
    }
    if out_full.is_file():
        try:
            prev = json.loads(out_full.read_text())
            if prev.get("format_version") in (2, 3) and isinstance(prev.get("results"), dict):
                payload["results"] = prev["results"]
                print(f"Loaded existing {out_full} (merge mode).", flush=True)
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    devices: list[tuple[str, bool]] = []
    if want_cpu:
        devices.append(("CPU", False))
    if want_gpu:
        devices.append(("GPU", True))

    for dataset_name, max_cap, station_counts in DATASET_SPECS:
        input_dir = REPO / "data" / dataset_name / TIMECHUNK
        if not input_dir.is_dir():
            print(f"\nSKIP dataset {dataset_name}: missing {input_dir}", flush=True)
            continue

        print(f"\n=== Dataset {dataset_name} (load up to {max_cap} streams) ===", flush=True)
        all_streams = preload_streams(input_dir, max_cap)
        print(f"  got {len(all_streams)} streams (cap {max_cap})", flush=True)
        if len(all_streams) < max_cap:
            print(
                f"  WARNING: fewer streams than cap; high-N rows may be missing",
                flush=True,
            )

        counts = [n for n in station_counts if n <= len(all_streams)]

        for dev_label, use_gpu in devices:
            payload["results"].setdefault(dev_label, {})
            if not counts:
                continue
            ds_block = payload["results"][dev_label].setdefault(
                dataset_name,
                {"station_counts_requested": counts, "models": {}},
            )
            ds_block["station_counts_requested"] = counts
            ds_block.setdefault("models", {})

            for parent, child, label in MODEL_SPECS:
                existing = ds_block["models"].get(label)
                if not args.rerun and _spotcheck_entry_complete(existing, counts):
                    print(
                        f"\n--- {dataset_name} | {dev_label} | {label} "
                        f"({parent}/{child}) — skip (already complete; use --rerun) ---",
                        flush=True,
                    )
                    continue

                print(
                    f"\n--- {dataset_name} | {dev_label} | {label} ({parent}/{child}) ---",
                    flush=True,
                )
                load_s, ann_s, cls_s = bench_model_session(
                    all_streams, counts, parent, child, use_gpu
                )
                ds_block["models"][label] = {
                    "load_s": round(load_s, 3),
                    "annotate_all_s": ann_s,
                    "classify_total_s": cls_s,
                }

                out_full.write_text(json.dumps(payload, indent=2))
                print(f"  (checkpoint) wrote {out_full}", flush=True)

    _write_legacy_files(payload, out_legacy_cpu, out_legacy_gpu)

    print(f"\nDone. Full results: {out_full}", flush=True)


if __name__ == "__main__":
    main()
