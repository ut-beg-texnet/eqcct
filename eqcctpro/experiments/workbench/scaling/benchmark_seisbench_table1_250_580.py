#!/usr/bin/env python3
"""
Supplemental Table 1 scaling: 228, 250, and 580 stations, all SeisBench models in Table 1.

Same methodology:
- Pre-copied streams from data/{230,250,580}_stations_*_min_dt/<TIMECHUNK>
  (228 stations use the first 228 directories from 230_stations_1_min_dt)
- Five runs, minimum for: Load, Annotate-All (merged stream), Classify (sequential per station)
- PhaseNet classify() uses threshold kwargs; other models use default classify kwargs

Classify timing: the model is loaded once; the timed loop is ``for stream in streams: model.classify(stream)``.
There is no per-station model reload. SeisBench ``classify`` runs ``annotate`` internally per call; full
PhaseNet is much heavier on CPU than PhaseNetLight (GPU times are closer). The JSON stores
``classify_total_s`` (full pass) and ``classify_per_station_mean_s`` (total / n_stations).

Writes docs/tables/seisbench_table1_scaling_228_250_580.json (checkpointed after each row).
"""
from __future__ import annotations

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
# (dataset_dir_name, n_stations to load) — 228 vs 250 should be close; 250 > 228 for classify totals
DATASETS = [
    ("230_stations_1_min_dt", 228),
    ("250_stations_1_min_dt", 250),
    ("580_stations_1_min_dt", 580),
]
N_RUNS = 5

# (parent, child, short label for JSON/LaTeX)
MODEL_SPECS = [
    ("PhaseNet", "original", "PhaseNet"),
    ("PhaseNetLight", "stead", "PhaseNetLight"),
    ("EQTransformer", "original", "EQTransformer"),
    ("EQTransformer", "original_nonconservative", "EQT-NC"),
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


def preload_streams(input_dir: Path, n_stations: int):
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
            print(f"  Skip {sta}: {e}", flush=True)
    return streams


def merge_streams_stream(streams: list) -> object:
    from obspy import Stream as ObsStream

    merged = ObsStream()
    for s in streams:
        merged += s
    return merged


def bench_load(parent: str, child: str, use_gpu: bool) -> float:
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


def bench_annotate(merged_stream, parent: str, child: str, use_gpu: bool) -> float:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_gpu else "-1"
    sb = SeisBenchModels(parent, child)
    model = sb.load_model()
    if use_gpu and torch and torch.cuda.is_available():
        model.to(torch.device("cuda"))
        cuda_sync()
    times = []
    for _ in range(N_RUNS):
        cuda_sync()
        t0 = time.perf_counter()
        model.annotate(merged_stream.copy())
        cuda_sync()
        times.append(time.perf_counter() - t0)
    del model, sb
    gc.collect()
    cuda_sync()
    return min(times)


def bench_classify(streams: list, parent: str, child: str, use_gpu: bool) -> tuple[float, float]:
    """Load model once; warmup one full pass; min over N_RUNS totals (seconds for all stations).

    Returns (classify_total_s, classify_per_station_mean_s).
    """
    kw = _classify_kw(parent)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_gpu else "-1"
    sb = SeisBenchModels(parent, child)
    model = sb.load_model()
    if use_gpu and torch and torch.cuda.is_available():
        model.to(torch.device("cuda"))
        cuda_sync()
    n = max(len(streams), 1)

    # Warmup: one full sequential pass (not timed) so cold-start is not conflated with model order.
    for stream in streams:
        model.classify(stream, **kw)
    cuda_sync()

    times = []
    for _ in range(N_RUNS):
        cuda_sync()
        t0 = time.perf_counter()
        for stream in streams:
            model.classify(stream, **kw)
        cuda_sync()
        times.append(time.perf_counter() - t0)
    del model, sb
    gc.collect()
    cuda_sync()
    total_min = min(times)
    return total_min, total_min / n


def _write_json(out_path: Path, rows: list) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))


def _row_key(row: dict) -> tuple:
    return (row["dataset"], row["n_stations"], row["model"], row["device"])


def main():
    out_path = REPO / "docs" / "tables" / "seisbench_table1_scaling_228_250_580.json"
    out: list[dict] = []
    completed: set[tuple] = set()

    if out_path.is_file():
        try:
            out = json.loads(out_path.read_text())
            completed = {_row_key(r) for r in out}
            print(
                f"Resume: loaded {len(out)} rows from {out_path}; skipping duplicates.",
                flush=True,
            )
        except (json.JSONDecodeError, OSError) as e:
            print(f"Could not read {out_path} ({e}); starting fresh.", flush=True)

    gpu_ok = bool(torch and torch.cuda.is_available())
    if not gpu_ok:
        print("WARNING: CUDA not available — skipping GPU rows.", flush=True)

    for dataset_name, n_stations in DATASETS:
        input_dir = REPO / "data" / dataset_name / TIMECHUNK
        if not input_dir.is_dir():
            print(f"SKIP {dataset_name}: missing {input_dir}", flush=True)
            continue

        streams = preload_streams(input_dir, n_stations)
        print(f"\n{dataset_name}: loaded {len(streams)} streams (target {n_stations})", flush=True)
        if len(streams) < n_stations:
            print("  WARNING: fewer streams than requested", flush=True)
        merged = merge_streams_stream(streams)

        for parent, child, label in MODEL_SPECS:
            for use_gpu in (False, True):
                if use_gpu and not gpu_ok:
                    continue
                dev = "GPU" if use_gpu else "CPU"
                skip_key = (dataset_name, n_stations, label, dev)
                if skip_key in completed:
                    print(f"\n  {label} {dev}: (skip — already in checkpoint)", flush=True)
                    continue

                print(f"\n  {label} {dev}:", flush=True)
                print(f"    load…", flush=True)
                load_s = bench_load(parent, child, use_gpu)
                print(f"      Load min: {load_s:.3f}", flush=True)
                print(f"    annotate…", flush=True)
                ann_s = bench_annotate(merged, parent, child, use_gpu)
                print(f"      Annotate min: {ann_s:.3f}", flush=True)
                print(f"    classify…", flush=True)
                cls_total, cls_per_stn = bench_classify(streams, parent, child, use_gpu)
                print(
                    f"      Classify min (total / mean per stn): {cls_total:.2f} s / {cls_per_stn:.4f} s",
                    flush=True,
                )

                row = {
                    "dataset": dataset_name,
                    "n_stations": n_stations,
                    "model": label,
                    "parent": parent,
                    "child": child,
                    "device": dev,
                    "load_s": round(load_s, 3),
                    "annotate_all_s": round(ann_s, 3),
                    "classify_total_s": round(cls_total, 2),
                    "classify_per_station_mean_s": round(cls_per_stn, 4),
                }
                out.append(row)
                completed.add(skip_key)
                _write_json(out_path, out)
                print(f"  (checkpoint) wrote {out_path}", flush=True)

    print(f"\nDone. Wrote {out_path}", flush=True)

    # Sanity: 250-station classify total should exceed 228 (more stations)
    by_key: dict[tuple, dict] = {}
    for row in out:
        k = (row["model"], row["device"])
        total = row.get("classify_total_s")
        if total is None:
            total = row.get("classify_per_station_s")  # legacy misnamed field = total
        if total is not None:
            by_key.setdefault(k, {})[row["n_stations"]] = total
    print("\nClassify total time check (250 should be > 228 when both exist):", flush=True)
    for k, d in sorted(by_key.items()):
        if 228 in d and 250 in d:
            a, b = d[228], d[250]
            ok = "OK" if b >= a * 0.98 else "CHECK"  # allow tiny noise
            print(f"  {k[0]} {k[1]}: 228={a:.2f} 250={b:.2f}  ratio={b/a:.3f}  {ok}", flush=True)


if __name__ == "__main__":
    main()
