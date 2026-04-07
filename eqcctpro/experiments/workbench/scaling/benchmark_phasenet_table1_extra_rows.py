#!/usr/bin/env python3
"""
PhaseNet-only Supplemental Table 1 rows at 250 and 580 stations.

Matches Table 1 methodology: pre-copied streams, five runs, minimum reported for
Load Time (model init), Annotate-All (single annotate on merged stream),
Classify-Per-Stn (sequential classify on each station stream).

Writes docs/tables/phasenet_table1_250_580.json and prints LaTeX \\midrule lines.
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
DATASETS = [("250_stations_1_min_dt", 250), ("580_stations_1_min_dt", 580)]
N_RUNS = 5
PARENT, CHILD = "PhaseNet", "original"


def _classify_kw():
    return dict(
        P_threshold=0.3,
        S_threshold=0.3,
        Detection_threshold=0.3,
        strict=False,
        flexible_horizontal_components=True,
    )


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


def bench_load(use_gpu: bool) -> float:
    times = []
    for _ in range(N_RUNS):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_gpu else "-1"
        gc.collect()
        cuda_sync()
        t0 = time.perf_counter()
        sb = SeisBenchModels(PARENT, CHILD)
        model = sb.load_model()
        if use_gpu and torch and torch.cuda.is_available():
            model.to(torch.device("cuda"))
            cuda_sync()
        times.append(time.perf_counter() - t0)
        del model, sb
        gc.collect()
        cuda_sync()
    return min(times)


def bench_annotate(merged_stream, use_gpu: bool) -> float:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_gpu else "-1"
    sb = SeisBenchModels(PARENT, CHILD)
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
    del sb
    gc.collect()
    return min(times)


def bench_classify(streams: list, use_gpu: bool) -> float:
    """One warm load; min wall time over N_RUNS full sequential classify passes (Table 1 style)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_gpu else "-1"
    sb = SeisBenchModels(PARENT, CHILD)
    model = sb.load_model()
    if use_gpu and torch and torch.cuda.is_available():
        model.to(torch.device("cuda"))
        cuda_sync()
    times = []
    for _ in range(N_RUNS):
        cuda_sync()
        t0 = time.perf_counter()
        for stream in streams:
            model.classify(stream, **_classify_kw())
        cuda_sync()
        times.append(time.perf_counter() - t0)
    del model, sb
    gc.collect()
    cuda_sync()
    return min(times)


def main():
    out: list[dict] = []
    for dataset_name, n_stations in DATASETS:
        input_dir = REPO / "data" / dataset_name / TIMECHUNK
        if not input_dir.is_dir():
            print(f"SKIP {dataset_name}: missing {input_dir}", flush=True)
            continue
        streams = preload_streams(input_dir, n_stations)
        print(f"\n{dataset_name}: loaded {len(streams)} streams (target {n_stations})", flush=True)
        if len(streams) < n_stations:
            print(f"  WARNING: fewer streams than requested", flush=True)
        merged = merge_streams_stream(streams)

        for use_gpu in (False, True):
            dev = "GPU" if use_gpu else "CPU"
            print(f"  {dev}: load…", flush=True)
            load_s = bench_load(use_gpu)
            print(f"    Load min: {load_s:.3f}", flush=True)
            print(f"  {dev}: annotate…", flush=True)
            ann_s = bench_annotate(merged, use_gpu)
            print(f"    Annotate min: {ann_s:.3f}", flush=True)
            print(f"  {dev}: classify…", flush=True)
            cls_s = bench_classify(streams, use_gpu)
            print(f"    Classify min: {cls_s:.2f}", flush=True)
            out.append(
                {
                    "dataset": dataset_name,
                    "n_stations": n_stations,
                    "device": dev,
                    "load_s": round(load_s, 3),
                    "annotate_all_s": round(ann_s, 3),
                    "classify_per_station_s": round(cls_s, 2),
                }
            )

    out_dir = REPO / "docs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phasenet_table1_250_580.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)

    print("\nLaTeX (insert after PhaseNet GPU 228 row, before PhaseNetLight):\n")
    for row in out:
        gpus = 1 if row["device"] == "GPU" else 0
        print(
            f"PhaseNet      & {row['device']} & {row['n_stations']} & 1 & {gpus} & "
            f"{row['load_s']:.3f} & {row['annotate_all_s']:.3f} & {row['classify_per_station_s']:.2f} \\\\"
        )


if __name__ == "__main__":
    main()
