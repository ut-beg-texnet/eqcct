#!/usr/bin/env python3
"""Create 250 and 580 station datasets by duplicating from 230_stations_1_min_dt."""
import os
import shutil

BASE = "/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro"
SRC = os.path.join(BASE, "data/230_stations_1_min_dt/20241215T120000Z_20241215T120100Z")


def create_dataset(target_n, name):
    stations = sorted([d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d))])
    n_src = len(stations)
    n_need = target_n - n_src
    out = os.path.join(BASE, "data", name, "20241215T120000Z_20241215T120100Z")
    os.makedirs(out, exist_ok=True)
    for sta in stations:
        dst = os.path.join(out, sta)
        if not os.path.exists(dst):
            shutil.copytree(os.path.join(SRC, sta), dst)
    for i in range(n_need):
        src_sta = stations[i % n_src]
        dup_name = f"{src_sta}_dup{i}"
        shutil.copytree(os.path.join(SRC, src_sta), os.path.join(out, dup_name))
    print(f"Created {name}: {len(os.listdir(out))} stations")


if __name__ == "__main__":
    create_dataset(250, "250_stations_1_min_dt")
    create_dataset(580, "580_stations_1_min_dt")
