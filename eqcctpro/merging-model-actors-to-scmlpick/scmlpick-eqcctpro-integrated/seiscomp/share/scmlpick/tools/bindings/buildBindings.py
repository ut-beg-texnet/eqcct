#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import csv, os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Union
from obspy.clients.fdsn import Client

FDSN_URL = "http://scarchive.beg.utexas.edu:8880"

PROFILES = {
    "DBW": dict(minlat=31.0, maxlat=32.0, minlon=-105.1, maxlon=-103.8),
#    "DBN": dict(minlat=31.5, maxlat=32.5, minlon=-104.4, maxlon=-102.8),
#    "DBS": dict(minlat=30.2, maxlat=31.9, minlon=-104.0, maxlon=-102.6)
}

STA2ADD_PER_PROFILE: Dict[str, List[str]] = {
    "DBW": ["TX_VHRN"],
#    "DBN": ["TX_ODSA", "TX_MB25", "TX_MB26", "4O_BB01", "TX_MNHN"],
#    "DBS": ["TX_VHRN"]
}

FILTERS_PROFILE_FILES: Dict[str, str] = {
    "DBW": "delaware/dbw/stations_filters_max.inp",
#    "DBN": "delaware/dbn/stations_filters_max.inp",
#    "DBS": "delaware/dbs/stations_filters_max.inp"
}

# -------------------------------------------------------------------
# OPTIONAL: Filters for LIST-based profiles (DELN/DELS/DELC/EF/MID...)
# You can provide ONE file (str) or MANY files (list[str]) per profile.
#
# Format expected per line (CSV):
#   NET_STA,LOW,HIGH
# Example:
#   4O_BP01,1,45
#   TX_VHRN,0.8,20
#
# Paths can be relative (recommended: relative to script directory).
# If multiple files are provided for the same profile, later files override
# earlier ones for duplicated stations.
# -------------------------------------------------------------------
LIST_FILTERS_PROFILE_FILES: Dict[str, Union[str, List[str]]] = {
    # "DELN": [
    #     "filters/delaware/dbn/stations_filters_max.inp",
    #     "filters/delaware/dbs/stations_filters_max.inp",
    #     "filters/delaware/dbw/stations_filters_max.inp",
    # ],
    # "DELS": [
    #     "filters/delaware/dbn/stations_filters_max.inp",
    #     "filters/delaware/dbs/stations_filters_max.inp",
    #     "filters/delaware/dbw/stations_filters_max.inp",
    # ],
    # "DELC": [
    #     "filters/delaware/dbn/stations_filters_max.inp",
    #     "filters/delaware/dbs/stations_filters_max.inp",
    #     "filters/delaware/dbw/stations_filters_max.inp",
    # ],
    "MID":  "filters/midland/stations_filters_max.inp",
    "EF":  "filters/eagleford/stations_filters_max.inp",
}

OUT_DIR = "key"

DEFAULT_FILTER_BW: Optional[Tuple[float, float]] = (1.0, 45.0)

@dataclass(frozen=True)
class ProfileBox:
    name: str
    minlat: float
    maxlat: float
    minlon: float
    maxlon: float


def load_filters_per_profile(files_by_profile: Dict[str, str]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for prof, path in files_by_profile.items():
        mapping: Dict[str, Tuple[float, float]] = {}
        if path and os.path.isfile(path):
            with open(path, "r", newline="", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if not row or row[0].strip().startswith("#") or len(row) < 3:
                        continue
                    key = row[0].strip()
                    try:
                        low = float(row[1].strip()); high = float(row[2].strip())
                    except ValueError:
                        continue
                    mapping[key] = (low, high)
        out[prof] = mapping
    return out


def query_profile_stations(client: Client, box: ProfileBox) -> Set[str]:
    inv = client.get_stations(
        minlatitude=box.minlat, maxlatitude=box.maxlat,
        minlongitude=box.minlon, maxlongitude=box.maxlon, level="station",
    )
    out: Set[str] = set()
    for net in inv:
        for sta in net.stations:
            out.add(f"{net.code}_{sta.code}")
    return out


def read_stations_from_dat(path: str) -> Set[str]:
    """
    Read stations from a .dat file containing station codes like:
        4O.BP01,\\
        TX.VHRN
        4O.BP01,
    - Accepts NET.STA or NET_STA
    - Ignores blank lines and lines starting with '#'
    - Strips trailing commas/backslashes
    Returns a set of station IDs in NET_STA format.
    """
    out: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            line = line.rstrip("\\").rstrip(",").strip()
            if not line:
                continue

            # allow multiple entries separated by commas
            for token in [t.strip() for t in line.split(",") if t.strip()]:
                out.add(token.replace(".", "_"))
    return out


def build_station_profiles_map(
    profiles_boxes: List[ProfileBox],
    fdsn_url: str,
    extras_per_profile: Dict[str, List[str]],
) -> Dict[str, Set[str]]:
    client = Client(fdsn_url)
    station_to_profiles: Dict[str, Set[str]] = {}
    for box in profiles_boxes:
        found = query_profile_stations(client, box)
        found |= set(extras_per_profile.get(box.name, []))
        for netsta in found:
            station_to_profiles.setdefault(netsta, set()).add(box.name)
    return station_to_profiles


def _fmt_float(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def resolve_filter_for(station: str, profile: str,
                       filt_station_per_profile: Dict[str, Dict[str, Tuple[float, float]]],
                       default_bw: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    prof_map = filt_station_per_profile.get(profile, {})
    if station in prof_map:
        return prof_map[station]
    return default_bw


def write_station_binding(
    path: str,
    profiles_for_station: List[str],
    station_code: str,
    filt_station_per_profile: Dict[str, Dict[str, Tuple[float, float]]],
    default_bw: Optional[Tuple[float, float]],
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines: List[str] = [f"profiles = {','.join(profiles_for_station)}", ""]

    for p in profiles_for_station:
        bw = resolve_filter_for(station_code, p, filt_station_per_profile, default_bw)
        if bw is None:
            lines += [
                "# Enables/disables picking on a station.",
                f"profiles.{p}.pickEnable = false",
                "",
                "# Defines the filter to be used for picking (no filter found).",
                '# profiles.{p}.filter = "BW(2,1,45)"',
                "",
            ]
        else:
            low, high = bw
            lines += [
                "# Enables/disables picking on a station.",
                f"profiles.{p}.pickEnable = true",
                "",
                "# Defines the filter to be used for picking.",
                f'profiles.{p}.filter = "BW(2,{_fmt_float(low)},{_fmt_float(high)})"',
                "",
            ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lists",
        nargs="+",
        help="Read stations from one or more .dat files (located next to this script unless absolute path). "
             "Profile name will be the .dat filename without extension.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load ORIGINAL filters (unchanged)
    filt_station_per_profile = load_filters_per_profile(FILTERS_PROFILE_FILES)

    # Load OPTIONAL list-based filters and merge (supports multiple files per profile)
    if LIST_FILTERS_PROFILE_FILES:
        for prof, paths in LIST_FILTERS_PROFILE_FILES.items():
            if isinstance(paths, str):
                paths = [paths]

            merged_mapping: Dict[str, Tuple[float, float]] = {}

            for pth in paths:
                full = pth if os.path.isabs(pth) else os.path.join(script_dir, pth)
                loaded = load_filters_per_profile({prof: full}).get(prof, {})
                # later files override earlier ones for same station
                merged_mapping.update(loaded)

            filt_station_per_profile.setdefault(prof, {}).update(merged_mapping)

    # ------------------------------------------------------------
    # NEW MODE: build bindings from .dat station lists
    # ------------------------------------------------------------
    if args.lists:
        station_profiles: Dict[str, Set[str]] = {}

        for item in args.lists:
            dat_path = item if os.path.isabs(item) else os.path.join(script_dir, item)
            profile = os.path.splitext(os.path.basename(dat_path))[0]
            stations = read_stations_from_dat(dat_path)

            for netsta in stations:
                station_profiles.setdefault(netsta, set()).add(profile)

        os.makedirs(OUT_DIR, exist_ok=True)

        for netsta, profs in sorted(station_profiles.items()):
            out_path = os.path.join(OUT_DIR, f"station_{netsta}")
            profiles_sorted = sorted(profs)

            write_station_binding(
                out_path, profiles_sorted, netsta,
                filt_station_per_profile, DEFAULT_FILTER_BW
            )
        return

    # ------------------------------------------------------------
    # ORIGINAL MODE: FDSN + region profiles (unchanged)
    # ------------------------------------------------------------
    boxes: List[ProfileBox] = [
        ProfileBox(name=n, minlat=lim["minlat"], maxlat=lim["maxlat"],
                   minlon=lim["minlon"], maxlon=lim["maxlon"])
        for n, lim in PROFILES.items()
    ]

    station_profiles = build_station_profiles_map(boxes, FDSN_URL, STA2ADD_PER_PROFILE)

    os.makedirs(OUT_DIR, exist_ok=True)
    missing_filters: List[str] = []
    for netsta, profs in sorted(station_profiles.items()):
        out_path = os.path.join(OUT_DIR, f"station_{netsta}")
        profiles_sorted = sorted(profs)
        all_none = True
        for p in profiles_sorted:
            if resolve_filter_for(netsta, p, filt_station_per_profile, DEFAULT_FILTER_BW) is not None:
                all_none = False
                break
        if all_none:
            missing_filters.append(netsta)

        write_station_binding(
            out_path, profiles_sorted, netsta,
            filt_station_per_profile, DEFAULT_FILTER_BW
        )


if __name__ == "__main__":
    main()