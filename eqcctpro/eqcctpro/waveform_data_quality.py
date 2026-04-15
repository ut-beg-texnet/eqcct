"""
Known marginal miniSEED / Steim issues (field data, exporter quirks, strict libmseed).

EQCCTPro logs a warning when a run's timechunk and station list overlap this registry.
Extend :data:`KNOWN_MSEED_QUALITY_BY_CHUNK` as new cases are confirmed.
"""
from __future__ import annotations

import logging
import os
from typing import Any

# Keys are timechunk directory ids: YYYYMMDDThhmmssZ_YYYYMMDDThhmmssZ
KNOWN_MSEED_QUALITY_BY_CHUNK: dict[str, dict[str, Any]] = {
    "20251001T000000Z_20251002T000000Z": {
        "stations_strict_decode_failed": (
            "BB01",
            "BB06",
            "BB08",
            "BB09",
            "BB12",
            "BB13",
        ),
        "stations_ok_same_chunk": ("BB02", "BB03", "BB04", "BB05", "BB07", "BB11"),
        "symptom": (
            "ObsPy ``InternalMSEEDError`` (e.g. only decoded N of M samples) on strict "
            "``obspy.read`` for some components; SeisComp may still display traces."
        ),
        "reference": (
            "Bluebonnet PhaseNet miniSEED; example BB08 CNE: msr_unpack_data expected "
            "22528 samples, decoded 412 before failure."
        ),
    },
    "20251002T000000Z_20251003T000000Z": {
        "stations_steim_warnings_observed": (
            "BB02",
            "BB03",
            "BB04",
            "BB06",
            "BB07",
            "BB08",
            "BB09",
            "BB11",
            "BB13",
        ),
        "symptom": (
            "ObsPy ``InternalMSEEDWarning``: Steim1 data integrity check failed and/or "
            "``Last reclen exceeds buflen`` during materialization or read; run may still complete."
        ),
        "reference": (
            "EQCCTPro driver log during flat-archive materialization (2026-04-14); "
            "BB01, BB05, BB12 had no warnings in captured log."
        ),
    },
}


def resolve_timechunk_id_for_registry(
    timechunk_id: str | None, input_dir: str | None
) -> str:
    tid = (timechunk_id or "").strip()
    if tid:
        return tid
    if not input_dir:
        return ""
    cand = os.path.basename(os.path.abspath(input_dir))
    return cand


def log_known_waveform_quality_issues(
    timechunk_id: str | None,
    input_dir: str | None,
    station_list: list[str],
    logger: logging.Logger | None,
) -> None:
    """
    If *timechunk_id* (or basename of *input_dir*) matches a registry key, log overlap
    with *station_list* so operators see known-problem dates/stations in every run.
    """
    if logger is None:
        return
    tid = resolve_timechunk_id_for_registry(timechunk_id, input_dir)
    meta = KNOWN_MSEED_QUALITY_BY_CHUNK.get(tid)
    if not meta:
        return
    st_set = {str(s).strip() for s in station_list}
    lines = [f"Known miniSEED quality notes for timechunk {tid} (see eqcctpro.waveform_data_quality)."]
    if "stations_strict_decode_failed" in meta:
        reg = set(meta["stations_strict_decode_failed"])
        hit = sorted(st_set & reg)
        lines.append(
            f"  Stations with documented strict ObsPy/libmseed decode failures: {sorted(reg)}. "
            f"This run overlaps: {hit if hit else 'none'}."
        )
    if "stations_ok_same_chunk" in meta:
        lines.append(f"  Same-day stations that decoded without hard failure in tests: {meta['stations_ok_same_chunk']}.")
    if "stations_steim_warnings_observed" in meta:
        reg = set(meta["stations_steim_warnings_observed"])
        hit = sorted(st_set & reg)
        lines.append(
            f"  Stations with observed Steim integrity warnings: {sorted(reg)}. "
            f"This run overlaps: {hit if hit else 'none'}."
        )
    lines.append(f"  Symptom: {meta.get('symptom', '')}")
    lines.append(f"  Reference: {meta.get('reference', '')}")
    logger.warning("\n".join(lines))
