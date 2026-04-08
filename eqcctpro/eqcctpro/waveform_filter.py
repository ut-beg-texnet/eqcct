"""
Waveform preprocessing filters (ObsPy ``Stream.filter``) shared by EQCCT and SeisBench paths.
"""

from __future__ import annotations

DEFAULT_WAVEFORM_FILTER_FREQMIN = 1.0
DEFAULT_WAVEFORM_FILTER_FREQMAX = 45.0
DEFAULT_WAVEFORM_FILTER_TYPE = "bandpass"
DEFAULT_WAVEFORM_FILTER_CORNERS = 2
DEFAULT_WAVEFORM_FILTER_ZEROPHASE = True

# ObsPy-supported types we expose (see ObsPy documentation for ``Stream.filter``).
SUPPORTED_WAVEFORM_FILTER_TYPES = frozenset(
    {"bandpass", "bandstop", "lowpass", "highpass"}
)


def resolve_waveform_filter_params(args: dict, station: str):
    """
    Return (filter_type, freqmin, freqmax, corners, zerophase).

    If ``args['stations_filters']`` is set, per-station ``hp`` / ``lp`` override the
    default frequencies (same as legacy behavior: high-pass corner, low-pass corner).
    """
    freqmin = float(args.get("waveform_filter_freqmin", DEFAULT_WAVEFORM_FILTER_FREQMIN))
    freqmax = float(args.get("waveform_filter_freqmax", DEFAULT_WAVEFORM_FILTER_FREQMAX))
    df = args.get("stations_filters")
    if df is not None:
        try:
            row = df[df.sta == station].iloc[0]
            freqmin = float(row["hp"])
            freqmax = float(row["lp"])
        except Exception:
            pass

    filter_type = str(
        args.get("waveform_filter_type", DEFAULT_WAVEFORM_FILTER_TYPE)
    ).lower().strip()
    corners = int(args.get("waveform_filter_corners", DEFAULT_WAVEFORM_FILTER_CORNERS))
    zerophase = bool(args.get("waveform_filter_zerophase", DEFAULT_WAVEFORM_FILTER_ZEROPHASE))
    return filter_type, freqmin, freqmax, corners, zerophase


def apply_waveform_filter(stream, filter_type: str, freqmin: float, freqmax: float, corners: int, zerophase: bool):
    """
    Apply ObsPy filter to an in-place ``Stream`` (all traces).

    * **bandpass** / **bandstop**: ``freqmin``, ``freqmax`` (Hz).
    * **lowpass**: corner frequency = ``freqmax`` (maps legacy "lp" / Nyquist-side corner).
    * **highpass**: corner frequency = ``freqmin`` (maps legacy "hp").
    """
    ft = (filter_type or DEFAULT_WAVEFORM_FILTER_TYPE).lower().strip()
    if ft not in SUPPORTED_WAVEFORM_FILTER_TYPES:
        raise ValueError(
            f"waveform_filter_type must be one of {sorted(SUPPORTED_WAVEFORM_FILTER_TYPES)}, got {filter_type!r}"
        )
    common = {"corners": corners, "zerophase": zerophase}
    if ft in ("bandpass", "bandstop"):
        stream.filter(ft, freqmin=freqmin, freqmax=freqmax, **common)
    elif ft == "lowpass":
        stream.filter("lowpass", freq=freqmax, **common)
    else:  # highpass
        stream.filter("highpass", freq=freqmin, **common)
