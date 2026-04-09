import os
import obspy
import numpy as np
import seisbench.models as sbm
import time
import random
import logging
from pathlib import Path

from eqcctpro.tools import read_station_waveform_file
from eqcctpro.waveform_filter import apply_waveform_filter, resolve_waveform_filter_params

class SeisBenchModels:
    def __init__(self, parent_model_name, child_model_name, validate_pretrained=True):
        """
        Parameters:
        -----------
        parent_model_name : str
            SeisBench model family (e.g. 'EQTransformer', 'PhaseNet')
        child_model_name : str
            Pretrained variant name (e.g. 'original', 'stead')
        validate_pretrained : bool
            If True (default), call list_pretrained() to verify child_model_name
            exists on the SeisBench server. Set to False when the caller has
            already validated (e.g. Ray actors after driver-side validation) to
            avoid redundant network calls and thundering-herd 500 errors.
        """
        self.models = {}
        self.parent_model_list = ['PhaseNet', 'PhaseNetLight', 'EQTransformer', 'CRED', 'GPD', 'LFEDetect', 'OBSTransformer']  # List of available models from SeisBench

        # Check if parent model is valid
        if parent_model_name not in self.parent_model_list:
            raise ValueError(
                f"Parent model {parent_model_name} not found in SeisBench. "
                f"Please choose from {self.parent_model_list}"
            )
        self.parent_model_name = parent_model_name

        # Check if child model is valid - use getattr to dynamically access the model class
        try:
            model_class = getattr(sbm, self.parent_model_name)
        except AttributeError:
            raise ValueError(
                f"Model class {self.parent_model_name} not found in seisbench.models. "
                f"Please check the model name."
            )

        if validate_pretrained:
            available_models = self._list_pretrained_with_retry(model_class)
            if available_models is not None and child_model_name not in available_models:
                raise ValueError(
                    f"Child model {child_model_name} not found in {parent_model_name}. "
                    f"Please choose from {available_models}"
                )

        self.child_model_name = child_model_name
        self.model = None  # Will be loaded in load_model()

    @staticmethod
    def _list_pretrained_with_retry(model_class, max_retries=3):
        """Fetch list_pretrained() with retries and jitter to handle transient server errors."""
        for i in range(max_retries):
            try:
                if i > 0:
                    time.sleep(random.uniform(1.0, 3.0))
                return model_class.list_pretrained()
            except Exception:
                if i == max_retries - 1:
                    return None

    def load_model(self):
        """
        Load the SeisBench model given the parent model name and its 'child' model subversion name.
        This follows the workflow from integration_phasenet.ipynb where models are loaded with from_pretrained().
        """
        if self.model is None:
            model_class = getattr(sbm, self.parent_model_name)
            self.model = model_class.from_pretrained(self.child_model_name)
        return self.model

    def annotate(self, stream, **kwargs):
        """
        Annotate a stream with phase probabilities (probability time series).
        This is the primary method used in integration_phasenet.ipynb.
        
        Parameters:
        -----------
        stream : obspy.Stream
            Input 3-component ObsPy Stream
        **kwargs : dict
            Additional arguments passed to model.annotate() (e.g., strict, overlap, stacking, etc.)
        
        Returns:
        --------
        obspy.Stream
            Stream with phase probability traces
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model.annotate(stream, **kwargs)

    def classify(self, stream, **kwargs):
        """
        Classify a stream and return picks directly.
        This method returns picks as a ClassifyOutput object.
        
        Parameters:
        -----------
        stream : obspy.Stream
            Input 3-component ObsPy Stream
        **kwargs : dict
            Additional arguments passed to model.classify()
        
        Returns:
        --------
        ClassifyOutput
            Object containing picks and metadata
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model.classify(stream, **kwargs)

    def predict(self, data):
        """
        Generic predict method for models that support it.
        
        Parameters:
        -----------
        data : obspy.Stream or numpy array
            Input data for prediction
        
        Returns:
        --------
        Model predictions (format depends on model type)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Try predict method if available, otherwise fall back to annotate
        if hasattr(self.model, 'predict'):
            return self.model.predict(data)
        elif hasattr(self.model, 'annotate'):
            return self.model.annotate(data)
        else:
            raise AttributeError(
                f"Model {self.parent_model_name} does not have predict() or annotate() methods."
            )


def resampling(st, antialias_lowpass_hz=45.0):
    """
    Perform resampling on ObsPy stream objects.
    Fallback resampling method when interpolate() fails.
    
    Parameters:
    -----------
    st : obspy.Stream
        Input ObsPy Stream to resample
    
    Returns:
    --------
    obspy.Stream
        Resampled stream at 100 Hz
    """
    need_resampling = [tr for tr in st if tr.stats.sampling_rate != 100.0]
    if len(need_resampling) > 0:
        for indx, tr in enumerate(need_resampling):
            if tr.stats.delta < 0.01:
                tr.filter('lowpass', freq=float(antialias_lowpass_hz), zerophase=True)
            tr.resample(100)
            tr.stats.sampling_rate = 100
            tr.stats.delta = 0.01
            tr.data.dtype = 'int32'
            st.remove(tr)
            st.append(tr)
    return st


def process_raw_station_stream_3c(args, st, station):
    """
    SeisBench preprocessing (taper → bandpass → resample → trim → 3C) from an in-memory
    ObsPy Stream that is already read, merged per file, and demeaned (same state as after
    the file-read loop in mseed2stream_3c). Used with ray.put shared Streams (scmlpick-style).
    """
    if st is None or len(st) == 0:
        raise ValueError(f"No traces for station {station} in shared Stream.")

    st = obspy.Stream(traces=[tr.copy() for tr in st])
    try:
        st.merge(method=1, fill_value=0)
    except Exception:
        try:
            st.merge(method=1, fill_value=0)
        except Exception:
            pass
    if len(st) == 0:
        raise ValueError(f"No valid data after merge for station {station}.")

    max_percentage = 5 / (st[0].stats.delta * st[0].stats.npts)
    st.taper(max_percentage=max_percentage, type="cosine")

    ftype, freqmin, freqmax, f_corners, f_zp = resolve_waveform_filter_params(args, station)
    apply_waveform_filter(st, ftype, freqmin, freqmax, f_corners, f_zp)

    if any(tr.stats.sampling_rate != 100.0 for tr in st):
        try:
            st.interpolate(100.0, method="linear")
        except Exception:
            st = resampling(st, antialias_lowpass_hz=freqmax)

    t0 = max(tr.stats.starttime for tr in st)
    t1 = min(tr.stats.endtime for tr in st)
    st.trim(t0, t1, pad=False)

    by_last: dict[str, list] = {}
    for tr in st:
        ch = (tr.stats.channel or "").strip()
        if len(ch) < 1:
            continue
        by_last.setdefault(ch[-1], []).append(tr)

    def _best_trace(letter: str):
        lst = by_last.get(letter, [])
        return lst[0] if lst else None

    trE = _best_trace("E") or _best_trace("1")
    trN = _best_trace("N") or _best_trace("2")
    trZ = _best_trace("Z")

    # Map observed traces by upper-case SEED channel name (first unused wins).
    by_code: dict[str, list] = {}
    for tr in st:
        key = (tr.stats.channel or "").strip().upper()
        if key:
            by_code.setdefault(key, []).append(tr)

    def _used_ids():
        return {id(x) for x in (trE, trN, trZ) if x is not None}

    def _first_unused_for_codes(codes: tuple[str, ...]):
        used = _used_ids()
        for code in codes:
            for tr in by_code.get(code, []):
                if id(tr) not in used:
                    return tr
        return None

    # Common broadband / weak-motion channel names (FDSN-style 3-letter codes).
    if trZ is None:
        trZ = _first_unused_for_codes(("CHZ", "HHZ", "BHZ", "SHZ", "CNZ", "EHZ"))
    if trE is None:
        trE = _first_unused_for_codes(("CHE", "HHE", "BHE", "SHE", "CNE", "HH1", "BH1", "EH1"))
    if trN is None:
        trN = _first_unused_for_codes(("CHN", "HHN", "BHN", "SHN", "CNN", "HH2", "BH2", "EH2"))

    def _unassigned_traces():
        sel = _used_ids()
        return [tr for tr in st if id(tr) not in sel]

    # Salvage: e.g. CHE + CHN + CNN where CNN is vertical or mis-tagged (no *Z code).
    _max_salvage = 4
    for _ in range(_max_salvage):
        u = _unassigned_traces()
        if trZ is None and trE is not None and trN is not None and len(u) == 1:
            trZ = u[0]
            continue
        u = _unassigned_traces()
        if trE is None and trN is not None and trZ is not None and len(u) == 1:
            c = (u[0].stats.channel or "")[-1:]
            if c in ("E", "1"):
                trE = u[0]
                continue
        u = _unassigned_traces()
        if trN is None and trE is not None and trZ is not None and len(u) == 1:
            c = (u[0].stats.channel or "")[-1:]
            if c in ("N", "2"):
                trN = u[0]
                continue
        break

    missing_components = []
    if trZ is None:
        missing_components.append("Z")
    if trE is None:
        missing_components.append("E (or 1)")
    if trN is None:
        missing_components.append("N (or 2)")

    _strict_3c = os.environ.get("EQCCTPRO_STRICT_3C", "").lower() in (
        "1",
        "true",
        "yes",
    )
    # Steim / partial decodes often leave only vertical (or N+Z). PhaseNet still
    # needs three inputs; duplicate an available trace with an explicit warning.
    if missing_components and not _strict_3c:
        logw = logging.getLogger("eqcctpro").warning
        if trE is None and trN is not None and trZ is not None:
            logw(
                "Station %s: east component missing; using north trace as synthetic E for PhaseNet (degraded picks).",
                station,
            )
            trE = trN.copy()
        elif trN is None and trE is not None and trZ is not None:
            logw(
                "Station %s: north component missing; using east trace as synthetic N for PhaseNet (degraded picks).",
                station,
            )
            trN = trE.copy()
        elif trE is None and trN is None and trZ is not None:
            logw(
                "Station %s: only vertical decoded; using Z as synthetic E and N for PhaseNet (degraded picks).",
                station,
            )
            trE = trZ.copy()
            trN = trZ.copy()
        missing_components = []
        if trZ is None:
            missing_components.append("Z")
        if trE is None:
            missing_components.append("E (or 1)")
        if trN is None:
            missing_components.append("N (or 2)")

    if missing_components and len(st) == 3 and not _strict_3c:
        chans = [tr.stats.channel for tr in st]
        logging.getLogger("eqcctpro").warning(
            "Station %s: could not map ENZ from channel names %s; "
            "using lexicographic order as E, N, Z for PhaseNet (set EQCCTPRO_STRICT_3C=1 to forbid).",
            station,
            chans,
        )
        ordered = sorted(list(st), key=lambda t: (t.stats.channel or "").upper())
        trE, trN, trZ = ordered[0], ordered[1], ordered[2]
        missing_components = []

    if missing_components:
        available_channels = [tr.stats.channel for tr in st]
        raise ValueError(
            f"Missing required components for station {station}: {', '.join(missing_components)}. "
            f"Available channels: {available_channels}. "
            f"Please ensure the mSEED files contain 3-component data (E/N/Z or 1/2/Z)."
        )

    out = obspy.Stream(traces=[trE.copy(), trN.copy(), trZ.copy()])
    out[0].stats.channel = out[0].stats.channel[:-1] + "E"
    out[1].stats.channel = out[1].stats.channel[:-1] + "N"
    out[2].stats.channel = out[2].stats.channel[:-1] + "Z"

    return out, freqmin, freqmax


def mseed2stream_3c(args, files_list, station):
    """
    Read miniSEED files and return a single 3-component ObsPy Stream
    (E/N/Z preferred, otherwise 1/2/Z), aligned in time, filtered, resampled.
    
    This function follows the preprocessing workflow from integration_phasenet.ipynb:
    1. Read and merge mSEED files
    2. Detrend (demean)
    3. Apply cosine taper (~5 seconds)
    4. Apply bandpass filter (1-45 Hz, or station-specific)
    5. Resample to 100 Hz
    6. Trim to intersection (common time window, no padding)
    7. Select best 3 components (E/N/Z or 1/2/Z)
    
    Parameters:
    -----------
    args : dict
        Dictionary containing optional 'stations_filters' key with pandas DataFrame
        containing station-specific filter parameters (columns: 'sta', 'hp', 'lp')
    files_list : list
        List of file paths (str or Path) to mSEED files for the station
    station : str
        Station code/name for filtering purposes
    
    Returns:
    --------
    tuple : (obspy.Stream, float, float) or None
        Returns (stream, freqmin, freqmax) if successful, None if no data or missing components
        - stream: 3-component ObsPy Stream with channels renamed to *E, *N, *Z
        - freqmin: Minimum frequency used in bandpass filter (Hz)
        - freqmax: Maximum frequency used in bandpass filter (Hz)
    
    Raises:
    ------
    ValueError
        If files_list is empty or no valid data is found
    """
    # Check if files_list is empty
    if not files_list or len(files_list) == 0:
        raise ValueError(
            f"No files found for station {station}. "
            f"Please check that the file paths are correct."
        )
    
    st = obspy.Stream()
    files_read = 0

    # --- 1) Read all input files into one stream ---
    for file in files_list:
        try:
            temp_st = read_station_waveform_file(
                str(file),
                logger=args.get("logger") if isinstance(args, dict) else None,
            )
            temp_st.merge(method=1, fill_value=0)   # merge fragments, fill gaps with zeros
            temp_st.detrend("demean")
            if len(temp_st) > 0:
                st += temp_st
                files_read += 1
        except Exception:
            continue

    if len(st) == 0:
        raise ValueError(
            f"No valid data found for station {station}. "
            f"Attempted to read {len(files_list)} file(s), successfully read {files_read}. "
            f"Please check that the mSEED files are valid and contain data."
        )

    return process_raw_station_stream_3c(args, st, station)