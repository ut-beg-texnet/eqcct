import obspy
import numpy as np
import seisbench.models as sbm
import time
import random
from pathlib import Path

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

    by_last = {}
    for tr in st:
        by_last.setdefault(tr.stats.channel[-1], []).append(tr)

    def _best_trace(letter):
        lst = by_last.get(letter, [])
        return lst[0] if lst else None

    trE = _best_trace("E") or _best_trace("1")
    trN = _best_trace("N") or _best_trace("2")
    trZ = _best_trace("Z")

    missing_components = []
    if trZ is None:
        missing_components.append("Z")
    if trE is None:
        missing_components.append("E (or 1)")
    if trN is None:
        missing_components.append("N (or 2)")

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
            temp_st = obspy.read(str(file))  # Convert Path to string if needed
            temp_st.merge(method=1, fill_value=0)   # merge fragments, fill gaps with zeros
            temp_st.detrend("demean")
            if len(temp_st) > 0:
                st += temp_st
                files_read += 1
        except Exception as e:
            # Continue to next file if one fails
            continue

    if len(st) == 0:
        raise ValueError(
            f"No valid data found for station {station}. "
            f"Attempted to read {len(files_list)} file(s), successfully read {files_read}. "
            f"Please check that the mSEED files are valid and contain data."
        )

    return process_raw_station_stream_3c(args, st, station)