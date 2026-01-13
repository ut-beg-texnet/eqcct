import obspy
import numpy as np
import seisbench.models as sbm
from pathlib import Path

class SeisBenchModels:
    def __init__(self, parent_model_name, child_model_name):
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
            available_models = model_class.list_pretrained()
            if child_model_name not in available_models:
                raise ValueError(
                    f"Child model {child_model_name} not found in {parent_model_name}. "
                    f"Please choose from {available_models}"
                )
        except AttributeError:
            raise ValueError(
                f"Model class {self.parent_model_name} not found in seisbench.models. "
                f"Please check the model name."
            )
        self.child_model_name = child_model_name
        self.model = None  # Will be loaded in load_model()

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


def resampling(st):
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
                tr.filter('lowpass', freq=45, zerophase=True)
            tr.resample(100)
            tr.stats.sampling_rate = 100
            tr.stats.delta = 0.01
            tr.data.dtype = 'int32'
            st.remove(tr)
            st.append(tr)
    return st


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

    # --- 2) Taper ---
    max_percentage = 5 / (st[0].stats.delta * st[0].stats.npts)  # taper ~5 sec worth
    st.taper(max_percentage=max_percentage, type="cosine")

    # --- 3) Bandpass (station-specific if provided) ---
    freqmin, freqmax = 1.0, 45.0
    if args.get("stations_filters") is not None:
        try:
            df_filters = args["stations_filters"]
            row = df_filters[df_filters.sta == station].iloc[0]
            freqmin, freqmax = float(row["hp"]), float(row["lp"])
        except Exception:
            pass

    st.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=2, zerophase=True)

    # --- 4) Resample to 100 Hz if needed ---
    if any(tr.stats.sampling_rate != 100.0 for tr in st):
        try:
            st.interpolate(100.0, method="linear")
        except Exception:
            st = resampling(st)  # fallback

    # --- 5) Force all traces to the SAME time window (intersection) ---
    # This is the correct choice for "combine 3 streams into one waveform"
    t0 = max(tr.stats.starttime for tr in st)
    t1 = min(tr.stats.endtime for tr in st)
    st.trim(t0, t1, pad=False)

    # --- 6) Pick the best 3 components: prefer E/N/Z, fallback to 1/2/Z ---
    by_last = {}
    for tr in st:
        by_last.setdefault(tr.stats.channel[-1], []).append(tr)

    def _best_trace(letter):
        """Choose the first trace for a given last-letter component."""
        lst = by_last.get(letter, [])
        return lst[0] if lst else None

    trE = _best_trace("E") or _best_trace("1")
    trN = _best_trace("N") or _best_trace("2")
    trZ = _best_trace("Z")

    # Check which components are missing and provide helpful error message
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

    # Optional: normalize naming so your plotting labels look nice
    # (This does NOT rotate; it only renames.)
    out[0].stats.channel = out[0].stats.channel[:-1] + "E"
    out[1].stats.channel = out[1].stats.channel[:-1] + "N"
    out[2].stats.channel = out[2].stats.channel[:-1] + "Z"

    return out, freqmin, freqmax