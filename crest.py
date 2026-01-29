import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import time
import subprocess
import json
import shutil
import os
import csv
from pathlib import Path
import math
import io
import contextlib
import re

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    import soundfile as sf
except ModuleNotFoundError:  # pragma: no cover
    sf = None

# Get CPU core count for parallelization
CPU_COUNT = mp.cpu_count()

# Check FFmpeg availability
def check_ffmpeg():
    """Check if FFmpeg is available in the system PATH"""
    return shutil.which("ffmpeg") is not None

FFMPEG_AVAILABLE = check_ffmpeg()

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".wave",
    ".flac",
    ".aif",
    ".aiff",
    ".caf",
    ".ogg",
    ".opus",
    ".mp3",
    ".m4a",
    ".mp4",
    ".aac",
    ".wma",
    ".alac",
}

def lin2dbfs(x):
    """Convert linear amplitude to dBFS (decibels relative to full scale)"""
    if np is None:
        raise RuntimeError("Missing dependency: numpy (pip install numpy)")
    return 20 * np.log10(x) if x > 0 else -np.inf

def remove_dc_offset(data):
    """Remove DC offset by subtracting the mean value from each channel"""
    if np is None:
        raise RuntimeError("Missing dependency: numpy (pip install numpy)")
    return data - np.mean(data, axis=0)

def ffmpeg_audio_analysis(file_path):
    """Perform audio analysis using FFmpeg to extract LUFS, True Peak, and LRA metrics"""
    if not FFMPEG_AVAILABLE:
        return None
    
    try:
        # Use FFmpeg's ebur128 filter to get EBU R128 metrics
        cmd = [
            'ffmpeg',
            '-i', file_path,
            '-af', 'ebur128=peak=true',
            '-f', 'null',
            '-',
            '-nostats'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        
        if result.returncode != 0:
            warnings.warn(f"FFmpeg execution failed: {result.stderr}")
            return None
        
        # Parse EBU R128 information from FFmpeg output
        output_lines = result.stderr.split('\n')
        
        analysis_results = {
            'integrated_lufs': None,
            'loudness_range': None,
            'true_peak_dbfs': None,
            'true_peak_dbfs_per_channel': None,
            'sample_peak_dbfs': None
        }
        
        # Find key information in Summary section
        in_summary = False
        per_channel_true_peaks = {}
        # Examples seen in the wild may include:
        #   "Peak:        -0.1 dBFS"
        #   "True peak:   -0.1 dBFS"
        #   "True peak:  -0.1 dBFS (ch=1)"
        #   "True peak:  -0.1 dBFS (L)" / "(R)"
        tp_regex = re.compile(r"^(True\s*peak|Peak):\s*([+-]?\d+(?:\.\d+)?)\s*dBFS", re.IGNORECASE)
        ch_regex = re.compile(r"\(.*?(?:ch\s*=\s*(\d+)|([LR]))\s*\)", re.IGNORECASE)
        for line in output_lines:
            line = line.strip()
            
            # Detect start of Summary section
            if 'Summary:' in line:
                in_summary = True
                continue
            
            if in_summary:
                # Parse Integrated loudness
                if line.startswith('I:') and 'LUFS' in line:
                    try:
                        # Format: "I:          -9.8 LUFS"
                        parts = line.split()
                        if len(parts) >= 2:
                            analysis_results['integrated_lufs'] = float(parts[1])
                    except (ValueError, IndexError):
                        pass
                
                # Parse Loudness range
                elif line.startswith('LRA:') and 'LU' in line:
                    try:
                        # Format: "LRA:         8.0 LU"
                        parts = line.split()
                        if len(parts) >= 2:
                            analysis_results['loudness_range'] = float(parts[1])
                    except (ValueError, IndexError):
                        pass
                
                # Parse (True) peak lines (overall and per-channel if present)
                elif ('dBFS' in line) and (line.lower().startswith('peak') or line.lower().startswith('true')):
                    m = tp_regex.match(line)
                    if not m:
                        continue
                    label = m.group(1).lower().replace(" ", "")
                    try:
                        value = float(m.group(2))
                    except Exception:
                        continue

                    if label.startswith("true"):
                        # Track overall max true peak
                        if analysis_results['true_peak_dbfs'] is None or value > analysis_results['true_peak_dbfs']:
                            analysis_results['true_peak_dbfs'] = value
                        # Attempt to capture per-channel true peak if annotated
                        chm = ch_regex.search(line)
                        if chm:
                            if chm.group(1):
                                ch_idx = int(chm.group(1)) - 1
                                per_channel_true_peaks[ch_idx] = max(per_channel_true_peaks.get(ch_idx, -1e9), value)
                            elif chm.group(2):
                                ch_key = chm.group(2).upper()
                                per_channel_true_peaks[ch_key] = max(per_channel_true_peaks.get(ch_key, -1e9), value)
                    else:
                        # "Peak:" may be sample peak in some outputs; keep for completeness.
                        if analysis_results['sample_peak_dbfs'] is None or value > analysis_results['sample_peak_dbfs']:
                            analysis_results['sample_peak_dbfs'] = value

        if per_channel_true_peaks:
            analysis_results["true_peak_dbfs_per_channel"] = per_channel_true_peaks
        
        return analysis_results
        
    except Exception as e:
        warnings.warn(f"FFmpeg analysis failed: {e}")
        return None

def discover_audio_files(directory, recursive=False):
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    candidates = [p for p in dir_path.glob(pattern) if p.is_file()]
    audio_files = [p for p in candidates if p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS]
    return sorted(audio_files)

def ffprobe_audio_stream_info(file_path):
    """Return (sample_rate, channels) for the first audio stream via ffprobe."""
    if not FFMPEG_AVAILABLE:
        return None, None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels",
        "-of",
        "json",
        file_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except Exception:
        return None, None

    if result.returncode != 0 or not result.stdout:
        return None, None

    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams") or []
        if not streams:
            return None, None
        stream = streams[0]
        sr = int(stream.get("sample_rate")) if stream.get("sample_rate") else None
        ch = int(stream.get("channels")) if stream.get("channels") else None
        return sr, ch
    except Exception:
        return None, None

def read_audio_any(file_path):
    """
    Read audio into float32 ndarray shape (samples, channels) and return (data, sample_rate).
    Tries SoundFile first; falls back to FFmpeg decode for formats SoundFile can't handle.
    """
    if np is None:
        raise RuntimeError("Missing dependency: numpy (pip install numpy)")
    try:
        if sf is None:
            raise RuntimeError("Missing dependency: soundfile (pip install soundfile)")
        data, samplerate = sf.read(file_path, always_2d=True)
        return data, samplerate
    except Exception:
        if not FFMPEG_AVAILABLE:
            raise

    samplerate, channels = ffprobe_audio_stream_info(file_path)
    if samplerate is None or channels is None or samplerate <= 0 or channels <= 0:
        raise RuntimeError("Unable to probe audio stream info via ffprobe")

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostats",
        "-i",
        file_path,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="ignore") if isinstance(result.stderr, (bytes, bytearray)) else str(result.stderr)
        raise RuntimeError(f"FFmpeg decode failed: {stderr.strip()}")

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("FFmpeg decode returned no audio samples")

    frames = audio.size // channels
    if frames <= 0:
        raise RuntimeError("FFmpeg decode produced insufficient samples for channel count")

    audio = audio[: frames * channels].reshape(frames, channels)
    return audio, samplerate

def _calculate_window_crest(args):
    """Calculate Crest Factor for a single window (for parallelization)"""
    segment, sr, start_idx = args
    peak = np.max(np.abs(segment))
    rms = np.sqrt(np.mean(segment**2))
    
    if rms > 0:
        cf_db = 20 * np.log10(peak / rms)
        return start_idx / sr, cf_db
    else:
        return start_idx / sr, None

def frame_crest_analysis_vectorized(data, sr, win_ms=50, hop_ms=12.5):
    """Vectorized short-term window Crest Factor analysis - high-performance version"""
    if np is None:
        raise RuntimeError("Missing dependency: numpy (pip install numpy)")
    win_samples = int(sr * win_ms / 1000)
    hop_samples = int(sr * hop_ms / 1000)
    
    # If multi-channel, mix to mono using power-based method
    if data.ndim > 1:
        data = np.sqrt(np.mean(data**2, axis=1))
    
    # Calculate number of windows
    num_windows = (len(data) - win_samples) // hop_samples + 1
    
    if num_windows <= 0:
        return np.array([]), np.array([])
    
    # Vectorized computation: create 2D array of all windows
    # Use stride tricks to avoid data copying
    from numpy.lib.stride_tricks import sliding_window_view
    
    # Create sliding window view
    windowed_data = sliding_window_view(data, window_shape=win_samples)[::hop_samples]
    
    if len(windowed_data) == 0:
        return np.array([]), np.array([])
    
    # Vectorized computation of peaks and RMS for all windows
    peaks = np.max(np.abs(windowed_data), axis=1)
    rms_values = np.sqrt(np.mean(windowed_data**2, axis=1))
    
    # Filter valid values (RMS > 0)
    valid_mask = rms_values > 0
    peaks = peaks[valid_mask]
    rms_values = rms_values[valid_mask]
    
    # Vectorized Crest Factor calculation
    crest_factors = 20 * np.log10(peaks / rms_values)
    
    # Calculate corresponding timestamps
    valid_indices = np.arange(len(windowed_data))[valid_mask]
    time_stamps = valid_indices * hop_samples / sr
    
    return time_stamps, crest_factors

def frame_crest_analysis(data, sr, win_ms=50, hop_ms=12.5, use_parallel=True):
    """Short-term window Crest Factor analysis - intelligent choice between vectorized and parallel processing"""
    # Prefer vectorized version (usually faster)
    try:
        return frame_crest_analysis_vectorized(data, sr, win_ms, hop_ms)
    except Exception as e:
        warnings.warn(f"Vectorized analysis failed, falling back to parallel version: {e}")
        
        # Fallback to original parallelized version
        win_samples = int(sr * win_ms / 1000)
        hop_samples = int(sr * hop_ms / 1000)
        
        # If multi-channel, mix to mono using power-based method
        if data.ndim > 1:
            data = np.sqrt(np.mean(data**2, axis=1))
        
        # Pre-calculate all window parameters
        window_args = []
        for i in range(0, len(data) - win_samples + 1, hop_samples):
            segment = data[i:i + win_samples]
            window_args.append((segment, sr, i))
        
        if not use_parallel or len(window_args) < 100:
            # Serial processing
            time_stamps = []
            crest_factors = []
            for segment, sr, start_idx in window_args:
                timestamp, cf = _calculate_window_crest((segment, sr, start_idx))
                if cf is not None:
                    time_stamps.append(timestamp)
                    crest_factors.append(cf)
        else:
            # Parallel processing
            max_workers = min(CPU_COUNT, len(window_args))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(_calculate_window_crest, window_args))
            
            # Filter valid results
            time_stamps = []
            crest_factors = []
            for timestamp, cf in results:
                if cf is not None:
                    time_stamps.append(timestamp)
                    crest_factors.append(cf)
        
        return np.array(time_stamps), np.array(crest_factors)

def convert_dbfs_to_linear(dbfs_value):
    """Convert dBFS value to linear amplitude"""
    if np is None:
        raise RuntimeError("Missing dependency: numpy (pip install numpy)")
    if dbfs_value is None:
        return None
    return 10 ** (dbfs_value / 20)

def calculate_pmf_dr(data, samplerate, peak_linear, block_seconds=3.0, top_fraction=0.2, rounding="nearest"):
    """
    Calculate PMF Dynamic Range (DR) in the style of the TT DR Meter:
      DR = Peak(dBFS) - RMS_top20%(dBFS)  == 20*log10(peak / rms_top20)

    Notes:
    - Audio is segmented into ~3s blocks; the loudest 20% (by RMS) are averaged.
    - For multi-channel audio, this can be computed either:
        - Per-channel (TT-style): compute DR per channel then average channel DR integers.
        - Mixdown: compute RMS on a power-mix of channels (may differ slightly vs TT-style).
    - `peak_linear` should be a linear amplitude (sample peak or true peak).
    """
    return calculate_pmf_dr_v2(
        data,
        samplerate,
        peak_linear=peak_linear,
        block_seconds=block_seconds,
        top_fraction=top_fraction,
        rounding=rounding,
        channel_mode="per_channel",
    )

def calculate_pmf_dr_v2(
    data,
    samplerate,
    peak_linear,
    block_seconds=3.0,
    hop_seconds=0.01,
    tau_seconds=3.0,
    top_fraction=0.2,
    rounding="nearest",
    channel_mode="per_channel",
    rms_mode="rect",
):
    """
    PMF/TT DR-style calculator with selectable multi-channel handling.

    channel_mode:
      - "per_channel": compute DR per channel; track DR is floor(mean(channel DR integers)).
      - "mix": compute RMS on a power-mix across channels (previous behavior).
    """
    if np is None:
        raise RuntimeError("Missing dependency: numpy (pip install numpy)")
    if data is None or samplerate is None or samplerate <= 0:
        return None

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim != 2:
        return None

    channels = int(data.shape[1])
    if channels <= 0:
        return None

    hop_len = int(round(hop_seconds * samplerate))
    if hop_len <= 0:
        hop_len = 1

    total_samples = data.shape[0]
    if rms_mode not in ("rect", "iir"):
        raise ValueError(f"Unknown rms_mode: {rms_mode}")

    if rms_mode == "rect":
        block_len = int(round(block_seconds * samplerate))
        if block_len <= 0:
            return None
        if total_samples < block_len:
            return None

        window_starts = np.arange(0, total_samples - block_len + 1, hop_len, dtype=np.int64)
        if window_starts.size <= 0:
            return None
        num_windows = int(window_starts.size)

        # Rolling RMS via cumulative sum of squares for efficiency (no huge window views)
        squares = data.astype(np.float64, copy=False) ** 2
        cs = np.vstack([np.zeros((1, channels), dtype=np.float64), np.cumsum(squares, axis=0)])
        window_sums = cs[window_starts + block_len, :] - cs[window_starts, :]
        mean_power_per_window_per_channel = window_sums / float(block_len)  # (num_windows, channels)
        rms_per_samplepoint_per_channel = np.sqrt(mean_power_per_window_per_channel)
        samplepoint_count = num_windows
    else:
        # IIR/ballistics RMS: EMA of power with time constant tau_seconds, updated at hop resolution.
        if tau_seconds is None or tau_seconds <= 0:
            return None
        a = float(np.exp(-hop_seconds / float(tau_seconds)))
        samples_used = (total_samples // hop_len) * hop_len
        if samples_used <= 0:
            return None
        trimmed = data[:samples_used, :]
        blocks = trimmed.reshape(samples_used // hop_len, hop_len, channels).astype(np.float64, copy=False)
        power_blocks = np.mean(blocks ** 2, axis=1)  # (num_steps, channels)
        y = np.empty_like(power_blocks)
        y[0, :] = power_blocks[0, :]
        for i in range(1, y.shape[0]):
            y[i, :] = (1.0 - a) * power_blocks[i, :] + a * y[i - 1, :]
        rms_per_samplepoint_per_channel = np.sqrt(y)
        samplepoint_count = int(rms_per_samplepoint_per_channel.shape[0])

    if channel_mode not in ("per_channel", "mix"):
        raise ValueError(f"Unknown channel_mode: {channel_mode}")

    def _round_dr(db):
        if rounding == "floor":
            return int(np.floor(db + 1e-12))
        if rounding == "ceil":
            return int(np.ceil(db - 1e-12))
        return int(np.round(db))

    if channel_mode == "mix":
        if peak_linear is None or float(peak_linear) <= 0:
            return None
        # Mixdown RMS: power-average across channels
        rms_series_mix = np.sqrt(np.mean(rms_per_samplepoint_per_channel ** 2, axis=1))
        valid = rms_series_mix > 0
        if not np.any(valid):
            return None
        rms_series_mix = rms_series_mix[valid]

        count = int(np.ceil(top_fraction * len(rms_series_mix)))
        count = max(1, min(count, len(rms_series_mix)))
        top_rms = np.sort(rms_series_mix)[-count:]
        top_rms_mean = float(np.mean(top_rms))
        if top_rms_mean <= 0:
            return None

        dr_db = float(20.0 * np.log10(float(peak_linear) / top_rms_mean))
        dr_value = _round_dr(dr_db)
        top_rms_mean_dbfs = float(20.0 * np.log10(top_rms_mean))
        return {
            "dr_db": dr_db,
            "dr_value": dr_value,
            "block_seconds": float(block_seconds),
            "hop_seconds": float(hop_seconds),
            "tau_seconds": float(tau_seconds),
            "top_fraction": float(top_fraction),
            "top_rms_mean": top_rms_mean,
            "top_rms_mean_dbfs": top_rms_mean_dbfs,
            "num_samples": int(samplepoint_count),
            "channel_mode": "mix",
            "rms_mode": str(rms_mode),
        }

    # per_channel (TT-style)
    if peak_linear is None:
        return None

    peak_arr = np.asarray(peak_linear)
    if peak_arr.ndim == 0:
        peak_arr = np.full((channels,), float(peak_arr), dtype=np.float64)
    elif peak_arr.ndim == 1 and peak_arr.shape[0] == channels:
        peak_arr = peak_arr.astype(np.float64, copy=False)
    else:
        return None

    if np.any(peak_arr <= 0):
        return None

    channel_results = []
    dr_values = []
    dr_dbs = []

    for ch in range(channels):
        rms_series = rms_per_samplepoint_per_channel[:, ch]
        valid = rms_series > 0
        if not np.any(valid):
            channel_results.append(
                {"channel": int(ch), "dr_db": None, "dr_value": None, "top_rms_mean": None}
            )
            continue
        rms_series = rms_series[valid]

        count = int(np.ceil(top_fraction * len(rms_series)))
        count = max(1, min(count, len(rms_series)))
        top_rms = np.sort(rms_series)[-count:]
        top_rms_mean = float(np.mean(top_rms))
        if top_rms_mean <= 0:
            channel_results.append(
                {"channel": int(ch), "dr_db": None, "dr_value": None, "top_rms_mean": None}
            )
            continue

        dr_db = float(20.0 * np.log10(float(peak_arr[ch]) / top_rms_mean))
        dr_value = _round_dr(dr_db)
        channel_results.append(
            {
                "channel": int(ch),
                "dr_db": dr_db,
                "dr_value": int(dr_value),
                "top_rms_mean": top_rms_mean,
                "top_rms_mean_dbfs": float(20.0 * np.log10(top_rms_mean)),
            }
        )
        dr_values.append(int(dr_value))
        dr_dbs.append(dr_db)

    if not dr_values:
        return None

    # TT-style track DR: average channel DR integers, then truncate down.
    track_dr_value = int(np.floor(float(np.mean(dr_values)) + 1e-12))
    track_dr_db = float(np.mean(dr_dbs)) if dr_dbs else None

    return {
        "dr_db": track_dr_db,
        "dr_value": track_dr_value,
        "block_seconds": float(block_seconds),
        "hop_seconds": float(hop_seconds),
        "tau_seconds": float(tau_seconds),
        "top_fraction": float(top_fraction),
        "num_samples": int(samplepoint_count),
        "channel_mode": "per_channel",
        "rms_mode": str(rms_mode),
        "channels": int(channels),
        "per_channel": channel_results,
        "per_channel_dr_values": dr_values,
    }

def _analysis_task_windowed(data, samplerate):
    """Short-term window analysis task (for parallelization)"""
    try:
        time_stamps, windowed_cf = frame_crest_analysis(data, samplerate)
        if len(windowed_cf) > 0:
            return {
                'time_stamps': time_stamps,
                'crest_factors': windowed_cf,
                'mean_cf': np.mean(windowed_cf),
                'std_cf': np.std(windowed_cf),
                'min_cf': np.min(windowed_cf),
                'max_cf': np.max(windowed_cf)
            }
        return None
    except Exception as e:
        warnings.warn(f"Short-term window analysis failed: {e}")
        return None

def _analysis_task_ffmpeg(file_path):
    """FFmpeg analysis task (for parallelization)"""
    return ffmpeg_audio_analysis(file_path)

def advanced_crest_analysis(
    file_path,
    enable_true_peak=True,
    enable_windowed=True,
    enable_lufs=True,
    enable_pmf_dr=False,
    pmf_dr_use_true_peak=False,
    pmf_dr_per_channel=True,
    pmf_dr_block_seconds=3.0,
    pmf_dr_hop_seconds=0.01,
    pmf_dr_tau_seconds=3.0,
    pmf_dr_rms_mode="rect",
    pmf_dr_compare=False,
    pmf_dr_top_fraction=0.2,
    pmf_dr_rounding="nearest",
    use_parallel=True,
):
    """Advanced Crest Factor analysis - FFmpeg + vectorized optimization version"""
    if np is None:
        raise RuntimeError("Missing dependency: numpy (pip install numpy)")
    try:
        # Force read as 2D array to preserve multi-channel information
        data, samplerate = read_audio_any(file_path)
        
        # Ensure data is float32 type and normalized
        if data.dtype != np.float32:
            if np.issubdtype(data.dtype, np.integer):
                # Integer types need normalization
                max_val = np.iinfo(data.dtype).max
                data = data.astype(np.float32) / max_val
            else:
                data = data.astype(np.float32)
        
        data_raw = data

        # Remove DC offset (keep raw for PMF DR alignment with TT tools)
        data = remove_dc_offset(data)
        
        # Basic calculations (always required) - vectorized optimization
        sample_peak = np.max(np.abs(data))
        sample_peak_per_channel = np.max(np.abs(data), axis=0) if data.shape[1] > 1 else np.array([sample_peak], dtype=np.float64)
        sample_peak_per_channel_raw = np.max(np.abs(data_raw), axis=0) if data_raw.shape[1] > 1 else np.array([np.max(np.abs(data_raw))], dtype=np.float64)
        
        # Calculate RMS (vectorized optimization)
        if data.shape[1] > 1:
            # Multi-channel: power average across channels first, then time average
            power_per_sample = np.mean(data**2, axis=1)
            rms = np.sqrt(np.mean(power_per_sample))
        else:
            # Mono: direct calculation
            rms = np.sqrt(np.mean(data**2))
        
        # Check validity
        if rms == 0:
            return None
        
        # Basic Crest Factor
        sample_crest_db = 20 * np.log10(sample_peak / rms)
        
        # Parallel execution: FFmpeg analysis + Python window analysis
        tasks = []
        task_names = []
        
        # FFmpeg analysis task (LUFS + True Peak)
        if (enable_lufs or enable_true_peak) and FFMPEG_AVAILABLE:
            tasks.append(partial(_analysis_task_ffmpeg, file_path))
            task_names.append('ffmpeg')
        
        # Python window analysis task
        if enable_windowed:
            tasks.append(partial(_analysis_task_windowed, data, samplerate))
            task_names.append('windowed')
        
        # Execute tasks
        results = {}
        if tasks and use_parallel:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=min(len(tasks), CPU_COUNT)) as executor:
                futures = [executor.submit(task) for task in tasks]
                for i, future in enumerate(futures):
                    results[task_names[i]] = future.result()
        else:
            # Serial execution
            for i, task in enumerate(tasks):
                results[task_names[i]] = task()
        
        # Extract FFmpeg results
        ffmpeg_results = results.get('ffmpeg', None)
        
        # Extract True Peak
        true_peak = None
        true_peak_dbfs = None
        if ffmpeg_results and enable_true_peak:
            true_peak_dbfs = ffmpeg_results.get('true_peak_dbfs', None)
            if true_peak_dbfs is not None:
                true_peak = convert_dbfs_to_linear(true_peak_dbfs)
        
        # Extract LUFS analysis
        lufs_analysis = None
        if ffmpeg_results and enable_lufs:
            integrated_lufs = ffmpeg_results.get('integrated_lufs', None)
            loudness_range = ffmpeg_results.get('loudness_range', None)
            if integrated_lufs is not None:
                lufs_analysis = {
                    'integrated_lufs': integrated_lufs,
                    'loudness_range': loudness_range,
                    'source': 'ffmpeg'
                }
        
        # Extract window analysis results
        windowed_analysis = results.get('windowed', None)
        
        # Calculate True Crest Factor
        true_crest_db = None
        if true_peak is not None:
            true_crest_db = 20 * np.log10(true_peak / rms)

        # PMF Dynamic Range (DR) calculation (TT DR-style)
        pmf_dr = None
        pmf_dr_alt = None
        pmf_dr_peak_source = None
        if enable_pmf_dr:
            pmf_data = data_raw
            dr_peak_linear = None
            if pmf_dr_use_true_peak:
                if true_peak is not None:
                    if pmf_dr_per_channel and ffmpeg_results and ffmpeg_results.get("true_peak_dbfs_per_channel"):
                        # Use per-channel true peaks if FFmpeg annotated them; fall back to global true peak otherwise.
                        per = ffmpeg_results.get("true_peak_dbfs_per_channel") or {}
                        tp_ch = []
                        # Support either numeric (0-based) or "L"/"R" keys from parsing.
                        for ch in range(int(pmf_data.shape[1])):
                            if ch in per:
                                tp_ch.append(convert_dbfs_to_linear(per[ch]))
                            elif ch == 0 and "L" in per:
                                tp_ch.append(convert_dbfs_to_linear(per["L"]))
                            elif ch == 1 and "R" in per:
                                tp_ch.append(convert_dbfs_to_linear(per["R"]))
                            else:
                                tp_ch.append(None)
                        if all(v is not None and v > 0 for v in tp_ch):
                            dr_peak_linear = np.array(tp_ch, dtype=np.float64)
                            pmf_dr_peak_source = "true_peak_per_channel"
                        else:
                            dr_peak_linear = true_peak
                            pmf_dr_peak_source = "true_peak"
                    else:
                        dr_peak_linear = true_peak
                        pmf_dr_peak_source = "true_peak"
                else:
                    dr_peak_linear = sample_peak_per_channel_raw if pmf_dr_per_channel else np.max(np.abs(pmf_data))
                    pmf_dr_peak_source = "sample_peak_fallback"
            else:
                dr_peak_linear = sample_peak_per_channel_raw if pmf_dr_per_channel else np.max(np.abs(pmf_data))
                pmf_dr_peak_source = "sample_peak"

            pmf_dr = calculate_pmf_dr_v2(
                pmf_data,
                samplerate,
                dr_peak_linear,
                block_seconds=pmf_dr_block_seconds,
                hop_seconds=pmf_dr_hop_seconds,
                tau_seconds=pmf_dr_tau_seconds,
                top_fraction=pmf_dr_top_fraction,
                rounding=pmf_dr_rounding,
                channel_mode=("per_channel" if pmf_dr_per_channel else "mix"),
                rms_mode=pmf_dr_rms_mode,
            )

            if pmf_dr_compare:
                alt_mode = "iir" if pmf_dr_rms_mode == "rect" else "rect"
                pmf_dr_alt = calculate_pmf_dr_v2(
                    pmf_data,
                    samplerate,
                    dr_peak_linear,
                    block_seconds=pmf_dr_block_seconds,
                    hop_seconds=pmf_dr_hop_seconds,
                    tau_seconds=pmf_dr_tau_seconds,
                    top_fraction=pmf_dr_top_fraction,
                    rounding=pmf_dr_rounding,
                    channel_mode=("per_channel" if pmf_dr_per_channel else "mix"),
                    rms_mode=alt_mode,
                )

        return {
            'file_path': file_path,
            'sample_rate': samplerate,
            'channels': data.shape[1],
            'duration': data.shape[0] / samplerate,
            'sample_peak': sample_peak,
            'true_peak': true_peak,
            'true_peak_dbfs': true_peak_dbfs,
            'rms': rms,
            'sample_crest_db': sample_crest_db,
            'true_crest_db': true_crest_db,
            'pmf_dr': pmf_dr,
            'pmf_dr_alt': pmf_dr_alt,
            'pmf_dr_peak_source': pmf_dr_peak_source,
            'windowed_analysis': windowed_analysis,
            'lufs_analysis': lufs_analysis,
            'ffmpeg_available': FFMPEG_AVAILABLE
        }
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def print_analysis_results(results):
    """Format and print analysis results"""
    if results is None:
        print("Analysis failed or audio file is invalid")
        return
    
    print(f"\n{'='*60}")
    print(f"File: {results['file_path']}")
    print(f"Sample Rate: {results['sample_rate']} Hz")
    print(f"Channels: {results['channels']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    print(f"{'='*60}")
    
    # Basic statistics
    print(f"\n📊 Basic Audio Statistics:")
    print(f"  Sample Peak: {results['sample_peak']:.6f} ({lin2dbfs(results['sample_peak']):.2f} dBFS)")
    
    # True Peak display (prioritize FFmpeg results)
    if results.get('true_peak_dbfs') is not None:
        print(f"  True Peak  : {results['true_peak']:.6f} ({results['true_peak_dbfs']:.2f} dBFS) [FFmpeg]")
    elif results['true_peak'] is not None:
        print(f"  True Peak  : {results['true_peak']:.6f} ({lin2dbfs(results['true_peak']):.2f} dBFS) [Python]")
    elif not results.get('ffmpeg_available', False):
        print(f"  True Peak  : Not calculated (FFmpeg unavailable)")
    
    print(f"  RMS        : {results['rms']:.6f} ({lin2dbfs(results['rms']):.2f} dBFS)")
    
    # Crest Factor
    print(f"\n🎯 Crest Factor:")
    print(f"  Sample CF  : {results['sample_crest_db']:.2f} dB")
    if results['true_crest_db'] is not None:
        print(f"  True CF    : {results['true_crest_db']:.2f} dB")

    # PMF Dynamic Range (DR)
    if results.get('pmf_dr') is not None:
        dr = results['pmf_dr']
        peak_src = results.get('pmf_dr_peak_source', 'unknown')
        src_label = {
            'true_peak': 'True Peak',
            'sample_peak': 'Sample Peak',
            'sample_peak_fallback': 'Sample Peak (True Peak unavailable)',
        }.get(peak_src, peak_src)
        ch_mode = dr.get("channel_mode") if isinstance(dr, dict) else None
        ch_mode_label = "Per-Channel" if ch_mode == "per_channel" else ("Mixdown" if ch_mode == "mix" else None)
        print(f"\n📏 PMF Dynamic Range (TT DR-style):")
        mode_tag = f", {ch_mode_label}" if ch_mode_label else ""
        dr_db_text = f"{dr['dr_db']:.2f} dB" if dr.get("dr_db") is not None else "n/a"
        print(f"  DR         : DR{dr['dr_value']} ({dr_db_text}) [{src_label}{mode_tag}]")
        if isinstance(dr, dict) and dr.get("per_channel_dr_values"):
            vals = ", ".join([f"DR{v}" for v in dr["per_channel_dr_values"]])
            print(f"  Channels   : {vals}")
        hop = dr.get("hop_seconds")
        tau = dr.get("tau_seconds")
        rms_mode = dr.get("rms_mode")
        hop_text = f", hop={hop:.4f}s" if isinstance(hop, (int, float)) else ""
        tau_text = f", tau={tau:.2f}s" if isinstance(tau, (int, float)) and rms_mode == "iir" else ""
        mode_text = f", rms={rms_mode}" if rms_mode else ""
        print(f"  Window     : {dr['block_seconds']:.1f}s blocks{hop_text}{tau_text}{mode_text}, top {int(dr['top_fraction']*100)}% RMS")
        if dr.get("top_rms_mean_dbfs") is not None:
            print(f"  Top20 RMS  : {dr['top_rms_mean_dbfs']:.2f} dBFS")

    if results.get("pmf_dr_alt") is not None:
        alt = results["pmf_dr_alt"]
        if isinstance(alt, dict):
            mode = alt.get("rms_mode", "alt")
            dr_db_text = f"{alt['dr_db']:.2f} dB" if alt.get("dr_db") is not None else "n/a"
            top_rms_text = f"{alt['top_rms_mean_dbfs']:.2f} dBFS" if alt.get("top_rms_mean_dbfs") is not None else "n/a"
            print(f"  Compare    : DR{alt.get('dr_value')} ({dr_db_text}), top20={top_rms_text} [rms={mode}]")

    # Short-term analysis results
    if results['windowed_analysis'] is not None:
        wa = results['windowed_analysis']
        print(f"\n🔍 Short-term Window Analysis (50ms windows):")
        print(f"  Mean CF    : {wa['mean_cf']:.2f} dB")
        print(f"  Std Dev    : {wa['std_cf']:.2f} dB")
        print(f"  Min CF     : {wa['min_cf']:.2f} dB")
        print(f"  Max CF     : {wa['max_cf']:.2f} dB")
        print(f"  Dynamic Range: {wa['max_cf'] - wa['min_cf']:.2f} dB")
    
    # LUFS loudness analysis results (FFmpeg priority)
    if results['lufs_analysis'] is not None:
        lufs = results['lufs_analysis']
        source_tag = f" [{lufs.get('source', 'Unknown')}]" if 'source' in lufs else ""
        print(f"\n🔊 LUFS Loudness Analysis (EBU R128){source_tag}:")
        
        if lufs.get('integrated_lufs') is not None and lufs['integrated_lufs'] > -70:
            print(f"  Integrated : {lufs['integrated_lufs']:.1f} LUFS")
        else:
            print(f"  Integrated : Invalid/too quiet")
        
        if lufs.get('loudness_range') is not None:
            print(f"  LRA        : {lufs['loudness_range']:.1f} LU")
            
        # If short-term loudness data is available (Python version only)
        if lufs.get('short_term_lufs') is not None:
            st = lufs['short_term_lufs']
            print(f"  Short-term :")
            print(f"    Mean     : {st['mean']:.1f} LUFS")
            print(f"    Max      : {st['max']:.1f} LUFS")
            print(f"    Min      : {st['min']:.1f} LUFS")
            print(f"    Std Dev  : {st['std']:.1f} LU")
    elif not results.get('ffmpeg_available', False):
        print(f"\n🔊 LUFS Loudness Analysis: FFmpeg unavailable")
    else:
        print(f"\n🔊 LUFS Loudness Analysis: Analysis failed or unsupported audio format")

def crest_factor_db(file_path):
    """Backward-compatible simple interface"""
    results = advanced_crest_analysis(file_path, enable_true_peak=False, enable_windowed=False, enable_lufs=False, use_parallel=False)
    if results is None:
        return None, None, None
    return results['sample_peak'], results['rms'], results['sample_crest_db']

def _safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def _safe_int(x):
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None

def _jsonify(x):
    if x is None:
        return None
    if np is not None and isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    return str(x)

def _fmt_num(value, fmt):
    if value is None:
        return None
    try:
        v = float(value)
        if not math.isfinite(v):
            return None
        return format(v, fmt)
    except Exception:
        return None

def _fmt_db(value, unit, fmt=".2f"):
    s = _fmt_num(value, fmt)
    return f"{s} {unit}" if s is not None else None

def write_album_summary_csv(directory, rows, output_name="crest_album_summary.csv"):
    out_path = Path(directory) / output_name
    fieldnames = [
        "file",
        "status",
        "error",
        "duration_s",
        "sample_rate_hz",
        "channels",
        "sample_peak_dbfs_text",
        "true_peak_dbfs_text",
        "sample_crest_db_text",
        "true_crest_db_text",
        "pmf_dr",
        "pmf_dr_db_text",
        "pmf_dr_peak_source",
        "pmf_dr_channel_mode",
        "pmf_dr_channels",
        "pmf_dr_window_s",
        "pmf_dr_hop_s",
        "integrated_lufs_text",
        "lra_lu_text",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") or "" for k in fieldnames})
    return str(out_path)

def write_album_summary_json(directory, rows, output_name="crest_album_summary.json"):
    out_path = Path(directory) / output_name
    payload = _jsonify(
        {
        "schema": "crest_calc_album_summary_v1",
        "generated_by": "crest.py",
        "directory": str(Path(directory)),
        "file_count": int(len(rows)),
        "rows": rows,
        }
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    return str(out_path)

def analyze_album(
    directory,
    recursive=False,
    simple_mode=False,
    enable_true_peak=True,
    enable_windowed=True,
    enable_lufs=True,
    enable_pmf_dr=False,
    pmf_dr_use_true_peak=False,
    pmf_dr_per_channel=True,
    pmf_dr_hop_seconds=0.01,
    pmf_dr_tau_seconds=3.0,
    pmf_dr_rms_mode="rect",
    pmf_dr_compare=False,
    use_parallel=True,
    summary_name="crest_album_summary.csv",
    album_jobs=1,
):
    files = discover_audio_files(directory, recursive=recursive)
    if not files:
        print(f"No supported audio files found in: {directory}")
        return None

    print(f"Found {len(files)} audio file(s) in: {directory}")
    summary_rows = []

    def _analyze_one(idx, path_obj):
        file_path = str(path_obj)
        buf = io.StringIO()
        row = None

        with contextlib.redirect_stdout(buf):
            print(f"\n[{idx}/{len(files)}] {file_path}")
            try:
                if simple_mode and not enable_pmf_dr:
                    peak, rms, crest_db = crest_factor_db(file_path)
                    if peak is None:
                        print("Audio file is invalid or contains only silence")
                        row = {
                            "file": file_path,
                            "status": "error",
                            "error": "invalid_or_silent",
                        }
                        return file_path, buf.getvalue(), row

                    print(f"File: {file_path}")
                    print(f"Peak: {peak:.6f}")
                    print(f"RMS: {rms:.6f}")
                    print(f"Crest Factor: {crest_db:.2f} dB")

                    row = {
                        "file": file_path,
                        "status": "ok",
                        "error": None,
                        "duration_s": None,
                        "sample_rate_hz": None,
                        "channels": None,
                        "sample_peak_dbfs": _safe_float(lin2dbfs(peak)),
                        "sample_peak_dbfs_text": _fmt_db(lin2dbfs(peak), "dBFS", fmt=".2f"),
                        "true_peak_dbfs": None,
                        "true_peak_dbfs_text": None,
                        "sample_crest_db": _safe_float(crest_db),
                        "sample_crest_db_text": _fmt_db(crest_db, "dB", fmt=".2f"),
                        "true_crest_db": None,
                        "true_crest_db_text": None,
                        "pmf_dr": None,
                        "pmf_dr_value": None,
                        "pmf_dr_db": None,
                        "pmf_dr_db_text": None,
                        "pmf_dr_peak_source": None,
                        "pmf_dr_channel_mode": None,
                        "pmf_dr_channels": None,
                        "integrated_lufs": None,
                        "integrated_lufs_text": None,
                        "lra_lu": None,
                        "lra_lu_text": None,
                    }
                    return file_path, buf.getvalue(), row

                # Avoid nested parallel oversubscription: if running songs in parallel, disable per-song task parallelism.
                internal_parallel = use_parallel if album_jobs <= 1 else False
                results = advanced_crest_analysis(
                    file_path,
                    enable_true_peak=enable_true_peak,
                    enable_windowed=enable_windowed,
                    enable_lufs=enable_lufs,
                    enable_pmf_dr=enable_pmf_dr,
                    pmf_dr_use_true_peak=pmf_dr_use_true_peak,
                    pmf_dr_per_channel=pmf_dr_per_channel,
                    pmf_dr_hop_seconds=pmf_dr_hop_seconds,
                    pmf_dr_tau_seconds=pmf_dr_tau_seconds,
                    pmf_dr_rms_mode=pmf_dr_rms_mode,
                    pmf_dr_compare=pmf_dr_compare,
                    use_parallel=internal_parallel,
                )
                if results is None:
                    print("Analysis failed or audio file is invalid")
                    row = {
                        "file": file_path,
                        "status": "error",
                        "error": "analysis_failed_or_invalid",
                    }
                    return file_path, buf.getvalue(), row

                print_analysis_results(results)

                lufs = results.get("lufs_analysis") or {}
                dr = results.get("pmf_dr") or {}
                sample_peak_dbfs = _safe_float(lin2dbfs(results.get("sample_peak"))) if results.get("sample_peak") is not None else None
                true_peak_dbfs = _safe_float(results.get("true_peak_dbfs")) if results.get("true_peak_dbfs") is not None else None
                sample_crest_db = _safe_float(results.get("sample_crest_db")) if results.get("sample_crest_db") is not None else None
                true_crest_db = _safe_float(results.get("true_crest_db")) if results.get("true_crest_db") is not None else None
                pmf_dr_value = _safe_int(dr.get("dr_value")) if dr.get("dr_value") is not None else None
                pmf_dr_db = _safe_float(dr.get("dr_db")) if dr.get("dr_db") is not None else None
                pmf_dr_channel_mode = dr.get("channel_mode") if isinstance(dr, dict) else None
                per_channel_vals = dr.get("per_channel_dr_values") if isinstance(dr, dict) else None
                pmf_dr_channels = ",".join([str(v) for v in per_channel_vals]) if isinstance(per_channel_vals, list) else None
                pmf_dr_window_s = dr.get("block_seconds") if isinstance(dr, dict) else None
                pmf_dr_hop_s = dr.get("hop_seconds") if isinstance(dr, dict) else None
                integrated_lufs = _safe_float(lufs.get("integrated_lufs")) if lufs.get("integrated_lufs") is not None else None
                lra_lu = _safe_float(lufs.get("loudness_range")) if lufs.get("loudness_range") is not None else None

                row = {
                    "file": file_path,
                    "status": "ok",
                    "error": None,
                    "duration_s": _safe_float(results.get("duration")),
                    "sample_rate_hz": _safe_int(results.get("sample_rate")),
                    "channels": _safe_int(results.get("channels")),
                    "sample_peak_dbfs": sample_peak_dbfs,
                    "sample_peak_dbfs_text": _fmt_db(sample_peak_dbfs, "dBFS", fmt=".2f"),
                    "true_peak_dbfs": true_peak_dbfs,
                    "true_peak_dbfs_text": _fmt_db(true_peak_dbfs, "dBFS", fmt=".2f"),
                    "sample_crest_db": sample_crest_db,
                    "sample_crest_db_text": _fmt_db(sample_crest_db, "dB", fmt=".2f"),
                    "true_crest_db": true_crest_db,
                    "true_crest_db_text": _fmt_db(true_crest_db, "dB", fmt=".2f"),
                    "pmf_dr": (f"DR{pmf_dr_value}" if pmf_dr_value is not None else None),
                    "pmf_dr_value": pmf_dr_value,
                    "pmf_dr_db": pmf_dr_db,
                    "pmf_dr_db_text": _fmt_db(pmf_dr_db, "dB", fmt=".2f"),
                    "pmf_dr_peak_source": results.get("pmf_dr_peak_source") if results.get("pmf_dr_peak_source") is not None else None,
                    "pmf_dr_channel_mode": pmf_dr_channel_mode,
                    "pmf_dr_channels": pmf_dr_channels,
                    "pmf_dr_window_s": _safe_float(pmf_dr_window_s),
                    "pmf_dr_hop_s": _safe_float(pmf_dr_hop_s),
                    "integrated_lufs": integrated_lufs,
                    "integrated_lufs_text": (_fmt_db(integrated_lufs, "LUFS", fmt=".1f") if integrated_lufs is not None else None),
                    "lra_lu": lra_lu,
                    "lra_lu_text": (_fmt_db(lra_lu, "LU", fmt=".1f") if lra_lu is not None else None),
                }
                return file_path, buf.getvalue(), row
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                row = {
                    "file": file_path,
                    "status": "error",
                    "error": str(e),
                }
                return file_path, buf.getvalue(), row

    jobs = max(1, min(int(album_jobs or 1), len(files)))
    if jobs == 1:
        for idx, path in enumerate(files, start=1):
            _, out_text, row = _analyze_one(idx, path)
            print(out_text, end="")
            summary_rows.append(row)
    else:
        print(f"Album parallelism: {jobs} song(s) at a time")
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_analyze_one, idx, path) for idx, path in enumerate(files, start=1)]
            for fut in as_completed(futures):
                _, out_text, row = fut.result()
                print(out_text, end="")
                summary_rows.append(row)

    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
    err_rows = [r for r in summary_rows if r.get("status") != "ok"]
    total_duration = sum([r.get("duration_s") or 0.0 for r in ok_rows])
    dr_db_vals = [r.get("pmf_dr_db") for r in ok_rows if r.get("pmf_dr_db") is not None]
    lufs_vals = [r.get("integrated_lufs") for r in ok_rows if r.get("integrated_lufs") is not None]

    print("\n" + "=" * 60)
    print("Album Summary:")
    print(f"  Directory  : {directory}")
    print(f"  Files      : {len(files)} (ok={len(ok_rows)}, error={len(err_rows)})")
    if total_duration > 0:
        print(f"  Duration   : {total_duration:.1f} s")
    if dr_db_vals:
        print(f"  DR (avg)   : {float(np.mean(dr_db_vals)):.2f} dB")
        print(f"  DR (min/max): {float(np.min(dr_db_vals)):.2f} / {float(np.max(dr_db_vals)):.2f} dB")
    if lufs_vals:
        print(f"  LUFS (avg) : {float(np.mean(lufs_vals)):.1f} LUFS")

    base = summary_name
    if base.lower().endswith(".csv"):
        csv_name = base
        json_name = base[:-4] + ".json"
    elif base.lower().endswith(".json"):
        json_name = base
        csv_name = base[:-5] + ".csv"
    else:
        csv_name = base + ".csv"
        json_name = base + ".json"

    csv_path = write_album_summary_csv(directory, summary_rows, output_name=csv_name)
    json_path = write_album_summary_json(directory, summary_rows, output_name=json_name)
    print(f"\nSummary written to: {csv_path}")
    print(f"Summary written to: {json_path}")
    return {"csv": csv_path, "json": json_path, "rows": summary_rows}

if __name__ == "__main__":
    if np is None or sf is None:
        if "--check-deps" not in sys.argv:
            missing = []
            if np is None:
                missing.append("numpy")
            if sf is None:
                missing.append("soundfile")
            print(f"Missing Python dependencies: {', '.join(missing)}")
            print("Install with: pip install numpy soundfile")
            sys.exit(2)

    if len(sys.argv) < 2:
        print("Usage: python crest.py <audio_file> [options]")
        print("       python crest.py --album <directory> [options]")
        print("       python crest.py --check-deps")
        print("  --simple: Use simple mode (backward compatible output)")
        print("  --no-true-peak: Disable True Peak calculation")
        print("  --no-windowed: Disable short-term window analysis")
        print("  --no-lufs: Disable LUFS loudness analysis")
        print("  --pmf-dr: Calculate PMF Dynamic Range (TT DR-style, Sample Peak)")
        print("  --pmf-dr-mk2: Calculate PMF Dynamic Range using True Peak (MkII-style)")
        print("  --pmf-dr-per-channel: TT-style channel handling (default)")
        print("  --pmf-dr-mix: Mixdown channel handling (legacy)")
        print("  --pmf-dr-hop S: Hop size in seconds for top-20% RMS sampling (default 0.01)")
        print("  --pmf-dr-rms {rect|iir}: RMS sampling mode (default rect)")
        print("  --pmf-dr-tau S: IIR RMS time constant in seconds (default 3.0, used when --pmf-dr-rms iir)")
        print("  --pmf-dr-compare: Print rect vs iir comparison")
        print("  --album <dir>: Analyze all audio files in a directory")
        print("  --recursive: When using --album, scan subdirectories too")
        print("  --album-jobs N: Song parallelism for --album (default 1)")
        print("  --no-parallel: Disable parallel processing")
        print("  --benchmark: Show performance benchmark information")
        print("  --check-deps: Check dependencies and FFmpeg availability")
        sys.exit(1)

    # Check special commands
    if "--check-deps" in sys.argv:
        print("🔧 Dependency Check:")
        print(f"  NumPy: {'✅' if np is not None else '❌ Missing'}")
        print(f"  SoundFile: {'✅' if sf is not None else '❌ Missing'}")
        print(f"  FFmpeg: {'✅ Available' if FFMPEG_AVAILABLE else '❌ Unavailable'}")
        if FFMPEG_AVAILABLE:
            try:
                result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, 
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                first_line = result.stdout.split('\n')[0] if result.stdout else "Version information unavailable"
                print(f"    {first_line}")
            except:
                print("    Version information unavailable")
        else:
            print("    Please install FFmpeg for optimal performance and authoritative LUFS/True Peak analysis")
        
        print(f"\n⚡ System Information:")
        print(f"  CPU Cores: {CPU_COUNT}")
        sys.exit(0)
    
    album_dir = None
    recursive_album = "--recursive" in sys.argv
    if "--album" in sys.argv:
        try:
            album_dir = sys.argv[sys.argv.index("--album") + 1]
        except Exception:
            print("Error: --album requires a directory argument")
            sys.exit(2)

    album_jobs = 1
    if "--album-jobs" in sys.argv:
        try:
            album_jobs = int(sys.argv[sys.argv.index("--album-jobs") + 1])
        except Exception:
            print("Error: --album-jobs requires an integer argument")
            sys.exit(2)

    pmf_dr_hop_seconds = 0.01
    if "--pmf-dr-hop" in sys.argv:
        try:
            pmf_dr_hop_seconds = float(sys.argv[sys.argv.index("--pmf-dr-hop") + 1])
        except Exception:
            print("Error: --pmf-dr-hop requires a numeric argument (seconds)")
            sys.exit(2)
        if pmf_dr_hop_seconds <= 0:
            print("Error: --pmf-dr-hop must be > 0")
            sys.exit(2)

    pmf_dr_rms_mode = "rect"
    if "--pmf-dr-rms" in sys.argv:
        try:
            pmf_dr_rms_mode = str(sys.argv[sys.argv.index("--pmf-dr-rms") + 1]).strip().lower()
        except Exception:
            print("Error: --pmf-dr-rms requires {rect|iir}")
            sys.exit(2)
        if pmf_dr_rms_mode not in ("rect", "iir"):
            print("Error: --pmf-dr-rms must be one of: rect, iir")
            sys.exit(2)

    pmf_dr_tau_seconds = 3.0
    if "--pmf-dr-tau" in sys.argv:
        try:
            pmf_dr_tau_seconds = float(sys.argv[sys.argv.index("--pmf-dr-tau") + 1])
        except Exception:
            print("Error: --pmf-dr-tau requires a numeric argument (seconds)")
            sys.exit(2)
        if pmf_dr_tau_seconds <= 0:
            print("Error: --pmf-dr-tau must be > 0")
            sys.exit(2)

    pmf_dr_compare = "--pmf-dr-compare" in sys.argv
    
    file_path = None if album_dir else sys.argv[1]
    
    # Parse command line arguments
    simple_mode = "--simple" in sys.argv
    enable_true_peak = "--no-true-peak" not in sys.argv
    enable_windowed = "--no-windowed" not in sys.argv
    enable_lufs = "--no-lufs" not in sys.argv
    enable_pmf_dr = ("--pmf-dr" in sys.argv) or ("--pmf-dr-mk2" in sys.argv)
    pmf_dr_use_true_peak = "--pmf-dr-mk2" in sys.argv
    pmf_dr_per_channel = "--pmf-dr-mix" not in sys.argv
    use_parallel = "--no-parallel" not in sys.argv
    show_benchmark = "--benchmark" in sys.argv

    if album_dir:
        # Default album mode to PMF DR MkII unless explicitly overridden.
        if not enable_pmf_dr:
            enable_pmf_dr = True
            pmf_dr_use_true_peak = True
        if show_benchmark:
            print("Note: --benchmark is ignored when using --album")

        analyze_album(
            album_dir,
            recursive=recursive_album,
            simple_mode=simple_mode,
            enable_true_peak=enable_true_peak,
            enable_windowed=enable_windowed,
            enable_lufs=enable_lufs,
            enable_pmf_dr=enable_pmf_dr,
            pmf_dr_use_true_peak=pmf_dr_use_true_peak,
            pmf_dr_per_channel=pmf_dr_per_channel,
            pmf_dr_hop_seconds=pmf_dr_hop_seconds,
            pmf_dr_tau_seconds=pmf_dr_tau_seconds,
            pmf_dr_rms_mode=pmf_dr_rms_mode,
            pmf_dr_compare=pmf_dr_compare,
            use_parallel=use_parallel,
            album_jobs=album_jobs,
        )
        sys.exit(0)
    
    if simple_mode:
        # Compatible mode: use original simple output
        if enable_pmf_dr:
            # Minimal analysis required for PMF DR (needs full read anyway)
            results = advanced_crest_analysis(
                file_path,
                enable_true_peak=pmf_dr_use_true_peak and enable_true_peak,
                enable_windowed=False,
                enable_lufs=False,
                enable_pmf_dr=True,
                pmf_dr_use_true_peak=pmf_dr_use_true_peak,
                pmf_dr_per_channel=pmf_dr_per_channel,
                pmf_dr_hop_seconds=pmf_dr_hop_seconds,
                pmf_dr_tau_seconds=pmf_dr_tau_seconds,
                pmf_dr_rms_mode=pmf_dr_rms_mode,
                pmf_dr_compare=pmf_dr_compare,
                use_parallel=False,
            )
            if results is None or results.get('pmf_dr') is None:
                print("Audio file is invalid or contains only silence")
            else:
                dr = results['pmf_dr']
                peak_src = results.get('pmf_dr_peak_source', 'unknown')
                algo = "PMF DR MkII (True Peak)" if peak_src == "true_peak" else "PMF DR (Sample Peak)"
                print(f"File: {file_path}")
                print(f"{algo}: DR{dr['dr_value']} ({dr['dr_db']:.2f} dB)")
        else:
            peak, rms, crest_db = crest_factor_db(file_path)
            if peak is None:
                print("Audio file is invalid or contains only silence")
            else:
                print(f"File: {file_path}")
                print(f"Peak: {peak:.6f}")
                print(f"RMS: {rms:.6f}")
                print(f"Crest Factor: {crest_db:.2f} dB")
    else:
        # Enhanced mode: use full analysis
        if show_benchmark:
            print(f"\n⚡ Performance Benchmark")
            print(f"CPU Cores: {CPU_COUNT}")
            print(f"Parallelization: {'Enabled' if use_parallel else 'Disabled'}")
            print("=" * 50)
            
            # Test serial version
            start_time = time.time()
            results_serial = advanced_crest_analysis(
                file_path,
                enable_true_peak,
                enable_windowed,
                enable_lufs,
                enable_pmf_dr=enable_pmf_dr,
                pmf_dr_use_true_peak=pmf_dr_use_true_peak,
                pmf_dr_per_channel=pmf_dr_per_channel,
                pmf_dr_hop_seconds=pmf_dr_hop_seconds,
                pmf_dr_tau_seconds=pmf_dr_tau_seconds,
                pmf_dr_rms_mode=pmf_dr_rms_mode,
                pmf_dr_compare=pmf_dr_compare,
                use_parallel=False,
            )
            serial_time = time.time() - start_time
            
            if use_parallel:
                # Test parallel version
                start_time = time.time()
                results_parallel = advanced_crest_analysis(
                    file_path,
                    enable_true_peak,
                    enable_windowed,
                    enable_lufs,
                    enable_pmf_dr=enable_pmf_dr,
                    pmf_dr_use_true_peak=pmf_dr_use_true_peak,
                    pmf_dr_per_channel=pmf_dr_per_channel,
                    pmf_dr_hop_seconds=pmf_dr_hop_seconds,
                    pmf_dr_tau_seconds=pmf_dr_tau_seconds,
                    pmf_dr_rms_mode=pmf_dr_rms_mode,
                    pmf_dr_compare=pmf_dr_compare,
                    use_parallel=True,
                )
                parallel_time = time.time() - start_time
                
                print(f"Serial Processing Time: {serial_time:.3f} seconds")
                print(f"Parallel Processing Time: {parallel_time:.3f} seconds")
                print(f"Performance Improvement: {serial_time/parallel_time:.2f}x")
                results = results_parallel
            else:
                print(f"Processing Time: {serial_time:.3f} seconds")
                results = results_serial
        else:
            # Normal mode
            start_time = time.time()
            results = advanced_crest_analysis(
                file_path,
                enable_true_peak,
                enable_windowed,
                enable_lufs,
                enable_pmf_dr=enable_pmf_dr,
                pmf_dr_use_true_peak=pmf_dr_use_true_peak,
                pmf_dr_per_channel=pmf_dr_per_channel,
                pmf_dr_hop_seconds=pmf_dr_hop_seconds,
                pmf_dr_tau_seconds=pmf_dr_tau_seconds,
                pmf_dr_rms_mode=pmf_dr_rms_mode,
                pmf_dr_compare=pmf_dr_compare,
                use_parallel=use_parallel,
            )
            end_time = time.time()
            
            if use_parallel:
                print(f"\n⚡ Processing Time: {end_time - start_time:.3f} seconds (parallelized)")
            else:
                print(f"\n⚡ Processing Time: {end_time - start_time:.3f} seconds (serial)")
        
        print_analysis_results(results)
