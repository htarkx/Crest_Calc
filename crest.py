import sys
import numpy as np
import soundfile as sf
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import multiprocessing as mp
import time
import subprocess
import json
import shutil
import os

# Get CPU core count for parallelization
CPU_COUNT = mp.cpu_count()

# Check FFmpeg availability
def check_ffmpeg():
    """Check if FFmpeg is available in the system PATH"""
    return shutil.which("ffmpeg") is not None

FFMPEG_AVAILABLE = check_ffmpeg()

def lin2dbfs(x):
    """Convert linear amplitude to dBFS (decibels relative to full scale)"""
    return 20 * np.log10(x) if x > 0 else -np.inf

def remove_dc_offset(data):
    """Remove DC offset by subtracting the mean value from each channel"""
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
            'sample_peak_dbfs': None
        }
        
        # Find key information in Summary section
        in_summary = False
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
                
                # Parse True peak
                elif line.startswith('Peak:') and 'dBFS' in line:
                    try:
                        # Format: "Peak:       -0.1 dBFS"
                        parts = line.split()
                        if len(parts) >= 2:
                            analysis_results['true_peak_dbfs'] = float(parts[1])
                    except (ValueError, IndexError):
                        pass
        
        return analysis_results
        
    except Exception as e:
        warnings.warn(f"FFmpeg analysis failed: {e}")
        return None

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
    if dbfs_value is None:
        return None
    return 10 ** (dbfs_value / 20)

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

def advanced_crest_analysis(file_path, enable_true_peak=True, enable_windowed=True, enable_lufs=True, use_parallel=True):
    """Advanced Crest Factor analysis - FFmpeg + vectorized optimization version"""
    try:
        # Force read as 2D array to preserve multi-channel information
        data, samplerate = sf.read(file_path, always_2d=True)
        
        # Ensure data is float32 type and normalized
        if data.dtype != np.float32:
            if np.issubdtype(data.dtype, np.integer):
                # Integer types need normalization
                max_val = np.iinfo(data.dtype).max
                data = data.astype(np.float32) / max_val
            else:
                data = data.astype(np.float32)
        
        # Remove DC offset
        data = remove_dc_offset(data)
        
        # Basic calculations (always required) - vectorized optimization
        sample_peak = np.max(np.abs(data))
        
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python crest.py <audio_file> [options] or python crest.py --check-deps")
        print("  --simple: Use simple mode (backward compatible output)")
        print("  --no-true-peak: Disable True Peak calculation")
        print("  --no-windowed: Disable short-term window analysis")
        print("  --no-lufs: Disable LUFS loudness analysis")
        print("  --no-parallel: Disable parallel processing")
        print("  --benchmark: Show performance benchmark information")
        print("  --check-deps: Check dependencies and FFmpeg availability")
        sys.exit(1)

    # Check special commands
    if "--check-deps" in sys.argv:
        print("🔧 Dependency Check:")
        print(f"  NumPy: ✅")
        print(f"  SoundFile: ✅")
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
    
    file_path = sys.argv[1]
    
    # Parse command line arguments
    simple_mode = "--simple" in sys.argv
    enable_true_peak = "--no-true-peak" not in sys.argv
    enable_windowed = "--no-windowed" not in sys.argv
    enable_lufs = "--no-lufs" not in sys.argv
    use_parallel = "--no-parallel" not in sys.argv
    show_benchmark = "--benchmark" in sys.argv
    
    if simple_mode:
        # Compatible mode: use original simple output
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
            results_serial = advanced_crest_analysis(file_path, enable_true_peak, enable_windowed, enable_lufs, use_parallel=False)
            serial_time = time.time() - start_time
            
            if use_parallel:
                # Test parallel version
                start_time = time.time()
                results_parallel = advanced_crest_analysis(file_path, enable_true_peak, enable_windowed, enable_lufs, use_parallel=True)
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
            results = advanced_crest_analysis(file_path, enable_true_peak, enable_windowed, enable_lufs, use_parallel)
            end_time = time.time()
            
            if use_parallel:
                print(f"\n⚡ Processing Time: {end_time - start_time:.3f} seconds (parallelized)")
            else:
                print(f"\n⚡ Processing Time: {end_time - start_time:.3f} seconds (serial)")
        
        print_analysis_results(results)
