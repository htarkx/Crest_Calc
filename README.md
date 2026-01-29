# Professional Audio Crest Factor Analyzer

A high-performance, production-ready audio analysis tool for professional audio engineers, mastering engineers, and broadcast professionals. This tool provides comprehensive audio analysis including Crest Factor, True Peak detection, LUFS loudness measurement, and dynamic range analysis.

## 🎯 Key Features

### Professional Audio Analysis
- **Crest Factor Analysis**: Sample Peak, True Peak, and windowed analysis
- **LUFS Loudness Measurement**: EBU R128/ITU-R BS.1770 compliant
- **True Peak Detection**: Industry-standard reconstruction filtering
- **PMF Dynamic Range (DR)**: TT DR-style `DR = Peak - (Top 20% RMS)` with Sample Peak or True Peak
- **Dynamic Range Analysis**: Short-term windowed crest statistics (not TT DR)
- **Multi-channel Support**: Proper power-based channel mixing

### High-Performance Architecture
- **FFmpeg Integration**: Authoritative audio processing with EBU R128 compliance
- **Vectorized Computing**: NumPy-optimized array operations for maximum speed
- **Parallel Processing**: Multi-threaded analysis for optimal CPU utilization
- **Memory Efficient**: Streaming analysis for large audio files

### Production-Ready Features
- **Industry Standards**: Broadcast and streaming platform compliance
- **Error Handling**: Graceful degradation and comprehensive error reporting
- **Cross-Platform**: Windows, macOS, and Linux support
- **Format Support**: All major audio formats via FFmpeg

## 📊 Performance Benchmarks

**Test File**: Radiohead - Paranoid Android (96kHz, 2-channel, 384s FLAC)
**System**: 32-core CPU

| Implementation | Processing Time | Speed Improvement | Key Optimizations |
|---------------|----------------|-------------------|-------------------|
| Original (pyloudnorm) | 11.92s | 1.0x | Baseline |
| Parallelized | 7.77s | 1.53x | Multi-threading |
| **FFmpeg + Vectorized** | **2.47s** | **4.83x** | Authority + Vectorization |

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure FFmpeg is installed and available in PATH
ffmpeg -version

# Install Python dependencies
pip install numpy soundfile
```

### Basic Usage
```bash
# Full analysis with all features
python crest.py audio_file.wav

# Analyze a whole directory (album)
python crest.py --album /path/to/album
#
# Writes a summary file into the directory:
#   crest_album_summary.csv
#   crest_album_summary.json

# Simple mode (backward compatibility)
python crest.py audio_file.wav --simple

# Performance benchmark
python crest.py audio_file.wav --benchmark

# Check system dependencies
python crest.py --check-deps
```

### Advanced Options
```bash
# Disable specific analysis modules
python crest.py audio_file.wav --no-true-peak --no-windowed --no-lufs

# PMF Dynamic Range (TT DR-style)
python crest.py audio_file.wav --pmf-dr          # Sample Peak
python crest.py audio_file.wav --pmf-dr-mk2      # True Peak (requires FFmpeg for True Peak)

# Album mode options
python crest.py --album /path/to/album --recursive
python crest.py --album /path/to/album --album-jobs 4

# PMF DR matching / experimentation
python crest.py audio_file.wav --pmf-dr-hop 0.01
python crest.py audio_file.wav --pmf-dr-rms iir --pmf-dr-tau 3.0
python crest.py audio_file.wav --pmf-dr-compare

# Disable parallel processing
python crest.py audio_file.wav --no-parallel

# Performance comparison
python crest.py audio_file.wav --benchmark
```

## 📈 Analysis Output

### Comprehensive Audio Statistics
```
============================================================
File: example.wav
Sample Rate: 44100 Hz
Channels: 2
Duration: 3.45 seconds
============================================================

📊 Basic Audio Statistics:
  Sample Peak: 0.987654 (-0.11 dBFS)
  True Peak  : 1.023456 (+0.20 dBFS) [FFmpeg]
  RMS        : 0.234567 (-12.58 dBFS)

🎯 Crest Factor:
  Sample CF  : 12.47 dB
  True CF    : 12.78 dB

📏 PMF Dynamic Range (TT DR-style):
  DR         : DR8 (7.62 dB) [True Peak]
  Window     : 3.0s blocks, hop=0.0100s, rms=rect, top 20% RMS
  Top20 RMS  : -7.62 dBFS

🔍 Short-term Window Analysis (50ms windows):
  Mean CF    : 11.23 dB
  Std Dev    : 2.45 dB
  Min CF     : 6.78 dB
  Max CF     : 18.90 dB
  Dynamic Range: 12.12 dB

🔊 LUFS Loudness Analysis (EBU R128) [ffmpeg]:
  Integrated: -23.4 LUFS
  LRA       : 8.0 LU
```

## 🔁 TT DR Meter Compatibility Notes

This project aims to be as close as practical to the classic TT DR (PMF) workflow (top-20% loudness statistics + peak), but there are still implementation details that can cause systematic offsets.

- **RMS integration (ballistics) differences:** TT DR implementations may behave closer to an RMS ballistics / time-constant tracker than a pure rectangular sliding-window RMS. This tool supports both for verification:
  - `--pmf-dr-rms rect` (rectangular sliding window)
  - `--pmf-dr-rms iir --pmf-dr-tau 3.0` (IIR/EMA power tracking, τ-based)
  - `--pmf-dr-compare` prints both results side-by-side.
- **Sine calibration offset (~3 dB):** Some TT DR tooling/reporting applies an approx. **3 dB sine-wave compensation** (often described as aligning sine RMS vs peak conventions). If you want to compare against **DR Database / TT DR log “DR dB” style values**, you may need to **manually subtract ~3 dB** from this tool’s `dr_db` before comparing.
- **Rounding amplifies small differences:** DR values are typically reported as integers (e.g. `DR8`). A small dB-level discrepancy in `top20 RMS` or peak handling can be **magnified after rounding**, resulting in a **±1 DR step** difference (“一个档位”).

## 🛠️ Technical Architecture

### Dual-Engine Design
- **FFmpeg Engine**: LUFS, True Peak, LRA (authoritative implementation)
- **Python Engine**: Crest Factor analysis (vectorized computation)

### Parallel Processing Strategy
```python
# Concurrent execution of analysis tasks
tasks = [
    ffmpeg_analysis(file_path),      # I-LUFS, LRA, True Peak
    python_windowed_analysis(data)    # Vectorized CF analysis
]
parallel_execution(tasks)  # Maximize CPU utilization
```

### Vectorized Crest Factor Analysis
```python
# NumPy-optimized sliding window analysis
from numpy.lib.stride_tricks import sliding_window_view

windowed_data = sliding_window_view(data, window_shape=win_samples)[::hop_samples]
peaks = np.max(np.abs(windowed_data), axis=1)           # Vectorized peaks
rms_values = np.sqrt(np.mean(windowed_data**2, axis=1)) # Vectorized RMS
crest_factors = 20 * np.log10(peaks / rms_values)       # Vectorized CF
```



### Performance Tuning
```python
# CPU core utilization
CPU_COUNT = mp.cpu_count()

# Parallel processing thresholds
if len(window_args) < 100:  # Small datasets use serial processing
    use_serial_processing()
else:
    use_parallel_processing()
```




## 🔍 Troubleshooting

### Common Issues
```bash
# Check FFmpeg availability
python crest.py --check-deps

# Verify audio file format
ffprobe audio_file.wav

# Test with simple mode
python crest.py audio_file.wav --simple
```
