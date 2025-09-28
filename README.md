# Professional Audio Crest Factor Analyzer

A high-performance, production-ready audio analysis tool for professional audio engineers, mastering engineers, and broadcast professionals. This tool provides comprehensive audio analysis including Crest Factor, True Peak detection, LUFS loudness measurement, and dynamic range analysis.

## 🎯 Key Features

### Professional Audio Analysis
- **Crest Factor Analysis**: Sample Peak, True Peak, and windowed analysis
- **LUFS Loudness Measurement**: EBU R128/ITU-R BS.1770 compliant
- **True Peak Detection**: Industry-standard reconstruction filtering
- **Dynamic Range Analysis**: Short-term windowed analysis with statistical metrics
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

## 📋 Professional Applications

### Audio Mastering
- **Dynamic Range Assessment**: Identify over-compressed sections
- **True Peak Compliance**: Ensure broadcast-safe levels
- **Loudness Standards**: Meet streaming platform requirements

### Broadcast Engineering
- **EBU R128 Compliance**: Integrated and short-term loudness
- **Peak Level Monitoring**: True Peak vs Sample Peak analysis
- **Dynamic Range Monitoring**: Real-time audio quality assessment

### Audio Quality Control
- **Compression Detection**: Identify over-limited audio
- **Dynamic Range Analysis**: Assess musical dynamics
- **Format Validation**: Ensure proper audio levels

## 🔧 Advanced Configuration

### FFmpeg Integration
The tool automatically detects and uses FFmpeg for authoritative audio analysis:
- **EBU R128 Loudness**: Industry-standard loudness measurement
- **True Peak Detection**: Reconstruction filtering for accurate peak detection
- **LRA Analysis**: Loudness Range Assessment for dynamic content

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

### Error Handling
- **Graceful Degradation**: Falls back to Python implementation if FFmpeg unavailable
- **Comprehensive Logging**: Detailed error reporting and warnings
- **Format Validation**: Automatic audio format detection and handling

## 📚 Technical Specifications

### Supported Audio Formats
- **Lossless**: FLAC, WAV, AIFF, ALAC
- **Lossy**: MP3, AAC, OGG, Opus
- **High-Resolution**: Up to 384kHz/32-bit
- **Multi-channel**: Up to 7.1 surround

### Analysis Parameters
- **Window Size**: 50ms (configurable)
- **Hop Size**: 12.5ms (75% overlap)
- **True Peak Oversampling**: 4x (FFmpeg standard)
- **LUFS Standard**: EBU R128/ITU-R BS.1770

### Performance Characteristics
- **Memory Usage**: Streaming analysis for large files
- **CPU Utilization**: Multi-threaded parallel processing
- **I/O Efficiency**: Optimized file reading and processing
- **Scalability**: Linear scaling with CPU cores

## 🎯 Industry Standards Compliance

### Broadcast Standards
- **EBU R128**: European Broadcasting Union loudness standard
- **ITU-R BS.1770**: International Telecommunication Union standard
- **ATSC A/85**: Advanced Television Systems Committee standard

### Streaming Platform Requirements
- **Spotify**: -14 LUFS integrated loudness
- **Apple Music**: -16 LUFS integrated loudness
- **YouTube**: -14 LUFS integrated loudness
- **Netflix**: -27 LUFS integrated loudness

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

### Performance Optimization
- **Large Files**: Use streaming analysis for files >1GB
- **High Sample Rates**: Consider downsampling for analysis
- **Batch Processing**: Process multiple files in parallel

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

We welcome contributions from the audio engineering community:
- **Bug Reports**: Use GitHub Issues
- **Feature Requests**: Submit detailed proposals
- **Code Contributions**: Follow our coding standards
- **Documentation**: Help improve our documentation

## 📞 Support

For professional support and custom implementations:
- **GitHub Issues**: Technical support and bug reports
- **Documentation**: Comprehensive guides and examples
- **Community**: Audio engineering discussions and best practices

---

**Built for professionals, by professionals.** 🎵⚡