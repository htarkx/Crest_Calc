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

# 获取CPU核心数，用于并行化
CPU_COUNT = mp.cpu_count()

# 检查FFmpeg可用性
def check_ffmpeg():
    """检查系统中是否有可用的FFmpeg"""
    return shutil.which("ffmpeg") is not None

FFMPEG_AVAILABLE = check_ffmpeg()

def lin2dbfs(x):
    """线性值转换为dBFS"""
    return 20 * np.log10(x) if x > 0 else -np.inf

def remove_dc_offset(data):
    """去除直流偏置"""
    return data - np.mean(data, axis=0)

def ffmpeg_audio_analysis(file_path):
    """使用FFmpeg进行音频分析，获取LUFS、True Peak、LRA等指标"""
    if not FFMPEG_AVAILABLE:
        return None
    
    try:
        # 使用FFmpeg的ebur128滤镜获取EBU R128指标
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
            warnings.warn(f"FFmpeg执行失败: {result.stderr}")
            return None
        
        # 解析FFmpeg输出中的EBU R128信息
        output_lines = result.stderr.split('\n')
        
        analysis_results = {
            'integrated_lufs': None,
            'loudness_range': None,
            'true_peak_dbfs': None,
            'sample_peak_dbfs': None
        }
        
        # 寻找Summary部分的关键信息
        in_summary = False
        for line in output_lines:
            line = line.strip()
            
            # 检测Summary部分开始
            if 'Summary:' in line:
                in_summary = True
                continue
            
            if in_summary:
                # 解析 Integrated loudness
                if line.startswith('I:') and 'LUFS' in line:
                    try:
                        # 格式: "I:          -9.8 LUFS"
                        parts = line.split()
                        if len(parts) >= 2:
                            analysis_results['integrated_lufs'] = float(parts[1])
                    except (ValueError, IndexError):
                        pass
                
                # 解析 Loudness range
                elif line.startswith('LRA:') and 'LU' in line:
                    try:
                        # 格式: "LRA:         8.0 LU"
                        parts = line.split()
                        if len(parts) >= 2:
                            analysis_results['loudness_range'] = float(parts[1])
                    except (ValueError, IndexError):
                        pass
                
                # 解析 True peak
                elif line.startswith('Peak:') and 'dBFS' in line:
                    try:
                        # 格式: "Peak:       -0.1 dBFS"
                        parts = line.split()
                        if len(parts) >= 2:
                            analysis_results['true_peak_dbfs'] = float(parts[1])
                    except (ValueError, IndexError):
                        pass
        
        return analysis_results
        
    except Exception as e:
        warnings.warn(f"FFmpeg分析失败: {e}")
        return None

def _calculate_window_crest(args):
    """计算单个窗口的Crest Factor（用于并行化）"""
    segment, sr, start_idx = args
    peak = np.max(np.abs(segment))
    rms = np.sqrt(np.mean(segment**2))
    
    if rms > 0:
        cf_db = 20 * np.log10(peak / rms)
        return start_idx / sr, cf_db
    else:
        return start_idx / sr, None

def frame_crest_analysis_vectorized(data, sr, win_ms=50, hop_ms=12.5):
    """向量化的短时窗口Crest Factor分析 - 极速版本"""
    win_samples = int(sr * win_ms / 1000)
    hop_samples = int(sr * hop_ms / 1000)
    
    # 如果是多声道，按功率合成单声道
    if data.ndim > 1:
        data = np.sqrt(np.mean(data**2, axis=1))
    
    # 计算窗口数量
    num_windows = (len(data) - win_samples) // hop_samples + 1
    
    if num_windows <= 0:
        return np.array([]), np.array([])
    
    # 向量化计算：创建所有窗口的2D数组
    # 使用stride技巧避免数据复制
    from numpy.lib.stride_tricks import sliding_window_view
    
    # 创建滑动窗口视图
    windowed_data = sliding_window_view(data, window_shape=win_samples)[::hop_samples]
    
    if len(windowed_data) == 0:
        return np.array([]), np.array([])
    
    # 向量化计算所有窗口的peak和RMS
    peaks = np.max(np.abs(windowed_data), axis=1)
    rms_values = np.sqrt(np.mean(windowed_data**2, axis=1))
    
    # 过滤有效值（RMS > 0）
    valid_mask = rms_values > 0
    peaks = peaks[valid_mask]
    rms_values = rms_values[valid_mask]
    
    # 向量化计算Crest Factor
    crest_factors = 20 * np.log10(peaks / rms_values)
    
    # 计算对应的时间戳
    valid_indices = np.arange(len(windowed_data))[valid_mask]
    time_stamps = valid_indices * hop_samples / sr
    
    return time_stamps, crest_factors

def frame_crest_analysis(data, sr, win_ms=50, hop_ms=12.5, use_parallel=True):
    """短时窗口Crest Factor分析 - 智能选择向量化或并行化"""
    # 优先使用向量化版本（通常更快）
    try:
        return frame_crest_analysis_vectorized(data, sr, win_ms, hop_ms)
    except Exception as e:
        warnings.warn(f"向量化分析失败，回退到并行化版本: {e}")
        
        # 回退到原并行化版本
        win_samples = int(sr * win_ms / 1000)
        hop_samples = int(sr * hop_ms / 1000)
        
        # 如果是多声道，按功率合成单声道
        if data.ndim > 1:
            data = np.sqrt(np.mean(data**2, axis=1))
        
        # 预计算所有窗口参数
        window_args = []
        for i in range(0, len(data) - win_samples + 1, hop_samples):
            segment = data[i:i + win_samples]
            window_args.append((segment, sr, i))
        
        if not use_parallel or len(window_args) < 100:
            # 串行处理
            time_stamps = []
            crest_factors = []
            for segment, sr, start_idx in window_args:
                timestamp, cf = _calculate_window_crest((segment, sr, start_idx))
                if cf is not None:
                    time_stamps.append(timestamp)
                    crest_factors.append(cf)
        else:
            # 并行处理
            max_workers = min(CPU_COUNT, len(window_args))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(_calculate_window_crest, window_args))
            
            # 过滤有效结果
            time_stamps = []
            crest_factors = []
            for timestamp, cf in results:
                if cf is not None:
                    time_stamps.append(timestamp)
                    crest_factors.append(cf)
        
        return np.array(time_stamps), np.array(crest_factors)

def convert_dbfs_to_linear(dbfs_value):
    """将dBFS值转换为线性值"""
    if dbfs_value is None:
        return None
    return 10 ** (dbfs_value / 20)

def _analysis_task_windowed(data, samplerate):
    """短时窗口分析任务（用于并行化）"""
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
        warnings.warn(f"短时窗口分析失败: {e}")
        return None

def _analysis_task_ffmpeg(file_path):
    """FFmpeg分析任务（用于并行化）"""
    return ffmpeg_audio_analysis(file_path)

def advanced_crest_analysis(file_path, enable_true_peak=True, enable_windowed=True, enable_lufs=True, use_parallel=True):
    """增强的Crest Factor分析 - FFmpeg + 向量化优化版本"""
    try:
        # 强制读取为2D数组，保持多声道信息
        data, samplerate = sf.read(file_path, always_2d=True)
        
        # 确保数据为float32类型并归一化
        if data.dtype != np.float32:
            if np.issubdtype(data.dtype, np.integer):
                # 整数类型需要归一化
                max_val = np.iinfo(data.dtype).max
                data = data.astype(np.float32) / max_val
            else:
                data = data.astype(np.float32)
        
        # 去除DC偏置
        data = remove_dc_offset(data)
        
        # 基本计算（总是需要的）- 向量化优化
        sample_peak = np.max(np.abs(data))
        
        # 计算RMS（向量化优化）
        if data.shape[1] > 1:
            # 多声道：先对每个采样点跨声道求功率平均
            power_per_sample = np.mean(data**2, axis=1)
            rms = np.sqrt(np.mean(power_per_sample))
        else:
            # 单声道：直接计算
            rms = np.sqrt(np.mean(data**2))
        
        # 检查有效性
        if rms == 0:
            return None
        
        # 基本Crest Factor
        sample_crest_db = 20 * np.log10(sample_peak / rms)
        
        # 并行执行任务：FFmpeg分析 + Python窗口分析
        tasks = []
        task_names = []
        
        # FFmpeg分析任务（LUFS + True Peak）
        if (enable_lufs or enable_true_peak) and FFMPEG_AVAILABLE:
            tasks.append(partial(_analysis_task_ffmpeg, file_path))
            task_names.append('ffmpeg')
        
        # Python窗口分析任务
        if enable_windowed:
            tasks.append(partial(_analysis_task_windowed, data, samplerate))
            task_names.append('windowed')
        
        # 执行任务
        results = {}
        if tasks and use_parallel:
            # 并行执行
            with ThreadPoolExecutor(max_workers=min(len(tasks), CPU_COUNT)) as executor:
                futures = [executor.submit(task) for task in tasks]
                for i, future in enumerate(futures):
                    results[task_names[i]] = future.result()
        else:
            # 串行执行
            for i, task in enumerate(tasks):
                results[task_names[i]] = task()
        
        # 提取FFmpeg结果
        ffmpeg_results = results.get('ffmpeg', None)
        
        # 提取True Peak
        true_peak = None
        true_peak_dbfs = None
        if ffmpeg_results and enable_true_peak:
            true_peak_dbfs = ffmpeg_results.get('true_peak_dbfs', None)
            if true_peak_dbfs is not None:
                true_peak = convert_dbfs_to_linear(true_peak_dbfs)
        
        # 提取LUFS分析
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
        
        # 提取窗口分析结果
        windowed_analysis = results.get('windowed', None)
        
        # 计算True Crest Factor
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
        print(f"错误处理文件 {file_path}: {e}")
        return None

def print_analysis_results(results):
    """格式化打印分析结果"""
    if results is None:
        print("分析失败或音频文件无效")
        return
    
    print(f"\n{'='*60}")
    print(f"文件: {results['file_path']}")
    print(f"采样率: {results['sample_rate']} Hz")
    print(f"声道数: {results['channels']}")
    print(f"时长: {results['duration']:.2f} 秒")
    print(f"{'='*60}")
    
    # 基本统计
    print(f"\n📊 基本音频统计:")
    print(f"  Sample Peak: {results['sample_peak']:.6f} ({lin2dbfs(results['sample_peak']):.2f} dBFS)")
    
    # True Peak显示（优先显示FFmpeg结果）
    if results.get('true_peak_dbfs') is not None:
        print(f"  True Peak  : {results['true_peak']:.6f} ({results['true_peak_dbfs']:.2f} dBFS) [FFmpeg]")
    elif results['true_peak'] is not None:
        print(f"  True Peak  : {results['true_peak']:.6f} ({lin2dbfs(results['true_peak']):.2f} dBFS) [Python]")
    elif not results.get('ffmpeg_available', False):
        print(f"  True Peak  : 未计算 (FFmpeg不可用)")
    
    print(f"  RMS        : {results['rms']:.6f} ({lin2dbfs(results['rms']):.2f} dBFS)")
    
    # Crest Factor
    print(f"\n🎯 Crest Factor:")
    print(f"  Sample CF  : {results['sample_crest_db']:.2f} dB")
    if results['true_crest_db'] is not None:
        print(f"  True CF    : {results['true_crest_db']:.2f} dB")
    
    # 短时分析结果
    if results['windowed_analysis'] is not None:
        wa = results['windowed_analysis']
        print(f"\n🔍 短时窗口分析 (50ms窗口):")
        print(f"  平均 CF    : {wa['mean_cf']:.2f} dB")
        print(f"  标准差     : {wa['std_cf']:.2f} dB")
        print(f"  最小 CF    : {wa['min_cf']:.2f} dB")
        print(f"  最大 CF    : {wa['max_cf']:.2f} dB")
        print(f"  动态范围   : {wa['max_cf'] - wa['min_cf']:.2f} dB")
    
    # LUFS响度分析结果（FFmpeg优先）
    if results['lufs_analysis'] is not None:
        lufs = results['lufs_analysis']
        source_tag = f" [{lufs.get('source', 'Unknown')}]" if 'source' in lufs else ""
        print(f"\n🔊 LUFS响度分析 (EBU R128){source_tag}:")
        
        if lufs.get('integrated_lufs') is not None and lufs['integrated_lufs'] > -70:
            print(f"  Integrated : {lufs['integrated_lufs']:.1f} LUFS")
        else:
            print(f"  Integrated : 无效/太安静")
        
        if lufs.get('loudness_range') is not None:
            print(f"  LRA        : {lufs['loudness_range']:.1f} LU")
            
        # 如果有短期响度数据（Python版本才有）
        if lufs.get('short_term_lufs') is not None:
            st = lufs['short_term_lufs']
            print(f"  短期响度   :")
            print(f"    平均     : {st['mean']:.1f} LUFS")
            print(f"    最大     : {st['max']:.1f} LUFS")
            print(f"    最小     : {st['min']:.1f} LUFS")
            print(f"    标准差   : {st['std']:.1f} LU")
    elif not results.get('ffmpeg_available', False):
        print(f"\n🔊 LUFS响度分析: FFmpeg不可用")
    else:
        print(f"\n🔊 LUFS响度分析: 分析失败或音频格式不支持")

def crest_factor_db(file_path):
    """保持向后兼容的简单接口"""
    results = advanced_crest_analysis(file_path, enable_true_peak=False, enable_windowed=False, enable_lufs=False, use_parallel=False)
    if results is None:
        return None, None, None
    return results['sample_peak'], results['rms'], results['sample_crest_db']

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python crest.py <audio_file> [选项] 或 python crest.py --check-deps")
        print("  --simple: 使用简单模式（兼容旧版本输出）")
        print("  --no-true-peak: 禁用True Peak计算")
        print("  --no-windowed: 禁用短时窗口分析")
        print("  --no-lufs: 禁用LUFS响度分析")
        print("  --no-parallel: 禁用并行化处理")
        print("  --benchmark: 显示性能基准测试信息")
        print("  --check-deps: 检查依赖项和FFmpeg可用性")
        sys.exit(1)

    # 检查特殊命令
    if "--check-deps" in sys.argv:
        print("🔧 依赖项检查:")
        print(f"  NumPy: ✅")
        print(f"  SoundFile: ✅")
        print(f"  FFmpeg: {'✅ 可用' if FFMPEG_AVAILABLE else '❌ 不可用'}")
        if FFMPEG_AVAILABLE:
            try:
                result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, 
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                first_line = result.stdout.split('\n')[0] if result.stdout else "版本信息获取失败"
                print(f"    {first_line}")
            except:
                print("    版本信息获取失败")
        else:
            print("    请安装FFmpeg以获得最佳性能和权威LUFS/True Peak分析")
        
        print(f"\n⚡ 系统信息:")
        print(f"  CPU核心数: {CPU_COUNT}")
        sys.exit(0)
    
    file_path = sys.argv[1]
    
    # 解析命令行参数
    simple_mode = "--simple" in sys.argv
    enable_true_peak = "--no-true-peak" not in sys.argv
    enable_windowed = "--no-windowed" not in sys.argv
    enable_lufs = "--no-lufs" not in sys.argv
    use_parallel = "--no-parallel" not in sys.argv
    show_benchmark = "--benchmark" in sys.argv
    
    if simple_mode:
        # 兼容模式：使用原始简单输出
        peak, rms, crest_db = crest_factor_db(file_path)
        if peak is None:
            print("音频文件无效或全是静音")
        else:
            print(f"文件: {file_path}")
            print(f"峰值: {peak:.6f}")
            print(f"RMS: {rms:.6f}")
            print(f"Crest Factor: {crest_db:.2f} dB")
    else:
        # 增强模式：使用完整分析
        if show_benchmark:
            print(f"\n⚡ 性能基准测试")
            print(f"CPU核心数: {CPU_COUNT}")
            print(f"并行化: {'启用' if use_parallel else '禁用'}")
            print("=" * 50)
            
            # 测试串行版本
            start_time = time.time()
            results_serial = advanced_crest_analysis(file_path, enable_true_peak, enable_windowed, enable_lufs, use_parallel=False)
            serial_time = time.time() - start_time
            
            if use_parallel:
                # 测试并行版本
                start_time = time.time()
                results_parallel = advanced_crest_analysis(file_path, enable_true_peak, enable_windowed, enable_lufs, use_parallel=True)
                parallel_time = time.time() - start_time
                
                print(f"串行处理时间: {serial_time:.3f} 秒")
                print(f"并行处理时间: {parallel_time:.3f} 秒")
                print(f"性能提升: {serial_time/parallel_time:.2f}x")
                results = results_parallel
            else:
                print(f"处理时间: {serial_time:.3f} 秒")
                results = results_serial
        else:
            # 普通模式
            start_time = time.time()
            results = advanced_crest_analysis(file_path, enable_true_peak, enable_windowed, enable_lufs, use_parallel)
            end_time = time.time()
            
            if use_parallel:
                print(f"\n⚡ 处理时间: {end_time - start_time:.3f} 秒 (并行化)")
            else:
                print(f"\n⚡ 处理时间: {end_time - start_time:.3f} 秒 (串行)")
        
        print_analysis_results(results)
