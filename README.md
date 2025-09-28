# 增强版 Crest Factor 分析器

这是一个专业级的音频 Crest Factor（波峰因子）分析工具，基于您的技术建议进行了全面优化。

## ✨ 主要改进

### 1. 🎵 更准确的多声道处理
- **Sample Peak**: 对各声道分别取绝对值最大值，再取全局最大
- **RMS 计算**: 按功率跨声道平均（对每个采样点先跨声道求功率平均，再跨时间平均）
- **避免相位抵消**: 不再简单求均值，保持真实的峰值特性

### 2. 📊 dBFS 报告
- 直观显示 Sample Peak 和 RMS 的 dBFS 值
- 更符合音频工程师的工作习惯

### 3. 🎯 True Peak 检测
- 通过 4× 过采样检测真实峰值
- 更准确地发现潜在的"重建削波"风险
- 比采样峰值更接近实际播放时的最大瞬态

### 4. ⏱️ 短时窗口分析
- 50ms 窗口，75% 重叠的 Crest Factor 分析
- 提供统计信息：平均值、标准差、最小值、最大值、动态范围
- 更有参考价值，能看到哪一段更被压缩、哪一段更有爆发感

### 5. 🔊 LUFS 响度测量
- 集成 EBU R128 标准的 LUFS 响度分析
- Integrated LUFS 和 Short-term LUFS 统计
- 比 RMS 更符合人耳主观感受的响度标准

### 6. 🛠️ 数据处理优化
- **数据类型规范化**: 确保 float32 格式，正确处理整数类型音频
- **DC 偏置去除**: 避免直流偏置影响 RMS 计算
- **多种格式兼容**: 通过 `always_2d=True` 强制保持多声道信息

## 📦 依赖包

```bash
pip install numpy soundfile scipy pyloudnorm
```

## 🚀 使用方法

### 增强模式（默认）
```bash
python crest.py audio_file.wav
```

提供完整的分析报告，包括：
- 基本音频统计（Sample Peak, True Peak, RMS，都含 dBFS）
- Crest Factor（Sample CF 和 True CF）
- 短时窗口分析统计
- LUFS 响度分析

### 简单模式（向后兼容）
```bash
python crest.py audio_file.wav --simple
```

输出与原版本兼容的简单格式。

### 选项控制
```bash
# 禁用 True Peak 计算（加快处理速度）
python crest.py audio_file.wav --no-true-peak

# 禁用短时窗口分析
python crest.py audio_file.wav --no-windowed

# 禁用 LUFS 分析
python crest.py audio_file.wav --no-lufs

# 组合使用
python crest.py audio_file.wav --no-true-peak --no-lufs
```

## 📈 输出示例

```
============================================================
文件: example.wav
采样率: 44100 Hz
声道数: 2
时长: 3.45 秒
============================================================

📊 基本音频统计:
  Sample Peak: 0.987654 (-0.11 dBFS)
  True Peak  : 1.023456 (+0.20 dBFS)
  RMS        : 0.234567 (-12.58 dBFS)

🎯 Crest Factor:
  Sample CF  : 12.47 dB
  True CF    : 12.78 dB

🔍 短时窗口分析 (50ms窗口):
  平均 CF    : 11.23 dB
  标准差     : 2.45 dB
  最小 CF    : 6.78 dB
  最大 CF    : 18.90 dB
  动态范围   : 12.12 dB

🔊 LUFS响度分析 (EBU R128):
  Integrated : -23.4 LUFS
  短期响度   :
    平均     : -22.8 LUFS
    最大     : -18.2 LUFS
    最小     : -28.9 LUFS
    标准差   : 2.3 LU
```

## 🔧 技术细节

### Crest Factor 意义
- **低 CF (< 6 dB)**: 高度压缩/限制的音频，动态范围小
- **中等 CF (6-12 dB)**: 适度压缩，平衡的动态范围
- **高 CF (> 12 dB)**: 动态范围大，瞬态突出

### True Peak vs Sample Peak
- **Sample Peak**: 数字采样点的最大值
- **True Peak**: 经过重建滤波后的真实峰值，更接近模拟输出
- True Peak 通常比 Sample Peak 高 0.1-3 dB

### LUFS vs RMS
- **RMS**: 简单的能量平均，技术指标
- **LUFS**: K-weighting + gating，更符合人耳感知的响度标准
- 广播、流媒体平台普遍采用 LUFS 作为响度标准

## 🎯 实际应用

1. **音频质量评估**: 通过 CF 判断压缩程度和动态范围
2. **母带制作**: True Peak 确保不超过 0 dBFS，避免削波
3. **广播准备**: LUFS 确保符合广播响度标准（如 -23 LUFS）
4. **流媒体优化**: 针对平台响度规范调整音频
5. **动态范围分析**: 短时 CF 分析找出过度压缩的片段

## 📝 注意事项

- True Peak 计算需要额外的计算资源，可用 `--no-true-peak` 禁用
- LUFS 分析需要至少 400ms 的音频，短音频可能无法计算
- 短时分析窗口默认 50ms，可根据需要在代码中调整
- 建议音频采样率 ≥ 44.1 kHz 以获得准确的 True Peak 结果
