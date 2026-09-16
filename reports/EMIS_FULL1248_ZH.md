# EMIS 全量 1,248 条时间外外部验证报告

日期：2026-09-16  
项目：KFAIR Audio Conflict Analysis

## 1. 为什么补这项实验

此前 EMIS 结果只基于严格平衡的 192 条子集，其中 144 条语义—声学冲突、48 条匹配。该子集适合快速验证时间外现象，但样本较少，外部修复的置信区间重叠，容易被质疑为抽样波动。本轮不引入新模型或新任务，只把同一 EMIS 外部验证扩展到官方发布的全部 1,248 条，并沿用已有双流干预与三类修复方法。

## 2. 数据来源与审计

- 官方论文：Corrêa et al., *Evaluating Emotion Recognition in Spoken Language Models on Emotionally Incongruent Speech*, arXiv:2510.25054。
- 官方 Zenodo：record 19207001。
- 音频包：`audios_EMIS.zip`，318,538,812 bytes，MD5 `a78a40ce73eae28dabafc1d16ec703bc`。
- 文本表：`text_samples_EMIS.csv`，MD5 `84a37ca7ae97859c809e50938b3b4b6f`。
- Zenodo 元数据许可证：GPL-3.0-or-later。
- 音频总数 1,248，无重名、无异常命名；三种生成器 COSY、F5TTS、STYLE 各 416 条。
- 语义标签 neutral/happy/sad/angry 各 312 条；声学标签四类也各 312 条。
- 936 条语义—声学冲突，312 条匹配；26 个 text ID、10 个 voice ID。

本地下载与服务器重组后的 MD5 均与 Zenodo 官方值完全一致；压缩包完整性测试无错误。

## 3. 运行环境与固定规则

- AutoDL，NVIDIA A800 80GB PCIe。
- Kimi-Audio-7B-Instruct；主干冻结；本地离线模型缓存。
- Python 3.12.3，PyTorch 2.5.1+cu124，Transformers 4.57.6。
- 分类提示、A/B/C/D 标签 token、模型前向方式与此前 EMIS 192 实验一致。
- 修复 checkpoint 固定使用 actor-CV fold-6；该规则在看到全量 EMIS 结果前已确定，与近期 TESS 和真实扰动外部测试一致。
- 种子 20260903、20260904、20260905。
- 统计以 `generator × text_id` 为 cluster，做 10,000 次 cluster bootstrap；修复与 baseline 还做逐样本配对 McNemar 精确检验。

## 4. 双流干预的全量复核

| 输入条件 | 整体声学准确率 | 冲突声学跟随 | 冲突语义跟随 | 匹配样本准确率 |
|---|---:|---:|---:|---:|
| 完整双流 | 52.24% | 43.59% | 37.18% | 78.21% |
| 去掉连续流 | 31.57% | 25.53% | 37.61% | 49.68% |
| 去掉离散流 | 49.84% | 40.17% | 42.09% | 78.85% |
| 错配连续流 | 43.19% | 32.48% | 47.97% | 75.32% |

完整双流在冲突样本上的声学跟随 95% cluster-bootstrap CI 为 39.95%–47.03%。去掉连续流后降至 25.53%（22.97%–27.89%）；把连续流错配到其他样本后降至 32.48%（28.98%–35.88%），同时语义跟随升到 47.97%。

通俗解释：连续流不仅提供了大量情绪信息，而且它必须与当前语音正确对齐；拿掉或错配都会让模型更少按语调判断。离散流也不是完全无用，去掉它后声学跟随略降，但影响明显小于去掉连续流。因此全量结果支持“连续声学流是主要情绪证据来源之一”，而不是“只有连续流有用”。

## 5. 三类修复的全量零样本迁移

所有修复只在 RAVDESS 上训练，训练过程从未见过 EMIS。

| 系统 | 整体声学准确率 | 冲突声学跟随 | 冲突语义跟随 | 匹配样本准确率 |
|---|---:|---:|---:|---:|
| 未修复 baseline | 52.24% | 43.59% | 37.18% | 78.21% |
| Decision adapter L25 | 57.21%±1.18% | **49.15%±1.34%** | **34.12%±1.36%** | 81.41%±1.28% |
| Full-sequence adapter L22 | 55.26%±1.13% | 47.01%±1.05% | 36.15%±1.13% | 80.02%±1.88% |
| Ordinary LoRA L22 | 57.13%±0.81% | 47.79%±0.80% | 37.25%±0.43% | **85.15%±0.93%** |

相对 baseline 的冲突声学跟随提升：

- Decision adapter：平均 +5.56 个百分点；三个种子分别 +5.66、+6.84、+4.17 pp。
- Full-sequence adapter：平均 +3.42 pp；三个种子分别 +3.10、+4.59、+2.56 pp。
- Ordinary LoRA：平均 +4.20 pp；三个种子分别 +3.42、+4.17、+5.02 pp。

936 个冲突样本的逐样本配对 McNemar 精确检验中，九个修复运行全部 `p<0.05`：

- Decision adapter：`8.03e-7`、`4.75e-8`、`4.32e-5`；
- Full-sequence adapter：`0.0113`、`1.50e-4`、`0.0149`；
- LoRA：`0.0185`、`0.00247`、`5.48e-4`。

## 6. 相比 192 子集，结论发生了什么变化

192 子集上，外部修复只能写成“方向性提升，置信区间重叠，证据有限”。全量 1,248 条显示：

1. 外部提升的绝对幅度仍不大，约 3–6 个百分点，因此不能宣称跨域问题已经解决。
2. 但提升在三种方法、三个种子上均为正，而且逐样本配对检验全部显著；因此“修复并非只记住 RAVDESS 演员”现在有了更稳定的证据。
3. Decision adapter 在冲突声学跟随和降低语义跟随上最好；LoRA 在匹配样本保持上最好；full-sequence adapter 的提升较小但方向稳定。
4. 全量基线的冲突声学跟随明显高于 192 子集，说明小平衡子集并不能代表 EMIS 的总体难度。论文中应优先报告全量结果，并把 192 子集保留为预先构造的均衡审计集或补充材料。

## 7. 论文应如何使用这项结果

建议表述：

> On all 1,248 post-release EMIS samples, RAVDESS-trained repairs consistently improve acoustic following under semantic–prosodic conflict by 3–6 percentage points across three seeds, with paired McNemar significance in every run. The modest absolute gains nevertheless leave substantial cross-domain headroom.

可以声称：全量时间外数据复现了连续流的重要作用；RAVDESS 训练的三类修复都出现跨域正迁移；这种提升跨种子一致且配对显著。

不能声称：EMIS 是真实自然录音；修复已经解决跨域语义偏置；adapter 全面优于 LoRA；全量 EMIS 可以替代真实冲突语料。

## 8. 可复现文件

- `code/run_emis_full1248.py`：正式评测。
- `code/validate_emis_full1248.py`：唯一性和跨种子聚合验证。
- `results/raw_predictions/emis_full1248_predictions.jsonl`：16,224 条逐样本预测。
- `results/statistics/emis_full1248_run_summary.jsonl`：13 个运行摘要。
- `results/statistics/emis_full1248_summary.csv`：generator 和 cluster-bootstrap 统计。
- `results/statistics/emis_full1248_paired_effects.csv`：配对提升与 McNemar 检验。
- `results/statistics/emis_full1248_analysis.json`：完整性与跨种子聚合。
- `provenance/emis_full1248_provenance.json`：官方来源、MD5、许可证与数据组成。
- `results/logs/emis_full1248.log`：完整运行日志。

完整性验证：16,224/16,224 个唯一系统—种子—文件键；1,248 个音频全部覆盖；13 个配置齐全；无空预测；九个修复运行的冲突配对检验全部 `p<0.05`。
