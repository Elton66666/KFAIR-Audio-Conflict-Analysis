# 三项补充实验详细报告：参数匹配、真实扰动与扩大 ASR

日期：2026-09-16  
项目：KFAIR Audio Conflict Analysis

## 1. 本轮目的与结论概览

本轮完成三项投稿增强实验：（1）在 Qwen2.5-Omni 上比较参数量严格相同的 full-sequence adapter 与普通 LoRA；（2）把此前较理想化的白噪声测试升级为真实房间噪声、真实办公室混响、G.711 μ-law 电话信道及其组合；（3）把原先 32 句 ASR 保持测试扩大为 RAVDESS 192 句和 TESS 192 句，并覆盖 96 个不同目标词。

主要结论如下。

1. 在 Qwen 的同域 held-out 演员测试上，57,344 参数的 full-sequence adapter 达到 91.67%±3.61%，相同参数量的普通 LoRA 为 88.54%±1.80%。这说明 adapter 的提升不能仅用“参数更多”解释，但样本量较小，不能声称它显著全面优于 LoRA。
2. 两者在 EMIS 零样本外部测试上都很低：adapter 27.08%±0.52%，LoRA 27.78%±0.30%。这是一项重要的负结果：同域演员泛化不等于跨语料泛化。
3. 在 Kimi 的真实扰动测试中，三种修复在所有困难扰动下总体明显优于未修复基线。最强组合扰动下，基线 37.50%，decision adapter 80.21%±1.80%，full-sequence adapter 71.88%±8.27%，普通 LoRA 83.33%±3.61%。LoRA 本轮仍略强，论文不应声称因果定位方法必然比 LoRA 更抗噪。
4. 扩大 ASR 测试表明修复基本不破坏语音内容。RAVDESS 上所有方法 WER 均不超过 0.087%；TESS 上基线 WER 2.73%，decision adapter 2.73%，full-sequence adapter 2.82%，LoRA 3.08%。变化很小，且 3,840 条输出中没有空转写。

## 2. 运行机器与软件环境

- 服务器：AutoDL；Linux；项目根目录 `/root/autodl-tmp/kfair`。
- GPU：NVIDIA A800 80GB PCIe，显存 81,920 MiB。
- GPU 驱动：590.48.01。
- Python：3.12.3（Anaconda，GCC 11.2.0）。
- PyTorch：2.5.1+cu124；torchaudio：2.5.1+cu124。
- Transformers：4.57.6；pandas：3.0.5；NumPy：2.1.3。
- 虚拟环境：`/root/autodl-tmp/kfair/fastenv2`。
- 模型与 Hugging Face 资源均使用服务器本地缓存；实验设置 `HF_HOME=/root/autodl-tmp/kfair/cache/huggingface`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，避免在线版本漂移。
- 随机种子：20260903、20260904、20260905。

## 3. 实验一：Qwen 参数匹配对照

### 3.1 问题

此前 Qwen 最小复现显示 full-sequence adapter 能修复情绪冲突判断。本实验问：效果是否只是因为它比普通 LoRA 使用了更多可训练参数？

### 3.2 数据和拆分

- 训练/验证/测试使用 RAVDESS，演员严格隔离：actor 1–16 为训练集 128 条，17–20 为验证集 32 条，21–24 为测试集 32 条。
- 外部零样本测试使用 EMIS 192 条。
- 测试同时报告整体声学情绪准确率、aligned 子集准确率、conflict 子集声学/语义跟随率和 acoustic margin。

### 3.3 两种严格同参数模型

- `full_sequence_adapter_layer22`：在 Qwen 第 22 层输入残差流上加入 rank=8 的低秩 adapter，作用于整段序列。
- `ordinary_lora_layer22`：在同一第 22 层 self-attention 的 `q_proj` 上加入 rank=8、alpha=8 的普通 LoRA。
- 两者可训练参数均为 **57,344**；主干冻结。
- 优化器 AdamW，学习率 1e-3，weight decay 1e-4，梯度裁剪 1.0；训练 4 epochs；以验证集声学准确率、再以 acoustic margin 作为择优规则。
- 每种方法运行 3 个种子，共 6 个 checkpoint。

### 3.4 结果

| 方法 | 参数 | RAVDESS test 准确率 | conflict 声学跟随率 | EMIS 零样本准确率 | EMIS conflict 声学跟随率 |
|---|---:|---:|---:|---:|---:|
| full-sequence adapter | 57,344 | 91.67%±3.61% | 93.06% | 27.08%±0.52% | 5.56% |
| ordinary LoRA | 57,344 | 88.54%±1.80% | 84.72% | 27.78%±0.30% | 5.32% |

逐种子 paired McNemar 检验显示，adapter 对 test 基线的 p 值分别为 0.0225、0.0654、0.0117；LoRA 分别为 0.0156、0.0078、0.0391。EMIS 上所有 p 值均不显著。

解释：同参数量下，针对整段残差流的 adapter 在本域测试上略高，并在 conflict 声学跟随率上有更明显优势，支持“作用位置和作用方式重要”。但普通 LoRA 也很强，且外部 EMIS 上二者都没有可靠迁移，不能夸大。

## 4. 实验二：真实扰动鲁棒性

### 4.1 数据与真实扰动来源

- 基础测试集：RAVDESS actor 21–24 的 32 条完整原始音频。
- OpenSLR SLR28 的真实小房间噪声：`RVB2014_type1_noise_smallroom1_1.wav`，SHA-256 `7825727b74f01d8d437ddf54188a2d0a25775eb787b9c1adb02b88509d1c7b37`。
- OpenSLR SLR28 的真实办公室双耳 RIR：`air_type1_air_binaural_office_0_1.wav`，SHA-256 `44d3bdd9b3caa52d156cf3f0ad297bd4aa1aeb5e6ddf7d07e766238ac2591a61`。
- 电话信道：8 kHz 下采样、G.711 μ-law 编码/解码、再恢复到模型输入采样率。
- SLR28 资产为 Apache 2.0；原始压缩包 MD5 为 `e6f48e257286e05de56413b4779d8ffb`。

### 4.2 六个条件

1. clean；2. real_noise_10db（真实噪声，10 dB SNR）；3. real_office_rir；4. g711_mulaw_8khz；5. rir_plus_noise；6. combined_rir_noise_g711。

### 4.3 模型与 checkpoint

- Kimi-Audio-7B-Instruct 主干冻结。
- 未修复 baseline 运行一次。
- `decision_adapter_layer25`、`full_sequence_adapter_layer22`、`ordinary_lora_layer22` 各使用 actor-CV 第 6 折的 3 个种子 checkpoint。
- 每个系统在 32 条×6 条件上测试；总计 1,920 条唯一预测、60 个系统-条件组合。

### 4.4 准确率结果

| 方法 | clean | 噪声10dB | 办公室RIR | G.711 | RIR+噪声 | 三者组合 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 43.75 | 56.25 | 28.13 | 50.00 | 31.25 | 37.50 |
| decision adapter | 91.67±1.80 | 87.50±6.25 | 86.46±7.86 | 87.50±3.13 | 85.42±4.77 | 80.21±1.80 |
| full-sequence adapter | 87.50±5.41 | 81.25±6.25 | 81.25±6.25 | 83.33±6.51 | 82.29±1.80 | 71.88±8.27 |
| ordinary LoRA | 92.71±3.61 | 89.58±1.80 | 87.50±5.41 | 92.71±3.61 | 84.38±5.41 | 83.33±3.61 |

注意：噪声或电话信道偶尔令 baseline 数值高于 clean，不代表扰动“有益”；样本只有 32 条且模型在冲突边界附近，扰动可能随机推动部分样本跨过决策边界。更稳妥的比较是不同修复方法在同一条件下的均值和跨种子稳定性。

### 4.5 结论边界

修复带来的情绪判断提升并未在真实声学退化下整体消失，因此不仅是干净合成条件里的偶然现象。但 LoRA 仍是最强或并列最强的鲁棒性基线；因果定位的主要价值应表述为“解释故障位置、指导极小规模且任务定向的干预”，而不是“永远优于通用 LoRA”。

## 5. 实验三：扩大 ASR 保持测试

### 5.1 数据与评测口径

- RAVDESS：全部 192 条实验音频，参考文本为数据集两条固定句子。
- TESS：192 条外部音频；96 个不同目标词×YAF/OAF 两位说话人，情绪标签轮换平衡。
- TESS 音频实际内容是载句 `Say the word <target>`，因此参考文本使用完整载句，而不是仅用文件名中的目标词。初次运行时发现参考口径错误后立即停止，删除无效输出，再按完整载句从头运行；无效数字未进入最终结果。
- 指标：micro WER、micro CER、规范化后整句 exact match。

### 5.2 checkpoint 防泄漏规则

- baseline 不加载修复参数。
- RAVDESS 对每位演员使用其所属 held-out fold 的 checkpoint，确保该演员未参与对应训练。
- TESS 是外部语料，三种修复统一使用预先固定的 actor-CV fold-6 checkpoint。
- 三种方法各 3 个种子；加 baseline 共 10 个系统配置，每个系统 384 条，总计 3,840 条唯一转写。

### 5.3 结果

| 方法 | RAVDESS WER | RAVDESS exact | TESS WER | TESS exact |
|---|---:|---:|---:|---:|
| baseline | 0.087% | 99.48% | 2.734% | 89.06% |
| decision adapter | 0.087%±0.000% | 99.48%±0.00% | 2.734%±0.000% | 89.06%±0.00% |
| full-sequence adapter | 0.087%±0.150% | 99.48%±0.90% | 2.821%±0.075% | 88.72%±0.30% |
| ordinary LoRA | 0.029%±0.050% | 99.83%±0.30% | 3.082%±0.493% | 88.89%±0.30% |

3,840 条转写中空输出为 0。总体差异远小于情绪识别的提升幅度，支持“修复改变的是冲突情绪决策倾向，而非通过损坏或重写语音内容取得高分”。

## 6. 完整性与可复现文件

- 完整性验证：Qwen 1,344 条均唯一；真实扰动 1,920 条均唯一、60 个组合齐全；ASR 3,840 条均唯一、20 个方法-种子-语料组合齐全且无空转写。
- 代码：`code/run_qwen_parameter_matched_controls.py`、`code/run_realistic_robustness.py`、`code/run_expanded_asr_preservation.py`、`code/validate_low_priority_outputs.py`。
- 汇总：`outputs/*summary.csv`；逐运行摘要：`outputs/*run_summary.jsonl`；原始预测：`outputs/*predictions.jsonl`。
- 真实扰动来源记录：`outputs/realistic_robustness_provenance.json`。
- 参数匹配 checkpoint：`outputs/qwen_parameter_matched_checkpoints/`。
- 完整性记录：`outputs/low_priority_validation.json`。

## 7. 对论文写作的直接建议

1. 把 Qwen 参数匹配实验作为“提升不是参数量堆出来的”证据，但承认普通 LoRA 是强基线。
2. 把 EMIS 的失败作为跨域局限，不隐藏负结果；它反而使结论边界更可信。
3. 用 SLR28+G.711 结果回应“只在理想合成条件有效”的质疑。
4. 用 384 条、96 个词的 ASR 测试替代原先 32 句结果，强调修复几乎不改变内容识别。
5. 仍不应把 32 条 held-out test 或单一外部语料写成普适结论；这些补充提高可信度，但不能替代更大规模、多模型、多真实冲突语料验证。
