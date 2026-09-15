# 投稿前高优先级补实验完整技术报告（2026-09-16）

> 本报告覆盖两项高优先级工作：① Kimi-Audio 的 6 折 held-out-actor 交叉验证；② Qwen2.5-Omni-7B 的第二模型最小复现。它是一份可独立交给老师、论文合作者或代码代理阅读的实验技术档案，记录机器、环境、模型、数据、训练、统计、耗时、故障恢复、结果与结论边界。

## 0. 完成状态与核心结论

两项工作均已完整完成：Kimi 共 6 折 × 3 方法 × 3 种子 = 54 次训练，保存 54 个 checkpoint、10,368 条修复测试预测和 1,152 条基线预测；Qwen 保存 384 条基线行为预测、3 个 adapter checkpoint 和 672 条修复预测。

- Kimi 原先单次最好 91.67% 不能代表普遍水平。六折基线为 **43.66% ± 9.21%**，三种 57,344 参数修复达到 **75.52%–77.95%**，“小参数修复有效”能跨演员成立。
- Qwen 同样存在语义—声学冲突时的系统性信任偏置。冻结主干、只加 57,344 参数后，RAVDESS held-out 演员准确率由 **65.63%** 提至 **84.38% ± 8.27%**。
- Qwen adapter 在从未训练过的 EMIS 上仅把冲突声学跟随从 **1.39%** 提到 **4.40% ± 1.45%**，所以它是明显的同域修复，不是可靠的跨域通用修复。
- 普通 LoRA 在 Kimi 六折平均准确率略高于定位式 adapter，论文不能写“只有因果定位才能修复”；定位的价值是给出可解释、可控、极低参数量的干预位置。

## 1. 运行机器与软件环境

项目根目录为 `/root/autodl-tmp/kfair`，运行在 AutoDL 单机单卡实例。

| 项目 | 实际配置 |
|---|---|
| 操作系统 | Ubuntu 22.04.4 LTS (Jammy) |
| Linux 内核 | `4.19.90-2107.6.0.0248.35.oe1.bclinux.x86_64` |
| CPU | Intel Xeon Gold 6348 @ 2.60 GHz |
| CPU 拓扑 | 2 sockets × 28 cores × 2 threads，共 112 logical CPUs |
| 内存 | 1.0 TiB；无 swap |
| GPU | NVIDIA A800 80GB PCIe，单卡 |
| GPU 显存 / Compute Capability | 81,920 MiB / 8.0 |
| 驱动 / 功率上限 | 590.48.01 / 300 W |
| 系统盘 | 30 GB，核对时剩余约 4.9 GB |
| 数据盘 | `/root/autodl-tmp` 50 GB，核对时剩余约 22 GB |

运行快照：Kimi 六折通常使用约 22.4–23.6 GB 显存，GPU 利用率约 72%–74%，温度约 60–64°C；Qwen 行为评测约 15.4 GB，adapter 阶段约 18.1 GB。这是抽样状态，不是全程平均。

虚拟环境：`/root/autodl-tmp/kfair/fastenv2`。

| 软件 | 版本 |
|---|---|
| Python | 3.12.3，Anaconda build，GCC 11.2.0 |
| PyTorch / CUDA runtime / cuDNN | 2.5.1+cu124 / 12.4 / 9.1.0 |
| torchaudio | 2.5.1+cu124 |
| Transformers / FlashAttention | 4.57.6 / 2.7.4.post1 |
| Accelerate | 1.14.0 |
| pandas / NumPy / SciPy | 3.0.5 / 2.1.3 / 1.18.1 |
| safetensors | 0.8.0 |

A800 支持 BF16。两套主干以 BF16 前向；新增 adapter/LoRA 权重为 FP32，增量再转回 hidden state dtype。

正式流水线设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，并固定 `HF_HOME=/root/autodl-tmp/kfair/cache/huggingface`、`TMPDIR=/root/autodl-tmp/tmp`、`XDG_CACHE_HOME=/root/autodl-tmp/cache` 和 `TORCH_HOME=/root/autodl-tmp/cache/torch`。首次 Kimi 启动曾因 GLM-4 voice tokenizer 缓存目录指错而尝试联网；切换正确缓存后解决。两次未完成启动未进入正式训练、未写入有效汇总。

## 2. 模型与固定版本

### 2.1 Kimi-Audio-7B-Instruct

- 路径 `/root/autodl-tmp/kfair/models/Kimi-Audio-7B-Instruct`，目录约 23.59 GB。
- 28 层，hidden size 3,584，intermediate size 18,944，28 attention heads、4 KV heads，最大位置 8,192，配置 dtype BF16。
- 启用 Whisper continuous feature，并含 Kimi 音频 token 与 MIMO 权重。
- 本轮仅做理解分类，`load_detokenizer=False`。
- `model.alm` 原参数全部 `requires_grad=False`，只更新新增 57,344 参数。
- branch mode 设为 `full`，分别控制离散音频输入和连续声学特征。

### 2.2 Qwen2.5-Omni-7B

- 标识 `Qwen/Qwen2.5-Omni-7B`，固定 revision `ae9e1690543ffd5c0221dc27f79834d0294cba00`。
- 本地 snapshot 约 22.38 GB。
- Thinker 文本部分 28 层，hidden size 3,584，intermediate size 18,944，28 attention heads、4 KV heads。
- 连续音频 encoder 32 层，`d_model=1280`，128 维 mel，输出 3,584 维。
- 关闭 `enable_audio_output`，只用 thinker，不运行 talker/token-to-wave。
- BF16、FlashAttention 2、`device_map="cuda:0"`、`low_cpu_mem_usage=True`。

Qwen 没有 Kimi 那种可直接互换的原生离散/连续双流，因此这里是跨架构行为复现和低秩决策层修复，不是 Kimi 双流交换的逐结构复刻。

首次调用 Qwen 自身 `from_pretrained` 时，Transformers 仍尝试读取 talker 的 `spk_dict.pt`；PyTorch 2.5.1 因 CVE-2025-32434 安全限制拒绝 `torch.load` 并要求至少 2.6。解决办法不是关闭检查，而是关闭 audio output，调用父类通用 safetensors 加载器，只加载 thinker，完全不读取无关 `.pt`。修正后从 5 个 safetensors shards 正常加载，理解权重未改变。

## 3. 数据、标签和预处理

四类固定为 `neutral, happy, sad, angry`，映射 `A/B/C/D`。两模型共用提示：

```text
Classify only the speaker's vocal emotion. Choose exactly one:
A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D.
```

不自由生成长回答；读取最后 decision position 上 A/B/C/D 分数。Qwen 端断言它们都是单 token，IDs 为 32、33、34、35。

### 3.1 RAVDESS

24 演员 × 2 固定句 × 4 情绪 = 192 条真人语音；四类各 48 条。全部 48 kHz、16-bit PCM；191 条单声道、1 条双声道；3.07–4.30 秒，均值 3.58 秒，中位数 3.54 秒，总计约 686.79 秒。清单为 `outputs/ravdess_expanded_manifest.csv` 和 `outputs/ravdess_statement2_manifest.csv`。按演员拆分，绝不音频级随机拆分。

### 3.2 EMIS

平衡 192 条合成冲突集：48 条语义/语调一致，144 条冲突。COSY、F5TTS、STYLE 各 64 条；语义与声学标签四类各 48；4 个 text ID、10 个 voice ID、12 个 generator×text cluster；冲突中 90 explicit、54 implicit。一律 24 kHz 单声道，128 条浮点 PCM、64 条整数 PCM；1.96–6.35 秒，均值 3.70 秒，总计 710.20 秒。清单位于 `data/emis_*manifest.jsonl`，音频在 `data/emis_balanced192/`。它不是自然人真实冲突语料。

### 3.3 Qwen 音频输入

torchaudio 读取，多声道先平均；RAVDESS 48 kHz 和 EMIS 24 kHz 均重采样为 processor 要求的 16 kHz。processor 使用 128 维特征、`n_fft=400`、`hop_length=160`、最大 30,000 帧、右侧 padding。浮点输入移到 GPU 后转 BF16。chat template 还含 system message `You are a careful speech emotion classifier.`。

## 4. 指标定义

本报告 `SER` 是 speech emotion recognition accuracy，不是 sentence error rate：预测等于声学/语调标签的比例。

- Aligned SER：AA/BB 一致单元准确率。
- Conflict acoustic follow：AB/BA 冲突时跟随声学标签比例。
- Conflict semantic follow：冲突时跟随语义或离散分支比例。
- Other：两者都未选。
- Acoustic margin：`score(acoustic)-score(semantic/discrete)`；正值偏声学，负值偏语义。

## 5. Kimi 六折 held-out-actor 交叉验证

### 5.1 析因构造

四类有 6 个无序标签对。每个演员、句子、标签对 A/B 构造 AA、AB、BA、BB：第一字母决定离散音频 token 来源，第二字母决定连续声学特征来源。总计 `24×2×6×4=1,152`，一致与冲突各 576。

每条原音频只预处理一次并缓存离散 IDs、text IDs、continuous mask 和 Whisper feature。混合单元从 `discrete_file` 取离散 token/text/mask，从 `continuous_file` 取连续特征；时间长度不同时用一维 linear interpolation 对齐。前向使用 `input_ids`、`text_input_ids`、`whisper_input_feature`、`is_continuous_mask`、逐位置 `position_ids`，`use_cache=False`，四类分数取文本 logits 最后位置。

### 5.2 六折拆分

固定组为 1–4、5–8、9–12、13–16、17–20、21–24。第 k 组测试，下一组验证，其余 16 人训练；第 6 折验证回绕 1–4。每折训练/验证/测试分别 768/192/192 单元。种子为 20260903、20260904、20260905。

### 5.3 三种 57,344 参数方法

参数量均为 `2×3584×8=57,344`，约为 7B 的 0.00082%。

1. **Layer 25 decision-token adapter**：forward pre-hook 只改最后 token，`h'=h+Wup GELU(Wdown h)`；3584→8→3584，down `std=0.01`，up 零初始化，无 bias，FP32。层 25 来自此前 activation patching。
2. **Layer 22 full-sequence adapter**：相同结构但修改全部序列位置；层 22 来自此前 adapter sweep。
3. **Layer 22 q-projection LoRA**：原 q_proj 冻结，A 3584→8、B 8→3584，rank=8、alpha=8、scale=1；A `std=0.01`、B 零初始化。它是参数匹配普通基线。

### 5.4 训练、损失和选模

4 epoch，batch size=1；AdamW，lr `1e-3`，weight decay `1e-4`，gradient clip 1.0。

Adapter 损失为 `L_SER + 0.5 L_pair + 0.2 L_preserve`。`L_SER` 是声学标签 CE；冲突时 `L_pair=L_SER`，所以分类项权重 1.5；一致时 `L_preserve` 是修复四类分布相对基线的 KL。LoRA 仅 CE。每 epoch 后验证，按 `(SER, conflict acoustic follow, aligned SER)` 字典序选 checkpoint。测试演员不参与选 epoch。

层 22/25 来自此前全项目实验，不是在每个外折内重选；彻底消除层选择偏差仍需 nested CV。

### 5.5 汇总统计

先在每折内平均 3 种子，再以 6 个互斥演员折计算样本标准差。每种方法和相同演员折基线配对，再对 6 个差值做双侧 Wilcoxon。只有 6 折时最小双侧 p 恰为 0.03125，因此它表示六折方向一致，不是高精度显著性估计。

### 5.6 结果

| 方法 | SER 均值±SD | 最差/最好折 | Aligned SER | 冲突声学跟随 | 冲突语义跟随 | margin |
|---|---:|---:|---:|---:|---:|---:|
| 基线 | 43.66%±9.21% | 29.17/54.69% | 44.79%±8.31% | 42.53%±10.28% | 18.75%±3.16% | 1.163±0.413 |
| Decision L25 | 75.52%±11.03% | 57.47/91.32% | 75.00%±11.49% | 76.04%±10.62% | 7.70%±3.68% | 3.374±0.721 |
| Full L22 | 76.13%±6.25% | 67.19/86.28% | 76.39%±6.84% | 75.87%±5.78% | 7.87%±2.26% | 3.491±0.445 |
| LoRA L22 | 77.95%±10.32% | 63.19/93.58% | 77.95%±10.19% | 77.95%±10.50% | 7.29%±3.41% | 5.440±1.070 |

| 测试演员 | 基线 | Decision | Full | LoRA |
|---|---:|---:|---:|---:|
| 1–4 | 54.69% | 78.30% | 78.65% | 79.17% |
| 5–8 | 39.06% | 75.52% | 74.31% | 76.22% |
| 9–12 | 29.17% | 78.82% | 75.87% | 83.51% |
| 13–16 | 51.04% | 71.70% | 74.48% | 72.05% |
| 17–20 | 46.88% | 57.47% | 67.19% | 63.19% |
| 21–24 | 41.15% | 91.32% | 86.28% | 93.58% |

| 方法 | SER 平均提升 | 六折提升范围 | Wilcoxon p | 冲突声学跟随提升 |
|---|---:|---:|---:|---:|
| Decision | +31.86 pp | +10.59 到 +50.17 | 0.03125 | +33.51 pp |
| Full | +32.47 pp | +20.31 到 +46.70 | 0.03125 | +33.33 pp |
| LoRA | +34.29 pp | +16.32 到 +54.34 | 0.03125 | +35.42 pp |

LoRA 平均最高，Full 标准差最低，Decision 干预范围最小但波动较大。第 5 折 Decision、种子 20260905 为 46.88%，等于基线；checkpoint 权重确实更新，所以是未成功优化而非代码漏训练。该负面种子被保留并计入统计。

### 5.7 耗时

54 次训练记录合计 7,748.51 秒，即 2 小时 9 分 9 秒；单次均值 143.49 秒、中位 145.70 秒、范围 131.09–158.59 秒。Decision/Full/LoRA 平均 135.04/148.32/147.11 秒。正式日志 14:59:39 开始加载、17:09 左右完成，墙钟约 2 小时 10 分。记录时间不含首次加载、192 条缓存和 1,152 条基线前向。54 个 checkpoint 合计约 12.52 MB。

## 6. Qwen 第二模型复现

### 6.1 行为设计与基线

RAVDESS 固定句语义统一记 neutral，主要评估声学情绪；EMIS 同时有明确语义和声学标签，144 条直接冲突。每条一次前向，A/B/C/D 做 log-softmax。RAVDESS 以 actor、EMIS 以 generator_text_id 聚类，做 10,000 次 cluster bootstrap，种子 20260915。

| 子集 | n | 声学准确率 | 语义跟随 | margin |
|---|---:|---:|---:|---:|
| RAVDESS 全体 | 192 | 68.23% | — | — |
| RAVDESS neutral | 48 | 100.00% | 100.00% | 0 |
| RAVDESS 非 neutral | 144 | 57.64% | 40.28% | +0.330 |
| 测试演员 21–24 | 32 | 65.63% | 59.38% | -0.168 |
| EMIS 全体 | 192 | 26.04% | 96.35% | -3.961 |
| EMIS 一致 | 48 | 100.00% | 100.00% | 0 |
| EMIS 冲突 | 144 | 1.39% | 95.14% | -5.281 |

EMIS 冲突 95% CI：声学跟随 0%–3.97%，语义跟随 91.98%–98.15%，margin -5.547 到 -4.874。三生成器冲突声学跟随 2.08%、2.08%、0%，不是单一生成器造成。

### 6.2 Adapter 训练

仅 RAVDESS：演员 1–16 训练 128 条，17–20 验证 32 条，21–24 测试 32 条；EMIS 192 条只作零样本外测。在 thinker 第 25 层 pre-hook 只改最后 token，rank=8、57,344 参数；down `std=0.02`、up 零初始化，无 bias，主干全冻结。

4 epoch、batch=1、AdamW、lr `1e-3`、wd `1e-4`、clip 1.0、CE。按验证 `(accuracy, margin)` 选 checkpoint，三个种子分别选第 3/4/2 epoch。每次前向重新读取和预处理音频。

### 6.3 同域结果与配对检验

| 模型 | 准确率 | 语义跟随 | margin | 提升 |
|---|---:|---:|---:|---:|
| 基线 | 65.63% | 59.38% | -0.168 | — |
| Seed 03 | 90.63% | 21.88% | 6.521 | +25.00 pp |
| Seed 04 | 87.50% | 37.50% | 5.059 | +21.88 pp |
| Seed 05 | 75.00% | 46.88% | 1.486 | +9.38 pp |
| 均值±SD | 84.38%±8.27% | — | 4.355±2.590 | +18.75 pp |

对相同 32 条样本做 10,000 次聚类配对 bootstrap 和精确 McNemar：Seed 03 提升 CI +15.63 到 +34.38 pp，改对/改错 10/2，p=0.0386；Seed 04 CI +6.25 到 +34.38，7/0，p=0.0156；Seed 05 CI 0 到 +18.75，3/0，p=0.25。第三种子未显著，需如实保留。

### 6.4 EMIS 零样本迁移

| 指标 | 基线 | Seed 03 | Seed 04 | Seed 05 | 均值±SD |
|---|---:|---:|---:|---:|---:|
| 全体准确率 | 26.04% | 27.08% | 29.17% | 27.08% | 27.78%±1.20% |
| 一致准确率 | 100.00% | 93.75% | 100.00% | 100.00% | 97.92%±3.61% |
| 冲突声学跟随 | 1.39% | 4.86% | 5.56% | 2.78% | 4.40%±1.45% |
| 冲突语义跟随 | 95.14% | 89.58% | 90.97% | 93.75% | 91.44%±2.13% |
| 全体 margin | -3.961 | -7.028 | -6.680 | -5.847 | -6.518±0.607 |

少数样本被改对，但总体 margin 更负，表示许多仍错样本反而更确信文字语义。EMIS 全体提升仅 +1.04/+3.13/+1.04 pp，对应 McNemar p=0.727/0.0313/0.5。正确结论是同域修复明确、跨域收益小且不稳定。

### 6.5 Qwen 耗时

恢复流水线：重算 Kimi 统计 2 秒；Qwen 384 条行为评测 48 秒（含加载）；三种子训练、验证、RAVDESS 与 EMIS 评测 4 分 28 秒（含加载）；最终统计 2 秒。三个 run 内部记录合计 255.19 秒，单种子 83.71–86.19 秒，均值 85.06 秒；3 个 checkpoint 约 695 KB。

## 7. 中断恢复与完整性

Kimi 每完成 `(fold,method,seed)` 就增量写 summary、预测、checkpoint，重启跳过已完成键；Qwen behavior 按 `dataset:filename` 检查；adapter 按 seed 保存；pipeline state 记录阶段状态。额度中断后先核对数量再恢复，最终状态 `all_high_priority_experiments: completed`。

| 内容 | 最终数量 |
|---|---:|
| Kimi 正式运行/checkpoint | 54/54 |
| Kimi 修复测试预测 | 10,368 |
| Kimi 基线预测 | 1,152 |
| Qwen 基线预测 | 384 |
| Qwen adapter 运行/checkpoint | 3/3 |
| Qwen adapter 预测 | 672 |

扩写前 GitHub 基线 commit 为 `b72c95c2f514dadd40e80457d2d1645d0f27cd59`；同步后将形成新 commit。

## 8. 代码和结果入口

核心代码：`code/run_actor_cv6.py`、`analyze_actor_cv6.py`、`run_qwen_behavioral_replication.py`、`run_qwen_adapter_replication.py`、`analyze_qwen_replication.py`、`run_high_priority_pipeline.py`。

Kimi：`outputs/actor_cv6_aggregate.csv`、`fold_seed_metrics.csv`、`actor_metrics.csv`、`paired_effects.csv`、`run_summary.jsonl`、`test_predictions.jsonl`、`baseline_predictions.jsonl`、`actor_cv6_checkpoints/`、`actor_cv6.log`。

Qwen：`outputs/qwen_behavioral_predictions.jsonl`、`qwen_behavioral_statistics.csv`、`qwen_behavioral_summary.json`、`qwen_adapter_predictions.jsonl`、`qwen_adapter_run_summary.jsonl`、`qwen_adapter_summary.json`、`qwen_replication_comparison.csv`、`qwen_replication_paired_effects.csv`、`qwen_replication_analysis.json`、`qwen_adapter_checkpoints/`、`high_priority_pipeline_resume.log`、`high_priority_pipeline_state.json`。

模型大权重和受许可约束原始数据不直接上传普通 GitHub；仓库保存代码、清单、统计、逐样本预测、小 checkpoint 和报告，复现者按许可准备资源。

## 9. 论文表述边界、限制与最终结论

可以写：Kimi 冻结主干、只训 57,344 参数，六折从 43.66% 提到 75.52%–77.95%；三种方法六折均正提升，p=0.03125；Qwen 也有冲突信任偏置和同域低成本修复；两模型偏向方向可不同，但信任分配失衡跨模型存在；EMIS 揭示域外边界。

不能写：91.67% 是普遍性能；定位 adapter 优于 LoRA；因果定位是提升必要条件；Qwen 复现了 Kimi 内部双流；Qwen adapter 已解决跨域；EMIS 是真实自然冲突。

限制包括：层位置未做外折内 nested selection；RAVDESS 仅 24 演员、两句；Qwen 测试仅 4 演员/32 条；EMIS 为合成；Qwen 未跑匹配 LoRA/full-sequence；Kimi adapter 损失与 LoRA 不完全相同。

本轮解决了投稿前两个最危险问题：Kimi 不再依赖单一高分划分，小参数修复在所有折总体有效，平均提高约 32–34 pp；项目也不再只有单模型证据，Qwen 同域复现成功，同时 EMIS 负结果清楚界定跨域限制。高优先级实验已完成；CASE/真实冲突、真实噪声与编解码、开放词汇 ASR、nested CV 和 DEAF 扩展属于提高论文上限的低优先级工作，不阻塞论文整理。
