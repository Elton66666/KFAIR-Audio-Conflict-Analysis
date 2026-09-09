# Kimi-Audio 双分支干预实验：会议版合并报告

## 摘要

我们在 Kimi-Audio-7B-Instruct 上直接干预连续声学表示与离散音频 token 的融合，并在 24 位说话人、96 条同文本受控 RAVDESS 语音上同时评估情绪分类与转写。最清晰的结果来自 ASR 对照：Full 与 No discrete 均达到 0% WER，而 No continuous 达到 121.7% WER，连续特征时序错位达到 57.8% WER。说明连续表示及其时序对齐对恢复语言内容至关重要；在这组短句上，仅连续表示已足以完成准确转写。情绪分类也随分支干预发生明显变化，但受生成式标签覆盖率及提示敏感性影响，应作为探索性结果解释。

## 1. 实验设计

- 模型与算力：Kimi-Audio-7B-Instruct；NVIDIA A800 80GB；推理峰值约 21–24 GiB。
- 数据：RAVDESS speech 1440 条中的受控子集。选取全部 24 位 actor，固定 statement=01、repetition=01、normal intensity。
- 情绪：neutral、happy、sad、angry，每类 24 条，共 96 条。
- 文本：所有样本均为 “Kids are talking by the door.”，从而隔离说话人和情绪变化。
- 四种条件：Full；No continuous；No discrete；Rolled continuous（循环错位连续特征，破坏时间对应但保留其分布）。
- 每项任务共 96×4=384 次推理；情绪与 ASR 合计 768 次正式推理。
- 置信区间：以 actor 为聚类单位进行 10,000 次 bootstrap，避免把同一说话人的四种情绪当成完全独立样本。

## 2. 情绪分类结果

| 条件 | Accuracy [95% CI] | 覆盖率 | 输出翻转率 | 相对 Full 差值 [95% CI] | McNemar p |
|---|---:|---:|---:|---:|---:|
| Full | 46.9% [39.6%, 54.2%] | 70.8% | — | — | — |
| No continuous | 38.5% [31.2%, 45.9%] | 99.0% | 58.3% | -8.3 pp [-17.7, +1.0] | 0.2295 |
| No discrete | 39.6% [31.2%, 47.9%] | 59.4% | 16.7% | -7.3 pp [-15.6, +1.0] | 0.0654 |
| Rolled continuous | 66.7% [59.4%, 72.9%] | 99.0% | 42.7% | +19.8 pp [+11.5, +27.1] | 0.0002 |

重要现象：No continuous 的预测大量塌缩到 sad，Full 与 No discrete 都存在输出题目外标签的问题；因此 accuracy 必须与覆盖率同时解读。Rolled continuous 的 accuracy 高于 Full，但 ASR 严重下降，不能将其解释为时序错位“改善模型”，更可能是生成偏置与该小型受控任务的偶然组合。

## 3. ASR 内容保真对照

| 条件 | WER [95% CI] | 完全匹配率 |
|---|---:|---:|
| Full | 0.0% [0.0%, 0.0%] | 100.0% |
| No continuous | 121.7% [101.0%, 143.4%] | 0.0% |
| No discrete | 0.0% [0.0%, 0.0%] | 100.0% |
| Rolled continuous | 57.8% [53.3%, 62.7%] | 1.0% |

![会议版主结果](stage2_main_results.svg)

ASR 给出了比情绪 accuracy 更稳定的结构性证据：

1. No discrete 与 Full 完全一致，96 条全部逐字正确，说明在该固定短句转写任务上连续表示足以保存语言内容。
2. No continuous 完全匹配率为 0%，并出现截断、替换和幻觉；离散 token 单独不足以可靠恢复内容。
3. Rolled continuous 几乎全部失败，说明不仅需要连续特征本身，还需要它与 token 位置的正确时间对齐。

## 4. 当前可支持与不可支持的结论

可以支持：连续表示及其时序对齐是 Kimi-Audio 在该任务上恢复语言内容的关键；分支干预会显著改变情绪输出；实验干预确实生效。

暂不能支持：离散分支专门编码情绪；某种干预提升了情绪识别；结果能直接推广到开放词汇、长语音或其他模型。情绪任务仍受到自由生成格式、标签覆盖率和单句数据设计的限制。

## 5. 下一阶段建议

优先采用候选标签条件似然或受约束解码，消除题外标签；加入第二条 RAVDESS statement 复现；再用不同语料测试 ASR，确认 No discrete 的 0% WER 是否只适用于极短、固定文本。若要研究“副语言与内容解耦”，还应加入 speaker identification、强度识别或音高/语速等连续声学指标。

## 6. 复现与附件

- `ravdess_expanded_manifest.csv`：96 条受控样本。
- `ravdess_expanded_predictions.jsonl`：384 条情绪原始输出。
- `ravdess_asr_predictions.jsonl`：384 条 ASR 原始输出。
- `stage2_emotion_statistics.csv`、`stage2_asr_statistics.csv`：置信区间和配对检验。
- `stage2_per_emotion.csv`：分情绪结果。
