# Kimi-Audio 双分支干预：阶段一实验报告

## 一句话结论

在 20 条严格配对的 RAVDESS 子集上，连续分支、离散分支和连续特征的时序对应关系都会显著改变模型输出；但模型对分类提示不够稳定、样本量较小，因此当前结果支持“分支确实影响副语言判断”，尚不足以支持“某一分支单独决定情绪”的强结论。

## 实验设置

- 模型：Kimi-Audio-7B-Instruct，A800 80GB，推理峰值约 21–22 GiB。
- 数据：RAVDESS speech 全集已下载（1440 条）；阶段一选取 actor 01–05、statement 01、repetition 01、normal intensity。
- 任务：neutral / happy / sad / angry 四分类，共 5 位说话人 × 4 情绪 = 20 条。
- 控制：每位说话人的文本、重复次序和强度条件一致，主要变化是情绪。
- 干预：Full；No continuous（去连续 Whisper 特征）；No discrete（去离散音频 token embedding）；Rolled continuous（仅打乱连续特征与 token 的时间对应）。
- 解码：确定性文本生成；每条样本在四种条件下各运行一次，共 80 次/协议。

## 结果一：自由标签协议

| 条件 | Accuracy | Macro-F1 | 合法标签覆盖率 | 相对 Full 翻转率 |
|---|---:|---:|---:|---:|
| Full | 55% | 0.551 | 70% | — |
| No continuous | 35% | 0.296 | 100% | 70% |
| No discrete | 35% | 0.437 | 40% | 30% |
| Rolled continuous | 65% | 0.567 | 100% | 35% |

## 结果二：显式四选一协议

| 条件 | Accuracy | Macro-F1 | 合法选项覆盖率 | 相对 Full 翻转率 |
|---|---:|---:|---:|---:|
| Full | 30% | 0.258 | 75% | — |
| No continuous | 45% | 0.382 | 100% | 65% |
| No discrete | 35% | 0.328 | 80% | 15% |
| Rolled continuous | 55% | 0.511 | 95% | 45% |

![阶段一结果](stage1_branch_results.svg)

## 可交差的观察

1. 干预是有效的。自由标签协议中移除连续分支后，相对 Full 的输出翻转率为 70%；四选一协议中为 65%。这不是“代码没有生效”的空消融。
2. 连续分支影响很大，但方向并不等同于简单的准确率提升。No continuous 在两套提示下出现明显的类别塌缩/偏置，且两套协议的 accuracy 方向不同，说明生成式分类的提示敏感性很强。
3. 仅保留连续分支（No discrete）时，自由标签覆盖率只有 40%，显示离散 token 对遵循输出格式和稳定生成可能很重要。
4. 打乱连续特征的时间对应仍造成 45% 的输出翻转，但小样本 accuracy 偶然高于 Full；这更适合解释为“时序对齐会改变决策”，不能解释为“打乱会改善模型”。

## 限制与下一步

- 每类仅 5 条，置信区间会很宽；当前阶段目标是跑通干预和发现信号，不是给最终统计结论。
- Kimi-Audio 偶尔输出题目之外的标签或字母，coverage 必须和 accuracy 同时报告。
- 下一阶段应扩展到更多 actor，并优先采用候选标签条件似然/强制解码，减少自由生成带来的提示偏差。
- 建议加入语音转写任务作为内容保真对照：如果某干预显著伤害情绪但较少伤害转写，才能更有力地区分“副语言信息”与“语言内容”。

## 可复现实件

- `ravdess_eval_manifest.csv`：20 条受控样本清单。
- `ravdess_branch_predictions.jsonl`：自由标签协议逐条原始输出。
- `ravdess_choice_predictions.jsonl`：显式四选一协议逐条原始输出。
- 两个 `*_summary.csv`：汇总指标。
