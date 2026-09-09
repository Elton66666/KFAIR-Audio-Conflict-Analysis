# K-FAIR 当前阶段实验总报告

**项目名称：** K-FAIR：Kimi-Audio 原生双流中的情绪证据析因干预与路径追踪  
**报告性质：** 阶段性中文总报告  
**当前状态：** 核心机制现象已建立；Base/Instruct × E/U 对照已完成；高层因果 patching、P 组与局部修复尚未完成  
**更新日期：** 2026 年 8 月 26 日

---

## 摘要

本项目研究 Kimi-Audio 在语音情绪识别中如何利用其原生的两条音频输入流：离散 semantic-dominant stream（离散语义主导流，记为 D）与连续 acoustic-rich stream（连续声学丰富流，记为 C）。项目没有停留在关闭单一分支的零化消融，而是利用同一说话人、同一文本、不同情绪的真实录音对，构造 `D_A/C_A、D_A/C_B、D_B/C_A、D_B/C_B` 四种组合，定量分解连续流、离散流及其交互对情绪判断的影响。

目前已在 RAVDESS（公开披露的 SER-SFT 暴露组 E）和 TESS（公开 SFT 清单未披露组 U）上完成 Kimi-Audio-7B-Instruct 的完整 2×2 析因实验，并在两组上完成 Base/Instruct 对照。Instruct 模型在 RAVDESS 和 TESS 上的连续流主效应分别为 2.361 和 4.428，而离散流主效应均接近零；Base 模型在相同固定读出下没有形成可用的四分类情绪决策，连续流效应分别为 -0.024 和 -0.099。逐层 logit-lens tracing 进一步显示，Base 中存在较弱的中高层连续声学表征，但最终读出消失；Instruct 则在第 25–27 层出现强烈放大，并将连续流效应保留到输出。

RAVDESS 反事实 ASR 验证表明，跨情绪交换连续流几乎不改变文本内容：AB 单元 WER 为 0，BA 单元 WER 为 0.0017。随机、零化、高斯噪声及同情绪换流控制进一步表明，真实配对析因结果不能简单归因于任意分布外破坏，但说话人及时间结构仍可能贡献部分变化。

现阶段可以支持的核心判断是：Kimi-Audio-7B-Instruct 在这两套受控语料上的情绪决策平均主要由连续声学丰富流驱动；这种任务可读的强连续流效应与后训练相关，并主要在语言模型主干末端形成或被放大。现阶段尚不能宣称第 25–27 层已被严格证明为因果仲裁层，也不能宣称局部修复、P 时间外评测或跨语料泛化已经完成。

---

## 1. 研究背景与路线调整

项目最初考虑以 CAGE（Counterfactual Acoustic Gating）门控模块作为论文中心。重新检索相关工作后发现，普通声学门控、双路径融合、层混合、选择性 LoRA、activation patching、声学神经元增强及跨语料 SER 均已有高度相近工作。因此，项目主线调整为 K-FAIR：

> 利用真实平行情绪语音，对 Kimi-Audio 原生离散与连续音频流进行模型内部 2×2 析因干预，比较 Base/Instruct 与训练语料暴露组，并在定位后的仲裁层进行最小化修复。

当前安全的创新表述不是“首次研究 Kimi 的情绪机制”或“首次发现文本/声学偏置”，而是：

> 据我们所知，本项目首次利用真实平行情绪语音，对 Kimi-Audio 原生离散语义主导流与连续声学丰富流进行系统的模型内析因干预，并比较 Base/Instruct 及公开 SFT 暴露条件下的路径变化。

投稿前仍需重新进行题名、摘要与关键词级查重。

---

## 2. 研究问题

当前实验主要回答以下问题：

1. Kimi-Audio 的情绪判断主要受连续流还是离散流影响？
2. 真实同文本异情绪换流时，模型会跟随哪一条流提供的情绪？
3. 连续流效应是否只存在于公开披露的 SER-SFT 数据上？
4. Base 与 Instruct 的双流依赖是否不同？
5. 连续情绪证据在哪些 Transformer 层变得可读并被放大？
6. 换流导致的情绪变化是否只是文本识别崩坏或任意分布外扰动？
7. 后续能否只在少数仲裁层进行局部修复，同时保持 SER 与 ASR？

前六个问题已经获得不同程度的证据；第七个问题尚未进入正式训练验证。

---

## 3. 模型、数据与术语

### 3.1 模型

- Kimi-Audio-7B-Instruct：用于主要情绪任务读出。
- Kimi-Audio-7B Base：用于比较后训练前后的表征与路径变化。
- 计算设备：AutoDL A800 80GB。
- SER 不需要生成音频，实验未加载不必要的语音生成 detokenizer。

### 3.2 两条原生输入流

- **D：离散语义主导流。** 来自 Kimi 的离散音频 token，但不能称为纯语义流。
- **C：连续声学丰富流。** 来自 Whisper 连续特征及 VQAdaptor，但同样可能包含词汇信息，不能称为纯声学流。

两路在进入共享 Transformer 前融合。本项目直接在该原生融合接口上控制和交换两路输入。

### 3.3 当前数据分组

| 数据集 | 暴露分组 | 当前用途 | 状态 |
|---|---|---|---|
| RAVDESS | E：公开 SER-SFT 已披露 | 核心 2×2、控制、ASR、Base/Instruct、逐层 tracing | 已完成 |
| TESS | U：公开 SFT 清单未披露 | 核心 2×2、Base/Instruct | 已完成 |
| EMIS/其他发布后语料 | P：时间外或更抗污染 | 时间外冲突评测 | 未完成 |
| ESD | E | 第三套真实平行语音复现 | 未完成 |
| CREMA-D、MELD、SAVEE、JL、MLEnd | E 或 U | 跨语料及基线评测 | 未完成 |

“U”仅表示没有出现在公开披露的 SER-SFT 清单中，不保证没有进入超过 1300 万小时的预训练语料。

### 3.4 标签与读出

主任务仅使用四类：

- neutral
- happy
- sad
- angry

Instruct 和 Base 使用同一提示，并将四类映射为单 token 候选 A/B/C/D，读取首个输出位置的 log-probability。这样避免自由生成标签带来的拼写、格式和多 token 长度差异。

---

## 4. 核心 2×2 析因方法

对同一说话人、同一文本、不同情绪的真实录音 A/B，分别提取：

- `D_A`、`C_A`
- `D_B`、`C_B`

运行四个单元：

1. `D_A/C_A`（AA）
2. `D_A/C_B`（AB）
3. `D_B/C_A`（BA）
4. `D_B/C_B`（BB）

以情绪 B 对 A 的 log-probability contrast 为：

`s(D,C) = log p(y_B|D,C) - log p(y_A|D,C)`

由四个单元计算：

- `ME_C = 0.5 × [(AB−AA) + (BB−BA)]`
- `ME_D = 0.5 × [(BA−AA) + (BB−AB)]`
- `I = BB−BA−AB+AA`

连续特征长度与离散流目标长度不一致时，采用线性时间重采样。该处理保证张量可交换，但仍可能产生一定分布偏移，因此额外设置了真实换流与噪声/零化控制。

---

## 5. 已完成工作的时间线

### 5.1 工程环境与模型接口验证

- 完成 Kimi-Audio-7B-Instruct 部署和推理环境配置。
- 定位原生离散/连续融合位置。
- 增加 Full、No-continuous、No-discrete、Rolled-continuous 等分支模式。
- 验证跨样本连续流替换确实改变模型候选 logits。
- 将自由生成读出替换为四个单 token 的强制候选读出。

### 5.2 早期分支零化消融

完成了 RAVDESS 小规模 Full/关闭连续流/关闭离散流等实验。这些结果用于验证接口和形成初步现象，但零化与高斯噪声属于明显分布外输入，因此没有被作为最终核心因果证据。

### 5.3 RAVDESS/E 真实 2×2 析因实验

- 24 位演员。
- 2 条固定文本。
- 每个演员/文本包含四种目标情绪。
- 6 种无序情绪对。
- 共 288 个真实平行对、1,152 个单元。

### 5.4 TESS/U 真实 2×2 析因实验

- 2 位说话人。
- 200 个词。
- 每个说话人/词包含四种目标情绪。
- 共 2,400 个真实平行对、9,600 个单元。

### 5.5 反事实 ASR 与分布控制

- 对 RAVDESS 96 个核心配对的 AB/BA 冲突单元进行 ASR。
- 完成同情绪其他说话人、随机其他说话人、零化、高斯噪声控制。

### 5.6 Base/Instruct × E/U 对照

- Base×RAVDESS/E：288 对、1,152 单元。
- Instruct×RAVDESS/E：288 对、1,152 单元。
- Base×TESS/U：2,400 对、9,600 单元。
- Instruct×TESS/U：2,400 对、9,600 单元。

### 5.7 Base/Instruct 逐层 logit-lens tracing

在 RAVDESS neutral–angry 与 happy–sad 两类核心对上：

- 96 个真实平行对；
- 384 个 2×2 单元；
- 输入嵌入加 28 个层级读出位置；
- Base 和 Instruct 均完成；
- 按 24 位演员聚类 bootstrap 10,000 次。

---

## 6. 核心实验结果

### 6.1 Instruct：E/U 双组结果

| 数据组 | ME_C | ME_D | 交互效应 | 连续流跟随 | 离散流跟随 | 第三类 | 匹配准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RAVDESS/E | 2.361 [1.844,2.870] | 0.035 [-0.050,0.119] | 0.153 [-0.008,0.313] | 42.5% | 18.8% | 38.7% | 44.8% |
| TESS/U | 4.428 [4.317,4.537] | -0.014 [-0.035,0.009] | -0.026 [-0.065,0.013] | 67.6% | 10.8% | 21.6% | 66.1% |

![Instruct E/U 结果](kfair_eu_results.svg)

两组中 `ME_C` 均显著大于零，`ME_D` 接近零。连续流主导没有局限于公开 SER-SFT 已披露的 RAVDESS，在 TESS/U 上反而更强、更稳定。

### 6.2 情绪对异质性

RAVDESS 的平均结果并不代表每种情绪对都相同：

- neutral–angry：`ME_C=5.998`，连续流跟随率 71.9%。
- happy–sad：`ME_C=-1.305`，连续流跟随率 10.4%。
- sad–angry：`ME_C` 也略为负。

TESS 中 happy–angry、happy–sad、sad–angry 的连续流效应较强，而 neutral–angry 较弱。因而论文应使用“平均连续流主导且存在情绪对异质性”，不能写成“所有情绪都完全由连续流决定”。

### 6.3 Base/Instruct × E/U 完整矩阵

| 模型 | RAVDESS/E：ME_C | TESS/U：ME_C | RAVDESS 匹配准确率 | TESS 匹配准确率 |
|---|---:|---:|---:|---:|
| Base | -0.024 [-0.085,0.034] | -0.099 [-0.123,-0.073] | 27.1% | 25.2% |
| Instruct | 2.361 [1.844,2.870] | 4.428 [4.317,4.537] | 44.8% | 66.1% |

![模型与暴露组矩阵](kfair_model_exposure_matrix.svg)

从 Base 到 Instruct，`ME_C` 在 RAVDESS/E 上增加约 2.385，在 TESS/U 上增加约 4.527。该模式与“后训练建立或强化连续声学流到情绪标签的任务读出”一致。

但 Base 的匹配准确率接近四分类 25% 机会水平，所以不能把 Base 的小幅负效应解释成可靠的反向情绪机制，也不能将 Base/Instruct 表格当作通常意义上的性能比较。

---

## 7. 内容保持与控制实验

### 7.1 反事实 ASR

| 冲突单元 | n | WER | 完全匹配率 |
|---|---:|---:|---:|
| AB：离散 A + 连续 B | 96 | 0 | 100% |
| BA：离散 B + 连续 A | 96 | 0.0017 | 98.96% |

A/B 本身为同人、同文本、不同情绪。连续流交换几乎没有破坏语言内容，却显著移动情绪证据，排除了“情绪变化只是转写崩坏”的简单解释。

### 7.2 分布破坏控制

| 连续流条件 | Accuracy | 相对原始预测翻转率 | 正确类 log-prob 变化 |
|---|---:|---:|---:|
| Original | 44.8% | 0.0% | 0.000 |
| Same emotion, other speaker | 34.9% | 44.8% | -0.150 |
| Random other speaker | 20.8% | 68.8% | -1.135 |
| Zero | 21.4% | 68.8% | +1.182 |
| Gaussian | 28.6% | 71.9% | +0.956 |

同情绪其他说话人的连续流比随机其他语音破坏更小，说明真实换流效应并非完全由任意 OOD 替换造成。但同情绪换流仍使准确率下降，表明说话人和时间结构也有贡献。Zero/Gaussian 条件还暴露出候选 logits 的校准与类别竞争问题，因此不能仅使用零化或单一 log-prob 均值证明流重要性。

---

## 8. 逐层路径结果

### 8.1 Instruct

- 第 0 层 `ME_C=0`。
- 第 16–17 层出现反向阶段，第 17 层为 -0.381 [-0.499,-0.261]。
- 第 20 层重新显著转正：0.361 [0.239,0.479]。
- 第 22 层升至 0.871 [0.595,1.151]。
- 第 25 层跃升至 3.491 [2.601,4.346]。
- 第 27 层为 3.577 [2.779,4.369]。
- 最终归一化读出为 2.346 [1.836,2.890]。

### 8.2 Base

- 第 20–23 层存在较弱连续声学可读信号。
- 第 23 层为 0.581 [0.453,0.710]。
- 第 26 层为 0.616 [0.407,0.833]。
- 最终归一化读出回到 -0.020 [-0.099,0.054]。

![Base 与 Instruct 逐层结果](kfair_base_instruct_layers.svg)

最明显的 Base/Instruct 分化位于第 25–27 层。这说明 Base 并非完全不含连续声学情绪信息，而是没有像 Instruct 一样在主干末端将其组织并放大为当前任务可读的标签决策。

必须强调：当前使用的是 logit lens。中间隐藏状态不是在训练时被要求直接接受最终 LM head，因此它能够定位“线性可读性变化”，但不能单独证明第 25–27 层就是因果仲裁层。真正的 activation patching 尚未完成。

---

## 9. 统计设计

- RAVDESS：按 24 位演员聚类 bootstrap，避免把同一演员的录音视作完全独立。
- TESS：按 200 个词聚类 bootstrap，同时保留同词两位说话人的依赖结构；只有两位说话人，不适合按说话人计算稳定区间。
- 重采样次数：10,000。
- 报告 95% 置信区间。
- 分情绪对报告效应异质性。

尚未进行训练实验，因此三随机种子、训练方差和 held-out-actor 调参尚不存在，必须在局部修复阶段补齐。

---

## 10. 当前可以支持的结论

1. 在 RAVDESS/E 与 TESS/U 上，Kimi-Audio-7B-Instruct 的平均情绪判断主要由连续声学丰富流推动。
2. 连续流主效应显著大于离散流主效应，冲突单元也更常跟随连续流。
3. 真实同文本换流几乎不破坏 ASR 内容，因此情绪效应不能简单归因于转写失败。
4. 结论跨 E/U 复现，削弱了“只是在 RAVDESS 上记忆公开 SER-SFT 数据”的单一解释。
5. Base 的中高层含有较弱连续情绪表征，但没有形成当前四选一任务的可靠最终读出。
6. Instruct 在第 25–27 层对连续声学情绪证据发生明显放大，提示后续局部干预应优先从主干末端开始。
7. 情绪对之间存在明显异质性，平均结果不能推广为每种情绪组合都连续流主导。

---

## 11. 当前不能支持的结论

1. 不能把 TESS 称为保证未见的 zero-shot 数据，只能称公开 SFT 清单未披露。
2. 不能仅凭 Base/Instruct 差异断言某一具体 SER-SFT 数据造成了连续流效应。
3. 不能说 Base 不含情绪信息；逐层结果显示其内部存在较弱表征。
4. 不能说第 25–27 层已被严格证明为因果仲裁层；目前只有 logit-lens 定位。
5. 不能说 K-FAIR 已经完成修复或提高跨语料 Macro-F1。
6. 不能说换流完全无分布偏移；线性时间重采样、说话人和时间结构仍是限制。
7. 不能把离散流称为纯语义、连续流称为纯声学。
8. 不能声称所有情绪对都由连续流控制。

---

## 12. 老师七大步骤状态

| 步骤 | 要求 | 状态 | 当前证据或缺口 |
|---:|---|---|---|
| 1 | 分离 Kimi 原生离散与连续流 | **已完成** | 已在原生融合点实现控制与跨样本换流 |
| 2 | 真实同人、同文本、异情绪配对 | **已完成主体** | RAVDESS 与 TESS 完成；ESD 尚缺 |
| 3 | `DA/CA、DA/CB、DB/CA、DB/CB` 真实 2×2 | **已完成** | 四个模型×数据组合均已完成 |
| 4 | 计算 ME_C、ME_D 与交互 | **已完成** | 含聚类 CI 与分情绪对结果 |
| 5 | 逐层信息路径追踪 | **部分完成** | Base/Instruct logit lens 完成；真正 activation patching 未完成 |
| 6 | Base/Instruct 与 E/U/P 暴露比较 | **部分完成** | Base/Instruct×E/U 已补齐；P 时间外组未完成 |
| 7 | 在定位后的仲裁层进行局部修复 | **未完成** | 尚未训练 adapter/LoRA/门控，也未做修复后 SER/ASR 评测 |

如果仅看 K-FAIR 核心诊断，当前约完成 75%；如果按完整机制论文与方法贡献计算，当前约完成 45%。

---

## 13. 剩余工作与优先级

### 优先级 1：第 24–27 层 activation patching

目标：确认这些层不仅与连续流效应相关，而且对冲突决策具有因果作用。

建议先在 RAVDESS 96 个核心对上完成：

- patch 第 24、25、26、27 层输入或输出状态；
- AA/BB 匹配状态向 AB/BA 冲突单元回补；
- 正向、反向、错误层、随机位置及无 patch 对照；
- 只 patch 音频相关 token 与 patch 全序列的比较；
- 报告 logit rescue、预测翻转、SER 与 ASR。

预计 2.5–3.5 小时，GPU 有效计算约 20–45 分钟。

### 优先级 2：P 时间外组

目标：补齐 E/U/P 暴露分析，降低预训练污染质疑。

候选包括 EMIS 或其他在 Kimi 初始权重发布后构造的受控冲突集。需要完成数据可用性、标签和任务格式审计后再运行。

预计 2–6 小时，主要不确定性在数据获取和标签清理，而非 GPU。

### 优先级 3：held-out-actor 局部修复

只有 activation patching 确认高层因果位置后再启动：

- 冻结 Kimi 主干；
- 仅在第 24–27 层加入低秩 residual adapter、branch-aware LoRA 或训练自由 residual correction；
- 演员级训练/验证/测试拆分；
- 联合 `L_SER + λ1 L_pair + λ2 L_preserve`；
- 对比简单校准、Gate-only、普通参数匹配 adapter/LoRA。

第一版原型预计 3–6 小时；可靠的多种子结果需要更长时间。

### 优先级 4：跨语料与强基线

仍需补充：

- ESD 第三套真实平行语音；
- CREMA-D、MELD、SAVEE 等跨语料测试；
- transcript-only；
- emotion2vec；
- WavLM/Whisper frozen probe；
- Base frozen linear probe；
- 普通 adapter/LoRA 与参数匹配基线；
- Average/Worst-Corpus Macro-F1、UA、WA、每类 recall、混淆矩阵。

### 优先级 5：投稿级复现与写作

- 固定模型 revision、权重 hash、代码 commit 和补丁 hash；
- 三随机种子；
- 训练超参数和演员拆分公开；
- 方法图、主结果表、消融表和误差分析；
- 投稿前重新查重并收紧创新表述。

---

## 14. 下一次会议可直接使用的汇报版本

### 30 秒版本

我们已经完成 Kimi-Audio 原生离散与连续流的真实 2×2 析因干预。RAVDESS/E 与 TESS/U 上，Instruct 的连续流主效应分别为 2.36 和 4.43，离散流效应接近零；反事实换流几乎不改变 ASR 内容。Base/Instruct 对照显示，Base 最终层没有稳定的四分类连续流效应，而 Instruct 在第 25–27 层对连续声学情绪证据发生强烈放大。下一步将通过高层 activation patching 判断该位置是否具有真正的因果仲裁作用，然后再做局部修复。

### 当前贡献

- 原生双流而非外挂双编码器；
- 真实同人同文本异情绪对；
- 完整 2×2 析因效应而非仅零化消融；
- Base/Instruct × E/U 对称矩阵；
- ASR 内容保持与 OOD 控制；
- 将后训练相关分化定位到第 25–27 层。

### 主动说明的限制

- P 组尚缺；
- tracing 还是 logit lens，不是 activation patching；
- Base 不是可直接使用的指令模型；
- TESS/U 不等于保证未见；
- 尚未完成局部修复和跨语料性能提升。

---

## 15. 主要产物索引

### 综合报告与图

- `KFAIR_中文阶段总报告.md`：本报告。
- `kfair_eu_comparison_report.md`：Instruct 的 E/U 双组报告。
- `kfair_base_instruct_comparison.md`：Base/Instruct 与逐层定位报告。
- `kfair_base_tess_report.md`：Base×TESS/U 补充报告。
- `kfair_meeting_talking_points.md`：会议口头提纲。
- `kfair_eu_results.svg`：Instruct E/U 对照图。
- `kfair_model_exposure_matrix.svg`：模型×暴露组矩阵图。
- `kfair_base_instruct_layers.svg`：Base/Instruct 逐层曲线。

### RAVDESS 核心结果

- `kfair_factorial_predictions.jsonl`
- `kfair_factorial_effects.csv`
- `kfair_factorial_statistics.csv`
- `kfair_by_emotion_pair.csv`
- `kfair_counterfactual_asr.jsonl`
- `kfair_counterfactual_asr_summary.csv`
- `kfair_controls.jsonl`
- `kfair_controls_summary.csv`

### TESS/Instruct 结果

- `tess_factorial_predictions.jsonl`
- `tess_factorial_effects.csv`
- `tess_factorial_statistics.csv`
- `tess_by_emotion_pair.csv`

### Base 对照结果

- `kfair_base_factorial_predictions.jsonl`
- `kfair_base_factorial_effects.csv`
- `kfair_base_factorial_statistics.csv`
- `tess_base_factorial_predictions.jsonl`
- `tess_base_factorial_effects.csv`
- `tess_base_factorial_statistics.csv`
- `tess_base_by_emotion_pair.csv`

### 逐层 tracing

- `kfair_layer_trace.jsonl`
- `kfair_layer_trace_effects.csv`
- `kfair_layer_trace_statistics.csv`
- `kfair_base_layer_trace.jsonl`
- `kfair_base_layer_trace_effects.csv`
- `kfair_base_layer_trace_statistics.csv`

---

## 16. 阶段结论

截至目前，项目已经越过“接口是否可行”和“现象是否存在”两个风险最高的阶段。真实 2×2 析因结果、E/U 复现、Base/Instruct 对照、ASR 保持和逐层 tracing 共同构成了一条相对完整的诊断证据链：强连续流情绪效应主要出现在 Instruct，并在第 25–27 层被显著放大。

项目尚未形成完整的方法闭环。下一步最重要的不是立即增加复杂模块，而是先用 activation patching 验证第 24–27 层的因果作用。若 patching 能稳定救回或改变冲突决策，再在这些层上进行最小化局部修复；若 patching 失败，则应修正当前的层级解释，而不是继续投入 LoRA 训练。

因此，当前最准确的项目状态是：

> **K-FAIR 的核心析因诊断与 Base/Instruct×E/U 证据已经完成；因果仲裁层验证、P 组和局部修复仍是后续决定论文是否形成完整贡献的关键工作。**
