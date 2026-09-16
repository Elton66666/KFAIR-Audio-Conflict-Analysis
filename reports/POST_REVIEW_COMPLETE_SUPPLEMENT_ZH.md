# 收到 Review 后的全部投稿补实验总报告

日期：2026-09-16  
项目：KFAIR Audio Conflict Analysis  
用途：供导师、论文合作者和 Claude Code/其他代码代理精确更新 ICASSP 稿件  
最终证据仓库：https://github.com/Elton66666/KFAIR-Audio-Conflict-Analysis  
本轮最终 Git 提交：`91081a1fc7c7eba23fa5b912eebb77a567a1d153`

> 本文档汇总收到 `REVIEW_zh.md` 后完成的全部补充工作。为防止二次概括丢失技术细节，文档前半部分提供统一叙事、审稿风险闭环、最终数字口径和论文修改建议；后半部分把三份原始详细报告完整收录。第一份高优先级报告作为最重要的技术主体全文保留，不删减机器环境、模型结构、数据、训练、统计、逐折结果、耗时、故障恢复、文件入口与结论边界。后两份报告记录其后完成的参数匹配、真实扰动、扩大 ASR 及 EMIS 全量验证。

---

## 一、Review 后究竟补了什么

Review 当时给出的接收概率约为 55%–65%，最严重风险依次是：单模型；只有 4 位 held-out 演员和两条固定句；冲突数据主要为构造或合成；普通 LoRA 在噪声下较强，可能削弱“因果定位是否必要”的说服力；ASR 保持只检查 32 句；EMIS 只用 192 条平衡子集。收到评估后，本项目没有另开无关的新任务，而是围绕原论文“诊断→因果定位→低成本修复→验证”主线完成以下补强。

1. **Kimi 六折 held-out-actor 交叉验证**：由单一 actor 21–24 测试扩展为 6 个互斥演员折，使 24 位演员都在某一折成为测试演员；三种等参数方法、三个随机种子，共 54 次训练。
2. **第二模型 Qwen2.5-Omni-7B 最小复现**：验证语义—声学冲突时的信任偏置和 57,344 参数低秩修复并非只存在于 Kimi。
3. **Qwen 参数严格匹配对照**：比较同为 57,344 参数的 layer-22 full-sequence adapter 与普通 q-projection LoRA，排除“效果只是参数更多”的简单解释。
4. **真实扰动鲁棒性**：将原有白噪声检查升级为 OpenSLR SLR28 真实房间噪声、真实办公室 RIR、G.711 μ-law 电话信道以及组合退化。
5. **扩大 ASR 保持测试**：由 32 句扩展为 RAVDESS 192 条和 TESS 192 条，TESS 覆盖 96 个不同目标词；总计 3,840 次转写。
6. **EMIS 全量 1,248 条时间外验证**：替代只依赖 192 条平衡子集的最终主结论；同时复核双流干预，并对三类 RAVDESS 训练修复做全量零样本迁移。

这些工作仍服务于同一故事：模型并非简单“听不到”韵律，而是在高层决策中对不同证据分配了不恰当的信任；原生双流析因和 activation patching 用于定位问题，小参数干预用于验证该位置是否可修复；跨演员、跨模型、真实扰动、ASR 保持和时间外语料用于排除更简单的替代解释。

## 二、Review 风险与当前闭环状态

| Review 风险 | 完成的补充工作 | 当前能支持的结论 | 仍需保留的限制 |
|---|---|---|---|
| 单模型 Kimi | Qwen 行为复现、decision adapter；随后补 full-sequence adapter 与同参数 LoRA | 冲突下的系统性信任偏置和低参数修复可跨到第二模型 | Qwen 未复刻 Kimi 原生双流交换和完整逐层 activation patching；不能宣称相同因果层普适 |
| 只有 4 位测试演员 | Kimi 6 折演员交叉验证，覆盖全部 24 位演员 | 三类 57,344 参数修复在六个测试折上均总体正提升 | 每折仍只有 4 位演员；RAVDESS 仍只有两条固定句 |
| 91.67% 单次高分可能偶然 | 主结果改为六折均值±折间 SD，并报告逐折和 Wilcoxon | 基线 43.66%±9.21%；修复 75.52%–77.95%；六折方向一致，p=0.03125 | 91.67%及相近数字仅能描述特定 actor 21–24 划分，现有证据不足以把它概括为普遍性能 |
| LoRA 是强基线，定位是否必要 | Kimi 六折等参数 LoRA；Qwen 等参数 full adapter/LoRA；真实扰动比较 | 因果定位的价值是解释故障并提供可控干预位置，不是宣称 adapter 永远优于 LoRA | LoRA 在部分指标和扰动条件上最佳；可在论文中作为方法边界明确说明 |
| 只有白噪声 | SLR28 真实噪声/RIR、G.711 和组合失真 | 修复收益在真实声学退化下没有整体消失 | 仍不是完整自然部署 benchmark；32 条基础音频较小 |
| ASR 只检查 32 句 | 384 条音频、96 个 TESS 词、10 个系统配置，共 3,840 条转写 | 修复没有通过破坏或重写语音文字内容获得情绪高分 | 不是 LibriSpeech 级通用 ASR 评测；TESS 只有两位女性说话人 |
| EMIS 只有 192 条 | 官方全部 1,248 条；936 conflict、312 aligned；16,224 条预测 | 外部提升幅度虽小，但三方法×三种子的冲突配对检验全部 p<0.05 | EMIS 仍是合成语音，不等于真实录制冲突 |
| 构造/合成冲突 | 时间外全量 EMIS、真实声学扰动加强了边界验证 | 可更稳健地讨论时间外与声学退化，不再只依赖单一小子集 | 尚未加入 CASE 等真实场景冲突；适合作为 limitation，现阶段不必临时扩展成新故事 |
| 层选择噪声 | 六折外部测试、随机早层/错误层历史对照、多方法比较 | 所选高层干预在跨演员评测中有效 | 层 22/25 不是每个外折内部重新选择；仍缺 nested CV |

## 三、统一运行机器与软件环境

所有本轮正式实验均运行于 AutoDL 单机单卡服务器，项目根目录 `/root/autodl-tmp/kfair`，虚拟环境 `/root/autodl-tmp/kfair/fastenv2`。

| 项目 | 配置 |
|---|---|
| 操作系统 | Ubuntu 22.04.4 LTS，Linux kernel `4.19.90-2107.6.0.0248.35.oe1.bclinux.x86_64` |
| CPU | Intel Xeon Gold 6348 @ 2.60 GHz；2 sockets×28 cores×2 threads，112 logical CPUs |
| 内存 | 1.0 TiB，无 swap |
| GPU | NVIDIA A800 80GB PCIe，81,920 MiB，Compute Capability 8.0 |
| 驱动/功率上限 | 590.48.01 / 300 W |
| Python | 3.12.3，Anaconda build，GCC 11.2.0 |
| PyTorch/CUDA/cuDNN | 2.5.1+cu124 / 12.4 / 9.1.0 |
| torchaudio | 2.5.1+cu124 |
| Transformers/FlashAttention | 4.57.6 / 2.7.4.post1 |
| Accelerate | 1.14.0 |
| pandas/NumPy/SciPy | 3.0.5 / 2.1.3 / 1.18.1 |
| safetensors | 0.8.0 |

Kimi 和 Qwen 主干均以 BF16 前向；新增 adapter/LoRA 权重为 FP32，增量转回 hidden-state dtype。正式流水线固定 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 和 `HF_HOME=/root/autodl-tmp/kfair/cache/huggingface`，避免模型版本漂移。

## 四、可供投稿取舍的最终数字口径

### 4.1 Kimi 主修复结果：六折结果是更稳健的主口径

| 方法 | 参数量 | 六折 SER | Aligned SER | Conflict acoustic follow | Conflict semantic follow |
|---|---:|---:|---:|---:|---:|
| Baseline | 0 | 43.66%±9.21% | 44.79%±8.31% | 42.53%±10.28% | 18.75%±3.16% |
| Decision adapter L25 | 57,344 | 75.52%±11.03% | 75.00%±11.49% | 76.04%±10.62% | 7.70%±3.68% |
| Full-sequence adapter L22 | 57,344 | 76.13%±6.25% | 76.39%±6.84% | 75.87%±5.78% | 7.87%±2.26% |
| Ordinary LoRA L22 | 57,344 | 77.95%±10.32% | 77.95%±10.19% | 77.95%±10.50% | 7.29%±3.41% |

三种修复相对基线平均提升 +31.86、+32.47、+34.29 pp；对六个折的双侧 Wilcoxon 均为 p=0.03125。若相关结果进入论文摘要、主表或结论，这些六折数字是更稳健的选择。actor 21–24 上 91.32%、86.28%、93.58%等结果更适合作为逐折结果，而不宜替代六折总体。

### 4.2 Qwen 第二模型：可按三个层次理解和表述

1. **行为基线**：RAVDESS 全体声学准确率 68.23%；actor 21–24 为 65.63%。EMIS 192 平衡子集上 aligned 100%，冲突声学跟随 1.39%、语义跟随 95.14%。
2. **Decision adapter 最小复现**：actor 21–24 从 65.63%提高到 84.38%±8.27%，平均 +18.75 pp；三个种子中两个 McNemar 显著，第三个 p=0.25。
3. **严格参数匹配对照**：57,344 参数 full-sequence adapter L22 为 91.67%±3.61%，同参数普通 LoRA L22 为 88.54%±1.80%；两者在 EMIS 192 上均约 27%，未形成可靠域外修复。

Qwen 支持的是跨架构“行为偏置＋低参数可修复”，不是 Kimi 双流结构和第 22–27 层因果轨迹的完整复制。

#### 4.2.1 Qwen 第二模型实验总览：做了什么、修复前后怎样、能说明什么

Qwen 部分围绕同一问题依次完成了三组实验，而不是新增一条独立研究故事。第一组是**行为基线复现**：用未修改的 Qwen2.5-Omni-7B 检查语义与语调发生冲突时模型更信哪一种证据。第二组是**Decision adapter 最小修复**：冻结整个主干，只在 thinker 第 25 层的最后 decision token 上加入 rank=8、57,344 参数的低秩 adapter。第三组是**严格参数匹配对照**：在第 22 层比较同为 57,344 参数的 full-sequence adapter 与普通 q-projection LoRA，从而区分“作用位置/方式”与“单纯增加参数量”。

三种修复均只用 RAVDESS 训练，演员 1–16 为训练集、17–20 为验证集、21–24 为测试集；测试集共 32 条，其中固定句文字语义统一视为 neutral，8 条 neutral 为语义—声学一致样本，另外 24 条非 neutral 为冲突样本。EMIS 192 条从未参与训练，其中 144 条为直接语义—声学冲突，用于零样本跨语料检查。下表把此前分散在不同章节的数字集中列出：

| Qwen 系统 | 可训练参数 | RAVDESS test 整体声学准确率（n=32） | RAVDESS 冲突声学跟随率（n=24） | EMIS 整体声学准确率（n=192） | EMIS 冲突声学跟随率（n=144） |
|---|---:|---:|---:|---:|---:|
| 未修复 baseline | 0 | 65.63% | 54.17% | 26.04% | 1.39% |
| Decision adapter L25 | 57,344 | 84.38%±8.27% | 81.94%±14.63% | 27.78%±1.20% | 4.40%±1.45% |
| Full-sequence adapter L22 | 57,344 | 91.67%±3.61% | 93.06%±2.41% | 27.08%±0.52% | 5.56%±0.69% |
| Ordinary LoRA L22 | 57,344 | 88.54%±1.80% | 84.72%±2.41% | 27.78%±0.30% | 5.32%±0.80% |

其中，RAVDESS baseline 与 Decision adapter 的冲突数字由逐样本预测按 `dataset=RAVDESS`、actor 21–24、`conflict=true` 重新核对：baseline 为 13/24，即 54.17%；Decision adapter 三个种子分别为 23/24、20/24、16/24，即 95.83%、83.33%、66.67%。Full adapter 三个种子为 95.83%、91.67%、91.67%；LoRA 为 83.33%、87.50%、83.33%。表中均值和样本标准差由这三个种子计算。

统计上，Decision adapter 对相同 32 条 RAVDESS 测试样本的三个种子分别提高 +25.00、+21.88、+9.38 个百分点；精确 McNemar `p=0.0386/0.0156/0.25`，即两个种子显著、一个种子不显著。参数匹配实验中，Full adapter 相对 baseline 的三个 `p=0.0225/0.0654/0.0117`，LoRA 为 `0.0156/0.0078/0.0391`。EMIS 上参数匹配方法的检验均不显著，Decision adapter 也只表现为小幅且不稳定的外部变化。

这组实验可支持四点相互衔接的判断：

1. **现象可以跨到第二模型。** Qwen 在 EMIS 一致样本上为 100%，但冲突时声学跟随只有 1.39%、语义跟随达到 95.14%，说明问题不是完全听不出情绪，而是冲突决策明显偏向文字语义。
2. **低参数同域修复可以跨架构成立。** 三种 57,344 参数方法都明显提高 RAVDESS held-out 演员的整体准确率和冲突声学跟随率，因此 Kimi 上的可修复性并非只来自单一模型或单一实现。
3. **提升不能只用参数数量解释。** Full adapter 与普通 LoRA 参数完全相同，但表现有所差异；不过测试只有 32 条，Full adapter 的小幅领先不足以证明它普遍或显著优于 LoRA。
4. **跨语料通用修复尚未成立。** 三种方法在 EMIS 冲突上的绝对提升只有约 3–4 个百分点，整体准确率仍约 27%。因此 Qwen 提供的是“冲突偏置和同域低成本修复具有一定跨架构性”的证据，而不是对 Kimi 原生双流机制、相同因果层位置或跨数据集通用修复的完整复现。

### 4.3 真实扰动结果

最困难的 RIR＋真实噪声＋G.711 条件：baseline 37.50%，decision adapter 80.21%±1.80%，full-sequence adapter 71.88%±8.27%，LoRA 83.33%±3.61%。正文可用这一列概括压力测试，其余五个条件保留在详细表或补充材料。

### 4.4 ASR 保持结果

RAVDESS 192 条：baseline、decision、full、LoRA 的 WER 分别为 0.087%、0.087%、0.087%、0.029%。TESS 192 条：2.734%、2.734%、2.821%、3.082%。3,840 条转写无空输出。核心结论是修复没有以破坏语音内容为代价。

### 4.5 EMIS 最终结果：全量 Kimi 结果是较稳妥的口径

| 系统 | 整体声学准确率 | 936 冲突声学跟随 | 冲突语义跟随 | 312 匹配准确率 |
|---|---:|---:|---:|---:|
| Baseline | 52.24% | 43.59% | 37.18% | 78.21% |
| Decision adapter L25 | 57.21%±1.18% | 49.15%±1.34% | 34.12%±1.36% | 81.41%±1.28% |
| Full-sequence adapter L22 | 55.26%±1.13% | 47.01%±1.05% | 36.15%±1.13% | 80.02%±1.88% |
| Ordinary LoRA L22 | 57.13%±0.81% | 47.79%±0.80% | 37.25%±0.43% | 85.15%±0.93% |

Decision/full/LoRA 的冲突声学跟随平均提升 +5.56/+3.42/+4.20 pp；九个修复种子相对相同 baseline 的冲突 McNemar 精确检验全部 p<0.05。正确结论是“幅度有限但跨种子一致、配对显著的外部迁移”，不是“跨域问题已解决”。192 条平衡子集仍可作为预先构造的审计子集；Qwen 目前仍只在该 192 子集上评测，不能把 Kimi 全量数字套到 Qwen。

## 五、补实验如何改变论文

### 5.1 摘要和贡献点

如篇幅允许，可从以下四方面选择或压缩补充贡献点：

1. 在 Kimi 原生语义主导离散流与声学丰富连续流接口上进行 2×2 析因换流，显示连续流承载主要可用韵律证据。
2. 用 logit tracing 提出层假设，并以 activation patching 因果确认高层冲突仲裁；logit lens 只负责提出假设，因果结论来自 patching。
3. 冻结 7B 主干，只训练 57,344 参数，在六折 held-out-actor 测试中把 SER 从 43.66%提高到 75.52%–77.95%。
4. 用 Qwen 第二模型、等参数 LoRA、全量时间外 EMIS、真实噪声/RIR/电话信道以及扩大 ASR 检查结论的跨架构性、域外边界、鲁棒性和能力保持。

### 5.2 主表调整

- 如果主修复结果被纳入正文，用六折均值±SD替代单一 actor 21–24 结果会更稳健；逐折结果可视篇幅放入补充材料。
- Qwen 结果可以压缩成一行或一个紧凑小表，涵盖 baseline、decision adapter、同参数 full adapter、同参数 LoRA，并注明测试仍为 32 条。
- 外部与能力保持部分可酌情选取 EMIS 全量 conflict acoustic follow、最强组合扰动和 RAVDESS/TESS WER；六种扰动完整表和逐种子 p 值可作为补充材料或仓库证据。

### 5.3 因果定位与 LoRA 的关系

现有实验不足以支持“只有因果定位才能获得性能”这一表述。普通 LoRA 是非常强的工程基线，在 Kimi 六折均值、部分真实扰动和匹配样本保持上可达到最高。更严谨的分工是：

- 因果定位回答**故障在何处形成、模型为何听到却没有信任声学证据**；
- Decision adapter 提供只修改决策 token 的可解释、低作用范围干预；
- Full-sequence adapter 提供因果层附近更广的残差修正；
- LoRA 提供强通用工程基线。

科学贡献是定位 perception–trust gap 并证明该位置可被低成本干预；工程贡献是三类 57,344 参数修复的可选 trade-off，而非 adapter 对 LoRA 的全面胜利。

### 5.4 应主动写入的限制

1. 详细层级因果定位只在 Kimi 完成，Qwen 是行为与修复复现。
2. RAVDESS 仍只有 24 位演员和两条固定句；Qwen 测试划分仍只有 4 位演员、32 条。
3. EMIS 虽已用全量 1,248 条，仍为合成语音；项目没有真实录制讽刺/反语冲突语料。
4. 层 22/25 来自先前全项目实验，并未在六折每个外折内部 nested selection。
5. LoRA 在多项指标中不弱于 adapter；因果定位的价值不能用“性能唯一必要条件”表述。
6. 真实扰动测试基础集只有 32 条；它是压力测试，不是完整部署 benchmark。

## 六、数据规模、运行量与可复现性总览

| 补充工作 | 运行/预测规模 | 主要产物 |
|---|---:|---|
| Kimi 六折演员 CV | 54 次训练；10,368 修复预测；1,152 baseline 预测；54 checkpoints | 六折汇总、逐折/逐演员指标、配对统计、完整日志 |
| Qwen 最小复现 | 384 baseline 行为预测；3 次训练；672 修复预测；3 checkpoints | 行为统计、adapter 结果、EMIS 192 零样本结果 |
| Qwen 参数匹配 | 6 次训练；1,344 预测；6 checkpoints | full adapter 与 LoRA 同参数对照、paired McNemar |
| 真实扰动 | 1,920 唯一预测；60 个系统—条件组合 | 六条件汇总、原始预测、SLR28/G.711 provenance |
| 扩大 ASR | 3,840 唯一转写；20 个方法—种子—语料组合 | WER/CER/exact match、逐条 transcript、无空输出验证 |
| EMIS 全量 | 13 配置；16,224 唯一预测；1,248 音频全覆盖 | 双流干预、三类修复、cluster bootstrap、McNemar、来源校验 |

所有关键代码、统计、逐样本预测、运行日志、小型 checkpoint 与报告均位于 GitHub。模型大权重和受数据许可约束的原始音频不进入普通 GitHub；仓库提供来源、MD5、manifest 和复现说明。

## 七、总报告阅读与引用规则

下面完整收录三份技术报告。它们按实验发生顺序保留，因此较早报告中的“尚未完成”“只能方向性表述”等状态，可能已被后续工作更新。老师或写作者如需根据篇幅取舍材料，可参考以下证据优先级：

1. Kimi 主修复性能：以第一份报告的六折结果为准。
2. Qwen 参数量公平性：以第二份报告的参数匹配结果为准。
3. 真实鲁棒性与 ASR：以第二份报告为准。
4. Kimi 的最终 EMIS 外部结论：以第三份全量 1,248 条报告为准。
5. Qwen 的 EMIS 结论：仍使用第一、二份报告中的 192 条平衡子集结果。
6. 第一份报告中的机器环境、模型结构、六折方法、训练配置、逐折数据、耗时与恢复记录仍是本轮最完整的技术依据，不因后续报告而省略。

---

# 技术卷 A：高优先级补实验完整报告（全文保留）

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

---

# 技术卷 B：参数匹配、真实扰动与扩大 ASR（全文保留）

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

## 7. 可供取舍的论文补充点

1. Qwen 参数匹配实验可作为“提升并非仅由新增参数量造成”的证据，同时普通 LoRA 的强基线表现也构成重要的结果边界。
2. EMIS 上有限甚至失败的迁移结果可以用于说明跨域局限；保留这一负结果有助于提高结论边界的可信度。
3. SLR28+G.711 结果可以回应“方法是否只在理想合成条件有效”的问题。
4. 384 条音频、96 个词的 ASR 测试比原先 32 句结果覆盖更充分，可用于说明修复几乎不改变内容识别。
5. 32 条 held-out test 和单一外部语料仍不足以支持普适性结论；这些补充提高了可信度，但无法替代更大规模、多模型、多真实冲突语料验证。

---

# 技术卷 C：EMIS 全量 1,248 条时间外验证（全文保留）

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
4. 全量基线的冲突声学跟随明显高于 192 子集，说明小平衡子集并不能代表 EMIS 的总体难度。若论文纳入这部分结果，全量结果是更有代表性的选择，192 子集则可作为预先构造的均衡审计集或补充材料。

## 7. 这项结果可提供的论文补充信息

建议表述：

> On all 1,248 post-release EMIS samples, RAVDESS-trained repairs consistently improve acoustic following under semantic–prosodic conflict by 3–6 percentage points across three seeds, with paired McNemar significance in every run. The modest absolute gains nevertheless leave substantial cross-domain headroom.

现有结果较稳妥地支持：全量时间外数据复现了连续流的重要作用；RAVDESS 训练的三类修复都出现跨域正迁移；这种提升跨种子一致且配对显著。

现有结果尚不足以支持：EMIS 是真实自然录音；修复已经解决跨域语义偏置；adapter 全面优于 LoRA；全量 EMIS 可以替代真实冲突语料。

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

---

## 八、可供论文修改时参考和取舍的补充点

以下内容用于说明各项补实验可能对已有论文产生的影响。受篇幅和整体叙事限制，老师或写作者可以结合论文重点自行选择，并不意味着所有内容都需要进入正文。

1. **主结果口径**：六折 Kimi 结果比单折 91.67%更适合作为摘要、主表或结论依据；逐折结果则可以作为补充证据。
2. **整体叙事位置**：诊断→因果定位→修复仍可作为主轴；Qwen、真实扰动、ASR 和 EMIS 更适合被理解为验证环节，而非四条彼此独立的新故事。
3. **Qwen 证据边界**：Qwen 提供的是跨架构行为复现证据，并不等同于复制 Kimi 原生双流机制或相同内部层位置。
4. **公平比较口径**：57,344 参数构成三类方法的共同预算。结果显示 adapter 与 LoRA 各有优势，尚不支持 adapter 全面优于 LoRA。
5. **外部结果选择**：Kimi 的 EMIS 全量 1,248 条结果比 192 条子集覆盖更充分；Qwen 的 EMIS 证据目前仍限于 192 条子集。
6. **外部迁移措辞**：Kimi 全量 EMIS 更符合“modest but consistent and paired-significant external improvement”的描述，同时仍受合成数据属性限制。
7. **鲁棒性证据**：最困难组合扰动可简洁展示收益在真实噪声、RIR 和电话信道下没有消失；baseline 加噪后的偶发上升更可能来自小样本和决策边界波动，而非噪声本身有益。
8. **能力保持证据**：扩大 ASR 测试为“修复没有依赖破坏文本内容获益”提供了更充分证据；TESS 统计采用完整载句 `Say the word <target>` 作为参考。
9. **可披露的局限**：真实录制冲突缺失、Qwen 测试规模较小、层选择并非 nested CV、LoRA 基线很强、真实扰动基础集较小，均有助于界定结论适用范围。
10. **数据可追溯性**：GitHub 中的 `results/statistics/` 与 `results/raw_predictions/` 保存了论文数字对应的统计和逐样本记录，可供写作时核查。

## 九、最终总判断

Review 后的补实验已经把原稿最危险的两项问题——单模型和单一四演员高分——实质性缓解，并进一步补上了参数量公平性、真实声学退化、扩大 ASR 和全量时间外语料。当前最稳健的论文结论是：

> Kimi-Audio 在语义—韵律冲突中并非缺少声学情绪信息；连续声学流携带可用证据，但高层决策可能错误分配信任。因果定位揭示了这一仲裁区域，冻结 7B 主干、只训练 57,344 参数即可在六折演员外测试中稳定改善声学忠实度。第二模型、等参数 LoRA、真实扰动、扩大 ASR 与全量 EMIS 共同表明该现象和修复并非单一划分或输入损坏的产物，同时也清楚限定了跨域、真实冲突和方法优越性的边界。

本轮不再需要通过增加第三模型、CASE、新任务或新副语言维度来扩张故事。投稿前剩余工作的重点应是把上述最终口径准确写入现有论文、更新表格和 claim trace，并由作者逐句核实。
