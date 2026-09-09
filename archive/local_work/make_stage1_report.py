import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

free = pd.read_csv(OUT / "ravdess_branch_summary.csv")
choice = pd.read_csv(OUT / "ravdess_choice_summary.csv")

labels = {
    "full": "Full",
    "no_continuous": "No continuous",
    "no_discrete": "No discrete",
    "rolled_continuous": "Rolled continuous",
}

colors = ["#4169E1", "#E67E22", "#2E8B57", "#8E44AD"]
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="460" viewBox="0 0 1100 460">',
       '<rect width="100%" height="100%" fill="white"/>',
       '<style>text{font-family:Arial,sans-serif;fill:#222}.small{font-size:12px}.title{font-size:20px;font-weight:bold}.sub{font-size:15px;font-weight:bold}</style>',
       '<text x="550" y="28" text-anchor="middle" class="title">RAVDESS controlled subset: branch intervention results (n=20)</text>']
for panel, (df, title) in enumerate(zip([free, choice], ["Free-label protocol", "Explicit-choice protocol"])):
    left = 55 + panel * 545
    top, height = 70, 290
    svg.append(f'<text x="{left+245}" y="52" text-anchor="middle" class="sub">{title}</text>')
    for tick in range(6):
        y = top + height - tick * height / 5
        svg.append(f'<line x1="{left}" y1="{y}" x2="{left+480}" y2="{y}" stroke="#ddd"/>')
        svg.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" class="small">{tick*20}%</text>')
    for i, (_, row) in enumerate(df.iterrows()):
        center = left + 65 + i * 115
        for j, (value, opacity) in enumerate([(row.accuracy, 1), (row.coverage, .35)]):
            h = value * height
            x = center - 26 + j * 28
            svg.append(f'<rect x="{x}" y="{top+height-h}" width="25" height="{h}" fill="{colors[i]}" opacity="{opacity}"/>')
        svg.append(f'<text x="{center}" y="{top+height+22}" text-anchor="middle" class="small">{labels[row["mode"]]}</text>')
svg += ['<rect x="420" y="420" width="18" height="12" fill="#4169E1"/><text x="445" y="431" class="small">Accuracy</text>',
        '<rect x="535" y="420" width="18" height="12" fill="#4169E1" opacity=".35"/><text x="560" y="431" class="small">Coverage</text>', '</svg>']
(OUT / "stage1_branch_results.svg").write_text("".join(svg), encoding="utf-8")

def table(df):
    rows = []
    for _, r in df.iterrows():
        flip = "—" if pd.isna(r.get("flip_rate_vs_full")) else f'{r["flip_rate_vs_full"]:.0%}'
        rows.append(f'| {labels[r["mode"]]} | {r["accuracy"]:.0%} | {r["macro_f1"]:.3f} | {r["coverage"]:.0%} | {flip} |')
    return "\n".join(rows)

report = f"""# Kimi-Audio 双分支干预：阶段一实验报告

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
{table(free)}

## 结果二：显式四选一协议

| 条件 | Accuracy | Macro-F1 | 合法选项覆盖率 | 相对 Full 翻转率 |
|---|---:|---:|---:|---:|
{table(choice)}

![阶段一结果](stage1_branch_results.svg)

## 可交差的观察

1. 干预是有效的。自由标签协议中移除连续分支后，相对 Full 的输出翻转率为 {free.loc[free['mode']=='no_continuous','flip_rate_vs_full'].iloc[0]:.0%}；四选一协议中为 {choice.loc[choice['mode']=='no_continuous','flip_rate_vs_full'].iloc[0]:.0%}。这不是“代码没有生效”的空消融。
2. 连续分支影响很大，但方向并不等同于简单的准确率提升。No continuous 在两套提示下出现明显的类别塌缩/偏置，且两套协议的 accuracy 方向不同，说明生成式分类的提示敏感性很强。
3. 仅保留连续分支（No discrete）时，自由标签覆盖率只有 {free.loc[free['mode']=='no_discrete','coverage'].iloc[0]:.0%}，显示离散 token 对遵循输出格式和稳定生成可能很重要。
4. 打乱连续特征的时间对应仍造成 {choice.loc[choice['mode']=='rolled_continuous','flip_rate_vs_full'].iloc[0]:.0%} 的输出翻转，但小样本 accuracy 偶然高于 Full；这更适合解释为“时序对齐会改变决策”，不能解释为“打乱会改善模型”。

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
"""
(OUT / "stage1_report.md").write_text(report, encoding="utf-8")
print(OUT / "stage1_report.md")
