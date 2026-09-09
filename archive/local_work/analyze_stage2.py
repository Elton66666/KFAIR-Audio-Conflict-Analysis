import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MODES = ["full", "no_continuous", "no_discrete", "rolled_continuous"]
NAMES = {"full": "Full", "no_continuous": "No continuous", "no_discrete": "No discrete", "rolled_continuous": "Rolled continuous"}
COLORS = {"full": "#4169E1", "no_continuous": "#E67E22", "no_discrete": "#2E8B57", "rolled_continuous": "#8E44AD"}

def read_jsonl(name, statement):
    df = pd.DataFrame(json.loads(x) for x in (OUT / name).read_text(encoding="utf-8").splitlines())
    df["statement"] = statement
    return df
emo = pd.concat([read_jsonl("ravdess_expanded_predictions.jsonl", 1), read_jsonl("ravdess_statement2_emotion_predictions.jsonl", 2)], ignore_index=True)
asr = pd.concat([read_jsonl("ravdess_asr_predictions.jsonl", 1), read_jsonl("ravdess_statement2_asr_predictions.jsonl", 2)], ignore_index=True)
emo["correct"] = emo.prediction == emo.label
actors = sorted(emo.actor.unique())
rng = np.random.default_rng(20260825)


def exact_mcnemar(a, b):
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    n = n10 + n01
    if not n:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return n10, n01, p


rows = []
full = emo[emo["mode"] == "full"].sort_values("filename")
for mode in MODES:
    p = emo[emo["mode"] == mode].sort_values("filename")
    actor_vals = p.groupby("actor").correct.mean().reindex(actors).to_numpy()
    full_actor_vals = full.groupby("actor").correct.mean().reindex(actors).to_numpy()
    sampled = rng.integers(0, len(actors), size=(10000, len(actors)))
    vals = actor_vals[sampled].mean(axis=1)
    diffs = (actor_vals - full_actor_vals)[sampled].mean(axis=1)
    n10, n01, mcp = exact_mcnemar(full.correct.reset_index(drop=True), p.correct.reset_index(drop=True))
    rows.append({"mode": mode, "n": len(p), "accuracy": p.correct.mean(),
                 "accuracy_ci_low": np.quantile(vals, .025), "accuracy_ci_high": np.quantile(vals, .975),
                 "delta_vs_full": p.correct.mean() - full.correct.mean(),
                 "delta_ci_low": np.quantile(diffs, .025), "delta_ci_high": np.quantile(diffs, .975),
                 "coverage": p.prediction.isin(["neutral", "happy", "sad", "angry"]).mean(),
                 "flip_rate": np.nan if mode == "full" else (p.prediction.reset_index(drop=True) != full.prediction.reset_index(drop=True)).mean(),
                 "mcnemar_full_better": n10, "mcnemar_mode_better": n01, "mcnemar_p": mcp})
stats = pd.DataFrame(rows)
stats.to_csv(OUT / "final_emotion_statistics.csv", index=False)

per_emotion = emo.groupby(["mode", "label"]).agg(n=("correct", "size"), accuracy=("correct", "mean"), coverage=("prediction", lambda x: x.isin(["neutral", "happy", "sad", "angry"]).mean())).reset_index()
per_emotion.to_csv(OUT / "final_per_emotion.csv", index=False)

asr_rows = []
for mode in MODES:
    p = asr[asr["mode"] == mode]
    actor_vals = p.groupby("actor").apply(lambda q: q.edits.sum()/q.ref_words.sum(), include_groups=False).reindex(actors).to_numpy()
    sampled = rng.integers(0, len(actors), size=(10000, len(actors)))
    vals = actor_vals[sampled].mean(axis=1)
    asr_rows.append({"mode": mode, "n": len(p), "wer": p.edits.sum()/p.ref_words.sum(),
                     "wer_ci_low": np.quantile(vals, .025), "wer_ci_high": np.quantile(vals, .975),
                     "exact_match": p.exact_match.mean()})
asr_stats = pd.DataFrame(asr_rows)
asr_stats.to_csv(OUT / "final_asr_statistics.csv", index=False)

# Dependency-free SVG: emotion accuracy with cluster-bootstrap CI and ASR WER.
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="470" viewBox="0 0 1100 470"><rect width="100%" height="100%" fill="white"/>',
       '<style>text{font-family:Arial,sans-serif;fill:#222}.t{font-size:20px;font-weight:bold}.s{font-size:15px;font-weight:bold}.x{font-size:12px}</style>',
       '<text x="550" y="28" text-anchor="middle" class="t">RAVDESS branch interventions (24 actors, 2 statements, n=192)</text>']
for panel, (df, metric, low, high, title, ymax) in enumerate([(stats,"accuracy","accuracy_ci_low","accuracy_ci_high","Emotion accuracy (95% actor-bootstrap CI)",1.0),(asr_stats,"wer","wer_ci_low","wer_ci_high","ASR word error rate (95% actor-bootstrap CI)",1.5)]):
    left, top, width, height = 65 + panel*545, 70, 460, 300
    svg.append(f'<text x="{left+230}" y="52" text-anchor="middle" class="s">{title}</text>')
    for i in range(6):
        y=top+height-i*height/5; val=i*ymax/5
        svg += [f'<line x1="{left}" y1="{y}" x2="{left+width}" y2="{y}" stroke="#ddd"/>',f'<text x="{left-8}" y="{y+4}" text-anchor="end" class="x">{val:.1f}</text>']
    for i,row in df.iterrows():
        x=left+60+i*110; val=row[metric]; y=top+height-val/ymax*height
        svg.append(f'<rect x="{x-25}" y="{y}" width="50" height="{top+height-y}" fill="{COLORS[row["mode"]]}"/>')
        yl=top+height-row[low]/ymax*height; yh=top+height-row[high]/ymax*height
        svg += [f'<line x1="{x}" y1="{yl}" x2="{x}" y2="{yh}" stroke="#111" stroke-width="2"/>',f'<line x1="{x-8}" y1="{yl}" x2="{x+8}" y2="{yl}" stroke="#111"/><line x1="{x-8}" y1="{yh}" x2="{x+8}" y2="{yh}" stroke="#111"/>',f'<text x="{x}" y="{top+height+22}" text-anchor="middle" class="x">{NAMES[row["mode"]]}</text>']
svg.append('</svg>')
(OUT / "final_main_results.svg").write_text("".join(svg), encoding="utf-8")


def pct(x): return f"{100*x:.1f}%"
def emo_table():
    out=[]
    for _,r in stats.iterrows():
        delta='—' if r['mode']=='full' else f"{100*r.delta_vs_full:+.1f} pp [{100*r.delta_ci_low:+.1f}, {100*r.delta_ci_high:+.1f}]"
        p='—' if r['mode']=='full' else f"{r.mcnemar_p:.4f}"
        flip='—' if r['mode']=='full' else pct(r.flip_rate)
        out.append(f"| {NAMES[r['mode']]} | {pct(r.accuracy)} [{pct(r.accuracy_ci_low)}, {pct(r.accuracy_ci_high)}] | {pct(r.coverage)} | {flip} | {delta} | {p} |")
    return '\n'.join(out)
def asr_table():
    return '\n'.join(f"| {NAMES[r['mode']]} | {pct(r.wer)} [{pct(r.wer_ci_low)}, {pct(r.wer_ci_high)}] | {pct(r.exact_match)} |" for _,r in asr_stats.iterrows())

report=f"""# Kimi-Audio 双分支干预实验：会议版最终报告

## 摘要

我们在 Kimi-Audio-7B-Instruct 上直接干预连续声学表示与离散音频 token 的融合，并在 24 位说话人、两条固定文本、192 条受控 RAVDESS 语音上同时评估情绪分类与转写。最清晰的结果来自 ASR 对照：Full WER 为 {pct(asr_stats.loc[asr_stats['mode']=='full','wer'].iloc[0])}，No discrete 为 {pct(asr_stats.loc[asr_stats['mode']=='no_discrete','wer'].iloc[0])}，而 No continuous 达到 {pct(asr_stats.loc[asr_stats['mode']=='no_continuous','wer'].iloc[0])}，连续特征时序错位达到 {pct(asr_stats.loc[asr_stats['mode']=='rolled_continuous','wer'].iloc[0])}。两条句子独立得到相同结构，说明连续表示及其时序对齐对恢复语言内容至关重要；仅连续表示已足以完成近乎准确的短句转写。情绪分类也随分支干预发生明显变化，但受生成式标签覆盖率及提示敏感性影响，应作为探索性结果解释。

## 1. 实验设计

- 模型与算力：Kimi-Audio-7B-Instruct；NVIDIA A800 80GB；推理峰值约 21–24 GiB。
- 数据：RAVDESS speech 1440 条中的受控子集。选取全部 24 位 actor、两条 statement，固定 repetition=01、normal intensity。
- 情绪：neutral、happy、sad、angry，每类 48 条，共 192 条。
- 文本：“Kids are talking by the door.” 与 “Dogs are sitting by the door.”；分别分析后再合并。
- 四种条件：Full；No continuous；No discrete；Rolled continuous（循环错位连续特征，破坏时间对应但保留其分布）。
- 每项任务共 192×4=768 次推理；情绪与 ASR 合计 1536 次正式推理。
- 置信区间：以 actor 为聚类单位进行 10,000 次 bootstrap，避免把同一说话人的四种情绪当成完全独立样本。

## 2. 情绪分类结果

| 条件 | Accuracy [95% CI] | 覆盖率 | 输出翻转率 | 相对 Full 差值 [95% CI] | McNemar p |
|---|---:|---:|---:|---:|---:|
{emo_table()}

重要现象：No continuous 的预测大量塌缩到 sad，Full 与 No discrete 都存在输出题目外标签的问题；因此 accuracy 必须与覆盖率同时解读。Rolled continuous 的 accuracy 高于 Full，但 ASR 严重下降，不能将其解释为时序错位“改善模型”，更可能是生成偏置与该小型受控任务的偶然组合。

## 3. ASR 内容保真对照

| 条件 | WER [95% CI] | 完全匹配率 |
|---|---:|---:|
{asr_table()}

![会议版主结果](final_main_results.svg)

ASR 给出了比情绪 accuracy 更稳定的结构性证据：

1. No discrete 与 Full 都接近零 WER，说明在这两条固定短句转写任务上连续表示足以保存语言内容。
2. No continuous 完全匹配率为 0%，并出现截断、替换和幻觉；离散 token 单独不足以可靠恢复内容。该结构在两条句子上重复出现。
3. Rolled continuous 几乎全部失败，说明不仅需要连续特征本身，还需要它与 token 位置的正确时间对齐。

## 4. 当前可支持与不可支持的结论

可以支持：连续表示及其时序对齐是 Kimi-Audio 在该任务上恢复语言内容的关键；分支干预会显著改变情绪输出；实验干预确实生效。

暂不能支持：离散分支专门编码情绪；某种干预提升了情绪识别；结果能直接推广到开放词汇、长语音或其他模型。情绪任务仍受到自由生成格式、标签覆盖率和单句数据设计的限制。

## 5. 下一阶段建议

优先采用候选标签条件似然或受约束解码，消除题外标签；加入第二条 RAVDESS statement 复现；再用不同语料测试 ASR，确认 No discrete 的 0% WER 是否只适用于极短、固定文本。若要研究“副语言与内容解耦”，还应加入 speaker identification、强度识别或音高/语速等连续声学指标。

## 6. 复现与附件

- 两个 `*_manifest.csv`：两条句子各 96 条受控样本。
- 两组 `*_emotion_predictions.jsonl`：合计 768 条情绪原始输出。
- 两组 `*_asr_predictions.jsonl`：合计 768 条 ASR 原始输出。
- `final_emotion_statistics.csv`、`final_asr_statistics.csv`：合并置信区间和配对检验。
- `final_per_emotion.csv`：合并后的分情绪结果。
"""
(OUT / "final_meeting_report.md").write_text(report, encoding="utf-8")
print(stats.to_string(index=False)); print(asr_stats.to_string(index=False))
