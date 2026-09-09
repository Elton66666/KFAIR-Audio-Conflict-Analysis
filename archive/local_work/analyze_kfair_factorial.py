import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"
pred=pd.DataFrame(json.loads(x) for x in (OUT/"kfair_factorial_predictions.jsonl").read_text(encoding="utf-8").splitlines())
eff=pd.read_csv(OUT/"kfair_factorial_effects.csv")
actors=sorted(eff.actor.unique()); rng=np.random.default_rng(20260826)

def boot_metric(df,col,n=10000):
    av=df.groupby("actor")[col].mean().reindex(actors).to_numpy()
    ix=rng.integers(0,len(actors),size=(n,len(actors)))
    vals=av[ix].mean(1)
    return float(av.mean()),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

summary=[]
for col in ["ME_C","ME_D","interaction"]:
    mean,lo,hi=boot_metric(eff,col); summary.append({"metric":col,"mean":mean,"ci_low":lo,"ci_high":hi})

cross=pred[pred.cell.isin(["AB","BA"])].copy()
cross["acoustic_follow"]=np.where(cross.cell.eq("AB"),cross.prediction.eq(cross.label_b),cross.prediction.eq(cross.label_a))
cross["discrete_follow"]=np.where(cross.cell.eq("AB"),cross.prediction.eq(cross.label_a),cross.prediction.eq(cross.label_b))
cross["other"]=~cross.acoustic_follow & ~cross.discrete_follow
for col in ["acoustic_follow","discrete_follow","other"]:
    mean,lo,hi=boot_metric(cross,col);summary.append({"metric":col,"mean":mean,"ci_low":lo,"ci_high":hi})

matched=pred[pred.cell.isin(["AA","BB"])].copy()
matched["correct"]=np.where(matched.cell.eq("AA"),matched.prediction.eq(matched.label_a),matched.prediction.eq(matched.label_b))
mean,lo,hi=boot_metric(matched,"correct");summary.append({"metric":"matched_accuracy","mean":mean,"ci_low":lo,"ci_high":hi})
stats=pd.DataFrame(summary);stats.to_csv(OUT/"kfair_factorial_statistics.csv",index=False)

pair_rows=[]
for (a,b),g in eff.groupby(["label_a","label_b"]):
    c=pred[(pred.label_a==a)&(pred.label_b==b)&pred.cell.isin(["AB","BA"])].copy()
    c["af"]=np.where(c.cell.eq("AB"),c.prediction.eq(c.label_b),c.prediction.eq(c.label_a))
    c["df"]=np.where(c.cell.eq("AB"),c.prediction.eq(c.label_a),c.prediction.eq(c.label_b))
    pair_rows.append({"emotion_pair":f"{a}-{b}","n_pairs":len(g),"ME_C":g.ME_C.mean(),"ME_D":g.ME_D.mean(),
                      "interaction":g.interaction.mean(),"acoustic_follow":c.af.mean(),"discrete_follow":c.df.mean()})
pair_stats=pd.DataFrame(pair_rows);pair_stats.to_csv(OUT/"kfair_by_emotion_pair.csv",index=False)

# Compact dependency-free SVG.
metrics=stats[stats.metric.isin(["ME_C","ME_D","interaction"])]
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="470"><rect width="100%" height="100%" fill="white"/>',
'<style>text{font-family:Arial,sans-serif;fill:#222}.t{font-size:20px;font-weight:bold}.s{font-size:13px}</style>',
'<text x="525" y="28" text-anchor="middle" class="t">K-FAIR native-stream factorial intervention (288 real parallel pairs)</text>']
left,top,w,h=75,70,420,310
for i in range(7):
    val=-2+i; y=top+h-(val+2)/6*h
    svg += [f'<line x1="{left}" y1="{y}" x2="{left+w}" y2="{y}" stroke="#ddd"/>',f'<text x="{left-8}" y="{y+4}" text-anchor="end" class="s">{val}</text>']
colors=['#4169E1','#E67E22','#8E44AD']
for i,(_,r) in enumerate(metrics.iterrows()):
    x=left+90+i*125; zero=top+h-2/6*h; y=top+h-(r['mean']+2)/6*h
    svg.append(f'<rect x="{x-30}" y="{min(y,zero)}" width="60" height="{abs(zero-y)}" fill="{colors[i]}"/>')
    yl=top+h-(r.ci_low+2)/6*h; yh=top+h-(r.ci_high+2)/6*h
    svg += [f'<line x1="{x}" y1="{yl}" x2="{x}" y2="{yh}" stroke="#111" stroke-width="2"/>',f'<text x="{x}" y="{top+h+24}" text-anchor="middle" class="s">{r.metric}</text>']
svg += ['<text x="750" y="65" text-anchor="middle" class="t">Conflict decisions</text>']
for i,name in enumerate(["acoustic_follow","discrete_follow","other"]):
    r=stats[stats.metric==name].iloc[0]; x=590+i*150; bh=r['mean']*260
    svg += [f'<rect x="{x}" y="{370-bh}" width="80" height="{bh}" fill="{colors[i]}"/>',f'<text x="{x+40}" y="395" text-anchor="middle" class="s">{name.replace("_"," ")}</text>',f'<text x="{x+40}" y="{360-bh}" text-anchor="middle" class="s">{100*r["mean"]:.1f}%</text>']
svg.append('</svg>');(OUT/"kfair_factorial_results.svg").write_text(''.join(svg),encoding="utf-8")

def stat(name): return stats[stats.metric==name].iloc[0]
mc,md,inter=stat("ME_C"),stat("ME_D"),stat("interaction")
af,df_,other,acc=stat("acoustic_follow"),stat("discrete_follow"),stat("other"),stat("matched_accuracy")
report=f"""# K-FAIR 核心析因实验：会议更新

## 本轮完成内容

在 Kimi-Audio-7B-Instruct 原生融合点上，对 RAVDESS 两条文本、24 位说话人、四种情绪构造真实同人同句异情绪对。每个 actor/statement 有 6 种情绪对，共 288 对；每对运行 AA、AB、BA、BB 四个双流组合，共 1152 个析因单元。

情绪评分不再使用自由生成，而是在 A=neutral、B=happy、C=sad、D=angry 四个候选首 token 上读取 log-probability。跨样本连续特征按离散流目标长度进行线性时间重采样。

## 核心结果

| 指标 | 均值 | actor-bootstrap 95% CI |
|---|---:|---:|
| 连续流主效应 ME_C | {mc['mean']:.3f} | [{mc.ci_low:.3f}, {mc.ci_high:.3f}] |
| 离散流主效应 ME_D | {md['mean']:.3f} | [{md.ci_low:.3f}, {md.ci_high:.3f}] |
| 交互效应 I | {inter['mean']:.3f} | [{inter.ci_low:.3f}, {inter.ci_high:.3f}] |
| 声学丰富流跟随率 | {100*af['mean']:.1f}% | [{100*af.ci_low:.1f}%, {100*af.ci_high:.1f}%] |
| 离散流跟随率 | {100*df_['mean']:.1f}% | [{100*df_.ci_low:.1f}%, {100*df_.ci_high:.1f}%] |
| 预测第三类情绪 | {100*other['mean']:.1f}% | [{100*other.ci_low:.1f}%, {100*other.ci_high:.1f}%] |
| 匹配单元准确率 | {100*acc['mean']:.1f}% | [{100*acc.ci_low:.1f}%, {100*acc.ci_high:.1f}%] |

![析因结果](kfair_factorial_results.svg)

## 当前解释

连续流的平均主效应远大于离散流，说明在真实同文本情绪对中，替换连续声学丰富流会系统性推动情绪 logit contrast 向连续流提供者的情绪移动。冲突单元中，模型跟随连续流的比例也高于跟随离散流。

但这仍是 RAVDESS 上的 Instruct 模型结果。RAVDESS 属于公开披露的 SER-SFT 数据，不能称为 zero-shot；时间重采样也可能引入分布偏移。下一步必须用随机/同情绪换流控制、未披露语料和 Base 模型验证。

## 与此前实验的关系

此前 Full/No continuous/No discrete 是零化消融；本轮首次完成老师方案要求的 DA/CA、DA/CB、DB/CA、DB/CB 四组合，并显式计算 ME_C、ME_D 和交互效应。因此这才是 K-FAIR 核心方法的第一轮结果。
"""
(OUT/"kfair_factorial_report.md").write_text(report,encoding="utf-8")
print(stats.to_string(index=False));print(pair_stats.to_string(index=False))
