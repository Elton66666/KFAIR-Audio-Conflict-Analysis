import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs"
raw=[json.loads(x) for x in (OUT/"kfair_activation_patching.jsonl").read_text(encoding="utf-8").splitlines()]
df=pd.DataFrame(raw).drop_duplicates(["pair_id","target_cell","condition","layer","scope"],keep="last")
assert len(df)==4800,len(df)
df["acoustic_follow"] = df.prediction.eq(df.acoustic_label)
df["discrete_follow"] = df.prediction.eq(df.discrete_label)
df["other"] = ~df.acoustic_follow & ~df.discrete_follow
actors=sorted(df.actor.unique());rng=np.random.default_rng(20260903)
def boot(g,col,n=10000):
    x=g.groupby("actor")[col].mean().reindex(actors).to_numpy();ix=rng.integers(0,len(x),size=(n,len(x)));z=x[ix].mean(1)
    return float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))
rows=[]
for keys,g in df.groupby(["condition","layer","scope"],dropna=False):
    for metric in ["acoustic_margin","rescue","acoustic_follow","discrete_follow","other"]:
        m,l,h=boot(g,metric);rows.append(dict(condition=keys[0],layer=keys[1],scope=keys[2],metric=metric,mean=m,ci_low=l,ci_high=h,n=len(g)))
stats=pd.DataFrame(rows);stats.to_csv(OUT/"kfair_activation_patching_statistics.csv",index=False)
main=stats[(stats.metric.isin(["rescue","acoustic_follow"])) & (stats.condition.isin(["acoustic","discrete"]))]
print(main.to_string(index=False))
