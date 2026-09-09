from pathlib import Path
import os
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"
PREFIX=os.environ.get("KFAIR_PREFIX","kfair")
df=pd.read_csv(OUT/f"{PREFIX}_layer_trace_effects.csv")
rng=np.random.default_rng(20260826); actors=sorted(df.actor.unique())
rows=[]
for layer,g in df.groupby("layer"):
    for metric in ["ME_C","ME_D","interaction"]:
        x=g.groupby("actor")[metric].mean().reindex(actors).to_numpy()
        ix=rng.integers(0,len(x),size=(10000,len(x)))
        z=x[ix].mean(1)
        rows.append(dict(layer=layer,metric=metric,mean=x.mean(),ci_low=np.quantile(z,.025),ci_high=np.quantile(z,.975)))
stats=pd.DataFrame(rows);stats.to_csv(OUT/f"{PREFIX}_layer_trace_statistics.csv",index=False)
print(stats[stats.metric.eq("ME_C")].to_string(index=False))
