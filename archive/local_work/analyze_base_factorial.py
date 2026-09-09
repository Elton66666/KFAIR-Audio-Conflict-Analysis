import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"
pred=pd.DataFrame(json.loads(x) for x in (OUT/"kfair_base_factorial_predictions.jsonl").read_text().splitlines())
eff=pd.read_csv(OUT/"kfair_base_factorial_effects.csv")
actors=sorted(eff.actor.unique());rng=np.random.default_rng(20260826)
def boot(df,col,n=10000):
    x=df.groupby("actor")[col].mean().reindex(actors).to_numpy();ix=rng.integers(0,len(x),size=(n,len(x)));z=x[ix].mean(1)
    return x.mean(),np.quantile(z,.025),np.quantile(z,.975)
rows=[]
for c in ["ME_C","ME_D","interaction"]:
    m,l,h=boot(eff,c);rows.append(dict(metric=c,mean=m,ci_low=l,ci_high=h))
cross=pred[pred.cell.isin(["AB","BA"])].copy()
cross["acoustic_follow"]=np.where(cross.cell.eq("AB"),cross.prediction.eq(cross.label_b),cross.prediction.eq(cross.label_a))
cross["discrete_follow"]=np.where(cross.cell.eq("AB"),cross.prediction.eq(cross.label_a),cross.prediction.eq(cross.label_b))
cross["other"]=~cross.acoustic_follow&~cross.discrete_follow
for c in ["acoustic_follow","discrete_follow","other"]:
    m,l,h=boot(cross,c);rows.append(dict(metric=c,mean=m,ci_low=l,ci_high=h))
matched=pred[pred.cell.isin(["AA","BB"])].copy()
matched["correct"]=np.where(matched.cell.eq("AA"),matched.prediction.eq(matched.label_a),matched.prediction.eq(matched.label_b))
m,l,h=boot(matched,"correct");rows.append(dict(metric="matched_accuracy",mean=m,ci_low=l,ci_high=h))
pd.DataFrame(rows).to_csv(OUT/"kfair_base_factorial_statistics.csv",index=False)
print(pd.DataFrame(rows).to_string(index=False))
