import json
from pathlib import Path
import pandas as pd
from scipy.stats import binomtest

ROOT=Path('/root/autodl-tmp/kfair'); OUT=ROOT/'outputs'
def read(name): return pd.DataFrame(json.loads(x) for x in (OUT/name).read_text(encoding='utf-8').splitlines())
def mcnemar(a,b):
 x=(~a & b).sum(); y=(a & ~b).sum(); n=int(x+y); p=1.0 if n==0 else binomtest(min(x,y),n,.5,alternative='two-sided').pvalue
 return int(x),int(y),float(p)
rows=[]
t=read('stage3_tess_clean_predictions.jsonl'); key=['filename']
base=t[t.method=='baseline'].set_index(key).correct.astype(bool)
for method in t.method.unique():
 if method=='baseline': continue
 other=t[t.method==method].set_index(key).correct.astype(bool).reindex(base.index); gain,loss,p=mcnemar(base,other)
 rows.append({'dataset':'TESS-clean','condition':'clean','comparison':f'{method} vs baseline','baseline_or_first_accuracy':base.mean(),'method_or_second_accuracy':other.mean(),'first_wrong_second_right':gain,'first_right_second_wrong':loss,'mcnemar_exact_p':p})
n=read('stage3_noise_robustness_predictions.jsonl'); key=['actor','statement','pair','cell']
for snr in ['clean',20,10]:
 sub=n[n.snr_db.astype(str)==str(snr)]; base=sub[sub.method=='baseline'].set_index(key).correct.astype(bool)
 for method in sub.method.unique():
  if method=='baseline': continue
  other=sub[sub.method==method].set_index(key).correct.astype(bool).reindex(base.index); gain,loss,p=mcnemar(base,other)
  rows.append({'dataset':'RAVDESS-factorial','condition':str(snr),'comparison':f'{method} vs baseline','baseline_or_first_accuracy':base.mean(),'method_or_second_accuracy':other.mean(),'first_wrong_second_right':gain,'first_right_second_wrong':loss,'mcnemar_exact_p':p})
for method in n.method.unique():
 clean=n[(n.method==method)&(n.snr_db.astype(str)=='clean')].set_index(key).correct.astype(bool)
 for snr in [20,10]:
  noisy=n[(n.method==method)&(n.snr_db.astype(str)==str(snr))].set_index(key).correct.astype(bool).reindex(clean.index); gain,loss,p=mcnemar(clean,noisy)
  rows.append({'dataset':'RAVDESS-factorial','condition':f'clean vs {snr}dB','comparison':method,'baseline_or_first_accuracy':clean.mean(),'method_or_second_accuracy':noisy.mean(),'first_wrong_second_right':gain,'first_right_second_wrong':loss,'mcnemar_exact_p':p})
pd.DataFrame(rows).to_csv(OUT/'stage3_paired_statistics.csv',index=False); print(pd.DataFrame(rows).to_string(index=False))
