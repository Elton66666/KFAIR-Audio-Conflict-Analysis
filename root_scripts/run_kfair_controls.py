import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from kimia_infer.api.kimia import KimiAudio

ROOT=Path("/root/autodl-tmp/kfair");OUT=ROOT/"outputs";PRED=OUT/"kfair_controls.jsonl"
LABELS=["neutral","happy","sad","angry"];CHOICES={"neutral":"A","happy":"B","sad":"C","angry":"D"}

def main():
    a=pd.read_csv(OUT/"ravdess_expanded_manifest.csv");b=pd.read_csv(OUT/"ravdess_statement2_manifest.csv")
    a["statement"],b["statement"]=1,2;df=pd.concat([a,b],ignore_index=True).sort_values(["statement","actor","label"]).reset_index(drop=True)
    model=KimiAudio(model_path=str(ROOT/"models/Kimi-Audio-7B-Instruct"),load_detokenizer=False)
    fusion=[m for m in model.alm.modules() if hasattr(m,"vq_adaptor")][0];fusion.kfair_branch_mode="full";pm=model.prompt_manager
    prompt="Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
    ids={k:pm.text_tokenizer.encode(v,bos=False,eos=False)[0] for k,v in CHOICES.items()}
    cache={}
    for i,r in enumerate(df.itertuples(),1):
        h=pm.get_prompt([{"role":"user","message_type":"text","content":prompt},{"role":"user","message_type":"audio","content":r.audio_path}],output_type="text")
        au,tx,ma,_,_=h.to_tensor();cache[r.filename]=(au.cpu(),tx.cpu(),ma.cpu(),h.continuous_feature[0].cpu())
        if i%24==0:print(f"CACHE {i}/192",flush=True)
    done=set()
    if PRED.exists():done={(x["filename"],x["control"]) for x in map(json.loads,PRED.read_text(encoding="utf-8").splitlines())}
    rng=np.random.default_rng(20260826)
    for i,r in enumerate(df.itertuples(),1):
        au,tx,ma,feat=cache[r.filename]
        same=df[(df.statement==r.statement)&(df.label==r.label)&(df.actor!=r.actor)].sort_values("actor")
        same_row=same.iloc[0] if r.actor!=int(same.iloc[0].actor) else same.iloc[1]
        random_pool=df[(df.statement==r.statement)&(df.actor!=r.actor)]
        random_row=random_pool.iloc[int(rng.integers(len(random_pool)))]
        specs={"original":feat,"zero":torch.zeros_like(feat),
               "gaussian":torch.randn_like(feat)*feat.float().std().to(feat.dtype)+feat.float().mean().to(feat.dtype),
               "same_emotion_other_speaker":cache[same_row.filename][3],"random_other_speaker":cache[random_row.filename][3]}
        for name,cf in specs.items():
            if (r.filename,name) in done:continue
            if cf.shape[1]!=feat.shape[1]:cf=F.interpolate(cf.transpose(1,2).float(),size=feat.shape[1],mode="linear",align_corners=False).transpose(1,2).to(feat.dtype)
            pos=torch.arange(au.shape[1],device="cuda").unsqueeze(0).long()
            with torch.inference_mode():_,logits,_=model.alm.forward(input_ids=au.cuda(),text_input_ids=tx.cuda(),whisper_input_feature=[cf.cuda()],is_continuous_mask=ma.cuda(),position_ids=pos,past_key_values=None,return_dict=False)
            lp=torch.log_softmax(logits[0,-1].float(),dim=-1);scores={k:float(lp[v].cpu()) for k,v in ids.items()};pred=max(scores,key=scores.get)
            row={"filename":r.filename,"actor":r.actor,"statement":r.statement,"label":r.label,"control":name,"prediction":pred,
                 "correct_logprob":scores[r.label],"scores":scores,"source_file":r.filename if name in ["original","zero","gaussian"] else (same_row.filename if name=="same_emotion_other_speaker" else random_row.filename)}
            with PRED.open("a",encoding="utf-8") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush()
        if i%24==0:print(f"SAMPLES {i}/192",flush=True)
    p=pd.DataFrame(json.loads(x) for x in PRED.read_text(encoding="utf-8").splitlines());base=p[p.control=="original"].set_index("filename")
    rows=[]
    for name,g in p.groupby("control"):
        q=g.set_index("filename");rows.append({"control":name,"n":len(q),"accuracy":(q.prediction==q.label).mean(),
          "mean_correct_logprob":q.correct_logprob.mean(),"flip_vs_original":0 if name=="original" else (q.prediction!=base.prediction).mean(),
          "delta_correct_logprob":0 if name=="original" else (q.correct_logprob-base.correct_logprob).mean()})
    s=pd.DataFrame(rows);s.to_csv(OUT/"kfair_controls_summary.csv",index=False);print(s.to_string(index=False),flush=True)
if __name__=="__main__":main()
