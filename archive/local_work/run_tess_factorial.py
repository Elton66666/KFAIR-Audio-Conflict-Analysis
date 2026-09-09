import glob,itertools,json,os,time
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from kimia_infer.api.kimia import KimiAudio

ROOT=Path("/root/autodl-tmp/kfair");OUT=ROOT/"outputs";AUDIO=ROOT/"data/tess/audio"
PREFIX=os.environ.get("KFAIR_PREFIX","tess")
MODEL_PATH=os.environ.get("KFAIR_MODEL_PATH",str(ROOT/"models/Kimi-Audio-7B-Instruct"))
PRED=OUT/f"{PREFIX}_factorial_predictions.jsonl"
MAP={"neutral":"neutral","joy":"happy","sadness":"sad","anger":"angry"};LABELS=["neutral","happy","sad","angry"];CHOICES={"neutral":"A","happy":"B","sad":"C","angry":"D"}

def prepare():
    AUDIO.mkdir(parents=True,exist_ok=True);raw=pd.concat([pd.read_parquet(x) for x in glob.glob(str(ROOT/"data/tess/data/*.parquet"))],ignore_index=True)
    rows=[]
    for r in raw.itertuples():
        if r.label not in MAP:continue
        obj=r.audio; fn=obj["path"]; parts=fn.removesuffix(".wav").split("_");speaker="OAF" if parts[0]=="OA" else parts[0];word="_".join(parts[1:-1]);out=AUDIO/fn
        if not out.exists():out.write_bytes(obj["bytes"])
        rows.append({"filename":fn,"audio_path":str(out),"speaker":speaker,"word":word,"label":MAP[r.label]})
    df=pd.DataFrame(rows).sort_values(["speaker","word","label"]);assert len(df)==1600
    df.to_csv(OUT/f"{PREFIX}_factorial_manifest.csv",index=False)
    pairs=[]
    for (sp,w),g in df.groupby(["speaker","word"]):
        d={x.label:x for x in g.itertuples()};assert set(d)==set(LABELS)
        for a,b in itertools.combinations(LABELS,2):
            pairs.append({"pair_id":f"{sp}_{w}_{a}_{b}","speaker":sp,"word":w,"label_a":a,"label_b":b,"file_a":d[a].filename,"file_b":d[b].filename})
    p=pd.DataFrame(pairs);assert len(p)==2400;p.to_csv(OUT/f"{PREFIX}_pair_manifest.csv",index=False);return df,p

def main():
    df,pairs=prepare();model=KimiAudio(model_path=MODEL_PATH,load_detokenizer=False);fusion=[m for m in model.alm.modules() if hasattr(m,"vq_adaptor")][0];fusion.kfair_branch_mode="full";pm=model.prompt_manager
    prompt="Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
    ids={k:pm.text_tokenizer.encode(v,bos=False,eos=False)[0] for k,v in CHOICES.items()};cache={}
    for i,r in enumerate(df.itertuples(),1):
        h=pm.get_prompt([{"role":"user","message_type":"text","content":prompt},{"role":"user","message_type":"audio","content":r.audio_path}],output_type="text");au,tx,ma,_,_=h.to_tensor();cache[r.filename]=(au.cpu(),tx.cpu(),ma.cpu(),h.continuous_feature[0].cpu(),r.label)
        if i%100==0:print(f"CACHE {i}/1600",flush=True)
    done=set()
    if PRED.exists():done={(x["pair_id"],x["cell"]) for x in map(json.loads,PRED.read_text(encoding="utf-8").splitlines())}
    for i,r in enumerate(pairs.itertuples(),1):
        for cell,dfi,cfi in [("AA",r.file_a,r.file_a),("AB",r.file_a,r.file_b),("BA",r.file_b,r.file_a),("BB",r.file_b,r.file_b)]:
            if (r.pair_id,cell) in done:continue
            au,tx,ma,dfeat,dl=cache[dfi];cfeat=cache[cfi][3];cl=cache[cfi][4]
            if cfeat.shape[1]!=dfeat.shape[1]:cfeat=F.interpolate(cfeat.transpose(1,2).float(),size=dfeat.shape[1],mode="linear",align_corners=False).transpose(1,2).to(dfeat.dtype)
            pos=torch.arange(au.shape[1],device="cuda").unsqueeze(0).long();started=time.time()
            with torch.inference_mode():_,logits,_=model.alm.forward(input_ids=au.cuda(),text_input_ids=tx.cuda(),whisper_input_feature=[cfeat.cuda()],is_continuous_mask=ma.cuda(),position_ids=pos,past_key_values=None,return_dict=False)
            lp=torch.log_softmax(logits[0,-1].float(),dim=-1);scores={k:float(lp[v].cpu()) for k,v in ids.items()};pred=max(scores,key=scores.get)
            row={"pair_id":r.pair_id,"speaker":r.speaker,"word":r.word,"label_a":r.label_a,"label_b":r.label_b,"file_a":r.file_a,"file_b":r.file_b,"cell":cell,"discrete_file":dfi,"continuous_file":cfi,"discrete_label":dl,"continuous_label":cl,"prediction":pred,"scores":scores,"contrast_b_minus_a":scores[r.label_b]-scores[r.label_a],"seconds":round(time.time()-started,4)}
            with PRED.open("a",encoding="utf-8") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush()
        if i%100==0:print(f"PAIRS {i}/2400",flush=True)
    x=pd.DataFrame(json.loads(z) for z in PRED.read_text(encoding="utf-8").splitlines());wide=x.pivot(index="pair_id",columns="cell",values="contrast_b_minus_a").reset_index();e=pairs.merge(wide,on="pair_id");e["ME_C"]=.5*((e.AB-e.AA)+(e.BB-e.BA));e["ME_D"]=.5*((e.BA-e.AA)+(e.BB-e.AB));e["interaction"]=e.BB-e.BA-e.AB+e.AA;e.to_csv(OUT/f"{PREFIX}_factorial_effects.csv",index=False)
    c=x[x.cell.isin(["AB","BA"])].copy();c["af"]=((c.cell=="AB")&(c.prediction==c.label_b))|((c.cell=="BA")&(c.prediction==c.label_a));c["df"]=((c.cell=="AB")&(c.prediction==c.label_a))|((c.cell=="BA")&(c.prediction==c.label_b))
    s={"n_pairs":len(e),"ME_C_mean":e.ME_C.mean(),"ME_D_mean":e.ME_D.mean(),"interaction_mean":e.interaction.mean(),"acoustic_follow_rate":c.af.mean(),"discrete_follow_rate":c.df.mean(),"other_rate":(~c.af&~c.df).mean()};(OUT/f"{PREFIX}_factorial_summary.json").write_text(json.dumps(s,indent=2),encoding="utf-8");print(json.dumps(s,indent=2),flush=True)
if __name__=="__main__":main()
