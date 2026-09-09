import json, gc
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from kimia_infer.api.kimia import KimiAudio

ROOT=Path("/root/autodl-tmp/kfair");OUT=ROOT/"outputs";PRED=OUT/"kfair_decision_token_patching.jsonl"
LABELS=["neutral","happy","sad","angry"];CHOICES={"neutral":"A","happy":"B","sad":"C","angry":"D"}
CORE=[("neutral","angry"),("happy","sad")];LAYERS=[12,18,22,24,25,26,27]

def main():
    a=pd.read_csv(OUT/"ravdess_expanded_manifest.csv");b=pd.read_csv(OUT/"ravdess_statement2_manifest.csv");a["statement"],b["statement"]=1,2
    manifest=pd.concat([a,b],ignore_index=True);pairs=[]
    for (actor,statement),g in manifest.groupby(["actor","statement"]):
        d={r.label:r for r in g.itertuples()}
        for la,lb in CORE:
            pairs.append(dict(pair_id=f"a{actor:02d}_s{statement}_{la}_{lb}",actor=int(actor),statement=int(statement),label_a=la,label_b=lb,file_a=d[la].filename,file_b=d[lb].filename))
    model=KimiAudio(model_path=str(ROOT/"models/Kimi-Audio-7B-Instruct"),load_detokenizer=False)
    [m for m in model.alm.modules() if hasattr(m,"vq_adaptor")][0].kfair_branch_mode="full";pm=model.prompt_manager
    prompt="Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
    ids={k:pm.text_tokenizer.encode(v,bos=False,eos=False)[0] for k,v in CHOICES.items()};cache={}
    for i,r in enumerate(manifest.itertuples(),1):
        h=pm.get_prompt([{"role":"user","message_type":"text","content":prompt},{"role":"user","message_type":"audio","content":r.audio_path}],output_type="text")
        au,tx,ma,_,_=h.to_tensor();cache[r.filename]=(au.cpu(),tx.cpu(),ma.cpu(),h.continuous_feature[0].cpu())
        if i%48==0:print(f"CACHE {i}/192",flush=True)
    def prep(df,cf):
        au,tx,ma,dfeat=cache[df];feat=cache[cf][3]
        if feat.shape[1]!=dfeat.shape[1]:feat=F.interpolate(feat.transpose(1,2).float(),size=dfeat.shape[1],mode="linear",align_corners=False).transpose(1,2).to(feat.dtype)
        au,tx,ma=au.cuda(),tx.cuda(),ma.cuda();pos=torch.arange(au.shape[1],device="cuda").unsqueeze(0).long();return au,tx,ma,feat.cuda(),pos
    def run(df,cf,capture=False,patch=None):
        au,tx,ma,feat,pos=prep(df,cf);states={};handles=[]
        if capture:
            for layer in LAYERS:
                def capture_hook(m,args,L=layer):
                    states[L]=args[0][0,-1].detach().cpu()
                    return None
                handles.append(model.alm.model.layers[layer].register_forward_pre_hook(capture_hook))
        if patch:
            layer,state=patch
            def hook(m,args):
                h=args[0].clone();h[0,-1]=state.to(h.device);return (h,)+args[1:]
            handles.append(model.alm.model.layers[layer].register_forward_pre_hook(hook))
        try:
            with torch.inference_mode():o=model.alm(input_ids=au,text_input_ids=tx,whisper_input_feature=[feat],is_continuous_mask=ma,position_ids=pos,use_cache=False,return_dict=True)
        finally:
            for h in handles:h.remove()
        lp=torch.log_softmax(o.logits[1][0,-1].float(),dim=-1);scores={k:float(lp[v].cpu()) for k,v in ids.items()};return scores,states
    with PRED.open("w",encoding="utf-8") as out:
        for pi,p in enumerate(pairs,1):
            aa,aas=run(p["file_a"],p["file_a"],True);bb,bbs=run(p["file_b"],p["file_b"],True)
            for cell,df,cf in [("AB",p["file_a"],p["file_b"]),("BA",p["file_b"],p["file_a"])]:
                base,_=run(df,cf);al=p["label_b"] if cell=="AB" else p["label_a"];dl=p["label_a"] if cell=="AB" else p["label_b"];bm=base[al]-base[dl]
                out.write(json.dumps({**p,"target_cell":cell,"condition":"no_patch","layer":None,"prediction":max(base,key=base.get),"acoustic_label":al,"discrete_label":dl,"acoustic_margin":bm,"rescue":0.0},ensure_ascii=False)+"\n")
                sources={"acoustic":bbs,"discrete":aas} if cell=="AB" else {"acoustic":aas,"discrete":bbs}
                for layer in LAYERS:
                    for direction,ss in sources.items():
                        scores,_=run(df,cf,patch=(layer,ss[layer]));margin=scores[al]-scores[dl]
                        out.write(json.dumps({**p,"target_cell":cell,"condition":direction,"layer":layer,"prediction":max(scores,key=scores.get),"acoustic_label":al,"discrete_label":dl,"acoustic_margin":margin,"baseline_margin":bm,"rescue":margin-bm},ensure_ascii=False)+"\n")
                out.flush()
            del aas,bbs;gc.collect();torch.cuda.empty_cache()
            if pi%8==0:print(f"PAIRS {pi}/96",flush=True)
    print("DONE",flush=True)
if __name__=="__main__":main()
