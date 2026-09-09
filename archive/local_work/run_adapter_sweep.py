import copy
import gc
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from kimia_infer.api.kimia import KimiAudio

ROOT = Path("/root/autodl-tmp/kfair"); OUT = ROOT / "outputs"
SUMMARY = OUT / "adapter_sweep_summary.jsonl"; RAW = OUT / "adapter_sweep_test_predictions.jsonl"
LABELS = ["neutral", "happy", "sad", "angry"]; CHOICES = {"neutral":"A","happy":"B","sad":"C","angry":"D"}
PAIRS = [(a,b) for i,a in enumerate(LABELS) for b in LABELS[i+1:]]
CONFIGS = [
    {"name":"layer12", "layer":12, "seed":20260903}, {"name":"layer18", "layer":18, "seed":20260903},
    {"name":"layer22", "layer":22, "seed":20260903}, {"name":"layer25", "layer":25, "seed":20260903},
    {"name":"layer27", "layer":27, "seed":20260903}, {"name":"random_layer5", "layer":5, "seed":20260903},
    {"name":"layer22_seed04", "layer":22, "seed":20260904}, {"name":"layer22_seed05", "layer":22, "seed":20260905},
    {"name":"layer22_no_pair", "layer":22, "seed":20260903, "pair_weight":0.0},
    {"name":"layer22_no_preserve", "layer":22, "seed":20260903, "preserve_weight":0.0},
    {"name":"layer22_full_sequence", "layer":22, "seed":20260903, "scope":"full_sequence"},
]


class Adapter(nn.Module):
    def __init__(self, width, rank=8):
        super().__init__(); self.down=nn.Linear(width,rank,bias=False,dtype=torch.float32); self.up=nn.Linear(rank,width,bias=False,dtype=torch.float32)
        nn.init.normal_(self.down.weight,std=.01); nn.init.zeros_(self.up.weight)
    def forward(self,x): return self.up(F.gelu(self.down(x.float()))).to(x.dtype)


def examples_from(manifest):
    result=[]
    for (actor,statement),g in manifest.groupby(["actor","statement"]):
        files={r.label:r.filename for r in g.itertuples()}
        for a,b in PAIRS:
            for cell,d,c in (("AA",a,a),("AB",a,b),("BA",b,a),("BB",b,b)):
                result.append({"actor":int(actor),"statement":int(statement),"pair":f"{a}_{b}","cell":cell,"discrete_label":d,"acoustic_label":c,"discrete_file":files[d],"continuous_file":files[c],"conflict":d!=c})
    return result


def key(e): return (e["actor"],e["statement"],e["pair"],e["cell"])


def metric(rows):
    f=pd.DataFrame(rows); f["correct"]=f.prediction.eq(f.acoustic_label); f["semantic"]=f.prediction.eq(f.discrete_label)
    c=f[f.conflict]; a=f[~f.conflict]
    return {"ser":float(f.correct.mean()),"aligned_ser":float(a.correct.mean()),"conflict_acoustic_follow":float(c.correct.mean()),"conflict_semantic_follow":float(c.semantic.mean()),"conflict_margin":float(c.acoustic_margin.mean())}


def main():
    first=pd.read_csv(OUT/"ravdess_expanded_manifest.csv"); first["statement"]=1
    second=pd.read_csv(OUT/"ravdess_statement2_manifest.csv"); second["statement"]=2
    manifest=pd.concat([first,second],ignore_index=True); examples=examples_from(manifest)
    train=[e for e in examples if e["actor"]<=16]; val=[e for e in examples if 17<=e["actor"]<=20]; test=[e for e in examples if e["actor"]>=21]
    model=KimiAudio(model_path=str(ROOT/"models/Kimi-Audio-7B-Instruct"),load_detokenizer=False)
    [m for m in model.alm.modules() if hasattr(m,"vq_adaptor")][0].kfair_branch_mode="full"
    for p in model.alm.parameters(): p.requires_grad_(False)
    prompt="Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
    token_ids=torch.tensor([model.prompt_manager.text_tokenizer.encode(CHOICES[x],bos=False,eos=False)[0] for x in LABELS],device="cuda")
    cache={}
    for i,r in enumerate(manifest.itertuples(),1):
        item=model.prompt_manager.get_prompt([{"role":"user","message_type":"text","content":prompt},{"role":"user","message_type":"audio","content":r.audio_path}],output_type="text")
        au,tx,ma,_,_=item.to_tensor(); cache[r.filename]=(au.cpu(),tx.cpu(),ma.cpu(),item.continuous_feature[0].cpu())
        if i%48==0: print(f"CACHE {i}/192",flush=True)
    def prepare(e):
        au,tx,ma,dfeat=cache[e["discrete_file"]]; feat=cache[e["continuous_file"]][3]
        if feat.shape[1]!=dfeat.shape[1]: feat=F.interpolate(feat.transpose(1,2).float(),size=dfeat.shape[1],mode="linear",align_corners=False).transpose(1,2).to(feat.dtype)
        au,tx,ma,feat=au.cuda(),tx.cuda(),ma.cuda(),feat.cuda(); pos=torch.arange(au.shape[1],device="cuda").unsqueeze(0).long(); return au,tx,ma,feat,pos
    def forward(e):
        au,tx,ma,feat,pos=prepare(e); out=model.alm(input_ids=au,text_input_ids=tx,whisper_input_feature=[feat],is_continuous_mask=ma,position_ids=pos,use_cache=False,return_dict=True)
        return out.logits[1][0,-1].index_select(0,token_ids).float()
    baseline={}
    with torch.inference_mode():
        for i,e in enumerate(examples,1): baseline[key(e)]=forward(e).cpu(); print(f"BASELINE {i}/1152",flush=True) if i%288==0 else None
    def rows_for(items, adapted):
        rows=[]
        with torch.inference_mode():
            for e in items:
                logits=forward(e) if adapted else baseline[key(e)].cuda(); scores={x:float(logits[j].cpu()) for j,x in enumerate(LABELS)}
                rows.append({**e,"prediction":max(scores,key=scores.get),"acoustic_margin":scores[e["acoustic_label"]]-scores[e["discrete_label"]]})
        return rows
    base_test=rows_for(test,False); print("BASE_TEST "+json.dumps(metric(base_test)),flush=True)
    SUMMARY.unlink(missing_ok=True); RAW.unlink(missing_ok=True)
    for ci,cfg0 in enumerate(CONFIGS,1):
        cfg={"pair_weight":.5,"preserve_weight":.2,"scope":"decision_token",**cfg0}; torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"])
        adapter=Adapter(model.alm.config.hidden_size).cuda(); optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-3,weight_decay=1e-4)
        def hook(module,args):
            h=args[0].clone()
            if cfg["scope"]=="full_sequence": h=h+adapter(h)
            else: h[:,-1]=h[:,-1]+adapter(h[:,-1])
            return (h,)+args[1:]
        handle=model.alm.model.layers[cfg["layer"]].register_forward_pre_hook(hook)
        best=None; history=[]; started=time.time()
        for epoch in range(1,5):
            random.shuffle(train); total=0.0
            for e in train:
                logits=forward(e); target=torch.tensor([LABELS.index(e["acoustic_label"])],device="cuda")
                ser=F.cross_entropy(logits.unsqueeze(0),target); pair=ser if e["conflict"] else logits.new_zeros(()); preserve=logits.new_zeros(())
                if not e["conflict"]: preserve=F.kl_div(F.log_softmax(logits,dim=-1),F.softmax(baseline[key(e)].cuda(),dim=-1),reduction="sum")
                loss=ser+cfg["pair_weight"]*pair+cfg["preserve_weight"]*preserve; loss.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True); total+=float(loss.detach().cpu())
            vm=metric(rows_for(val,True)); history.append({"epoch":epoch,"loss":total/len(train),**vm}); score=(vm["ser"],vm["conflict_acoustic_follow"],vm["aligned_ser"])
            if best is None or score>best["score"]: best={"score":score,"epoch":epoch,"state":copy.deepcopy(adapter.state_dict()),"validation":vm}
            print(f"CONFIG {ci}/{len(CONFIGS)} {cfg['name']} epoch={epoch} loss={total/len(train):.4f} val={vm['ser']:.4f}",flush=True)
        adapter.load_state_dict(best["state"]); test_rows=rows_for(test,True); tm=metric(test_rows)
        checkpoint=OUT/f"adapter_sweep_{cfg['name']}.pt"; torch.save({"config":cfg,"best_epoch":best["epoch"],"state_dict":best["state"],"validation":best["validation"],"test":tm},checkpoint)
        record={"config":cfg,"parameters":sum(p.numel() for p in adapter.parameters()),"best_epoch":best["epoch"],"history":history,"validation":best["validation"],"test":tm,"seconds":time.time()-started}
        with SUMMARY.open("a",encoding="utf-8") as f: f.write(json.dumps(record,ensure_ascii=False)+"\n")
        with RAW.open("a",encoding="utf-8") as f:
            for row in test_rows: f.write(json.dumps({"config":cfg["name"],**row},ensure_ascii=False)+"\n")
        handle.remove(); del adapter,optimizer; gc.collect(); torch.cuda.empty_cache(); print("RESULT "+json.dumps(record,ensure_ascii=False),flush=True)
    with RAW.open("a",encoding="utf-8") as f:
        for row in base_test: f.write(json.dumps({"config":"baseline",**row},ensure_ascii=False)+"\n")
    print("DONE",flush=True)


if __name__=="__main__": main()
