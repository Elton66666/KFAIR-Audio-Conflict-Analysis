import copy, json, random, time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from kimia_infer.api.kimia import KimiAudio

ROOT=Path("/root/autodl-tmp/kfair");OUT=ROOT/"outputs";RESULT=OUT/"practical_baselines_summary.json";RAW=OUT/"practical_baselines_predictions.jsonl"
LABELS=["neutral","happy","sad","angry"];CHOICES={"neutral":"A","happy":"B","sad":"C","angry":"D"};PAIRS=[(a,b) for i,a in enumerate(LABELS) for b in LABELS[i+1:]]
PROMPTS=[
"Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D.",
"Ignore the literal meaning of the words. Judge only vocal tone, pitch, rhythm, intensity, and prosody. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D.",
"The words may deliberately conflict with the speaker's emotion. Identify exclusively the emotion expressed by the voice, not the text. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
]

class LoRALinear(nn.Module):
 def __init__(self,base,rank=8,alpha=8):
  super().__init__();self.base=base;self.scale=alpha/rank;self.A=nn.Linear(base.in_features,rank,bias=False,dtype=torch.float32);self.B=nn.Linear(rank,base.out_features,bias=False,dtype=torch.float32);nn.init.normal_(self.A.weight,std=.01);nn.init.zeros_(self.B.weight)
 def forward(self,x):return self.base(x)+self.B(self.A(x.float())).to(x.dtype)*self.scale

def examples(frame):
 out=[]
 for (actor,statement),g in frame.groupby(["actor","statement"]):
  files={r.label:r.filename for r in g.itertuples()}
  for a,b in PAIRS:
   for cell,d,c in (("AA",a,a),("AB",a,b),("BA",b,a),("BB",b,b)):out.append({"actor":int(actor),"statement":int(statement),"pair":f"{a}_{b}","cell":cell,"discrete_label":d,"acoustic_label":c,"discrete_file":files[d],"continuous_file":files[c],"conflict":d!=c})
 return out
def k(e):return(e["actor"],e["statement"],e["pair"],e["cell"])
def metrics(rows):
 f=pd.DataFrame(rows);f["correct"]=f.prediction.eq(f.acoustic_label);f["semantic"]=f.prediction.eq(f.discrete_label);c=f[f.conflict];a=f[~f.conflict]
 return{"ser":float(f.correct.mean()),"aligned_ser":float(a.correct.mean()),"conflict_acoustic_follow":float(c.correct.mean()),"conflict_semantic_follow":float(c.semantic.mean()),"conflict_margin":float(c.acoustic_margin.mean())}

def main():
 first=pd.read_csv(OUT/"ravdess_expanded_manifest.csv");first["statement"]=1;second=pd.read_csv(OUT/"ravdess_statement2_manifest.csv");second["statement"]=2;manifest=pd.concat([first,second],ignore_index=True);ex=examples(manifest);train=[e for e in ex if e["actor"]<=16];val=[e for e in ex if 17<=e["actor"]<=20];test=[e for e in ex if e["actor"]>=21]
 model=KimiAudio(model_path=str(ROOT/"models/Kimi-Audio-7B-Instruct"),load_detokenizer=False);[m for m in model.alm.modules() if hasattr(m,"vq_adaptor")][0].kfair_branch_mode="full"
 for p in model.alm.parameters():p.requires_grad_(False)
 token_ids=torch.tensor([model.prompt_manager.text_tokenizer.encode(CHOICES[x],bos=False,eos=False)[0] for x in LABELS],device="cuda")
 def make_cache(prompt):
  cache={}
  for r in manifest.itertuples():
   item=model.prompt_manager.get_prompt([{"role":"user","message_type":"text","content":prompt},{"role":"user","message_type":"audio","content":r.audio_path}],output_type="text");au,tx,ma,_,_=item.to_tensor();cache[r.filename]=(au.cpu(),tx.cpu(),ma.cpu(),item.continuous_feature[0].cpu())
  return cache
 cache=make_cache(PROMPTS[0])
 def forward(e,scale=1.0,active_cache=None):
  c=active_cache or cache;au,tx,ma,dfeat=c[e["discrete_file"]];feat=c[e["continuous_file"]][3]
  if feat.shape[1]!=dfeat.shape[1]:feat=F.interpolate(feat.transpose(1,2).float(),size=dfeat.shape[1],mode="linear",align_corners=False).transpose(1,2).to(feat.dtype)
  au,tx,ma,feat=au.cuda(),tx.cuda(),ma.cuda(),(feat*scale).cuda();pos=torch.arange(au.shape[1],device="cuda").unsqueeze(0).long();o=model.alm(input_ids=au,text_input_ids=tx,whisper_input_feature=[feat],is_continuous_mask=ma,position_ids=pos,use_cache=False,return_dict=True);return o.logits[1][0,-1].index_select(0,token_ids).float()
 def make_rows(items,fn):
  rows=[]
  with torch.inference_mode():
   for e in items:
    z=fn(e);s={x:float(z[i].cpu()) for i,x in enumerate(LABELS)};rows.append({**e,"prediction":max(s,key=s.get),"acoustic_margin":s[e["acoustic_label"]]-s[e["discrete_label"]]})
  return rows
 base={}
 with torch.inference_mode():
  for i,e in enumerate(ex,1):base[k(e)]=forward(e).cpu();print(f"BASE {i}/1152",flush=True) if i%288==0 else None
 summaries={"baseline":metrics(make_rows(test,lambda e:base[k(e)].cuda()))};allrows=[]
 # Calibration-only: optimize four class biases and temperature on training logits, select L2 on validation.
 best=None
 for l2 in (0.0,.001,.01,.1):
  bias=torch.zeros(4,requires_grad=True);logtemp=torch.zeros((),requires_grad=True);opt=torch.optim.LBFGS([bias,logtemp],max_iter=100)
  X=torch.stack([base[k(e)] for e in train]);y=torch.tensor([LABELS.index(e["acoustic_label"]) for e in train])
  def closure():
   opt.zero_grad();loss=F.cross_entropy((X+bias)/logtemp.exp().clamp(.25,4),y)+l2*bias.square().mean();loss.backward();return loss
  opt.step(closure)
  def cal(e):return(base[k(e)]+bias.detach())/logtemp.detach().exp().clamp(.25,4)
  vm=metrics(make_rows(val,cal));score=(vm["ser"],vm["conflict_acoustic_follow"],vm["aligned_ser"])
  if best is None or score>best[0]:best=(score,l2,bias.detach().clone(),logtemp.detach().clone(),vm)
 _,l2,bias,logtemp,vm=best;cal=lambda e:(base[k(e)]+bias)/logtemp.exp().clamp(.25,4);tr=make_rows(test,cal);summaries["calibration"]={"l2":l2,"temperature":float(logtemp.exp()),"validation":vm,"test":metrics(tr)}
 # Gate-only: choose a scalar multiplier for continuous features on validation actors.
 best=None
 for scale in (0.0,.25,.5,.75,1.0,1.25,1.5,2.0):
  vm=metrics(make_rows(val,lambda e,s=scale:forward(e,s)));score=(vm["ser"],vm["conflict_acoustic_follow"],vm["aligned_ser"]);print(f"GATE {scale} {vm}",flush=True)
  if best is None or score>best[0]:best=(score,scale,vm)
 _,scale,vm=best;tr=make_rows(test,lambda e:forward(e,scale));summaries["gate"]={"scale":scale,"validation":vm,"test":metrics(tr)}
 # Prompt-only: choose among three fixed, label-free instructions on validation actors.
 best=None
 for pi,prompt in enumerate(PROMPTS):
  pc=cache if pi==0 else make_cache(prompt);vm=metrics(make_rows(val,lambda e,c=pc:forward(e,active_cache=c)));score=(vm["ser"],vm["conflict_acoustic_follow"],vm["aligned_ser"]);print(f"PROMPT {pi} {vm}",flush=True)
  if best is None or score>best[0]:best=(score,pi,vm,pc)
 _,pi,vm,pc=best;tr=make_rows(test,lambda e:forward(e,active_cache=pc));summaries["prompt"]={"prompt_index":pi,"prompt":PROMPTS[pi],"validation":vm,"test":metrics(tr)}
 # Ordinary rank-8 LoRA on layer-22 attention q projection.
 random.seed(20260903);torch.manual_seed(20260903);attn=model.alm.model.layers[22].self_attn;original=attn.q_proj;lora=LoRALinear(original).cuda();attn.q_proj=lora;opt=torch.optim.AdamW([*lora.A.parameters(),*lora.B.parameters()],lr=1e-3,weight_decay=1e-4);best=None;history=[]
 for epoch in range(1,5):
  random.shuffle(train);total=0
  for e in train:
   z=forward(e);target=torch.tensor([LABELS.index(e["acoustic_label"])],device="cuda");loss=F.cross_entropy(z.unsqueeze(0),target);loss.backward();torch.nn.utils.clip_grad_norm_([*lora.A.parameters(),*lora.B.parameters()],1);opt.step();opt.zero_grad(set_to_none=True);total+=float(loss.detach().cpu())
  vm=metrics(make_rows(val,forward));history.append({"epoch":epoch,"loss":total/len(train),**vm});score=(vm["ser"],vm["conflict_acoustic_follow"],vm["aligned_ser"]);print(f"LORA {epoch} {vm}",flush=True)
  if best is None or score>best[0]:best=(score,epoch,{"A":copy.deepcopy(lora.A.state_dict()),"B":copy.deepcopy(lora.B.state_dict())},vm)
 _,epoch,state,vm=best;lora.A.load_state_dict(state["A"]);lora.B.load_state_dict(state["B"]);tr=make_rows(test,forward);summaries["lora"]={"parameters":sum(p.numel() for p in lora.A.parameters())+sum(p.numel() for p in lora.B.parameters()),"best_epoch":epoch,"history":history,"validation":vm,"test":metrics(tr)};torch.save({"best_epoch":epoch,"state_dict":state,"summary":summaries["lora"]},OUT/"ordinary_lora_layer22_rank8.pt");attn.q_proj=original
 RESULT.write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
