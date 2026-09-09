import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from kimia_infer.api.kimia import KimiAudio

ROOT=Path('/root/autodl-tmp/kfair'); OUT=ROOT/'outputs'; AUG=ROOT/'data/stage3_noise'; AUG.mkdir(exist_ok=True)
PRED=OUT/'stage3_noise_robustness_predictions.jsonl'; SUMMARY=OUT/'stage3_noise_robustness_summary.csv'
LABELS=['neutral','happy','sad','angry']; CH={'neutral':'A','happy':'B','sad':'C','angry':'D'}; PAIRS=[(a,b) for i,a in enumerate(LABELS) for b in LABELS[i+1:]]
class Adapter(nn.Module):
 def __init__(self,w,r=8): super().__init__(); self.down=nn.Linear(w,r,bias=False,dtype=torch.float32); self.up=nn.Linear(r,w,bias=False,dtype=torch.float32)
 def forward(self,x): return self.up(F.gelu(self.down(x.float()))).to(x.dtype)
class LoRALinear(nn.Module):
 def __init__(self,b,r=8,a=8): super().__init__(); self.base=b; self.scale=a/r; self.A=nn.Linear(b.in_features,r,bias=False,dtype=torch.float32); self.B=nn.Linear(r,b.out_features,bias=False,dtype=torch.float32)
 def forward(self,x): return self.base(x)+self.B(self.A(x.float())).to(x.dtype)*self.scale
def examples(m):
 out=[]
 for (actor,statement),g in m.groupby(['actor','statement']):
  fs={r.label:r.filename for r in g.itertuples()}
  for a,b in PAIRS:
   for cell,d,c in [('AA',a,a),('AB',a,b),('BA',b,a),('BB',b,b)]: out.append({'actor':int(actor),'statement':int(statement),'pair':f'{a}_{b}','cell':cell,'discrete_label':d,'acoustic_label':c,'discrete_file':fs[d],'continuous_file':fs[c],'conflict':d!=c})
 return out
def main():
 a=pd.read_csv(OUT/'ravdess_expanded_manifest.csv'); a['statement']=1; b=pd.read_csv(OUT/'ravdess_statement2_manifest.csv'); b['statement']=2; m=pd.concat([a,b]); m=m[m.actor>=21].copy(); ex=examples(m); assert len(ex)==192
 paths={r.filename:r.audio_path for r in m.itertuples()}; snrs=['clean',20,10]
 for name,path in list(paths.items()):
  x,sr=sf.read(path,dtype='float32'); rms=np.sqrt(np.mean(x*x)+1e-12)
  for snr in [20,10]:
   rng=np.random.default_rng(int(hashlib.sha256(f'{name}-{snr}'.encode()).hexdigest()[:8],16)); noise=rng.standard_normal(x.shape).astype('float32'); noise*=rms/(10**(snr/20)*np.sqrt(np.mean(noise*noise)+1e-12)); y=np.clip(x+noise,-1,1); p=AUG/f'{snr}db_{name}'; sf.write(p,y,sr,subtype='PCM_16'); paths[(name,snr)]=str(p)
 model=KimiAudio(model_path=str(ROOT/'models/Kimi-Audio-7B-Instruct'),load_detokenizer=False); [x for x in model.alm.modules() if hasattr(x,'vq_adaptor')][0].kfair_branch_mode='full'
 for p in model.alm.parameters(): p.requires_grad_(False)
 prompt="Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."; ids=torch.tensor([model.prompt_manager.text_tokenizer.encode(CH[x],bos=False,eos=False)[0] for x in LABELS],device='cuda'); cache={}
 for si,snr in enumerate(snrs):
  for i,r in enumerate(m.itertuples(),1):
   path=r.audio_path if snr=='clean' else paths[(r.filename,snr)]; item=model.prompt_manager.get_prompt([{'role':'user','message_type':'text','content':prompt},{'role':'user','message_type':'audio','content':path}],output_type='text'); au,tx,ma,_,_=item.to_tensor(); cache[(r.filename,snr)]=(au.cpu(),tx.cpu(),ma.cpu(),item.continuous_feature[0].cpu())
  print(f'CACHE snr={snr}',flush=True)
 def forward(e,snr):
  au,tx,ma,df=cache[(e['discrete_file'],snr)]; feat=cache[(e['continuous_file'],snr)][3]
  if feat.shape[1]!=df.shape[1]: feat=F.interpolate(feat.transpose(1,2).float(),size=df.shape[1],mode='linear',align_corners=False).transpose(1,2).to(feat.dtype)
  au,tx,ma,feat=au.cuda(),tx.cuda(),ma.cuda(),feat.cuda(); pos=torch.arange(au.shape[1],device='cuda').unsqueeze(0).long(); o=model.alm(input_ids=au,text_input_ids=tx,whisper_input_feature=[feat],is_continuous_mask=ma,position_ids=pos,use_cache=False,return_dict=True); return o.logits[1][0,-1].index_select(0,ids).float()
 adapter=Adapter(model.alm.config.hidden_size).cuda(); holder={}; attn=model.alm.model.layers[22].self_attn; original=attn.q_proj
 def on(method):
  if method.startswith('decision'):
   adapter.load_state_dict(torch.load(OUT/'adapter_sweep_layer25.pt',map_location='cpu')['state_dict']); adapter.eval()
   def hook(mod,args): h=args[0].clone(); h[:,-1]=h[:,-1]+adapter(h[:,-1]); return (h,)+args[1:]
   holder['h']=model.alm.model.layers[25].register_forward_pre_hook(hook)
  elif method.startswith('full'):
   adapter.load_state_dict(torch.load(OUT/'adapter_sweep_layer22_full_sequence.pt',map_location='cpu')['state_dict']); adapter.eval()
   def hook(mod,args): h=args[0]; return (h+adapter(h),)+args[1:]
   holder['h']=model.alm.model.layers[22].register_forward_pre_hook(hook)
  elif method.startswith('ordinary'):
   l=LoRALinear(original).cuda(); s=torch.load(OUT/'ordinary_lora_layer22_rank8.pt',map_location='cpu')['state_dict']; l.A.load_state_dict(s['A']); l.B.load_state_dict(s['B']); l.eval(); holder['l']=l; attn.q_proj=l
 def off(method):
  if 'h' in holder: holder.pop('h').remove()
  if method.startswith('ordinary'): attn.q_proj=original; holder.pop('l')
 rows=[]; methods=['baseline','decision_adapter_layer25','full_sequence_adapter_layer22','ordinary_lora_layer22']
 for method in methods:
  on(method)
  with torch.inference_mode():
   for snr in snrs:
    for i,e in enumerate(ex,1):
     z=forward(e,snr); pred=LABELS[int(z.argmax())]; rows.append({**e,'method':method,'snr_db':snr,'prediction':pred,'correct':pred==e['acoustic_label'],'semantic':pred==e['discrete_label'],'margin':float(z[LABELS.index(e['acoustic_label'])]-z[LABELS.index(e['discrete_label'])])})
    print(f'{method} snr={snr} done',flush=True)
  off(method)
 PRED.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8'); f=pd.DataFrame(rows); out=[]
 for (method,snr),g in f.groupby(['method','snr_db'],sort=False):
  c=g[g.conflict]; a=g[~g.conflict]; out.append({'method':method,'snr_db':snr,'n':len(g),'ser':g.correct.mean(),'aligned_ser':a.correct.mean(),'conflict_acoustic_follow':c.correct.mean(),'conflict_semantic_follow':c.semantic.mean(),'conflict_margin':c.margin.mean()})
 pd.DataFrame(out).to_csv(SUMMARY,index=False); print(pd.DataFrame(out).to_string(index=False),flush=True)
if __name__=='__main__': main()
