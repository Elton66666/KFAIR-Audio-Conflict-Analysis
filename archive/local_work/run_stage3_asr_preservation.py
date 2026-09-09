import json, re, time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from kimia_infer.api.kimia import KimiAudio

ROOT=Path('/root/autodl-tmp/kfair'); OUT=ROOT/'outputs'
PRED=OUT/'stage3_asr_preservation_predictions.jsonl'; SUMMARY=OUT/'stage3_asr_preservation_summary.csv'

class Adapter(nn.Module):
 def __init__(self,width,rank=8):
  super().__init__(); self.down=nn.Linear(width,rank,bias=False,dtype=torch.float32); self.up=nn.Linear(rank,width,bias=False,dtype=torch.float32)
 def forward(self,x): return self.up(F.gelu(self.down(x.float()))).to(x.dtype)
class LoRALinear(nn.Module):
 def __init__(self,base,rank=8,alpha=8):
  super().__init__(); self.base=base; self.scale=alpha/rank; self.A=nn.Linear(base.in_features,rank,bias=False,dtype=torch.float32); self.B=nn.Linear(rank,base.out_features,bias=False,dtype=torch.float32)
 def forward(self,x): return self.base(x)+self.B(self.A(x.float())).to(x.dtype)*self.scale
def words(s): return re.findall(r'[a-z]+',s.lower())
def dist(a,b):
 p=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  c=[i]
  for j,y in enumerate(b,1): c.append(min(c[-1]+1,p[j]+1,p[j-1]+(x!=y)))
  p=c
 return p[-1]
def main():
 a=pd.read_csv(OUT/'ravdess_expanded_manifest.csv'); a['statement']=1
 b=pd.read_csv(OUT/'ravdess_statement2_manifest.csv'); b['statement']=2
 m=pd.concat([a,b]); m=m[m.actor>=21].copy(); assert len(m)==32
 model=KimiAudio(model_path=str(ROOT/'models/Kimi-Audio-7B-Instruct'),load_detokenizer=False)
 for p in model.alm.parameters(): p.requires_grad_(False)
 adapter=Adapter(model.alm.config.hidden_size).cuda(); holder={}
 attn=model.alm.model.layers[22].self_attn; original=attn.q_proj
 def activate(method):
  if method=='decision_adapter_layer25':
   ck=torch.load(OUT/'adapter_sweep_layer25.pt',map_location='cpu'); adapter.load_state_dict(ck['state_dict']); adapter.eval()
   def hook(mod,args):
    h=args[0].clone(); h[:,-1]=h[:,-1]+adapter(h[:,-1]); return (h,)+args[1:]
   holder['hook']=model.alm.model.layers[25].register_forward_pre_hook(hook)
  elif method=='full_sequence_adapter_layer22':
   ck=torch.load(OUT/'adapter_sweep_layer22_full_sequence.pt',map_location='cpu'); adapter.load_state_dict(ck['state_dict']); adapter.eval()
   def hook(mod,args):
    h=args[0]; return (h+adapter(h),)+args[1:]
   holder['hook']=model.alm.model.layers[22].register_forward_pre_hook(hook)
  elif method=='ordinary_lora_layer22':
   lora=LoRALinear(original).cuda(); ck=torch.load(OUT/'ordinary_lora_layer22_rank8.pt',map_location='cpu')['state_dict']; lora.A.load_state_dict(ck['A']); lora.B.load_state_dict(ck['B']); lora.eval(); holder['lora']=lora; attn.q_proj=lora
 def deactivate(method):
  if 'hook' in holder: holder.pop('hook').remove()
  if method=='ordinary_lora_layer22': attn.q_proj=original; holder.pop('lora',None)
 methods=['baseline','decision_adapter_layer25','full_sequence_adapter_layer22','ordinary_lora_layer22']; rows=[]
 prompt='Transcribe the spoken words exactly. Reply with only the transcript.'
 for method in methods:
  activate(method)
  for i,r in enumerate(m.to_dict('records'),1):
   msg=[{'role':'user','message_type':'text','content':prompt},{'role':'user','message_type':'audio','content':r['audio_path']}]
   t=time.time(); _,hyp=model.generate(msg,output_type='text',max_new_tokens=48,text_temperature=0.0,text_top_k=5)
   ref='kids are talking by the door' if r['statement']==1 else 'dogs are sitting by the door'; e=dist(words(ref),words(hyp))
   row={**r,'method':method,'reference':ref,'transcript':hyp,'edits':e,'ref_words':len(words(ref)),'wer':e/len(words(ref)),'exact_match':words(ref)==words(hyp),'seconds':time.time()-t}; rows.append(row)
   print(f'{method} {i}/32 wer={row["wer"]:.3f}',flush=True)
  deactivate(method)
 PRED.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8'); f=pd.DataFrame(rows)
 out=[]
 for method,g in f.groupby('method',sort=False): out.append({'method':method,'n':len(g),'micro_wer':g.edits.sum()/g.ref_words.sum(),'mean_wer':g.wer.mean(),'exact_match':g.exact_match.mean(),'mean_seconds':g.seconds.mean()})
 pd.DataFrame(out).to_csv(SUMMARY,index=False); print(pd.DataFrame(out).to_string(index=False),flush=True)
if __name__=='__main__': main()
