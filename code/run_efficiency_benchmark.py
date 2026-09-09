import json, statistics, time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from kimia_infer.api.kimia import KimiAudio

ROOT=Path('/root/autodl-tmp/kfair'); OUT=ROOT/'outputs'
LABELS=['neutral','happy','sad','angry']; CHOICES={'neutral':'A','happy':'B','sad':'C','angry':'D'}

class Adapter(nn.Module):
 def __init__(self,width,rank=8):
  super().__init__(); self.down=nn.Linear(width,rank,bias=False,dtype=torch.float32); self.up=nn.Linear(rank,width,bias=False,dtype=torch.float32)
 def forward(self,x): return self.up(F.gelu(self.down(x.float()))).to(x.dtype)

class LoRALinear(nn.Module):
 def __init__(self,base,rank=8,alpha=8):
  super().__init__(); self.base=base; self.scale=alpha/rank; self.A=nn.Linear(base.in_features,rank,bias=False,dtype=torch.float32); self.B=nn.Linear(rank,base.out_features,bias=False,dtype=torch.float32)
 def forward(self,x): return self.base(x)+self.B(self.A(x.float())).to(x.dtype)*self.scale

def main():
 a=pd.read_csv(OUT/'ravdess_expanded_manifest.csv'); a['statement']=1
 b=pd.read_csv(OUT/'ravdess_statement2_manifest.csv'); b['statement']=2
 m=pd.concat([a,b]); m=m[m.actor>=21]
 pairs=[]
 for (actor,statement),g in m.groupby(['actor','statement']):
  fs={r.label:r.filename for r in g.itertuples()}
  for x,y in [('neutral','happy'),('neutral','sad'),('neutral','angry')]:
   pairs += [(fs[x],fs[x]),(fs[x],fs[y]),(fs[y],fs[x])]
 samples=pairs[:24]
 model=KimiAudio(model_path=str(ROOT/'models/Kimi-Audio-7B-Instruct'),load_detokenizer=False)
 [x for x in model.alm.modules() if hasattr(x,'vq_adaptor')][0].kfair_branch_mode='full'
 for p in model.alm.parameters(): p.requires_grad_(False)
 prompt="Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
 cache={}
 for r in m.itertuples():
  item=model.prompt_manager.get_prompt([{'role':'user','message_type':'text','content':prompt},{'role':'user','message_type':'audio','content':r.audio_path}],output_type='text')
  au,tx,ma,_,_=item.to_tensor(); cache[r.filename]=(au.cpu(),tx.cpu(),ma.cpu(),item.continuous_feature[0].cpu())
 def forward(pair):
  df,cf=pair; au,tx,ma,dfeat=cache[df]; feat=cache[cf][3]
  if feat.shape[1]!=dfeat.shape[1]: feat=F.interpolate(feat.transpose(1,2).float(),size=dfeat.shape[1],mode='linear',align_corners=False).transpose(1,2).to(feat.dtype)
  au,tx,ma,feat=au.cuda(),tx.cuda(),ma.cuda(),feat.cuda(); pos=torch.arange(au.shape[1],device='cuda').unsqueeze(0).long()
  return model.alm(input_ids=au,text_input_ids=tx,whisper_input_feature=[feat],is_continuous_mask=ma,position_ids=pos,use_cache=False,return_dict=True)
 def bench(name,setup=None,cleanup=None):
  if setup: setup()
  with torch.inference_mode(): forward(samples[0]); torch.cuda.synchronize()
  torch.cuda.reset_peak_memory_stats(); times=[]
  with torch.inference_mode():
   for s in samples:
    t=time.perf_counter(); forward(s); torch.cuda.synchronize(); times.append((time.perf_counter()-t)*1000)
  row={'method':name,'n':len(times),'mean_ms':statistics.mean(times),'median_ms':statistics.median(times),'p95_ms':sorted(times)[int(.95*(len(times)-1))],'peak_allocated_gb':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gb':torch.cuda.max_memory_reserved()/2**30}
  if cleanup: cleanup()
  return row
 rows=[bench('baseline')]
 state_dec=torch.load(OUT/'adapter_sweep_layer25.pt',map_location='cpu')['state_dict']; adapter=Adapter(model.alm.config.hidden_size).cuda(); adapter.load_state_dict(state_dec)
 holder={}
 def setup_dec():
  def hook(mod,args):
   h=args[0].clone(); h[:,-1]=h[:,-1]+adapter(h[:,-1]); return (h,)+args[1:]
  holder['h']=model.alm.model.layers[25].register_forward_pre_hook(hook)
 rows.append(bench('decision_adapter_layer25_rank8',setup_dec,lambda:holder.pop('h').remove()))
 state_full=torch.load(OUT/'adapter_sweep_layer22_full_sequence.pt',map_location='cpu')['state_dict']; adapter.load_state_dict(state_full)
 def setup_full():
  def hook(mod,args):
   h=args[0]; return (h+adapter(h),)+args[1:]
  holder['h']=model.alm.model.layers[22].register_forward_pre_hook(hook)
 rows.append(bench('full_sequence_adapter_layer22_rank8',setup_full,lambda:holder.pop('h').remove()))
 attn=model.alm.model.layers[22].self_attn; original=attn.q_proj; lora=LoRALinear(original).cuda(); st=torch.load(OUT/'ordinary_lora_layer22_rank8.pt',map_location='cpu')['state_dict']; lora.A.load_state_dict(st['A']); lora.B.load_state_dict(st['B'])
 rows.append(bench('ordinary_lora_q_layer22_rank8',lambda:setattr(attn,'q_proj',lora),lambda:setattr(attn,'q_proj',original)))
 pd.DataFrame(rows).to_csv(OUT/'efficiency_benchmark.csv',index=False); (OUT/'efficiency_benchmark.json').write_text(json.dumps(rows,indent=2),encoding='utf-8'); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
