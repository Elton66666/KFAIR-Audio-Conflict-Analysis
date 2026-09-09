import hashlib, json
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from kimia_infer.api.kimia import KimiAudio

ROOT=Path('/root/autodl-tmp/kfair'); OUT=ROOT/'outputs'; LABELS=['neutral','happy','sad','angry']; CH={'neutral':'A','happy':'B','sad':'C','angry':'D'}
PRED=OUT/'stage3_tess_clean_predictions.jsonl'; SUMMARY=OUT/'stage3_tess_clean_summary.csv'
class Adapter(nn.Module):
 def __init__(self,w,r=8): super().__init__(); self.down=nn.Linear(w,r,bias=False,dtype=torch.float32); self.up=nn.Linear(r,w,bias=False,dtype=torch.float32)
 def forward(self,x): return self.up(F.gelu(self.down(x.float()))).to(x.dtype)
class LoRALinear(nn.Module):
 def __init__(self,b,r=8,a=8): super().__init__(); self.base=b; self.scale=a/r; self.A=nn.Linear(b.in_features,r,bias=False,dtype=torch.float32); self.B=nn.Linear(r,b.out_features,bias=False,dtype=torch.float32)
 def forward(self,x): return self.base(x)+self.B(self.A(x.float())).to(x.dtype)*self.scale
def main():
 m=pd.read_csv(OUT/'tess_factorial_manifest.csv'); m=m[m.label.isin(LABELS)].copy(); m['order']=m.filename.map(lambda x:hashlib.sha256(x.encode()).hexdigest()); m=m.sort_values('order').groupby(['speaker','label']).head(25).reset_index(drop=True); assert len(m)==200
 model=KimiAudio(model_path=str(ROOT/'models/Kimi-Audio-7B-Instruct'),load_detokenizer=False); [x for x in model.alm.modules() if hasattr(x,'vq_adaptor')][0].kfair_branch_mode='full'
 for p in model.alm.parameters(): p.requires_grad_(False)
 prompt="Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
 ids=torch.tensor([model.prompt_manager.text_tokenizer.encode(CH[x],bos=False,eos=False)[0] for x in LABELS],device='cuda'); cache={}
 for i,r in enumerate(m.itertuples(),1):
  item=model.prompt_manager.get_prompt([{'role':'user','message_type':'text','content':prompt},{'role':'user','message_type':'audio','content':r.audio_path}],output_type='text'); cache[r.filename]=item
  if i%50==0: print(f'CACHE {i}/200',flush=True)
 def forward(r):
  item=cache[r['filename']]; au,tx,ma,_,_=item.to_tensor(); au,tx,ma=au.cuda(),tx.cuda(),ma.cuda(); feat=[x.cuda() for x in item.continuous_feature]; pos=torch.arange(au.shape[1],device='cuda').unsqueeze(0).long(); o=model.alm(input_ids=au,text_input_ids=tx,whisper_input_feature=feat,is_continuous_mask=ma,position_ids=pos,use_cache=False,return_dict=True); return o.logits[1][0,-1].index_select(0,ids).float()
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
   for i,r in enumerate(m.to_dict('records'),1):
    z=forward(r); pred=LABELS[int(z.argmax())]; rows.append({**r,'method':method,'prediction':pred,'correct':pred==r['label']});
    if i%50==0: print(f'{method} {i}/200',flush=True)
  off(method)
 PRED.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8'); f=pd.DataFrame(rows); out=[]
 for method,g in f.groupby('method',sort=False):
  rec={'method':method,'n':len(g),'accuracy':g.correct.mean()}; rec.update({f'acc_{x}':g[g.label==x].correct.mean() for x in LABELS}); out.append(rec)
 pd.DataFrame(out).to_csv(SUMMARY,index=False); print(pd.DataFrame(out).to_string(index=False),flush=True)
if __name__=='__main__': main()
